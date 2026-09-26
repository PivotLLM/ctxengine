// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/PivotLLM/ctxengine/logger"
	"github.com/PivotLLM/ctxengine/memory"
	"github.com/PivotLLM/ctxengine/session"
	"github.com/PivotLLM/spawnllm"
)

// Manager implements ContextManager over a SessionStore. The system prompt is
// composed from the Layers the host passes on every Assemble; the manager owns
// the rendered summary, the history and the injections.
type Manager struct {
	// mu serialises every public operation: one Add, Assemble, Compact, Reset
	// or read at a time per session. Held across compaction, model call
	// included, so a concurrent Add or Assemble waits for the pass to finish.
	mu sync.Mutex

	sessionKey string
	store      session.SessionStore
	cfg        managerConfig

	// lastChannel/lastChatID are the conversation of the most recent Assemble,
	// remembered so the automatic compaction path can deliver its report.
	lastChannel string
	lastChatID  string

	// trigger state — in-memory
	msgCount          int  // total messages added since last compression
	compressedAtCount int  // msgCount at last compression (count trigger reset window)
	cooling           bool // true while in the post-compression cooldown window
	coolingSinceCount int  // msgCount when cooling started

	// failure circuit breaker — in-memory, resets on restart. Counts consecutive
	// failed automatic compactions; once it reaches
	// defaultMaxConsecutiveCompactFailures the automatic normal-trigger path is
	// suppressed until msgCount advances past breakerTrippedUntilCount. The
	// safety-net path, a manual /compact (Compact) and the 413-recovery path
	// (ForceCompress) bypass the breaker; see compress for why.
	consecutiveCompactFailures int
	breakerTrippedUntilCount   int // 0 = not tripped

	// ageTriggerFloor is the oldest live timestamp left behind by the last
	// age-triggered pass, and suppresses the age trigger until the window moves
	// past it. In-memory: a restart costs at most one extra pass, and only for a
	// session dormant enough to age out in the first place.
	//
	// Compaction cannot always cut back to the age cap. retainMinMessages keeps
	// the last messages whatever their age, and the last-user clamp keeps the
	// latest user turn even when it is older than the cap — deliberately, since
	// a payload with no user message is a non-retriable 400. A session dormant
	// past trigger.days therefore comes back with an over-age message the pass
	// was never going to remove, and without this the very next message would
	// trigger another pass against the same boundary, and the next, and the next.
	ageTriggerFloor time.Time

	// caller is the host's summarization model; nil disables compaction.
	caller ModelCaller

	// refusedModels tracks summarization models that refused this session's
	// content on content-policy grounds. Such models are skipped on subsequent
	// compactions for this session so we stop sending material to a model that
	// will not handle it. In-memory and per-session: cleared when the
	// ContextManager is rebuilt (config reload / restart). Guarded by refusedMu.
	refusedModels map[string]bool
	refusedMu     sync.Mutex

	// compression outcome tracking
	lastCompressedAt    time.Time
	lastCompressionGain float64
	lastReport          *CompactionReport // report from the most recent doCompress

	// archive is opened lazily on first write; nil until then.
	// archiveMu guards lazy initialisation; read operations do not need it.
	archive   *memory.ArchiveStore
	archiveMu sync.Mutex

	// compressHook is called by compress() when non-nil. Only for testing.
	compressHook func(safetyNet bool)

	// toolDefTokens is the estimated token cost of the tool schemas sent with
	// every request. The agent loop sets it per turn via SetToolDefinitionTokens;
	// the Manager cannot derive it because it never sees the toolset. Counted by
	// contextTokens so the triggers measure the real request, not just history.
	toolDefTokens int

	// builtOverheadTokens is the measured token cost of everything Build() adds
	// on top of raw history — the system prompt, the rendered summary, the
	// injected memory blocks. Recorded by Build() and reused by the history-only
	// trigger paths, which would otherwise ignore it entirely. Zero until the
	// first Build of a session.
	builtOverheadTokens int
}

// New constructs a ContextManager. Options are applied over package defaults.
// Validation order: (a) zero percent → default; (b) safety ≤ normal → WARN;
// (c) min ≥ normal → WARN, set min = normalPercent/2.
func New(
	sessionKey string,
	store session.SessionStore,
	opts ...Option,
) ContextManager {
	cfg := defaultManagerConfig()
	for _, o := range opts {
		o(&cfg)
	}

	// (a) Zero percent thresholds → replace with defaults.
	if cfg.minPercent == 0 {
		cfg.minPercent = defaultMinPercent
	}
	if cfg.normalPercent == 0 {
		cfg.normalPercent = defaultNormalPercent
	}
	if cfg.safetyPercent == 0 {
		cfg.safetyPercent = defaultSafetyPercent
	}

	// (b) safety ≤ normal → WARN only; safety net will never fire.
	if cfg.safetyPercent <= cfg.normalPercent {
		logger.WarnCF("llmcontext", "compress_safety_percent <= compress_normal_percent; safety net will never fire",
			map[string]any{"safety": cfg.safetyPercent, "normal": cfg.normalPercent})
	}

	// (c) min ≥ normal → WARN + clamp to normal/2.
	if cfg.minPercent >= cfg.normalPercent {
		logger.WarnCF("llmcontext", "compress_min_percent >= compress_normal_percent; clamping min to normal/2",
			map[string]any{"min": cfg.minPercent, "normal": cfg.normalPercent})
		cfg.minPercent = cfg.normalPercent / 2
	}

	// (d) retain ≥ min → WARN + clamp to min/2.
	//
	// The retained tail is what a compaction leaves behind, and minPercent is the
	// floor below which compaction refuses to fire. When they are equal every
	// pass shaves back to exactly the floor, the next message crosses it again,
	// and the session pins to the boundary forever — compacting constantly while
	// never getting smaller. The tail must land clearly below the floor so each
	// pass buys real headroom.
	if cfg.retainTokenPercent >= cfg.minPercent {
		logger.WarnCF("llmcontext", "compress retain_token_percent >= trigger min_percent; clamping retain to min/2 (equal values pin the session to the floor and compact on every message)",
			map[string]any{"retain": cfg.retainTokenPercent, "min": cfg.minPercent})
		cfg.retainTokenPercent = cfg.minPercent / 2
	}

	// (e) trigger.days < retain.max_age_days → WARN + clamp trigger up.
	//
	// Firing before anything is old enough to cut produces a pass that summarizes
	// nothing. Equal values are legal but degenerate for the same reason as (d):
	// the tail sits exactly at the trigger age, so it re-fires as soon as the
	// oldest message ages another instant.
	if cfg.triggerDays > 0 && cfg.retainMaxAgeDays > 0 {
		switch {
		case cfg.triggerDays < cfg.retainMaxAgeDays:
			logger.WarnCF("llmcontext", "compress trigger.days < retain.max_age_days; clamping trigger up (it would fire with nothing old enough to summarize)",
				map[string]any{"trigger_days": cfg.triggerDays, "retain_max_age_days": cfg.retainMaxAgeDays})
			cfg.triggerDays = cfg.retainMaxAgeDays
		case cfg.triggerDays == cfg.retainMaxAgeDays:
			logger.WarnCF("llmcontext", "compress trigger.days == retain.max_age_days; the session will re-compact as soon as the tail ages, set trigger.days higher for hysteresis",
				map[string]any{"days": cfg.triggerDays})
		}
	}

	// (f) every retention bound disabled → WARN.
	//
	// Each bound can legitimately be 0: retaining by age alone, or by tokens
	// alone, are both sensible. Turning off all three is not — the tail becomes
	// unbounded and compaction can never shrink the window, which presents as
	// "compaction is broken" rather than as a configuration choice.
	if cfg.retainTokenPercent <= 0 && cfg.retainMaxTokens <= 0 && cfg.retainMaxAgeDays <= 0 {
		logger.WarnCF("llmcontext", "every compression retain bound is disabled; the live window is unbounded and compaction cannot shrink it",
			map[string]any{
				"retain_token_percent": cfg.retainTokenPercent,
				"retain_max_tokens":    cfg.retainMaxTokens,
				"retain_max_age_days":  cfg.retainMaxAgeDays,
			})
	}

	m := &Manager{
		sessionKey: sessionKey,
		store:      store,
		cfg:        cfg,
		caller:     cfg.caller,
	}

	// 9c. Load durable compaction state if the store supports it.
	if cs, ok := store.(CompactionStateStore); ok {
		state, err := cs.GetCompactionState(sessionKey)
		if err == nil {
			m.msgCount = state.MeaningfulCount
			m.compressedAtCount = state.CompressedAtMeaningfulCount
			m.cooling = state.Cooling
			m.coolingSinceCount = state.CoolingSinceCount
			m.lastCompressedAt = state.SummaryGeneratedAt
		}
	}

	return m
}

// classifyRefusal applies the configured refusal classifier, or the built-in
// one when none was set (including on a zero Manager).
func (m *Manager) classifyRefusal(finishReason, content string) (bool, string) {
	if m.cfg.refusalClassifier != nil {
		return m.cfg.refusalClassifier(finishReason, content)
	}
	return defaultRefusalClassifier(finishReason, content)
}

// noiseKey returns the configured NoiseKeyFunc, or the no-op default.
func (m *Manager) noiseKey() NoiseKeyFunc {
	if m.cfg.noiseKey != nil {
		return m.cfg.noiseKey
	}
	return noNoiseKey
}

// noteAdded advances the meaningful-message count after a store write. A
// store that keeps compaction state owns the count, since it is the one that
// decides what is noise (a repeated scheduled fire, an identical reply), so
// the manager reads the count back instead of counting every message itself.
// A store without compaction state counts every message.
func (m *Manager) noteAdded() {
	if cs, ok := m.store.(CompactionStateStore); ok {
		if st, err := cs.GetCompactionState(m.sessionKey); err == nil {
			m.msgCount = st.MeaningfulCount
			return
		}
	}
	m.msgCount++
}

// updateCompactionState rewrites the durable compaction-state fields the
// manager owns and leaves the rest of the record as it is: the meaningful
// count the store keeps, and the host's per-session settings that share the
// record (active model index, reasoning and tool-activity toggles). A store
// without compaction state is a no-op. op names the caller for the log.
func (m *Manager) updateCompactionState(op string, apply func(st *memory.CompactionState)) {
	cs, ok := m.store.(CompactionStateStore)
	if !ok {
		return
	}
	st, err := cs.GetCompactionState(m.sessionKey)
	if err != nil {
		logger.WarnCF("llmcontext", op+": failed to load compaction state", map[string]any{
			"session_key": m.sessionKey,
			"error":       err.Error(),
		})
		return
	}
	apply(&st)
	if err := cs.SetCompactionState(m.sessionKey, st); err != nil {
		logger.WarnCF("llmcontext", op+": failed to persist compaction state", map[string]any{
			"session_key": m.sessionKey,
			"error":       err.Error(),
		})
	}
}

// addMessage writes msg to the store and, once it has a seq, to the archive,
// and advances the message count. A store write failure is returned and
// nothing else happens: an unstored message must not be archived or counted.
func (m *Manager) addMessage(op string, msg spawnllm.Message) (int64, error) {
	seq, err := m.store.AddFullMessage(m.sessionKey, msg)
	if err != nil {
		return 0, fmt.Errorf("llmcontext: %s: %w", op, err)
	}
	m.archiveAppend(seq, msg)
	m.noteAdded()
	return seq, nil
}

func (m *Manager) AddUserMessage(ctx context.Context, msg spawnllm.Message) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	seq, err := m.addMessage("add user message", msg)
	if err != nil {
		return 0, err
	}
	if err := m.triggerCheck(ctx); err != nil {
		// Automatic triggers log and continue — do not block the LLM call.
		logger.WarnCF("llmcontext", "compression error on AddUserMessage (continuing)", map[string]any{
			"session_key": m.sessionKey,
			"error":       err.Error(),
		})
	}
	return seq, nil
}

func (m *Manager) AddAssistantMessage(ctx context.Context, msg spawnllm.Message) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	seq, err := m.addMessage("add assistant message", msg)
	if err != nil {
		return 0, err
	}
	if err := m.triggerCheck(ctx); err != nil {
		// Automatic triggers log and continue — do not block the LLM call.
		logger.WarnCF("llmcontext", "compression error on AddAssistantMessage (continuing)", map[string]any{
			"session_key": m.sessionKey,
			"error":       err.Error(),
		})
	}
	return seq, nil
}

// AddToolCallMessage records the assistant turn containing tool calls.
// Writes to session store and archive and advances the message count.
// Does NOT trigger a compression check — compression is deferred to the next
// Assemble so that the check runs once per dispatch rather than after every
// tool-call message.
func (m *Manager) AddToolCallMessage(_ context.Context, msg spawnllm.Message) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.addMessage("add tool call message", msg)
}

// AddToolResult records a tool result message.
// Writes to session store and archive and advances the message count.
// Does NOT trigger a compression check — compression is deferred to the next
// Assemble.
func (m *Manager) AddToolResult(_ context.Context, msg spawnllm.Message) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.addMessage("add tool result", msg)
}

// Assemble is the single per-dispatch entry point: eviction sweep, emergency
// compaction before and after the build, and injection placement, in that
// order. See ContextManager.Assemble.
//
// The two emergency checks are kept distinct on purpose. The pre-build check
// measures stored history and catches a turn whose tool results have grown
// past the safety line; the post-build check measures the actual request
// (system prompt, summary, injections, tool schemas, completion reserve) and
// catches the case where history alone is under the line but the request is
// not. When the first one compacts, the second is skipped for this call: the
// pass just ran against the same window and would only report nothing to do.
func (m *Manager) Assemble(ctx context.Context, req AssembleRequest) (Assembly, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.setToolDefinitionTokens(req.ToolDefinitionTokens)
	if req.Channel != "" {
		m.lastChannel, m.lastChatID = req.Channel, req.ChatID
	}
	out := Assembly{Evictions: m.sweepEvictions(ctx)}

	if m.emergencyCompactOnHistory(ctx) {
		out.Compacted = true
	}
	msgs, err := m.build(req)
	if err != nil {
		return out, err
	}
	if !out.Compacted && m.emergencyCompactOnBuilt(ctx, msgs) {
		out.Compacted = true
		if msgs, err = m.build(req); err != nil {
			return out, err
		}
	}
	out.Messages = msgs
	return out, nil
}

// emergencyCompactOnHistory runs the safety-net compaction when stored history
// alone is already past the safety line. Returns whether a pass ran and
// succeeded; a failure is logged and the caller proceeds with what it has.
func (m *Manager) emergencyCompactOnHistory(ctx context.Context) bool {
	if m.cfg.contextWindow <= 0 {
		return false
	}
	history := m.store.GetHistory(m.sessionKey)
	if m.contextPercent(history) < float64(m.cfg.safetyPercent) {
		return false
	}
	if err := m.compress(ctx, true); err != nil && !errors.Is(err, ErrNothingToCompress) {
		logger.WarnCF("llmcontext", "pre-build safety compaction failed (continuing)", map[string]any{
			"session_key": m.sessionKey,
			"error":       err.Error(),
		})
		return false
	}
	return true
}

// emergencyCompactOnBuilt runs the safety-net compaction when the built request
// plus the reserve and the tool schemas is past the safety line. Returns
// whether a pass ran and succeeded.
func (m *Manager) emergencyCompactOnBuilt(ctx context.Context, built []spawnllm.Message) bool {
	if m.cfg.contextWindow <= 0 {
		return false
	}
	// `built` already carries the system prompt, summary and injections, so add
	// only what it cannot: the reserve and the tool schemas.
	tokens := m.estTokens(built) + m.cfg.overheadTokens + m.toolDefTokens
	if float64(tokens)*100.0/float64(m.cfg.contextWindow) < float64(m.cfg.safetyPercent) {
		return false
	}
	if err := m.compress(ctx, true); err != nil && !errors.Is(err, ErrNothingToCompress) {
		logger.WarnCF("llmcontext", "post-build safety compaction failed (continuing)", map[string]any{
			"session_key": m.sessionKey,
			"error":       err.Error(),
		})
		return false
	}
	return true
}

// PreDispatchCheck is the history-only emergency check: it compacts when
// stored history is past the safety line and returns a fresh build, else
// current unchanged. Assemble runs the same check; this remains for callers
// and tests that drive the primitives one at a time.
func (m *Manager) PreDispatchCheck(ctx context.Context, current []spawnllm.Message) ([]spawnllm.Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.emergencyCompactOnHistory(ctx) {
		return current, nil
	}
	built, err := m.build(AssembleRequest{})
	if err != nil {
		return current, err
	}
	return built, nil
}

// CheckAndCompress is the built-request emergency check: it compacts when the
// built slice plus reserve and tool schemas is past the safety line and returns
// a fresh build, else built unchanged. Assemble runs the same check.
func (m *Manager) CheckAndCompress(ctx context.Context, built []spawnllm.Message) ([]spawnllm.Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.emergencyCompactOnBuilt(ctx, built) {
		return built, nil
	}
	fresh, err := m.build(AssembleRequest{})
	if err != nil {
		return built, err
	}
	return fresh, nil
}

// estimateTokens estimates token count using the package defaults
// (~4 chars/token, no safety margin). It is retained for callers and tests that
// have no Manager config in scope; Manager methods should use estTokens so the
// configured divisor and safety margin apply.
func estimateTokens(msgs []spawnllm.Message) int {
	return estimateTokensWith(msgs, defaultCharsPerToken, defaultTokenSafetyMargin)
}

// estimateTokensWith estimates token count by dividing the total rune count by
// charsPerToken and inflating by safetyMargin. Non-positive tuning values fall
// back to the defaults.
//
// It counts every field that is actually replayed to the provider on the next
// request:
//
//   - Content.
//   - The JSON-serialized arguments of any ToolCalls, so tool-call turns are not
//     under-counted (a file_write payload lives here, not in Content).
//   - ReasoningContent, which openai_compat serializes back for every historical
//     assistant message. Omitting it understated reasoning-model sessions by
//     roughly half.
//   - ResponsesReasoning, the opaque OpenAI Responses reasoning items replayed
//     before this turn's function_call items.
//
// Attachments are deliberately NOT counted: they are archive-side metadata and
// are never sent to the LLM. Media is counted per item (mediaTokensPerItem) rather than by the
// length of its data: URI, because providers bill images by resolution tiles and
// a base64 payload's rune count overstates that by orders of magnitude.
func estimateTokensWith(msgs []spawnllm.Message, charsPerToken, safetyMargin float64) int {
	if charsPerToken <= 0 {
		charsPerToken = defaultCharsPerToken
	}
	if safetyMargin <= 0 {
		safetyMargin = defaultTokenSafetyMargin
	}
	total := 0
	mediaItems := 0
	for _, m := range msgs {
		total += len([]rune(m.Content))
		total += len([]rune(m.ReasoningContent))
		if len(m.ToolCalls) > 0 {
			if data, err := json.Marshal(m.ToolCalls); err == nil {
				total += len([]rune(string(data)))
			}
		}
		for _, r := range m.ResponsesReasoning {
			total += len([]rune(string(r)))
		}
		mediaItems += len(m.Media)
	}
	return int(float64(total)/charsPerToken*safetyMargin) + mediaItems*mediaTokensPerItem
}

// estTokens estimates token count for msgs using this Manager's configured
// chars-per-token divisor and safety margin.
func (m *Manager) estTokens(msgs []spawnllm.Message) int {
	return estimateTokensWith(msgs, m.cfg.charsPerToken, m.cfg.tokenSafetyMargin)
}

// EstimateToolDefinitionTokens estimates the token cost of a tool-schema set as
// the provider will serialize it. Callers pass the same slice they hand to the
// provider, so the figure tracks the real request rather than a fixed guess.
// Returns 0 on a marshalling failure — an under-count is preferable to a panic
// on a path that runs every turn.
func EstimateToolDefinitionTokens(defs []spawnllm.ToolDefinition) int {
	if len(defs) == 0 {
		return 0
	}
	data, err := json.Marshal(defs)
	if err != nil {
		return 0
	}
	return int(float64(len([]rune(string(data)))) / defaultCharsPerToken)
}

// SetToolDefinitionTokens records the estimated token cost of the tool schemas
// accompanying each request. The agent loop calls this once per turn; the
// Manager has no other way to learn it, and 46 tool definitions are worth tens
// of thousands of tokens. Negative values are ignored.
func (m *Manager) setToolDefinitionTokens(n int) {
	if n < 0 {
		return
	}
	m.toolDefTokens = n
}

// recordBuiltOverhead stores the token cost of everything Build() adds on top of
// raw history (system prompt, rendered summary, memory blocks) so the
// history-only trigger paths can account for it. Called by Build().
func (m *Manager) recordBuiltOverhead(built, history []spawnllm.Message) {
	overhead := m.estTokens(built) - m.estTokens(history)
	if overhead < 0 {
		overhead = 0
	}
	m.builtOverheadTokens = overhead
}

// nonHistoryTokens is everything in a dispatch that is not the stored history:
// the configured reserve (completion budget), the tool schemas, and the measured
// Build() overhead. All three trigger paths add it so they agree on what
// "percent of the context window" means.
func (m *Manager) nonHistoryTokens() int {
	return m.cfg.overheadTokens + m.toolDefTokens + m.builtOverheadTokens
}

// contextTokens estimates the full request cost for a given stored history: the
// history itself plus everything Build() and the provider add around it.
func (m *Manager) contextTokens(history []spawnllm.Message) int {
	return m.estTokens(history) + m.nonHistoryTokens()
}

// contextPercent expresses contextTokens as a percentage of the context window.
// Returns 0 when no window is configured (callers treat that as "no limit").
func (m *Manager) contextPercent(history []spawnllm.Message) float64 {
	if m.cfg.contextWindow <= 0 {
		return 0
	}
	return float64(m.contextTokens(history)) * 100.0 / float64(m.cfg.contextWindow)
}

// oldestAge returns the age of the oldest message in the live window, and false
// when the window is empty or carries no usable timestamp. System messages are
// skipped: a session's system message is rewritten in place and its age says
// nothing about how stale the conversation is.
func oldestAge(stored []memory.StoredMessage, now time.Time) (time.Duration, bool) {
	for _, sm := range stored {
		if sm.Role == "system" || sm.CreatedAt.IsZero() {
			continue
		}
		return now.Sub(sm.CreatedAt), true
	}
	return 0, false
}

// oldestCreatedAt returns the timestamp of the oldest live message, skipping
// system messages for the same reason oldestAge does: a session's system message
// is rewritten in place and its age says nothing about the conversation.
func oldestCreatedAt(stored []memory.StoredMessage) (time.Time, bool) {
	for _, sm := range stored {
		if sm.Role == "system" || sm.CreatedAt.IsZero() {
			continue
		}
		return sm.CreatedAt, true
	}
	return time.Time{}, false
}

// getOrOpenArchive returns the ArchiveStore for this session, opening it lazily
// on first call. Returns nil when archiveDir is empty or the open fails.
func (m *Manager) getOrOpenArchive() *memory.ArchiveStore {
	if m.cfg.archiveDir == "" {
		return nil
	}
	m.archiveMu.Lock()
	defer m.archiveMu.Unlock()
	if m.archive != nil {
		return m.archive
	}
	path := memory.ArchivePath(m.cfg.archiveDir, m.sessionKey)
	store, err := memory.Open(path)
	if err != nil && !errors.Is(err, memory.ErrArchiveUnavailable) {
		logger.WarnCF("llmcontext", "archive open failed", map[string]any{
			"session_key": m.sessionKey,
			"path":        path,
			"error":       err.Error(),
		})
	}
	m.archive = store
	// Apply retention once per manager lifecycle, on the path that actually opens
	// a fresh archive (not on subsequent cached returns), so long-lived growth is
	// trimmed at startup.
	m.pruneArchive(store)
	return m.archive
}

// pruneArchive applies the four retention caps (message count/age and summary
// count/age) to the given archive using the manager's configured limits. Each
// cap is skipped when its configured value is <= 0. Best-effort: errors are
// logged and otherwise ignored so pruning never fails compaction or open.
func (m *Manager) pruneArchive(a *memory.ArchiveStore) {
	if a == nil {
		return
	}

	if n := m.cfg.archiveMessageCount; n > 0 {
		if err := a.PruneMessagesToCount(n); err != nil {
			logger.WarnCF("llmcontext", "archive prune messages to count failed", map[string]any{
				"session_key": m.sessionKey,
				"count":       n,
				"error":       err.Error(),
			})
		}
	}
	if d := m.cfg.archiveDays; d > 0 {
		cutoff := time.Now().AddDate(0, 0, -d)
		if err := a.PruneMessagesBefore(cutoff); err != nil {
			logger.WarnCF("llmcontext", "archive prune messages before failed", map[string]any{
				"session_key": m.sessionKey,
				"days":        d,
				"error":       err.Error(),
			})
		}
	}
	if n := m.cfg.summaryMaxCount; n > 0 {
		if err := a.PruneSummariesToCount(n); err != nil {
			logger.WarnCF("llmcontext", "archive prune summaries to count failed", map[string]any{
				"session_key": m.sessionKey,
				"count":       n,
				"error":       err.Error(),
			})
		}
	}
	if d := m.cfg.summaryRetentionDays; d > 0 {
		cutoff := time.Now().AddDate(0, 0, -d)
		if err := a.PruneSummariesBefore(cutoff); err != nil {
			logger.WarnCF("llmcontext", "archive prune summaries before failed", map[string]any{
				"session_key": m.sessionKey,
				"days":        d,
				"error":       err.Error(),
			})
		}
	}
}

// archiveAppend writes msg to the archive (if one is configured) keyed by the
// MEMORY seq assigned to the message by the session store. Using the memory seq
// (not a separate archive-private counter) keeps a single sequence space across
// the memory store, the archive, and the summaries that cite them: the seq the
// summarizer cites == the seq the strip validates == the seq stored in (and
// retrievable from) the archive.
//
// Cutover note: archives written before this change stored rows under an OLD,
// separate archive-space counter (1, 2, 3, ...). New messages now use the memory
// seq, which is typically far higher on a long-lived session. As a result,
// Bounds() returns a mixed min(legacy_low)..max(memory) range until those old
// rows age out of the retrieval window. This is harmless: the previous bug was
// the archive upper bound being too LOW (so fresh summary refs were stripped);
// now max tracks the memory seq, so fresh refs survive. The legacy low-seq rows
// are inert. No data migration is required.
//
// Best-effort: errors are logged but not returned to callers. A non-positive seq
// (e.g. a store that failed to assign one) is skipped to avoid corrupting the
// archive with seq 0.
func (m *Manager) archiveAppend(seq int64, msg spawnllm.Message) {
	a := m.getOrOpenArchive()
	if a == nil {
		return
	}
	if seq <= 0 {
		logger.WarnCF("llmcontext", "archive append skipped: non-positive seq", map[string]any{
			"session_key": m.sessionKey,
			"seq":         seq,
		})
		return
	}
	msg = archiveTruncateContent(msg, m.archiveContentLimit())
	if err := a.Append(seq, msg, time.Now()); err != nil {
		logger.WarnCF("llmcontext", "archive append failed", map[string]any{
			"session_key": m.sessionKey,
			"seq":         seq,
			"error":       err.Error(),
		})
	}
}

// archiveContentMaxBytes is the default maximum number of content bytes stored
// per message in the archive for TOOL results. Messages whose Content exceeds
// this limit are truncated before writing; the LLM already saw the full content
// in the active context window, so only a compact summary is needed for history.
// Tool results that contain large file payloads are the primary use-case, and
// they are re-retrievable — the file is still on disk.
// Override per-agent via WithArchiveContentMaxBytes.
const archiveContentMaxBytes = 4096

// archiveConversationMaxBytes is the cap applied to user and assistant messages
// instead. Conversation is not re-retrievable and it is what the archive exists
// to preserve — it also feeds cognitive-memory consolidation, which distils
// long-term memory from these rows, so a clipped instruction yields a memory
// built on a fragment. Measured across production archives, user and assistant
// content sits at ~1.2KB and ~2.7KB at the 99th percentile: a 4KB cap sits just
// inside the distribution and clips only the longest, most substantive turns,
// while this cap clears it entirely at a cost of a few hundred KB per archive.
// Tool results keep the tighter cap because they carry the real bulk (a single
// production row measured 5.7MB).
const archiveConversationMaxBytes = 16384

// archiveContentLimit returns the effective per-message archive content cap,
// falling back to archiveContentMaxBytes when unconfigured.
func (m *Manager) archiveContentLimit() int {
	if m.cfg.archiveContentMaxBytes > 0 {
		return m.cfg.archiveContentMaxBytes
	}
	return archiveContentMaxBytes
}

// archiveTruncateContent returns a shallow copy of msg with Content truncated
// to maxBytes if it exceeds that limit. The original msg is not mutated.
func archiveTruncateContent(msg spawnllm.Message, maxBytes int) spawnllm.Message {
	if maxBytes <= 0 {
		maxBytes = archiveContentMaxBytes
	}
	// Conversation gets the larger cap; tool results keep the configured one.
	// An explicit per-agent setting above the conversation cap wins for both,
	// so raising the limit never silently lowers it for user/assistant text.
	if msg.Role != "tool" && maxBytes < archiveConversationMaxBytes {
		maxBytes = archiveConversationMaxBytes
	}
	if len(msg.Content) <= maxBytes {
		return msg
	}
	original := len(msg.Content)
	msg.Content = msg.Content[:maxBytes] +
		fmt.Sprintf("\n[content truncated: %d bytes total, first %d shown]", original, maxBytes)
	return msg
}

// archiveWindow returns the effective [minSeq, maxSeq] range of retrievable
// archive messages for this session, applying the archiveMessageCount cap.
// Returns (0, 0) if no archive exists or the archive is unavailable.
func (m *Manager) archiveWindow() (minSeq, maxSeq int64) {
	a := m.getOrOpenArchive()
	if a == nil {
		return 0, 0
	}
	var err error
	minSeq, maxSeq, err = a.Bounds()
	if err != nil || maxSeq == 0 {
		return 0, 0
	}
	// Cutover note: because the archive is now keyed by the memory seq, Bounds()
	// may return a mixed range whose minSeq is an old archive-space value while
	// maxSeq is a high memory seq. That is expected and harmless — see
	// archiveAppend for the full explanation.
	if m.cfg.archiveMessageCount > 0 {
		floor := maxSeq - int64(m.cfg.archiveMessageCount) + 1
		if floor > minSeq {
			minSeq = floor
		}
	}
	if m.cfg.archiveDays > 0 && m.archive != nil {
		cutoff := time.Now().AddDate(0, 0, -m.cfg.archiveDays)
		if dayFloor, err := m.archive.MinSeqAfter(cutoff); err == nil && dayFloor > minSeq {
			minSeq = dayFloor
		}
	}
	return
}

// compress logs the compression trigger and dispatches to the LLM-based
// compressor. If compressHook is set (tests only), the hook short-circuits
// before doCompress so trigger tests remain independent of LLM behavior.
func (m *Manager) compress(ctx context.Context, safetyNet bool) error {
	// Failure circuit breaker: after repeated automatic-compaction failures,
	// suppress the normal-trigger path until enough new messages accumulate. A
	// manual /compact (Compact) and the 413-recovery path (ForceCompress) bypass
	// this, and so does the safety net: it is the guarantee that the request
	// fits before it reaches the provider, and it has its own drop-only fallback
	// when the model fails, so suppressing it would not save a model call so
	// much as let an oversized request through. Returning nil here for a
	// safety-net pass was exactly that bug: the emergency paths read nil as
	// "compacted" and skipped their fallback. The breaker exists to stop the
	// optional normal path from hammering a failing model; the emergency path
	// is not optional.
	if !safetyNet && m.autoCompactionSuppressed() {
		logger.InfoCF("llmcontext", "automatic compaction suppressed by circuit breaker", map[string]any{
			"session_key":          m.sessionKey,
			"consecutive_failures": m.consecutiveCompactFailures,
			"resume_at_msg_count":  m.breakerTrippedUntilCount,
		})
		return nil
	}

	history := m.store.GetHistory(m.sessionKey)
	logger.InfoCF("llmcontext", "compression triggered", map[string]any{
		"session_key":     m.sessionKey,
		"safety_net":      safetyNet,
		"context_pct":     m.contextPercent(history),
		"history_tokens":  m.estTokens(history),
		"overhead_tokens": m.nonHistoryTokens(),
		"msg_count":       m.msgCount,
	})
	if m.compressHook != nil {
		m.compressedAtCount = m.msgCount
		m.compressHook(safetyNet)
		return nil
	}
	err := m.doCompress(ctx, safetyNet)
	m.recordCompactionOutcome(err)
	// Deliver the report for the automatic path. The manual /compact path returns
	// the report to the caller instead and leaves reportCallback unset here.
	// Skip the "nothing to compress" non-event — auto compaction firing with
	// nothing to do is noise (and produced the bogus "0 messages (0 B)" notice).
	if m.cfg.reportCallback != nil && m.lastReport != nil && m.lastReport.Outcome != "nothing" {
		m.cfg.reportCallback(m.lastChannel, m.lastChatID, m.lastReport.String())
	}
	return err
}

// autoCompactionSuppressed reports whether the failure circuit breaker is
// currently tripped for the automatic compaction path. It clears the breaker
// once enough new messages have accumulated, allowing a fresh attempt.
func (m *Manager) autoCompactionSuppressed() bool {
	if m.breakerTrippedUntilCount == 0 {
		return false
	}
	if m.msgCount >= m.breakerTrippedUntilCount {
		m.breakerTrippedUntilCount = 0
		m.consecutiveCompactFailures = 0
		return false
	}
	return true
}

// recordCompactionOutcome updates the failure circuit breaker from a doCompress
// result. A genuine compression failure increments the consecutive-failure
// counter and trips the breaker at the threshold; success or a benign no-op
// (nothing to compress) resets it. Transient/other errors leave it unchanged.
func (m *Manager) recordCompactionOutcome(err error) {
	switch {
	case err == nil || errors.Is(err, ErrNothingToCompress):
		m.consecutiveCompactFailures = 0
		m.breakerTrippedUntilCount = 0
	case errors.Is(err, ErrCompressionFailed):
		m.consecutiveCompactFailures++
		if m.consecutiveCompactFailures >= defaultMaxConsecutiveCompactFailures && m.breakerTrippedUntilCount == 0 {
			m.breakerTrippedUntilCount = m.msgCount + defaultCompactFailureCooldownMessages
			logger.WarnCF("llmcontext", "compaction circuit breaker tripped; suppressing automatic compaction", map[string]any{
				"session_key":          m.sessionKey,
				"consecutive_failures": m.consecutiveCompactFailures,
				"resume_at_msg_count":  m.breakerTrippedUntilCount,
			})
			if m.cfg.breakerTrippedHook != nil {
				m.cfg.breakerTrippedHook(m.sessionKey, m.consecutiveCompactFailures)
			}
		}
	}
}

// triggerCheck runs the unified compression trigger. Called at the end of
// AddUserMessage and AddAssistantMessage.
func (m *Manager) triggerCheck(ctx context.Context) error {
	if m.cfg.contextWindow <= 0 {
		return nil
	}
	// GetHistoryWithSeqs costs the same as GetHistory (both re-read and re-parse
	// the whole session file) and carries the CreatedAt stamps the age trigger
	// needs, so take the timestamped form and derive the plain slice from it.
	stored := m.store.GetHistoryWithSeqs(m.sessionKey)
	history := storedToPlain(stored)
	contextPct := m.contextPercent(history)

	// Safety net overrides everything, floor included.
	if contextPct >= float64(m.cfg.safetyPercent) {
		return m.compress(ctx, true)
	}

	// Age trigger: fires regardless of how little of the window the history
	// occupies, and so must be evaluated BEFORE the percentage floor. This is the
	// only trigger that reaches a low-volume session — one that never approaches
	// any percentage threshold and whose message count creeps up too slowly to
	// matter will otherwise keep weeks-old messages in the window indefinitely.
	if m.ageTriggered(stored) {
		logger.InfoCF("llmcontext", "compaction age trigger fired", map[string]any{
			"session_key":  m.sessionKey,
			"trigger_days": m.cfg.triggerDays,
			"context_pct":  contextPct,
		})
		err := m.compress(ctx, false)
		m.noteAgeTriggerBoundary()
		return err
	}

	// Floor: no compression regardless of the remaining triggers.
	if contextPct < float64(m.cfg.minPercent) {
		return nil
	}

	// Cooldown suppresses normal and count triggers.
	if m.cooling {
		coolAge := m.msgCount - m.coolingSinceCount
		if coolAge >= defaultCooldownMessages {
			m.cooling = false
		}
	}

	countTriggered := m.cfg.messageThreshold > 0 &&
		(m.msgCount-m.compressedAtCount) >= m.cfg.messageThreshold

	if (contextPct >= float64(m.cfg.normalPercent) || countTriggered) && !m.cooling {
		return m.compress(ctx, false)
	}

	return nil
}

// ageTriggered reports whether the oldest live message has passed triggerDays.
// The cooldown deliberately does not suppress it: cooling is counted in messages
// and a session quiet enough to age out is, by definition, not producing them.
func (m *Manager) ageTriggered(stored []memory.StoredMessage) bool {
	if m.cfg.triggerDays <= 0 {
		return false
	}
	now := time.Now()
	age, ok := oldestAge(stored, now)
	if !ok {
		return false
	}
	if age < time.Duration(m.cfg.triggerDays)*24*time.Hour {
		return false
	}
	// Already compacted against this boundary and it did not move: the message
	// still holding it is one the pass chose to keep, and running again would
	// summarize nothing and keep summarizing nothing on every message that
	// follows. Wait for the window to move past it instead.
	if oldest, ok := oldestCreatedAt(stored); ok &&
		!m.ageTriggerFloor.IsZero() && !oldest.After(m.ageTriggerFloor) {
		return false
	}
	return true
}

// noteAgeTriggerBoundary records where an age-triggered pass left the window, so
// the trigger does not fire again until something older than that has gone.
// Clears the floor once the window is back inside the trigger age, which is the
// ordinary outcome — the floor exists for the pass that could not get there.
func (m *Manager) noteAgeTriggerBoundary() {
	stored := m.store.GetHistoryWithSeqs(m.sessionKey)
	oldest, ok := oldestCreatedAt(stored)
	if !ok {
		m.ageTriggerFloor = time.Time{}
		return
	}
	if time.Since(oldest) < time.Duration(m.cfg.triggerDays)*24*time.Hour {
		m.ageTriggerFloor = time.Time{}
		return
	}
	m.ageTriggerFloor = oldest
	logger.InfoCF("llmcontext", "compaction left a message older than the trigger; suppressing the age trigger until the window moves past it", map[string]any{
		"session_key":  m.sessionKey,
		"trigger_days": m.cfg.triggerDays,
		"oldest_age":   time.Since(oldest).Round(time.Hour).String(),
	})
}

// SetTestCompressHook sets a hook function that is called whenever compress()
// fires. Only for use in tests.
func (m *Manager) SetTestCompressHook(fn func(safetyNet bool)) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.compressHook = fn
}

// Build assembles the request with no layers or injections. Assemble is the
// per-dispatch entry point; Build remains for callers and tests that want the
// bare slice (summary block, if any, plus the sanitised history).
func (m *Manager) Build(_ context.Context) ([]spawnllm.Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.build(AssembleRequest{})
}

// systemSeparator joins the blocks of the system message. Every block — a
// host layer, the summary, a stable injection — is separated by it, so the
// message reads as a sequence of sections.
const systemSeparator = "\n\n---\n\n"

// summaryBlock returns the rendered summary section for the system message:
// the stored summary rendered as Markdown, or a minimal archive-bounds note
// when there is no summary yet but the archive holds rows (so the agent always
// knows the archive exists and which seq range is queryable). "" when there is
// neither.
func (m *Manager) summaryBlock() string {
	archiveMin, archiveMax := m.archiveWindow()
	rendered := renderSummaryFromRaw(m.store.GetSummary(m.sessionKey), archiveMin, archiveMax)
	if rendered == "" && archiveMax > 0 {
		rendered = fmt.Sprintf(
			"## Session Archive\n\nMessages #%d–#%d are stored in the archive. "+
				"Use `mcp__claw__get_session_messages` with `seq_start`/`seq_end` to retrieve them, "+
				"or `mcp__claw__search_session_messages` to search by keyword.",
			archiveMin, archiveMax)
	}
	if rendered == "" {
		return ""
	}
	return "CONTEXT_SUMMARY: The following is an approximate summary of prior conversation " +
		"for reference only. It may be incomplete or outdated — always defer to explicit instructions.\n\n" +
		rendered
}

// composeSystem joins the system message from its parts, in order of
// increasing volatility: the layers placed before the summary, the summary
// block, the layers placed after it, then the stable injections. Empty parts
// are skipped. Ordering is load-bearing: every HTTP provider caches by
// longest-common-prefix, and this message precedes the whole history.
func composeSystem(layers []Layer, summary string, injections []Injection) string {
	parts := make([]string, 0, len(layers)+len(injections)+1)
	for _, l := range layers {
		if !l.AfterSummary && l.Text != "" {
			parts = append(parts, l.Text)
		}
	}
	if summary != "" {
		parts = append(parts, summary)
	}
	for _, l := range layers {
		if l.AfterSummary && l.Text != "" {
			parts = append(parts, l.Text)
		}
	}
	for _, inj := range injections {
		if inj.Placement == PlaceSystemStable && inj.Text != "" {
			parts = append(parts, inj.Text)
		}
	}
	return strings.Join(parts, systemSeparator)
}

// build assembles the full message slice: one system message (the host's
// layers around the rendered summary, then the stable injections), the
// sanitised history, and the per-turn injections on the latest user message.
// Everything system-side is a single message for provider compatibility: the
// Anthropic adapter maps messages[0] to the top-level system parameter and
// Codex maps only the first system message to its instructions field.
func (m *Manager) build(req AssembleRequest) ([]spawnllm.Message, error) {
	history := m.store.GetHistory(m.sessionKey)
	system := composeSystem(req.Layers, m.summaryBlock(), req.Injections)

	msgs := make([]spawnllm.Message, 0, len(history)+1)
	if system != "" {
		msgs = append(msgs, spawnllm.Message{Role: "system", Content: system})
	}
	msgs = append(msgs, sanitizeHistoryForProvider(history)...)

	// Per-turn content rides on the latest user message so it never sits ahead
	// of the history.
	for _, inj := range req.Injections {
		if inj.Placement == PlaceCurrentUser && inj.Text != "" {
			attachRoutedMemory(msgs, inj.Text)
		}
	}

	logger.DebugCF("llmcontext", "request assembled", map[string]any{
		"session_key":  m.sessionKey,
		"layers":       len(req.Layers),
		"system_chars": len(system),
		"history":      len(history),
		"messages":     len(msgs),
	})

	// Record what this build added on top of raw history so the history-only
	// trigger paths can charge for it. Build is read-only with respect to the
	// store; this is in-memory bookkeeping only.
	m.recordBuiltOverhead(msgs, history)

	return msgs, nil
}

// attachRoutedMemory folds the per-turn memory block into the LAST user message
// of the built slice, in place.
//
// It is deliberately not its own message: a trailing user block would put two
// user turns back to back (which some providers merge or reject), and a trailing
// system message is not accepted by every adapter. Folding it into the existing
// turn sidesteps both and keeps the block fixed for every iteration of the turn,
// so within-turn caching still works.
//
// The mutation is safe and must stay confined to the built slice: msgs comes
// from GetHistory, which returns a copy, and nothing writes it back. If this
// block ever reached the store, history would accumulate one stale memory dump
// per turn — silently and cumulatively.
//
// With no user message (a turn that opens on tool plumbing) the block is
// dropped rather than forced somewhere invalid; the next user turn re-routes it.
func attachRoutedMemory(msgs []spawnllm.Message, routed string) {
	if routed == "" {
		return
	}
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != "user" {
			continue
		}
		msgs[i].Content += systemSeparator + routed
		return
	}
}

// Compact triggers a normal LLM-based compression pass, identical to what
// runs when the regular compression threshold is crossed. It is the manual
// /compact entry point and bypasses the failure circuit breaker, but a
// successful manual compaction still resets the breaker so the automatic path
// resumes.
func (m *Manager) Compact(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	err := m.doCompress(ctx, false)
	m.recordCompactionOutcome(err)
	return err
}

// RenderedSummary returns the current session summary rendered as Markdown
// (the same block Build() injects into the system prompt), or "" when there is
// no summary. Used by session_compact to show the agent what was just preserved.
func (m *Manager) RenderedSummary() string {
	m.mu.Lock()
	defer m.mu.Unlock()

	archiveMin, archiveMax := m.archiveWindow()
	return renderSummaryFromRaw(m.store.GetSummary(m.sessionKey), archiveMin, archiveMax)
}

// LastCompactionReport returns the report produced by the most recent
// compaction pass, or nil if none has run on this manager.
func (m *Manager) LastCompactionReport() *CompactionReport {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.lastReport
}

// splitSystemMessage separates a stored system message at the head of the
// window from the conversation that follows it.
func splitSystemMessage(stored []memory.StoredMessage) (*memory.StoredMessage, []memory.StoredMessage) {
	if len(stored) > 0 && stored[0].Role == "system" {
		sys := stored[0]
		return &sys, stored[1:]
	}
	return nil, stored
}

// ForceCompress is the host's recovery when a provider rejects the request as
// too large. It measures the request the way Assemble does — stored history
// plus the reserve, the tool schemas and the measured build overhead — and,
// when that is past the safety line, runs the safety-net pass: a summary
// through the model when one is configured, then the drop-only fallback that
// removes the oldest turn groups whole until the request fits. Without a
// model it goes straight to the drops and the existing summary stays as it
// is. It bypasses the failure circuit breaker, and the retained messages keep
// their seqs. Returns ErrCompressionFailed when the request is still over the
// line afterwards, which means the newest turn group alone does not fit.
func (m *Manager) ForceCompress(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.cfg.contextWindow <= 0 {
		return nil
	}
	sysMsg, conv := splitSystemMessage(m.store.GetHistoryWithSeqs(m.sessionKey))
	if len(conv) == 0 {
		return nil
	}
	if m.contextPercent(storedToPlain(conv)) < float64(m.cfg.safetyPercent) {
		return nil
	}

	before := len(conv)
	if m.caller != nil {
		err := m.doCompress(ctx, true)
		m.recordCompactionOutcome(err)
		if err != nil && !errors.Is(err, ErrCompressionPartial) {
			return fmt.Errorf("force compress: %w", err)
		}
	} else {
		conv = m.dropOldestStoredGroups(ctx, conv)
		m.applyLargeMsgChecksStored(conv)
		if err := m.persistStoredResult(sysMsg, conv, nil); err != nil {
			return fmt.Errorf("force compress: %w", err)
		}
	}
	m.compressedAtCount = m.msgCount

	_, after := splitSystemMessage(m.store.GetHistoryWithSeqs(m.sessionKey))
	logger.WarnCF("llmcontext", "force compression executed", map[string]any{
		"session_key":  m.sessionKey,
		"summarized":   m.caller != nil,
		"removed_msgs": before - len(after),
		"new_count":    len(after),
	})
	if m.contextPercent(storedToPlain(after)) >= float64(m.cfg.safetyPercent) {
		return fmt.Errorf("%w: current turn group alone exceeds context window (%d tokens)",
			ErrCompressionFailed, m.cfg.contextWindow)
	}
	return nil
}

func (m *Manager) Stats() ContextStats {
	m.mu.Lock()
	defer m.mu.Unlock()

	history := m.store.GetHistory(m.sessionKey)
	tokens := m.estTokens(history)
	pct := 0.0
	if m.cfg.contextWindow > 0 {
		pct = float64(tokens) * 100.0 / float64(m.cfg.contextWindow)
	}
	// Estimate summary token count from the raw stored summary string using the
	// configured chars-per-token divisor (no safety margin — this is a stat).
	summaryTokens := 0
	rawSummary := m.store.GetSummary(m.sessionKey)
	if rawSummary != "" {
		cpt := m.cfg.charsPerToken
		if cpt <= 0 {
			cpt = defaultCharsPerToken
		}
		summaryTokens = int(float64(len([]rune(rawSummary))) / cpt)
	}
	return ContextStats{
		TotalMessages:       len(history),
		MeaningfulMessages:  m.msgCount,
		EstimatedTokens:     tokens,
		ContextWindowPct:    pct,
		LastCompressedAt:    m.lastCompressedAt,
		LastCompressionGain: m.lastCompressionGain,
		CompressionCooling:  m.cooling,
		CoolingSinceCount:   m.coolingSinceCount,
		SummaryTokens:       summaryTokens,
	}
}

// Close flushes durable compaction state and closes the archive connection.
// After Close the manager must not be used; it is called by the eviction
// goroutine and during AgentLoop shutdown.
//
// Close is safe to call on a manager that has never opened an archive.
func (m *Manager) Close(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Flush the compaction state the manager owns so a new manager created
	// for the same session restores the trigger window and cooldown.
	m.updateCompactionState("Close", func(st *memory.CompactionState) {
		st.CompressedAtMeaningfulCount = m.compressedAtCount
		st.Cooling = m.cooling
		st.CoolingSinceCount = m.coolingSinceCount
		st.SummaryGeneratedAt = m.lastCompressedAt
	})
	if _, ok := m.store.(CompactionStateStore); ok {
		if err := m.store.Save(m.sessionKey); err != nil {
			logger.WarnCF("llmcontext", "Close: failed to save session", map[string]any{
				"session_key": m.sessionKey,
				"error":       err.Error(),
			})
		}
	}

	// Close the archive connection.
	m.archiveMu.Lock()
	defer m.archiveMu.Unlock()
	if m.archive != nil {
		if err := m.archive.Close(); err != nil {
			logger.WarnCF("llmcontext", "Close: archive close failed", map[string]any{
				"session_key": m.sessionKey,
				"error":       err.Error(),
			})
		}
		m.archive = nil
	}
	return nil
}

// Reset clears the active conversation — history window, current rolling
// summary, and in-memory compression state — but PRESERVES the durable archive
// (long-term memory) and the summary log. After Reset the session starts a fresh
// conversation while retaining full recall via session_messages and
// session_summary_*. A hard wipe (erase long-term memory) is done by deleting the
// per-session .archive.db file manually; there is no destructive clear.
func (m *Manager) Reset(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// 1. Clear in-memory compression state.
	m.msgCount = 0
	m.compressedAtCount = 0
	m.cooling = false
	m.coolingSinceCount = 0
	m.lastCompressedAt = time.Time{}
	m.lastCompressionGain = 0

	// 2. Clear the pending-turn flag so no stale flag survives the reset.
	if err := m.store.ClearPendingTurn(m.sessionKey); err != nil {
		logger.WarnCF("llmcontext", "Reset: ClearPendingTurn failed", map[string]any{
			"session_key": m.sessionKey,
			"error":       err.Error(),
		})
	}

	// 3. Wipe the active history window and the current rolling summary. The
	// archive (keyed by memory seq) and the summary log are intentionally left
	// intact — the agent keeps its long-term memory across a clear; new messages
	// continue under the next memory seq the store assigns.
	if err := m.store.TruncateHistory(m.sessionKey, 0); err != nil {
		return fmt.Errorf("llmcontext: reset: truncate history: %w", err)
	}
	if err := m.store.SetSummary(m.sessionKey, ""); err != nil {
		return fmt.Errorf("llmcontext: reset: clear summary: %w", err)
	}

	// 4. Zero the compaction counters in the durable state. The host's
	// per-session settings in the same record survive a clear.
	m.updateCompactionState("Reset", func(st *memory.CompactionState) {
		st.MeaningfulCount = 0
		st.CompressedAtMeaningfulCount = 0
		st.Cooling = false
		st.CoolingSinceCount = 0
		st.SummaryGeneratedAt = time.Time{}
		st.SummaryModel = ""
	})

	return m.store.Save(m.sessionKey)
}

// SetToolDefinitionTokens records the token cost of the tool schemas the host
// sends with every request, so the built-request checks can account for it.
func (m *Manager) SetToolDefinitionTokens(n int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.setToolDefinitionTokens(n)
}
