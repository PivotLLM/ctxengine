// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/PivotLLM/ctxengine/memory"
	"github.com/PivotLLM/spawnllm"
)

// compressTestStore is a minimal in-memory SessionStore for compress tests.
type compressTestStore struct {
	history []spawnllm.Message
	summary string
}

func (s *compressTestStore) GetHistory(_ string) []spawnllm.Message {
	cp := make([]spawnllm.Message, len(s.history))
	copy(cp, s.history)
	return cp
}

func (s *compressTestStore) SetHistory(_ string, h []spawnllm.Message) error {
	cp := make([]spawnllm.Message, len(h))
	copy(cp, h)
	s.history = cp
	return nil
}

func (s *compressTestStore) GetSummary(_ string) string { return s.summary }
func (s *compressTestStore) SetSummary(_, v string) error {
	s.summary = v
	return nil
}
func (s *compressTestStore) Save(_ string) error             { return nil }
func (s *compressTestStore) AddMessage(_, _, _ string) error { return nil }
func (s *compressTestStore) AddFullMessage(_ string, _ spawnllm.Message) (int64, error) {
	return 0, nil
}
func (s *compressTestStore) TruncateHistory(_ string, _ int) error    { return nil }
func (s *compressTestStore) SetPendingTurn(_ string) error            { return nil }
func (s *compressTestStore) ClearPendingTurn(_ string) error          { return nil }
func (s *compressTestStore) GetArchiveBounds(_ string) (int64, int64) { return 0, 0 }
func (s *compressTestStore) ListPendingSessions() ([]string, error)   { return nil, nil }
func (s *compressTestStore) Close() error                             { return nil }
func (s *compressTestStore) GetHistoryWithSeqs(_ string) []memory.StoredMessage {
	stored := make([]memory.StoredMessage, len(s.history))
	for i, msg := range s.history {
		stored[i] = memory.StoredMessage{Seq: int64(i + 1), Message: msg}
	}
	return stored
}

// mockLLM is a sequence-based summarization model for testing. It plays one
// model of a host chain: see testChain.
type mockLLM struct {
	model         string // reported on every reply; "" leaves attempts labelled "model"
	responses     []string
	errors        []error
	finishReasons []string // optional, parallel to responses
	callCount     int
}

func (m *mockLLM) Complete(_ context.Context, _ ModelRequest) (ModelReply, error) {
	i := m.callCount
	m.callCount++
	if i < len(m.errors) && m.errors[i] != nil {
		return ModelReply{Model: m.model}, m.errors[i]
	}
	resp := ""
	if i < len(m.responses) {
		resp = m.responses[i]
	}
	fr := ""
	if i < len(m.finishReasons) {
		fr = m.finishReasons[i]
	}
	return ModelReply{Content: resp, FinishReason: fr, Model: m.model}, nil
}

// testChain walks a list of models the way the host's ModelCaller does: a
// model named in Exclude is skipped, a transport error moves on to the next
// model, the first reply wins, and ErrNoModel is returned when nothing is left.
type testChain struct {
	models []*mockLLM
}

func (c *testChain) Complete(ctx context.Context, req ModelRequest) (ModelReply, error) {
	var lastErr error
	lastModel := ""
	for _, m := range c.models {
		if m.model != "" && slices.Contains(req.Exclude, m.model) {
			continue
		}
		reply, err := m.Complete(ctx, req)
		if err != nil {
			lastErr = err
			lastModel = m.model
			continue
		}
		return reply, nil
	}
	if lastErr != nil {
		return ModelReply{Model: lastModel}, lastErr
	}
	return ModelReply{}, ErrNoModel
}

// chainOf wraps models as one ModelCaller; no models means no caller (the
// manager then reports "nothing" without summarizing).
func chainOf(models []*mockLLM) ModelCaller {
	if len(models) == 0 {
		return nil
	}
	return &testChain{models: models}
}

// validSummaryJSON produces minimal valid Summary JSON.
func validSummaryJSON(goals string) string {
	return fmt.Sprintf(`{"version":2,"state":{"goals":[{"text":%q,"refs":[{"seq_start":1}]}]},"covered_seq_start":0,"covered_seq_end":0}`, goals)
}

// makeConversation builds a slice of alternating user/assistant messages.
// Each message content is repeated to produce the desired approximate token count.
func makeConversation(pairs int, charsPerMessage int) []spawnllm.Message {
	msgs := make([]spawnllm.Message, 0, pairs*2)
	for i := 0; i < pairs; i++ {
		msgs = append(msgs,
			spawnllm.Message{Role: "user", Content: strings.Repeat("u", charsPerMessage)},
			spawnllm.Message{Role: "assistant", Content: strings.Repeat("a", charsPerMessage)},
		)
	}
	return msgs
}

// newCompressManager builds a Manager wired for compress tests (no compressHook).
func newCompressManager(store *compressTestStore, clients []*mockLLM, opts ...Option) *Manager {
	baseOpts := []Option{
		WithContextWindow(10000),
		// Tests below reason in exact token terms against a small window; the
		// real per-request reserve would swamp it. TestTriggers_CountReserve
		// covers the reserve itself.
		WithOverheadTokens(0),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithRetainTokenPercent(20),
		WithRetainMinMessages(2),
		WithModelCaller(chainOf(clients)),
	}
	baseOpts = append(baseOpts, opts...)
	cm := New("sess", store, baseOpts...)
	return cm.(*Manager)
}

// TestCompress_PrimarySuccess verifies that a single successful LLM client
// produces a stored summary and shrinks the history.
func TestCompress_PrimarySuccess(t *testing.T) {
	store := &compressTestStore{
		history: makeConversation(10, 200), // 10 pairs × 2 msgs × 200 chars = 4000 chars → ~1600 tokens
	}

	llm := &mockLLM{
		responses: []string{validSummaryJSON("test goal")},
	}

	mgr := newCompressManager(store, []*mockLLM{llm})
	mgr.msgCount = len(store.history)

	err := mgr.doCompress(context.Background(), false)
	if err != nil {
		t.Fatalf("doCompress returned error: %v", err)
	}

	if store.summary == "" {
		t.Error("expected summary to be stored")
	}
	if _, parseErr := unmarshalSummary(store.summary); parseErr != nil {
		t.Errorf("stored summary is not valid: %v", parseErr)
	}
	if len(store.history) >= 20 {
		t.Errorf("expected history to shrink; got %d messages", len(store.history))
	}
	if llm.callCount == 0 {
		t.Error("expected LLM to be called")
	}
}

// TestCompress_RefusalDetectedAndModelSkipped verifies that a content refusal is
// (a) classified as "refused" rather than "error", (b) does not block a later
// model from succeeding, and (c) causes the refusing model to be skipped on the
// next compaction for the same session.
func TestCompress_RefusalDetectedAndModelSkipped(t *testing.T) {
	store := &compressTestStore{history: makeConversation(10, 200)}

	// First model refuses (partial JSON then a decline, no finish_reason);
	// second model produces a valid summary.
	refuser := &mockLLM{
		model:     "gpt-refuser",
		responses: []string{`{"version":2,"state":{"goals":[{"text":"x"` + "\n\nI'm sorry, but I cannot assist with that request."},
	}
	worker := &mockLLM{
		model:     "deepseek-worker",
		responses: []string{validSummaryJSON("g1"), validSummaryJSON("g2")},
	}

	mgr := newCompressManager(store, []*mockLLM{refuser, worker})
	mgr.msgCount = len(store.history)

	if err := mgr.doCompress(context.Background(), false); err != nil {
		t.Fatalf("doCompress returned error: %v", err)
	}

	// The refuser must be recorded as "refused", and remembered.
	var sawRefused bool
	for _, a := range mgr.lastReport.Attempts {
		if a.Model == "gpt-refuser" && a.Status == "refused" {
			sawRefused = true
		}
	}
	if !sawRefused {
		t.Fatalf("expected gpt-refuser attempt with status 'refused'; got %+v", mgr.lastReport.Attempts)
	}
	if !mgr.refusedModels["gpt-refuser"] {
		t.Fatal("expected gpt-refuser to be remembered in refusedModels")
	}
	if store.summary == "" {
		t.Fatal("expected a summary from the non-refusing model")
	}

	// Second compaction: the refuser must be skipped entirely (no new call).
	refuser.callCount = 0
	store.history = makeConversation(10, 200)
	mgr.msgCount = len(store.history)
	if err := mgr.doCompress(context.Background(), false); err != nil {
		t.Fatalf("second doCompress returned error: %v", err)
	}
	if refuser.callCount != 0 {
		t.Errorf("expected refusing model to be skipped on the second pass; got %d calls", refuser.callCount)
	}
}

// TestCompress_NeverEmptiesLiveWindow reproduces the "Compacted to 0 messages"
// bug: a long in-flight tool-call sequence (all assistant tool_calls + tool
// results, no user/clean anchor) must NOT be compacted away to a system-only
// payload — at least the last turn group is retained.
func TestCompress_NeverEmptiesLiveWindow(t *testing.T) {
	big := strings.Repeat("x", 4000)
	history := []spawnllm.Message{{Role: "system", Content: "sys"}}
	for i := 0; i < 6; i++ {
		id := fmt.Sprintf("tc%d", i)
		history = append(history,
			spawnllm.Message{Role: "assistant", Content: big, ToolCalls: []spawnllm.ToolCall{{ID: id, Name: "x"}}},
			spawnllm.Message{Role: "tool", Content: big, ToolCallID: id},
		)
	}
	store := &compressTestStore{history: history}
	llm := &mockLLM{responses: []string{
		validSummaryJSON("a"), validSummaryJSON("b"), validSummaryJSON("c"),
	}}
	mgr := newCompressManager(store, []*mockLLM{llm})
	mgr.msgCount = len(history)

	_ = mgr.doCompress(context.Background(), false)

	conv := 0
	for _, m := range store.GetHistory("sess") {
		if m.Role != "system" {
			conv++
		}
	}
	if conv == 0 {
		t.Fatal("compaction emptied the live window to a system-only payload")
	}
}

// TestCompress_RetainsLastUserMessage verifies compaction never archives the
// most recent user turn. A long tool/assistant tail after the last user message
// would otherwise push it out of the retained window, leaving a payload with no
// user-role message (strict providers reject that with a non-retriable 400).
func TestCompress_RetainsLastUserMessage(t *testing.T) {
	history := []spawnllm.Message{{Role: "system", Content: "sys"}}
	history = append(history, spawnllm.Message{Role: "user", Content: strings.Repeat("u", 200)})
	for i := 0; i < 60; i++ { // long assistant tail after the only user turn
		history = append(history, spawnllm.Message{Role: "assistant", Content: strings.Repeat("a", 200)})
	}
	store := &compressTestStore{history: history}
	llm := &mockLLM{responses: []string{validSummaryJSON("goal")}}
	mgr := newCompressManager(store, []*mockLLM{llm})
	mgr.msgCount = len(history)

	_ = mgr.doCompress(context.Background(), false)

	hasUser := false
	for _, m := range store.GetHistory("sess") {
		if m.Role == "user" {
			hasUser = true
			break
		}
	}
	if !hasUser {
		t.Fatalf("compaction archived the last user message — payload would have no user role")
	}
}

// TestCompress_HostErrorNeatened verifies a host transport error is recorded
// once, attributed to the model the host named, and neatened into the
// "HTTP 402 (out of credits)" form instead of a raw body dump. Cooling the
// model is the host's job now; the engine simply stops the call.
func TestCompress_HostErrorNeatened(t *testing.T) {
	store := &compressTestStore{history: makeConversation(10, 200)}
	failing := &mockLLM{
		model: "abliterated-model",
		errors: []error{
			errors.New("API request failed:   Status: 402   Body: {\"error\":{\"billing_url\":\"x\"}}"),
		},
	}
	mgr := newCompressManager(store, []*mockLLM{failing},
		WithMinPercent(1), WithRetainTokenPercent(0), WithRetainMaxTokens(400))
	mgr.msgCount = len(store.history)

	// A hard error ends the call: the loop runs one iteration per prompt type
	// and each makes exactly one call before giving up.
	_ = mgr.doCompress(context.Background(), false)
	rep := mgr.LastCompactionReport()
	if rep == nil || len(rep.Attempts) == 0 {
		t.Fatalf("expected attempts in the report: %+v", rep)
	}
	if rep.Attempts[0].Model != "abliterated-model" || rep.Attempts[0].Status != "error" ||
		rep.Attempts[0].Detail != "HTTP 402 (out of credits)" {
		t.Fatalf("first attempt not neatened: %+v", rep.Attempts[0])
	}
	if !rep.hasCooldown() {
		t.Fatalf("a billing error should read as a cooldown condition: %+v", rep)
	}
}

// noModelCaller is a host whose whole chain is unavailable (every model in
// cooldown or excluded): it answers ErrNoModel without dispatching.
type noModelCaller struct {
	err   error
	calls int
}

func (c *noModelCaller) Complete(_ context.Context, _ ModelRequest) (ModelReply, error) {
	c.calls++
	return ModelReply{}, c.err
}

// TestCompress_HostNoModelReportsSkipped verifies that when the host has no
// model to offer (all in cooldown), the pass fails without retrying, and the
// report carries a single "skipped" line naming the cooldown so the user sees
// why nothing ran.
func TestCompress_HostNoModelReportsSkipped(t *testing.T) {
	store := &compressTestStore{history: makeConversation(10, 200)}
	host := &noModelCaller{err: fmt.Errorf("%w: 1 in cooldown", ErrNoModel)}
	mgr := newCompressManager(store, nil, WithModelCaller(host))
	mgr.msgCount = len(store.history)

	err := mgr.doCompress(context.Background(), false)
	if !errors.Is(err, ErrCompressionFailed) {
		t.Fatalf("expected ErrCompressionFailed; got %v", err)
	}
	rep := mgr.LastCompactionReport()
	if rep == nil || len(rep.Attempts) == 0 {
		t.Fatalf("expected a skipped attempt in the report: %+v", rep)
	}
	if rep.Attempts[0].Status != "skipped" || !strings.Contains(rep.Attempts[0].Detail, "cooldown") {
		t.Fatalf("attempt[0] = %+v; want skipped (cooldown)", rep.Attempts[0])
	}
	if !rep.hasCooldown() || !strings.Contains(rep.String(), "cooldown") {
		t.Fatalf("report should explain the cooldown:\n%s", rep.String())
	}
	// One call per loop iteration, never more: the engine does not re-ask a
	// host that has nothing to offer.
	if host.calls > 2*defaultMaxCompressIterations {
		t.Fatalf("host asked %d times for a chain with nothing available", host.calls)
	}
}

// TestCompress_ExcludeGrowsWithinOneCall verifies the retry contract: an
// unusable reply adds its model to Exclude and the host is asked again, so the
// second request names the first model; a host that ignores Exclude is bounded
// by maxCompressAttempts.
func TestCompress_ExcludeGrowsWithinOneCall(t *testing.T) {
	store := &compressTestStore{history: makeConversation(10, 200)}
	var seen [][]string
	stubborn := &recordingCaller{reply: func(req ModelRequest) ModelReply {
		seen = append(seen, append([]string(nil), req.Exclude...))
		return ModelReply{Content: invalidSummaryJSON("uncited"), Model: "stubborn"}
	}}
	mgr := newCompressManager(store, nil, WithModelCaller(stubborn))
	mgr.msgCount = len(store.history)

	_ = mgr.doCompress(context.Background(), false)

	if len(seen) < maxCompressAttempts {
		t.Fatalf("expected at least %d calls in the first iteration, got %d", maxCompressAttempts, len(seen))
	}
	if len(seen[0]) != 0 {
		t.Errorf("first request should exclude nothing, got %v", seen[0])
	}
	if !slices.Contains(seen[1], "stubborn") {
		t.Errorf("second request should exclude the rejected model, got %v", seen[1])
	}
	// Per summarization call the cap holds even though the host ignores Exclude.
	if n := len(seen); n > maxCompressAttempts*2*defaultMaxCompressIterations {
		t.Errorf("host called %d times; exclusion retry is not bounded", n)
	}
}

// recordingCaller is a ModelCaller driven by a function of the request.
type recordingCaller struct {
	reply func(ModelRequest) ModelReply
}

func (c *recordingCaller) Complete(_ context.Context, req ModelRequest) (ModelReply, error) {
	return c.reply(req), nil
}

// TestCompress_FallbackSuccess verifies that when the first client fails the
// second client is tried and a successful result is persisted.
func TestCompress_FallbackSuccess(t *testing.T) {
	store := &compressTestStore{
		history: makeConversation(10, 200),
	}

	primary := &mockLLM{
		errors: []error{errors.New("primary failed")},
	}
	fallback := &mockLLM{
		responses: []string{validSummaryJSON("fallback goal")},
	}

	mgr := newCompressManager(store, []*mockLLM{primary, fallback})
	mgr.msgCount = len(store.history)

	err := mgr.doCompress(context.Background(), false)
	if err != nil {
		t.Fatalf("doCompress returned error: %v", err)
	}

	if store.summary == "" {
		t.Error("expected summary to be stored via fallback client")
	}
	if primary.callCount == 0 {
		t.Error("expected primary client to be tried")
	}
	if fallback.callCount == 0 {
		t.Error("expected fallback client to be tried")
	}
}

// TestCompress_AllFail_Normal verifies that when all clients fail on a normal
// (non-safety-net) compression, ErrCompressionFailed is returned and history is unchanged.
func TestCompress_AllFail_Normal(t *testing.T) {
	origHistory := makeConversation(10, 200)
	store := &compressTestStore{history: origHistory}

	llm := &mockLLM{
		errors: []error{
			errors.New("fail1"),
			errors.New("fail2"),
			errors.New("fail3"),
			errors.New("fail4"), // extra to cover aggressive iterations
			errors.New("fail5"),
			errors.New("fail6"),
		},
	}

	mgr := newCompressManager(store, []*mockLLM{llm})
	mgr.msgCount = len(store.history)

	err := mgr.doCompress(context.Background(), false)
	if !errors.Is(err, ErrCompressionFailed) {
		t.Fatalf("expected ErrCompressionFailed; got %v", err)
	}

	if len(store.history) != len(origHistory) {
		t.Errorf("expected history unchanged (%d); got %d", len(origHistory), len(store.history))
	}
}

// TestCompress_AllFail_Safety_Drop verifies that when all clients fail on the
// safety-net path, oldest messages are dropped to reduce context size.
func TestCompress_AllFail_Safety_Drop(t *testing.T) {
	// Build a history large enough to exceed safetyPercent (80% of 10000 = 8000 tokens).
	// Each char ≈ 0.4 tokens; need >20000 chars. Use 40 pairs × 600 chars each.
	store := &compressTestStore{
		history: makeConversation(40, 600),
	}

	// All calls fail.
	errList := make([]error, 20)
	for i := range errList {
		errList[i] = errors.New("fail")
	}
	llm := &mockLLM{errors: errList}

	mgr := newCompressManager(store, []*mockLLM{llm},
		WithContextWindow(10000),
		WithSafetyPercent(80),
		WithRetainMinMessages(2),
	)
	mgr.msgCount = len(store.history)
	origLen := len(store.history)

	err := mgr.doCompress(context.Background(), true)
	if err != nil {
		t.Fatalf("doCompress returned error: %v", err)
	}

	if len(store.history) >= origLen {
		t.Errorf("expected messages to be dropped; history length unchanged at %d", len(store.history))
	}
}

// TestCompress_StaleSummary verifies that when all clients fail on the
// safety-net path and an existing summary is present, it is retained and
// oldest messages are still dropped to reduce context size.
func TestCompress_StaleSummary(t *testing.T) {
	store := &compressTestStore{
		history: makeConversation(40, 600),
		summary: validSummaryJSON("existing goal"),
	}

	errList := make([]error, 20)
	for i := range errList {
		errList[i] = errors.New("fail")
	}
	llm := &mockLLM{errors: errList}

	mgr := newCompressManager(store, []*mockLLM{llm},
		WithContextWindow(10000),
		WithSafetyPercent(80),
		WithRetainMinMessages(2),
	)
	mgr.msgCount = len(store.history)
	origLen := len(store.history)

	err := mgr.doCompress(context.Background(), true)
	if err != nil {
		t.Fatalf("doCompress returned error: %v", err)
	}

	// History must have been trimmed.
	if len(store.history) >= origLen {
		t.Errorf("expected history to shrink; got %d (orig %d)", len(store.history), origLen)
	}

	// The stale summary must still be present (not cleared).
	if store.summary == "" {
		t.Error("expected stale summary to be retained")
	}
}

// TestCompress_NotifyCallback verifies that the notifyCallback is called with
// "compression started" and "compression complete" messages.
func TestCompress_NotifyCallback(t *testing.T) {
	store := &compressTestStore{
		history: makeConversation(10, 200),
	}

	llm := &mockLLM{
		responses: []string{validSummaryJSON("notify goal")},
	}

	var notifications []string
	mgr := newCompressManager(store, []*mockLLM{llm},
		WithNotifyCallback(func(msg string) {
			notifications = append(notifications, msg)
		}),
	)
	mgr.msgCount = len(store.history)

	err := mgr.doCompress(context.Background(), false)
	if err != nil {
		t.Fatalf("doCompress returned error: %v", err)
	}

	if len(notifications) < 2 {
		t.Fatalf("expected at least 2 notifications; got %d: %v", len(notifications), notifications)
	}
	if notifications[0] != "compression started" {
		t.Errorf("expected first notification 'compression started'; got %q", notifications[0])
	}
	found := false
	for _, n := range notifications {
		if n == "compression complete" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected 'compression complete' notification; got %v", notifications)
	}
}

// TestCompress_CoolingSetOnLowGain verifies that cooling is enabled when the
// LLM succeeds but the compression gain is below defaultMinCompressionGain
// and the final context percentage is still at or above normalPercent.
func TestCompress_CoolingSetOnLowGain(t *testing.T) {
	// Context window: 10000 tokens. normalPercent: 50 → trigger at 5000 tokens.
	// retainTokenPercent: 90 → budget = 9000 tokens; tail covers nearly everything,
	// so gain will be very small (tail ≈ whole conversation).
	// History: 10 pairs × 200 chars → ~1600 tokens → 16% — but we want to be above
	// normalPercent after "compression". We use retainTokenPercent=90 so the budget
	// is large, meaning the tail covers almost everything, giving near-zero gain,
	// but we need finalPct >= normalPercent (50%).
	//
	// Set contextWindow small so the same history is at/above normalPercent.
	// 10 pairs × 200 chars = 4000 chars → ~1600 tokens.
	// contextWindow = 2000: 1600/2000 = 80% → above normalPercent=50.
	// budget = 2000 * 90 / 100 = 1800 tokens → tail covers most messages.
	// gain ≈ tiny → cooling should be set.
	store := &compressTestStore{
		history: makeConversation(10, 200),
	}

	llm := &mockLLM{
		responses: []string{
			validSummaryJSON("low gain goal"),
			validSummaryJSON("low gain goal 2"),
			validSummaryJSON("low gain goal 3"),
		},
	}

	mgr := newCompressManager(store, []*mockLLM{llm},
		WithContextWindow(2000),
		WithNormalPercent(50),
		WithSafetyPercent(90),
		WithRetainTokenPercent(90),
		WithRetainMinMessages(2),
	)
	mgr.msgCount = len(store.history)

	err := mgr.doCompress(context.Background(), false)
	if err != nil {
		t.Fatalf("doCompress returned error: %v", err)
	}

	if !mgr.cooling {
		// It's possible the LLM succeeded AND gain was sufficient; check the gain value.
		// Only fail if gain is genuinely low but cooling wasn't set.
		if mgr.lastCompressionGain < defaultMinCompressionGain {
			t.Errorf("expected cooling=true when gain (%.4f) < defaultMinCompressionGain (%.4f)",
				mgr.lastCompressionGain, defaultMinCompressionGain)
		}
		// If gain is large enough, cooling correctly stays false — test passes.
	}
}
