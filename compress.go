/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine/logger"
	"github.com/PivotLLM/ctxengine/memory"
	"github.com/PivotLLM/ctxengine/session"
)

// maxCompressAttempts bounds the number of ModelCaller.Complete calls one
// summarization call may make. Each unusable reply (refusal, invalid JSON, a
// summary that fails validation) adds its model to the exclusion list and the
// host is asked again; this cap keeps a host that ignores Exclude, or a long
// chain of unusable models, from burning calls without limit.
const maxCompressAttempts = 4

// ErrNoModel is the error a ModelCaller returns when nothing in its chain can
// serve the request — every model is excluded, in cooldown, or unconfigured.
// The engine records it once per summarization call rather than as one failed
// attempt per model.
var ErrNoModel = errors.New("no summarization model available")

// doCompress performs LLM-based compression of the conversation history.
// It summarizes the oldest messages into a structured Summary, retains a tail
// of recent messages, and persists the result. safetyNet=true enables fallback
// behavior (drop oldest groups) when the model produces no usable summary.
func (m *Manager) doCompress(ctx context.Context, safetyNet bool) error {
	if m.cfg.notifyCallback != nil {
		m.cfg.notifyCallback("compression started")
	}
	// Compaction rewrites the window and so the cached prefix anyway, which
	// makes it the free moment to apply every pending eviction: the batched
	// per-dispatch sweep may have been holding them back for the cache. The
	// events are DEBUG-logged; the compaction report covers the pass.
	m.sweepEvictions(ctx, true)
	defer func() {
		m.compressedAtCount = m.msgCount
		if m.cfg.notifyCallback != nil {
			m.cfg.notifyCallback("compression complete")
		}
	}()

	if m.caller == nil {
		m.lastReport = &CompactionReport{SessionKey: m.sessionKey, Outcome: "nothing"}
		return nil // no summarization model configured
	}

	// Separate system message from conversation.
	// Use GetHistoryWithSeqs to preserve seq numbers for the summarizer.
	storedHistory := m.store.GetHistoryWithSeqs(m.sessionKey)
	var sysMsg *memory.StoredMessage
	storedConversation := storedHistory
	if len(storedHistory) > 0 && storedHistory[0].Role == "system" {
		sys := storedHistory[0]
		sysMsg = &sys
		storedConversation = storedHistory[1:]
	}

	if len(storedConversation) == 0 {
		m.lastReport = &CompactionReport{SessionKey: m.sessionKey, Outcome: "nothing"}
		return nil
	}

	// Per-pass recorder: collects one entry per LLM invocation for the report,
	// and (when debug capture is on) appends verbatim request/response to
	// <workspace>/compact.jsonl.
	debugPath := ""
	if m.cfg.compactDebug && m.cfg.compressionProfileDir != "" {
		debugPath = filepath.Join(m.cfg.compressionProfileDir, "compact.jsonl")
	}
	rec := &compactionRecorder{
		sessionKey:  m.sessionKey,
		debugPath:   debugPath,
		failureDump: m.cfg.failureDump,
	}
	beforeMsgs := len(storedConversation)
	beforeBytes := storedBytes(storedConversation)
	dateFrom, dateTo := storedDateRange(storedConversation)

	conversation := storedToPlain(storedConversation)
	tokensBeforeCompress := m.estTokens(conversation)
	targetPct := m.compressTargetPercent()
	budget := m.retainBudgetTokens()
	maxAge := m.retainMaxAge()
	now := time.Now()

	archiveMin, archiveMax := m.archiveWindow() //nolint:contextcheck // archive bounds come from memory.ArchiveStore, whose read API takes no context
	existingSummary, parseErr := unmarshalSummary(m.store.GetSummary(m.sessionKey))
	if parseErr != nil {
		// A stored summary that is not a valid structured summary (legacy
		// prose, a corrupt write) is not merged; this pass starts afresh.
		logger.DebugCF("llmcontext", "existing summary not parsed; compacting without it",
			map[string]any{"session_key": m.sessionKey, "error": parseErr.Error()})
		existingSummary = nil
	}

	// Load the agent's compression profile once per compression pass.
	compressionProfile := loadCompressionProfile(m.cfg.compressionProfileDir)

	// Compression loop: iteratively summarize the oldest portion of the
	// conversation until we reach the target percentage or exhaust iterations.
	// Item 13: enforce the max summary token budget. Computed once and passed to
	// callModel so an oversized summary is rejected within the model call
	// (excluding that model and asking again) rather than after it.
	summaryTokenLimit := m.cfg.maxSummaryTokens
	if summaryTokenLimit <= 0 && m.cfg.contextWindow > 0 {
		summaryTokenLimit = m.cfg.contextWindow * 20 / 100
	}

	currentStored := storedConversation
	currentConversation := conversation
	latestSummary := existingSummary
	llmSucceeded := false
	attemptedLLM := false // true once the LLM was actually invoked for a summary
	aggressive := false
	iterCount := 0
	tokensBeforeIteration := tokensBeforeCompress

	for {
		if iterCount >= defaultMaxCompressIterations {
			if !aggressive {
				aggressive = true
				iterCount = 0
				tokensBeforeIteration = m.estTokens(currentConversation)
				continue
			}
			break // exhausted both standard and aggressive prompt types
		}

		storedTail, tailStart := selectTail(currentStored, budget, m.cfg.retainMinMessages, maxAge, now, m.estTokens, m.noiseKey())
		// Never archive past the most recent user message: the live window must
		// always retain the latest user turn. Otherwise the next dispatch sends a
		// payload of only system+assistant+tool messages, which strict providers
		// reject with "messages must contain at least one item with role='user'"
		// (a non-retriable 400 that kills the turn). This clamp overrides the age
		// cap too: an old-but-latest user message is still the anchor of the turn.
		// It checks the messages actually kept, not an index: the tail is
		// collapsed, so lu >= tailStart alone does not prove lu survived.
		if lu := lastUserStoredIndex(currentStored); lu >= 0 && !containsSeq(storedTail, currentStored[lu].Seq) {
			tailStart = min(tailStart, lu)
			storedTail = currentStored[tailStart:] // keep tail/tailStart consistent
		}
		// Never empty the live window. In a long in-flight tool-call sequence the
		// retained tail can be all tool plumbing, which selectTail trims away to
		// nothing; with no user/clean anchor to clamp to that would archive the
		// whole conversation and leave a system-only payload (the model loses the
		// entire thread mid-turn). Keep at least the last turn group instead.
		if tailStart >= len(currentStored) && len(currentStored) > 0 {
			g := resolveGroup(currentConversation, len(currentStored)-1)
			tailStart = g.start
			storedTail = currentStored[tailStart:]
			logger.WarnCF("llmcontext", "compaction would empty the live window; retaining the last turn group", map[string]any{
				"session_key": m.sessionKey,
				"kept":        len(storedTail),
			})
		}
		// The summarizer gets the prefix only. Repeats the tail collapsed out of
		// currentStored[tailStart:] are not summarized: each is an adjacent copy
		// of a message the window keeps, it stays in the archive, and sent on
		// its own a scheduled fire reads like an open request.
		toSummarize := currentStored[:tailStart]

		if len(toSummarize) == 0 {
			break // tail covers everything; nothing left to summarize
		}

		attemptedLLM = true
		newSummary, ok := m.callModel(ctx, latestSummary, toSummarize, archiveMin, archiveMax, aggressive, compressionProfile, summaryTokenLimit, rec)
		if !ok {
			iterCount++
			continue
		}
		if newSummary.Model == "" {
			newSummary.Model = m.cfg.compressModel.Primary
		}

		llmSucceeded = true
		latestSummary = newSummary
		if dropped := droppedRepeatSeqs(currentStored[tailStart:], storedTail); len(dropped) > 0 {
			logger.InfoCF("llmcontext", "compaction dropped adjacent repeats from the live window (kept in the archive)", map[string]any{
				"session_key": m.sessionKey,
				"seqs":        dropped,
			})
		}
		currentStored = storedTail
		currentConversation = storedToPlain(storedTail)

		tokensCurrent := m.estTokens(currentConversation)
		if m.cfg.contextWindow > 0 {
			pctCurrent := float64(tokensCurrent) * 100 / float64(m.cfg.contextWindow)
			if pctCurrent < targetPct {
				break // reached target
			}
		}

		gain := 1.0
		if tokensBeforeIteration > 0 {
			gain = 1 - float64(tokensCurrent)/float64(tokensBeforeIteration)
		}
		if gain < defaultMinLoopGain {
			if !aggressive {
				aggressive = true
				iterCount = 0
				tokensBeforeIteration = tokensCurrent
			} else {
				break // stalled even on aggressive prompt
			}
		} else {
			tokensBeforeIteration = tokensCurrent
		}

		iterCount++
	}

	var err error
	if !safetyNet {
		err = m.handleNormalPostLoop(ctx, sysMsg, currentStored, latestSummary, llmSucceeded, attemptedLLM, tokensBeforeCompress, targetPct)
	} else {
		err = m.handleSafetyNetPostLoop(ctx, sysMsg, storedConversation, currentStored, latestSummary, existingSummary, llmSucceeded)
	}
	m.lastReport = m.buildReport(rec, beforeMsgs, beforeBytes, dateFrom, dateTo, err)
	return err
}

// compressTargetPercent is the compaction loop's stop condition: iterate until
// the live window falls below this percentage of the context window. An explicit
// targetPercent wins; otherwise it is derived from normalPercent, preserving the
// historic behaviour for configs that do not set it.
//
// Note the derived value interacts with the retain budget: with the defaults
// (normal 50 → target 25, retain 10) the first pass already lands under target,
// so the loop exits after one iteration by design.
func (m *Manager) compressTargetPercent() float64 {
	if m.cfg.targetPercent > 0 {
		return float64(m.cfg.targetPercent)
	}
	return float64(m.cfg.normalPercent) * defaultCompressTargetFactor
}

// retainBudgetTokens resolves the token budget for the retained tail: the
// percentage of the context window, capped by the absolute retainMaxTokens when
// set. The absolute cap exists because percentages scale with the window — a
// 1M-token model would otherwise inherit a tail budget tuned for 128k and retain
// an enormous amount purely because it can. Returns 0 (no budget check) only
// when no window is configured and no absolute cap is set.
func (m *Manager) retainBudgetTokens() int {
	budget := 0
	if m.cfg.contextWindow > 0 {
		budget = m.cfg.contextWindow * m.cfg.retainTokenPercent / 100
	}
	if limit := m.cfg.retainMaxTokens; limit > 0 && (budget <= 0 || limit < budget) {
		budget = limit
	}
	return budget
}

// retainMaxAge resolves the age cap for the retained tail. Zero disables it.
func (m *Manager) retainMaxAge() time.Duration {
	if m.cfg.retainMaxAgeDays <= 0 {
		return 0
	}
	return time.Duration(m.cfg.retainMaxAgeDays) * 24 * time.Hour
}

// refusedModelList returns the models that refused this session's content, as
// the Exclude list for the next request. Nil when none has.
func (m *Manager) refusedModelList() []string {
	m.refusedMu.Lock()
	defer m.refusedMu.Unlock()
	if len(m.refusedModels) == 0 {
		return nil
	}
	out := make([]string, 0, len(m.refusedModels))
	for model := range m.refusedModels {
		out = append(out, model)
	}
	return out
}

// noteRefusedModel records a model that refused this session's content so
// later compactions for this session exclude it. Unnamed models cannot be
// excluded and are not recorded.
func (m *Manager) noteRefusedModel(model, detail string) {
	if model == "" {
		return
	}
	m.refusedMu.Lock()
	defer m.refusedMu.Unlock()
	if m.refusedModels == nil {
		m.refusedModels = make(map[string]bool)
	}
	if m.refusedModels[model] {
		return
	}
	m.refusedModels[model] = true
	logger.WarnCF("llmcontext", "summarization model refused this session's content; skipping it for future compactions", map[string]any{
		"session_key": m.sessionKey,
		"model":       model,
		"detail":      detail,
	})
}

// lastUserStoredIndex returns the index of the most recent user-role message in
// the (system-stripped) conversation slice, or -1 if there is none.
func lastUserStoredIndex(stored []memory.StoredMessage) int {
	for i, s := range slices.Backward(stored) {
		if s.Role == "user" {
			return i
		}
	}
	return -1
}

// containsSeq reports whether stored holds the message with sequence number seq.
func containsSeq(stored []memory.StoredMessage, seq int64) bool {
	return slices.ContainsFunc(stored, func(sm memory.StoredMessage) bool { return sm.Seq == seq })
}

// droppedRepeatSeqs returns the seqs of window, the messages from the tail's
// start onwards, that the retained tail does not hold: the adjacent repeats
// collapseStoredNoise removed. They are neither summarized nor kept in the
// window; the archive still holds them.
func droppedRepeatSeqs(window, tail []memory.StoredMessage) []int64 {
	if len(window) == len(tail) {
		return nil
	}
	kept := make(map[int64]bool, len(tail))
	for _, sm := range tail {
		kept[sm.Seq] = true
	}
	var out []int64
	for _, sm := range window {
		if !kept[sm.Seq] {
			out = append(out, sm.Seq)
		}
	}
	return out
}

// handleNormalPostLoop handles post-loop logic for the normal (non-safety-net) path.
func (m *Manager) handleNormalPostLoop(
	ctx context.Context,
	sysMsg *memory.StoredMessage,
	currentStored []memory.StoredMessage,
	latestSummary *Summary,
	llmSucceeded bool,
	attemptedLLM bool,
	tokensBeforeCompress int,
	targetPct float64,
) error {
	_ = ctx // reserved for future use
	currentConversation := storedToPlain(currentStored)
	if !llmSucceeded {
		if !attemptedLLM {
			// The retained tail already covers the whole conversation; there was
			// nothing to summarize. This is a benign no-op, not a failure.
			logger.InfoCF("llmcontext", "compression: nothing to compress (tail already covers conversation)", map[string]any{
				"session_key": m.sessionKey,
			})
			return ErrNothingToCompress
		}
		logger.WarnCF("llmcontext", "compression failed: every summarization model rejected the summary", map[string]any{
			"session_key": m.sessionKey,
		})
		return ErrCompressionFailed
	}

	if err := m.persistStoredResult(sysMsg, currentStored, latestSummary); err != nil {
		return err
	}

	tokensFinal := m.estTokens(currentConversation)
	overallGain := 0.0
	if tokensBeforeCompress > 0 {
		overallGain = 1 - float64(tokensFinal)/float64(tokensBeforeCompress)
	}
	m.lastCompressionGain = overallGain
	m.lastCompressedAt = time.Now()

	finalPct := 0.0
	if m.cfg.contextWindow > 0 {
		finalPct = float64(tokensFinal) * 100 / float64(m.cfg.contextWindow)
	}
	if overallGain < defaultMinCompressionGain && finalPct >= float64(m.cfg.normalPercent) {
		m.cooling = true
		m.coolingSinceCount = m.msgCount
	}

	return nil
}

// handleSafetyNetPostLoop handles post-loop logic for the safety-net path.
func (m *Manager) handleSafetyNetPostLoop(
	ctx context.Context,
	sysMsg *memory.StoredMessage,
	originalStored []memory.StoredMessage,
	currentStored []memory.StoredMessage,
	latestSummary *Summary,
	existingSummary *Summary,
	llmSucceeded bool,
) error {
	if llmSucceeded {
		if err := m.persistStoredResult(sysMsg, currentStored, latestSummary); err != nil {
			return err
		}
	} else if existingSummary != nil {
		logger.WarnCF("llmcontext", "safety-net compression: all LLM clients failed; using stale summary", map[string]any{
			"session_key":   m.sessionKey,
			"stale_summary": true,
		})
		currentStored = originalStored
	} else {
		logger.WarnCF("llmcontext", "safety-net compression: all LLM clients failed; no summary available", map[string]any{
			"session_key": m.sessionKey,
			"no_summary":  true,
		})
		currentStored = originalStored
	}

	// The safety net is judged by the measure that fired it: the whole request
	// (history plus reserve, tool schemas and build overhead), not history
	// alone, or a pass could stop with history under the line and the request
	// still over it.
	if m.contextPercent(storedToPlain(currentStored)) < float64(m.cfg.safetyPercent) {
		// Compression was sufficient; clear cooling and update stats.
		m.cooling = false
		m.lastCompressionGain = 0
		m.lastCompressedAt = time.Now()
		return nil
	}

	// Still at or above safety threshold — drop oldest groups, then apply large
	// message checks, then persist. Order: group-drop -> applyLargeMsgChecks ->
	// persistStoredResult -> compute finalPct. When no model succeeded the
	// summary in the store is the stale one and stays as it is: it is not
	// handed to persist, which would checkpoint it a second time.
	var newSummary *Summary
	if llmSucceeded {
		newSummary = latestSummary
	}
	currentStored = m.dropOldestStoredGroups(ctx, currentStored)
	m.applyLargeMsgChecksStored(currentStored)
	if err := m.persistStoredResult(sysMsg, currentStored, newSummary); err != nil {
		return err
	}

	// Recheck after drops.
	if m.contextPercent(storedToPlain(currentStored)) < float64(m.cfg.safetyPercent) {
		m.cooling = false
		m.lastCompressedAt = time.Now()
		return nil
	}

	return ErrCompressionPartial
}

// callModel asks the host's ModelCaller for a summary and returns the first
// one that passes validation. A reply whose summary fails to parse, lacks cited
// material, or cannot be trimmed within summaryTokenLimit is recorded, its
// model is added to the request's Exclude list, and the host is asked again —
// so a fully-configured summarization chain actually falls through on a model
// that returns an unacceptable summary, not just on a hard transport error. A
// content refusal is remembered for the session so later compactions exclude
// that model from the start. A host error ends the call. Returns (nil, false)
// when no attempt produced an acceptable summary.
//
// toSummarize is a []memory.StoredMessage so the prompt can include [#N] seq
// prefixes for each message. CoveredSeqStart and CoveredSeqEnd are set from the
// actual min/max seq of the slice, not from any LLM-emitted values.
func (m *Manager) callModel(
	ctx context.Context,
	existing *Summary,
	toSummarize []memory.StoredMessage,
	archiveMin, archiveMax int64,
	aggressive bool,
	compressionProfile string,
	summaryTokenLimit int,
	rec *compactionRecorder,
) (*Summary, bool) {
	if len(toSummarize) == 0 || m.caller == nil {
		return nil, false
	}
	sessionKey := m.sessionKey

	// Compute the actual seq range from the slice (do not trust LLM output).
	coveredStart := toSummarize[0].Seq
	coveredEnd := toSummarize[len(toSummarize)-1].Seq
	for _, sm := range toSummarize {
		if sm.Seq < coveredStart {
			coveredStart = sm.Seq
		}
		if sm.Seq > coveredEnd {
			coveredEnd = sm.Seq
		}
	}

	req := ModelRequest{
		System:     buildSummarizationPrompt(existing, archiveMin, archiveMax, aggressive, compressionProfile),
		User:       "Messages to summarize:\n\n" + formatStoredMessagesForSummary(toSummarize, m.noiseKey()),
		JSONObject: true,
		Exclude:    m.refusedModelList(),
	}
	// What an exact (verbatim) field may quote: the user's own words in this
	// range, or a value an earlier pass already accepted. Anything else — a
	// tool result, the assistant's own text — is not a user instruction and
	// is dropped from exact before the summary is stored.
	exactSources := userTexts(toSummarize)
	if existing != nil {
		exactSources = append(exactSources, existing.exactValues()...)
	}
	// The recorder captures the call as the two-message exchange the host
	// sends, so debug capture and failure dumps keep their historical shape.
	messages := []spawnllm.Message{
		{Role: "system", Content: req.System},
		{Role: "user", Content: req.User},
	}

	// exclude adds model to the request's exclusion list for the rest of this
	// call. An unnamed model cannot be excluded: asking again would hit the
	// same model, so the call stops instead.
	exclude := func(model string) bool {
		if model == "" {
			return false
		}
		req.Exclude = append(req.Exclude, model)
		return true
	}

	for range maxCompressAttempts {
		start := time.Now()
		reply, err := m.caller.Complete(ctx, req)
		dur := time.Since(start)
		model := reply.Model
		if err != nil {
			if errors.Is(err, ErrNoModel) {
				// Nothing left to try. Report it once when it is the whole
				// story (every model excluded or cooling before any call);
				// after a recorded attempt it only says the chain ran out.
				if len(rec.attempts) == 0 {
					rec.record(model, "skipped", shortErr(err), dur, messages, "")
				}
				return nil, false
			}
			// Classify so the report shows a clean "HTTP 402 (out of credits)"
			// instead of a raw body dump. Cooling the model is the host's job.
			detail := shortErr(err)
			if fe := spawnllm.ClassifyError(err, "", model); fe != nil {
				if fe.Status > 0 {
					detail = fmt.Sprintf("HTTP %d (%s)", fe.Status, spawnllm.ReasonText(fe.Reason))
				} else {
					detail = spawnllm.ReasonText(fe.Reason)
				}
			}
			rec.record(model, "error", detail, dur, messages, "")
			return nil, false
		}

		summary, perr := validateAndUnmarshalLLMResponse(reply.Content)
		if perr != nil {
			// A response that did not yield a valid summary AND carries refusal
			// signals (finish_reason or a decline phrase) is a content refusal, not
			// a flaky model. Record it distinctly so the user is alerted, and stop
			// sending this session's content to this model.
			if refused, detail := m.classifyRefusal(reply.FinishReason, reply.Content); refused {
				rec.record(model, "refused", detail, dur, messages, reply.Content)
				m.noteRefusedModel(model, detail)
			} else {
				rec.record(model, "error", "invalid JSON response: "+shortErr(perr), dur, messages, reply.Content)
			}
			if !exclude(model) {
				return nil, false
			}
			continue
		}

		if dropped := summary.DropUnsourcedExact(exactSources); dropped > 0 {
			logger.DebugCF("llmcontext", "summary exact values not quoted from a user message dropped", map[string]any{
				"session_key": sessionKey,
				"model":       model,
				"dropped":     dropped,
			})
		}

		// Set coverage from actual seq ranges. Do NOT use coverage values
		// emitted by the LLM itself.
		applySummaryCoverage(summary, existing, coveredStart, coveredEnd, toSummarize)
		summary.GeneratedAt = time.Now()
		summary.Profile = profileFingerprint(compressionProfile)

		// Strip seq references outside the valid range, then require the summary
		// to carry cited material. A model that returns an un-cited summary is
		// excluded so the next call reaches the next model in the host's chain.
		summary.StripOutOfRangeSeqRefs(archiveMin, archiveMax)
		if !summary.HasMaterial() || !summary.HasEvidence() {
			rec.record(model, "rejected", "missing citations", dur, messages, reply.Content)
			if !exclude(model) {
				return nil, false
			}
			continue
		}

		// Item 13: enforce the max summary token budget. Truncate, then discard
		// (excluding the model) if it still does not fit.
		if summaryTokenLimit > 0 {
			if summary.TruncateToFit(summaryTokenLimit) {
				logger.WarnCF("llmcontext", "summary truncated to fit token budget", map[string]any{
					"session_key": sessionKey,
					"limit":       summaryTokenLimit,
				})
			}
			if data, merr := json.Marshal(summary); merr == nil {
				if len([]rune(string(data)))/4 > summaryTokenLimit {
					logger.WarnCF("llmcontext", "summary still oversized after truncation — discarding", map[string]any{
						"session_key": sessionKey,
						"tokens":      len([]rune(string(data))) / 4,
						"limit":       summaryTokenLimit,
					})
					rec.record(model, "rejected", "summary too large", dur, messages, reply.Content)
					if !exclude(model) {
						return nil, false
					}
					continue
				}
			}
		}

		rec.record(model, "ok", "", dur, messages, reply.Content)
		if model != "" && summary.Model == "" {
			summary.Model = model
		}
		return summary, true
	}

	return nil, false
}

// profileFingerprint returns a short sha256 hex[:8] fingerprint of a compression
// profile's content, or "" when no profile is in effect. Stamped into the summary
// so a reader can tell which profile shaped it and detect when the profile changed.
func profileFingerprint(profile string) string {
	p := strings.TrimSpace(profile)
	if p == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(p))
	return hex.EncodeToString(sum[:])[:8]
}

// loadCompressionProfile reads COMPRESSION.md (or the legacy compression.md)
// from dir if it exists. HTML comments are stripped so the template's
// human-facing documentation never reaches the summarizer; only real
// role-specific guidance is appended to the prompt. Returns "" when neither file
// is present, unreadable, or comment-only.
func loadCompressionProfile(dir string) string {
	if dir == "" {
		return ""
	}
	// Prefer the uppercase name to match the other workspace files (AGENTS.md,
	// SOUL.md, MEMORY.md, …); fall back to the legacy lowercase name.
	data, err := os.ReadFile(filepath.Join(dir, "COMPRESSION.md")) //nolint:gosec // G304: dir is the host-configured compression profile directory
	if err != nil {
		data, err = os.ReadFile(filepath.Join(dir, "compression.md")) //nolint:gosec // G304: dir is the host-configured compression profile directory
		if err != nil {
			return ""
		}
	}
	return strings.TrimSpace(stripHTMLComments(string(data)))
}

// stripHTMLComments removes <!-- ... --> blocks (including multi-line and
// unterminated ones) from s.
func stripHTMLComments(s string) string {
	for {
		start := strings.Index(s, "<!--")
		if start < 0 {
			break
		}
		rest := s[start+len("<!--"):]
		_, after, ok := strings.Cut(rest, "-->")
		if !ok {
			s = s[:start] // unterminated comment: drop to end
			break
		}
		s = s[:start] + after
	}
	return s
}

// repetitiveRunThreshold is the minimum number of consecutive near-identical
// messages required before they are collapsed into a count annotation.
const repetitiveRunThreshold = 3

// normalizeForComparison returns a whitespace-collapsed lowercase version of s
// used to detect near-identical messages in repetitive run detection.
func normalizeForComparison(s string) string {
	return strings.ToLower(strings.Join(strings.Fields(s), " "))
}

// userTexts returns the content of every user-role message in stored.
func userTexts(stored []memory.StoredMessage) []string {
	var out []string
	for _, sm := range stored {
		if sm.Role == "user" && strings.TrimSpace(sm.Content) != "" {
			out = append(out, sm.Content)
		}
	}
	return out
}

func storedToPlain(stored []memory.StoredMessage) []spawnllm.Message {
	msgs := make([]spawnllm.Message, len(stored))
	for i, sm := range stored {
		msgs[i] = sm.Message
	}
	return msgs
}

// cronNoOpReplyMaxLen bounds how long an assistant reply may be (after trim)
// to still count as a "routine" no-op reply eligible for cron run collapse.
// This avoids collapsing substantive replies that merely happen to repeat.
const cronNoOpReplyMaxLen = 200

// collapseRepetitiveRuns scans stored messages for two kinds of repetition and
// replaces each run with a single counted anchor so the LLM summarizer handles
// them correctly instead of silently ignoring them:
//
//  1. Cron no-op runs: maximal runs of [cron-marker user message with the SAME
//     key] → [assistant reply] pairs where every assistant reply in the run is
//     mutually identical (trimmed-equal) and short. Each cron fire embeds a
//     different timestamp, so a byte-identical check would miss them entirely;
//     the marker key (fingerprint, or payload for legacy fires) lets us group
//     them. Seq of the first message is preserved.
//  2. Byte-identical same-role runs: the original behavior, applied to anything
//     not consumed by cron collapse.
func collapseRepetitiveRuns(stored []memory.StoredMessage, noise NoiseKeyFunc) []memory.StoredMessage {
	if len(stored) < repetitiveRunThreshold {
		return stored
	}
	if noise == nil {
		noise = noNoiseKey
	}
	result := make([]memory.StoredMessage, 0, len(stored))
	i := 0
	for i < len(stored) {
		if anchor, next, ok := collapseCronRun(stored, i, noise); ok {
			result = append(result, anchor)
			i = next
			continue
		}

		j := i + 1
		norm := normalizeForComparison(stored[i].Content)
		for j < len(stored) &&
			stored[j].Role == stored[i].Role &&
			normalizeForComparison(stored[j].Content) == norm {
			j++
		}
		runLen := j - i
		if runLen >= repetitiveRunThreshold {
			// Emit a single representative entry with a count annotation.
			collapsed := stored[i]
			collapsed.Content = fmt.Sprintf(
				"[REPEATED %d TIMES — content identical to above]\n%s",
				runLen, stored[i].Content,
			)
			result = append(result, collapsed)
		} else {
			result = append(result, stored[i])
			j = i + 1
		}
		i = j
	}
	return result
}

// collapseCronRun attempts to detect a scheduled-job no-op run starting at
// index start. A qualifying run is a maximal sequence of consecutive [user
// message with the SAME noise key] → [assistant reply] pairs where all the
// assistant replies are mutually trimmed-equal and short (<= cronNoOpReplyMaxLen).
// It returns the synthetic counted anchor, the index immediately after the run,
// and true. If no qualifying run of >= repetitiveRunThreshold fires begins at
// start, ok is false.
func collapseCronRun(stored []memory.StoredMessage, start int, noise NoiseKeyFunc) (memory.StoredMessage, int, bool) {
	key, isCron := noise(stored[start].Content)
	if stored[start].Role != "user" || !isCron {
		return memory.StoredMessage{}, start, false
	}

	var reply string
	haveReply := false
	count := 0
	i := start

	for i+1 < len(stored) {
		userMsg := stored[i]
		replyMsg := stored[i+1]

		if userMsg.Role != "user" {
			break
		}
		k, ok := noise(userMsg.Content)
		if !ok || k != key {
			break
		}
		if replyMsg.Role != "assistant" {
			break
		}
		// Action detection: a reply that issued tool calls means the LLM DID
		// something in response to this fire (restarted a service, sent a message,
		// wrote a file, ...) — it is not routine noise. Stop the run here so the
		// acting fire and the entire tool exchange that follows it (tool results,
		// follow-up assistant turns) are preserved verbatim, never folded into the
		// collapsed count. A reply with no tool calls is a complete [cron → reply]
		// turn; the loop's own cron-key check ends the run when the next message
		// isn't another fire of the same job.
		if len(replyMsg.ToolCalls) > 0 {
			break
		}
		r := strings.TrimSpace(replyMsg.Content)
		if len(r) > cronNoOpReplyMaxLen {
			break
		}
		if !haveReply {
			reply = r
			haveReply = true
		} else if r != reply {
			// A differing reply means something actually happened — it breaks
			// the run. The accumulated uniform prefix is collapsed; this pair is
			// preserved verbatim by the caller continuing from the run boundary.
			break
		}

		count++
		i += 2
	}

	if count < repetitiveRunThreshold {
		return memory.StoredMessage{}, start, false
	}

	// The run spans stored[start] (first cron message) through stored[i-1] (last
	// consumed assistant reply). The anchor carries the seq of the FIRST message
	// and notes the full [first-last] seq range in its text. Seqs are permanent
	// identities — they are never renumbered — so the collapsed-away messages
	// simply do not appear inline in the retained tail; they remain intact in the
	// archive and stay retrievable by seq via session_messages.
	firstSeq := stored[start].Seq
	lastSeq := stored[i-1].Seq

	anchor := stored[start] // carry the seq of the first message in the run.
	anchor.Role = "user"
	anchor.Content = cronRunAnchor(key, count, firstSeq, lastSeq, reply)
	anchor.ToolCalls = nil
	anchor.ToolCallID = ""
	return anchor, i, true
}

// cronRunAnchorKeyMaxLen bounds the noise key rendered in a run anchor. A key
// is usually a short job fingerprint; a legacy key is the whole payload, which
// is clipped so the anchor stays one line.
const cronRunAnchorKeyMaxLen = 40

// cronRunAnchor renders the counted anchor string for a collapsed no-op run of
// a scheduled job identified by key. It states the count and the
// [firstSeq-lastSeq] range so a reader knows exactly which archived messages
// were elided and can retrieve them via session_messages.
func cronRunAnchor(key string, count int, firstSeq, lastSeq int64, reply string) string {
	shortReply := truncateRunes(reply, 60)
	label := truncateRunes(strings.Join(strings.Fields(key), " "), cronRunAnchorKeyMaxLen)
	if label != "" {
		return fmt.Sprintf(
			"[scheduled job %s fired ×%d (#%d-#%d); routine, replies identical: %q]",
			label, count, firstSeq, lastSeq, shortReply,
		)
	}
	return fmt.Sprintf(
		"[scheduled job fired ×%d (#%d-#%d); routine, replies identical: %q]",
		count, firstSeq, lastSeq, shortReply,
	)
}

// collapseRetainedCronRuns collapses runs of repeated cron no-op fires in the
// RETAINED live tail into a single counted anchor — the same transformation the
// summarizer input receives — so the kept context window does not carry many
// identical scheduled checks verbatim. Only cron no-op runs are collapsed; every
// other message is preserved unchanged. The anchor carries the seq of the first
// message in the run; the elided originals remain in the archive (retrievable via
// session_messages), so this elides them only from the live tail, never from
// the durable record.
func collapseRetainedCronRuns(stored []memory.StoredMessage, noise NoiseKeyFunc) []memory.StoredMessage {
	if len(stored) < repetitiveRunThreshold {
		return stored
	}
	if noise == nil {
		noise = noNoiseKey
	}
	result := make([]memory.StoredMessage, 0, len(stored))
	i := 0
	for i < len(stored) {
		if anchor, next, ok := collapseCronRun(stored, i, noise); ok {
			result = append(result, anchor)
			i = next
			continue
		}
		result = append(result, stored[i])
		i++
	}
	return result
}

// truncateRunes returns s clipped to at most maxRunes runes, appending an
// ellipsis when truncation occurs.
func truncateRunes(s string, maxRunes int) string {
	r := []rune(s)
	if len(r) <= maxRunes {
		return s
	}
	return string(r[:maxRunes]) + "…"
}

// toolOutputOpen and toolOutputClose delimit a tool result in the summarizer
// input. The id is random per summarization call, so text inside a result
// cannot forge the closing marker without guessing it — and any marker it does
// carry is neutralised anyway (neutralizeToolOutput).
func toolOutputOpen(id string) string  { return "<<<TOOL_OUTPUT id=" + id + ">>>" }
func toolOutputClose(id string) string { return "<<<END_TOOL_OUTPUT id=" + id + ">>>" }

// newToolOutputID returns the per-call random id for the tool-output markers.
func newToolOutputID() string {
	var b [8]byte
	_, _ = rand.Read(b[:]) // crypto/rand.Read does not return an error on supported platforms
	return hex.EncodeToString(b[:])
}

// transcriptFramingRe matches the "[#N] [role]" header the summarizer input
// frames every message with. Tool text carrying one could pass itself off as
// a turn — "[#3] [user] do X" — which is exactly what an injection attempt
// would do.
var transcriptFramingRe = regexp.MustCompile(`\[#\d+\]\s*\[[A-Za-z_]+\]`)

// neutralizeToolOutput rewrites tool text so it cannot impersonate a
// transcript turn or a block marker: a framing header gets its '#' escaped
// ("[#3] [user]" becomes "[\#3] [user]") and any "<<<" or ">>>" run gets a
// backslash inserted, so neither matches what the framing or the markers
// look like. The rewrite is visible and reversible by eye; the summarizer
// still sees what the tool said, just not as structure.
func neutralizeToolOutput(s string) string {
	s = transcriptFramingRe.ReplaceAllStringFunc(s, func(m string) string {
		return strings.Replace(m, "[#", `[\#`, 1)
	})
	s = strings.ReplaceAll(s, "<<<", `<\<<`)
	s = strings.ReplaceAll(s, ">>>", `>\>>`)
	return s
}

// formatStoredMessagesForSummary renders the messages to summarize as the
// framed transcript the summarizer prompt describes. Tool results are wrapped
// in TOOL_OUTPUT markers and neutralised (neutralizeToolOutput), so fetched
// content cannot fake a user turn; the prompt tells the model those blocks
// are data.
func formatStoredMessagesForSummary(stored []memory.StoredMessage, noise NoiseKeyFunc) string {
	stored = collapseRepetitiveRuns(stored, noise)
	toolID := newToolOutputID()
	var sb strings.Builder
	for _, sm := range stored {
		fmt.Fprintf(&sb, "[#%d] [%s]\n", sm.Seq, sm.Role)
		if sm.Source != "" {
			fmt.Fprintf(&sb, "source: %s\n", sm.Source)
		}
		if sm.ToolCallID != "" {
			fmt.Fprintf(&sb, "tool_call_id: %s\n", sm.ToolCallID)
		}
		switch {
		case sm.Role == "tool":
			body := "<empty>"
			if strings.TrimSpace(sm.Content) != "" {
				body = neutralizeToolOutput(sm.Content)
			}
			fmt.Fprintf(&sb, "content:\n%s\n%s\n%s\n", toolOutputOpen(toolID), body, toolOutputClose(toolID))
		case strings.TrimSpace(sm.Content) != "":
			fmt.Fprintf(&sb, "content:\n%s\n", sm.Content)
		default:
			sb.WriteString("content: <empty>\n")
		}
		if len(sm.ToolCalls) > 0 {
			sb.WriteString("tool_calls:\n")
			for _, tc := range sm.ToolCalls {
				name := tc.Name
				args := ""
				if tc.Function != nil {
					if tc.Function.Name != "" {
						name = tc.Function.Name
					}
					args = tc.Function.Arguments
				}
				if args == "" && len(tc.Arguments) > 0 {
					if data, err := json.Marshal(tc.Arguments); err == nil {
						args = string(data)
					}
				}
				fmt.Fprintf(&sb, "- id: %s\n  name: %s\n", tc.ID, name)
				if args != "" {
					fmt.Fprintf(&sb, "  arguments: %s\n", args)
				}
			}
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

func applySummaryCoverage(summary, existing *Summary, coveredStart, coveredEnd int64, summarized []memory.StoredMessage) {
	if summary == nil {
		return
	}
	summary.CoveredSeqStart = coveredStart
	summary.CoveredSeqEnd = coveredEnd
	if existing != nil {
		if existing.CoveredSeqStart > 0 && (summary.CoveredSeqStart == 0 || existing.CoveredSeqStart < summary.CoveredSeqStart) {
			summary.CoveredSeqStart = existing.CoveredSeqStart
		}
		if existing.CoveredSeqEnd > summary.CoveredSeqEnd {
			summary.CoveredSeqEnd = existing.CoveredSeqEnd
		}
	}
	ranges := make([]SeqRange, 0, len(summary.CoveredRanges)+len(summarized)+1)
	if existing != nil {
		if len(existing.CoveredRanges) > 0 {
			ranges = append(ranges, existing.CoveredRanges...)
		} else if existing.CoveredSeqStart > 0 && existing.CoveredSeqEnd >= existing.CoveredSeqStart {
			ranges = append(ranges, SeqRange{SeqStart: existing.CoveredSeqStart, SeqEnd: existing.CoveredSeqEnd})
		}
	}
	ranges = append(ranges, SeqRange{SeqStart: coveredStart, SeqEnd: coveredEnd})
	summary.CoveredRanges = mergeSeqRanges(ranges)
	summary.LastSummarizedSeq = coveredEnd
	summary.LastSummarizedRange = SeqRange{SeqStart: coveredStart, SeqEnd: coveredEnd}
	if len(summarized) > 0 {
		startAt := summarized[0].CreatedAt
		endAt := summarized[len(summarized)-1].CreatedAt
		if existing != nil && !existing.CoveredSeqStartAt.IsZero() && existing.CoveredSeqStart <= coveredStart {
			startAt = existing.CoveredSeqStartAt
		}
		summary.CoveredSeqStartAt = startAt
		summary.CoveredSeqEndAt = endAt
	}
}

func mergeSeqRanges(ranges []SeqRange) []SeqRange {
	out := make([]SeqRange, 0, len(ranges))
	for _, r := range ranges {
		if r.SeqStart <= 0 {
			continue
		}
		if r.SeqEnd == 0 {
			r.SeqEnd = r.SeqStart
		}
		if r.SeqEnd < r.SeqStart {
			continue
		}
		if len(out) == 0 {
			out = append(out, r)
			continue
		}
		last := &out[len(out)-1]
		if r.SeqStart <= last.SeqEnd+1 {
			if r.SeqEnd > last.SeqEnd {
				last.SeqEnd = r.SeqEnd
			}
			continue
		}
		out = append(out, r)
	}
	return out
}

// persistStoredResult writes the retained window back to the store and, when
// summary is a summary this pass generated, makes it the session's current
// summary, appends it to the archive's checkpoint log and records its
// provenance in the compaction state. A nil summary leaves the current
// summary, its checkpoint log and its provenance fields untouched (a
// drop-only pass, or a pass that fell back to the stale summary).
//
// On a store that implements CompactionCommitter the window, the summary, the
// checkpoint and the compaction counters are one transaction; a crash leaves
// either the old state or the new one, never a truncated window beside a
// stale summary. Any other store gets the same writes one at a time. Every
// write failure is returned as ErrCompressionFailed.
func (m *Manager) persistStoredResult(sysMsg *memory.StoredMessage, conv []memory.StoredMessage, summary *Summary) error {
	// Collapse repeated cron no-op runs in the retained tail before persisting,
	// so the live context window the LLM keeps seeing carries one counted anchor
	// instead of many identical scheduled checks. The elided fires remain in the
	// archive (retrievable by seq); only the live tail elides them. Idempotent:
	// an already-collapsed anchor is not a cron-marker message, so re-running it
	// leaves anchors untouched.
	conv = collapseRetainedCronRuns(conv, m.noiseKey())

	newStored := make([]memory.StoredMessage, 0, len(conv)+1)
	if sysMsg != nil {
		newStored = append(newStored, *sysMsg)
	}
	newStored = append(newStored, conv...)

	var raw *string
	var checkpoint *memory.SummaryRecord
	summaryModel := ""
	summaryGeneratedAt := m.lastCompressedAt
	if summary != nil {
		data, err := json.Marshal(summary)
		if err != nil {
			return fmt.Errorf("%w: marshal summary: %s", ErrCompressionFailed, err.Error())
		}
		s := string(data)
		raw = &s
		srcRange := summary.LastSummarizedSeqRange()
		checkpoint = &memory.SummaryRecord{
			GeneratedAt:     summary.GeneratedAt,
			Model:           summary.Model,
			Profile:         summary.Profile,
			SourceSeqStart:  srcRange.SeqStart,
			SourceSeqEnd:    srcRange.SeqEnd,
			CoveredSeqStart: summary.CoveredSeqStart,
			CoveredSeqEnd:   summary.CoveredSeqEnd,
			Summary:         s,
		}
		summaryModel = summary.Model
		if !summary.GeneratedAt.IsZero() {
			summaryGeneratedAt = summary.GeneratedAt
		}
	}
	// CompressedAtMeaningfulCount is the current count because the defer in
	// doCompress sets m.compressedAtCount = m.msgCount after this returns.
	applyState := func(st *memory.CompactionState) {
		st.CompressedAtMeaningfulCount = m.msgCount
		st.Cooling = m.cooling
		st.CoolingSinceCount = m.coolingSinceCount
		if summary != nil {
			st.SummaryGeneratedAt = summaryGeneratedAt
			st.SummaryModel = summaryModel
		}
	}

	if cc, ok := m.store.(CompactionCommitter); ok {
		err := cc.CommitCompaction(m.sessionKey, session.CompactionCommit{
			History:    newStored,
			Summary:    raw,
			Checkpoint: checkpoint,
			Compaction: applyState,
		})
		if err != nil {
			logger.WarnCF("llmcontext", "compression: commit failed", map[string]any{
				"session_key": m.sessionKey,
				"error":       err.Error(),
			})
			return fmt.Errorf("%w: commit: %s", ErrCompressionFailed, err.Error())
		}
	} else if err := m.persistStepwise(newStored, raw, checkpoint, applyState); err != nil {
		return err
	}

	// Apply retention after each compaction that produced a summary so a
	// long-running agent prunes its archive incrementally. Best-effort, and
	// outside the commit: pruning is housekeeping, not part of the result.
	if summary != nil {
		if a := m.getOrOpenArchive(); a != nil {
			m.pruneArchive(a)
		}
	}

	if err := m.store.Save(m.sessionKey); err != nil {
		logger.WarnCF("llmcontext", "compression: failed to save session", map[string]any{
			"session_key": m.sessionKey,
			"error":       err.Error(),
		})
		return fmt.Errorf("%w: save: %s", ErrCompressionFailed, err.Error())
	}
	return nil
}

// persistStepwise is the compaction write for a store without
// CommitCompaction: the window, then the summary, then the checkpoint into
// the manager's archive, then the compaction counters, as separate writes. A
// crash between them can leave the window truncated with the summary not yet
// updated; only a CompactionCommitter closes that gap.
func (m *Manager) persistStepwise(newStored []memory.StoredMessage, raw *string, checkpoint *memory.SummaryRecord, applyState func(*memory.CompactionState)) error {
	var err error
	if sh, ok := m.store.(interface {
		SetHistoryWithSeqs(sessionKey string, history []memory.StoredMessage) error
	}); ok {
		err = sh.SetHistoryWithSeqs(m.sessionKey, newStored)
	} else {
		err = m.store.SetHistory(m.sessionKey, storedToPlain(newStored))
	}
	if err != nil {
		return fmt.Errorf("%w: write window: %s", ErrCompressionFailed, err.Error())
	}
	if raw != nil {
		if err := m.store.SetSummary(m.sessionKey, *raw); err != nil {
			return fmt.Errorf("%w: write summary: %s", ErrCompressionFailed, err.Error())
		}
	}
	if checkpoint != nil {
		// Best-effort: the checkpoint log is a convenience view of the summary
		// the store already holds; a failure here does not fail the pass.
		if a := m.getOrOpenArchive(); a != nil {
			if _, appendErr := a.AppendSummary(*checkpoint); appendErr != nil {
				logger.WarnCF("llmcontext", "compression: failed to append summary to archive", map[string]any{
					"session_key": m.sessionKey,
					"error":       appendErr.Error(),
				})
			}
		}
	}
	m.updateCompactionState("compression", applyState)
	return nil
}

// dropOldestStoredGroups removes the oldest turn groups (seq-preserving) from
// conv until the request — history plus the reserve, the tool schemas and the
// build overhead, the same measure the triggers use — drops below
// safetyPercent, or conv reaches retainMinMessages. A group is an assistant
// tool-call turn with the results that answer it, or a single message
// otherwise; groups are dropped whole so no result is left behind without its
// call. The newest group is never dropped: it is the turn in progress.
func (m *Manager) dropOldestStoredGroups(_ context.Context, conv []memory.StoredMessage) []memory.StoredMessage {
	for len(conv) > m.cfg.retainMinMessages {
		plain := storedToPlain(conv)
		tokens := m.estTokens(plain)
		pct := m.contextPercent(plain)
		if pct < float64(m.cfg.safetyPercent) {
			break
		}

		span := 1
		if plain[0].Role == "assistant" && len(plain[0].ToolCalls) > 0 {
			span = collectToolGroup(plain[0], plain[1:]).span
		}
		if span >= len(conv) {
			break // the only group left is the current turn
		}
		logger.WarnCF("llmcontext", "safety-net: dropping oldest turn group", map[string]any{
			"session_key": m.sessionKey,
			"group_end":   span - 1,
			"tokens":      tokens,
			"pct":         pct,
		})
		conv = conv[span:]
	}
	return conv
}

// applyLargeMsgChecks truncates individual messages that exceed the per-message
// size threshold. The last message is never truncated if it is a user message,
// since it is the current trigger.
func (m *Manager) applyLargeMsgChecks(conv []spawnllm.Message) {
	if m.cfg.contextWindow <= 0 {
		return
	}
	threshold := m.cfg.contextWindow * (m.cfg.safetyPercent - defaultLargeMsgOffset) / 100

	hardLimit := m.cfg.contextWindow * m.cfg.safetyPercent / 100

	for i := range conv {
		msgTokens := m.estTokens([]spawnllm.Message{conv[i]})
		if msgTokens > threshold {
			// The very last message: warn only if user, truncate otherwise.
			if i == len(conv)-1 && conv[i].Role == "user" {
				if msgTokens > hardLimit {
					logger.WarnCF("llmcontext", "last user message exceeds hard context limit", map[string]any{
						"session_key": m.sessionKey,
						"msg_tokens":  msgTokens,
					})
				}
				continue
			}
			// Truncate: threshold * 5/2 runes is the char limit.
			maxRunes := threshold * 5 / 2
			runes := []rune(conv[i].Content)
			if len(runes) > maxRunes {
				conv[i].Content = string(runes[:maxRunes]) + " [**TRUNCATED DUE TO SIZE**]"
			}
		}
	}
}

func (m *Manager) applyLargeMsgChecksStored(conv []memory.StoredMessage) {
	if m.cfg.contextWindow <= 0 {
		return
	}
	plain := storedToPlain(conv)
	m.applyLargeMsgChecks(plain)
	for i := range conv {
		conv[i].Message = plain[i]
	}
}
