// ctxengine
// License: MIT

package ctxengine

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine/logger"
	"github.com/PivotLLM/ctxengine/memory"
)

// CompactionAttempt records the outcome of a single summarization LLM
// invocation during a compaction pass.
type CompactionAttempt struct {
	Model      string `json:"model"`
	Status     string `json:"status"` // "ok" | "rejected" | "error" | "refused" | "skipped"
	Detail     string `json:"detail,omitempty"`
	DurationMs int64  `json:"duration_ms"`
}

// CompactionReport is a human-facing summary of a single compaction pass: the
// context that went in, one entry per LLM invocation (in order, including
// retries against the same model), and the final result. It is assembled on
// every compaction path so the user can see exactly which models were tried,
// which failed and why, and what was burned in the process.
type CompactionReport struct {
	SessionKey  string              `json:"session_key"`
	BeforeMsgs  int                 `json:"before_msgs"`
	BeforeBytes int                 `json:"before_bytes"`
	DateFrom    time.Time           `json:"date_from"`
	DateTo      time.Time           `json:"date_to"`
	Attempts    []CompactionAttempt `json:"attempts"`
	AfterMsgs   int                 `json:"after_msgs"`
	AfterBytes  int                 `json:"after_bytes"`
	Outcome     string              `json:"outcome"` // "success" | "nothing" | "failed" | "partial"
}

// String renders the report as the multi-line notice shown to the user.
func (r *CompactionReport) String() string {
	if r == nil {
		return ""
	}
	var b strings.Builder

	// When the pass exited before reading the conversation (e.g. every model is
	// in cooldown), there are no before-stats — omit the misleading
	// "0 messages (0 B)" and show a bare header.
	var header string
	if r.BeforeMsgs == 0 && r.BeforeBytes == 0 {
		header = "Compaction:"
	} else if dr := formatDateRange(r.DateFrom, r.DateTo); dr != "" {
		header = fmt.Sprintf("Compaction: %d messages (%s)\n%s", r.BeforeMsgs, formatBytes(r.BeforeBytes), dr)
	} else {
		header = fmt.Sprintf("Compaction: %d messages (%s)", r.BeforeMsgs, formatBytes(r.BeforeBytes))
	}
	b.WriteString(header)

	for _, a := range r.Attempts {
		status := a.Status
		if a.Detail != "" {
			status = fmt.Sprintf("%s (%s)", a.Status, a.Detail)
		}
		fmt.Fprintf(&b, "\nModel %s: %s", a.Model, status)
	}

	b.WriteString("\n")
	switch r.Outcome {
	case "success":
		fmt.Fprintf(&b, "Compacted to %d messages (%s)", r.AfterMsgs, formatBytes(r.AfterBytes))
	case "partial":
		fmt.Fprintf(&b, "Partially compacted (still above safety threshold) — %d messages (%s)", r.AfterMsgs, formatBytes(r.AfterBytes))
	case "nothing":
		b.WriteString("Nothing to compress — already compact.")
	default:
		switch {
		case r.hasRefusal():
			b.WriteString("Failed — model(s) refused to summarize this content (content policy). Refusing models will be skipped for this session.")
		case r.hasCooldown():
			b.WriteString("Failed — summarization model unavailable (billing/auth/rate-limit); it is in cooldown and compaction will retry later.")
		default:
			b.WriteString("Failed — no model produced an acceptable summary.")
		}
	}
	return b.String()
}

// hasRefusal reports whether any attempt in the pass was a content refusal.
func (r *CompactionReport) hasRefusal() bool {
	for _, a := range r.Attempts {
		if a.Status == "refused" {
			return true
		}
	}
	return false
}

// hasCooldown reports whether any attempt was skipped for, or failed into, a
// cooldown (billing/auth/rate-limit/overload). The host's ModelCaller names
// the cooldown in its error when it skipped every model for one.
func (r *CompactionReport) hasCooldown() bool {
	for _, a := range r.Attempts {
		if strings.Contains(a.Detail, "cooldown") {
			return true
		}
		if a.Status == "error" && (strings.Contains(a.Detail, "out of credits") ||
			strings.Contains(a.Detail, "rate limited") || strings.Contains(a.Detail, "auth failed") ||
			strings.Contains(a.Detail, "overloaded")) {
			return true
		}
	}
	return false
}

// formatBytes renders a byte count as a compact human-readable string.
func formatBytes(n int) string {
	switch {
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%d KB", n/(1<<10))
	default:
		return fmt.Sprintf("%d B", n)
	}
}

// formatDateRange renders the inclusive span of message timestamps. Returns ""
// when either bound is unset.
func formatDateRange(from, to time.Time) string {
	if from.IsZero() || to.IsZero() {
		return ""
	}
	if from.Year() == to.Year() && from.YearDay() == to.YearDay() {
		return from.Format("Jan 2, 2006")
	}
	return from.Format("Jan 2, 2006") + " to " + to.Format("Jan 2, 2006")
}

// compactionRecorder accumulates per-invocation attempts during a compaction
// pass and, when debug capture is enabled, appends the verbatim request and
// response of each invocation to a JSONL file in the agent workspace.
type compactionRecorder struct {
	sessionKey string
	debugPath  string // "" disables verbatim capture
	// failureDump, when set, receives the request + raw response of each FAILED
	// attempt (status != "ok").
	failureDump FailureDumpFunc
	attempts    []CompactionAttempt
}

// record logs one LLM invocation. req/resp are only persisted when debug
// capture is enabled; they are never retained in memory beyond the call.
func (r *compactionRecorder) record(model, status, detail string, dur time.Duration, req []spawnllm.Message, resp string) {
	if model == "" {
		model = "model"
	}
	r.attempts = append(r.attempts, CompactionAttempt{
		Model:      model,
		Status:     status,
		Detail:     detail,
		DurationMs: dur.Milliseconds(),
	})
	// Always log the per-model outcome (independent of debug capture) so claw.log
	// shows which summarization model succeeded or failed for each attempt.
	if status == "ok" {
		logger.InfoCF("llmcontext", "compression model succeeded", map[string]any{
			"model":       model,
			"session":     r.sessionKey,
			"duration_ms": dur.Milliseconds(),
		})
	} else {
		logger.WarnCF("llmcontext", "compression model failed", map[string]any{
			"model":       model,
			"status":      status,
			"detail":      detail,
			"session":     r.sessionKey,
			"duration_ms": dur.Milliseconds(),
		})
	}
	if r.failureDump != nil && status != "ok" {
		r.dumpFailure(model, status, detail, dur, req, resp)
	}
	if r.debugPath == "" {
		return
	}
	rec := map[string]any{
		"ts":          time.Now().Format(time.RFC3339Nano),
		"session":     r.sessionKey,
		"attempt":     len(r.attempts),
		"model":       model,
		"status":      status,
		"detail":      detail,
		"duration_ms": dur.Milliseconds(),
		"request":     req,
		"response":    resp,
	}
	data, err := json.Marshal(rec)
	if err != nil {
		return
	}
	f, err := os.OpenFile(r.debugPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		logger.WarnCF("llmcontext", "compaction debug capture: open failed", map[string]any{
			"session_key": r.sessionKey,
			"path":        r.debugPath,
			"error":       err.Error(),
		})
		return
	}
	if _, err := f.Write(append(data, '\n')); err != nil {
		logger.WarnCF("llmcontext", "compaction debug capture: write failed", map[string]any{
			"session_key": r.sessionKey,
			"error":       err.Error(),
		})
	}
	if err := f.Close(); err != nil {
		logger.WarnCF("llmcontext", "compaction debug capture: close failed", map[string]any{
			"session_key": r.sessionKey,
			"error":       err.Error(),
		})
	}
}

// dumpFailure hands a diagnostic snapshot of one failed summarization attempt
// (request + raw response) to the host's failure-dump sink. The raw response
// is JSON-encoded as a string so the dump stays valid JSON even when the model
// returned non-JSON (the common failure mode).
func (r *compactionRecorder) dumpFailure(model, status, detail string, dur time.Duration, req []spawnllm.Message, resp string) {
	input := dumpJSON(req)
	output := dumpJSON(resp)
	meta := map[string]any{
		"session":     r.sessionKey,
		"model":       model,
		"status":      status,
		"detail":      detail,
		"duration_ms": dur.Milliseconds(),
	}
	if err := r.failureDump("compress_fail", meta, input, output); err != nil {
		logger.WarnCF("llmcontext", "failed-compression dump: write failed", map[string]any{
			"session_key": r.sessionKey,
			"error":       err.Error(),
		})
	}
}

// dumpJSON encodes v for a failure dump. A value that cannot be encoded is
// recorded as a JSON string naming the encoding error, so the dump is still
// written and still valid JSON.
func dumpJSON(v any) string {
	b, err := json.Marshal(v)
	if err == nil {
		return string(b)
	}
	msg, msgErr := json.Marshal("unencodable: " + err.Error())
	if msgErr != nil {
		return `"unencodable"`
	}
	return string(msg)
}

// buildReport assembles the final report from the per-pass recorder, the
// before-stats captured at the start, and the post-compaction store state. The
// outcome is derived from the doCompress return error.
func (m *Manager) buildReport(rec *compactionRecorder, beforeMsgs, beforeBytes int, from, to time.Time, err error) *CompactionReport {
	afterStored := m.store.GetHistoryWithSeqs(m.sessionKey)
	if len(afterStored) > 0 && afterStored[0].Role == "system" {
		afterStored = afterStored[1:]
	}

	outcome := "success"
	switch {
	case errors.Is(err, ErrNothingToCompress):
		outcome = "nothing"
	case errors.Is(err, ErrCompressionPartial):
		outcome = "partial"
	case err != nil:
		outcome = "failed"
	}

	return &CompactionReport{
		SessionKey:  m.sessionKey,
		BeforeMsgs:  beforeMsgs,
		BeforeBytes: beforeBytes,
		DateFrom:    from,
		DateTo:      to,
		Attempts:    rec.attempts,
		AfterMsgs:   len(afterStored),
		AfterBytes:  storedBytes(afterStored),
		Outcome:     outcome,
	}
}

// shortErr returns a single-line, length-bounded form of an error for display.
func shortErr(err error) string {
	if err == nil {
		return ""
	}
	s := strings.ReplaceAll(err.Error(), "\n", " ")
	if len(s) > 120 {
		s = s[:117] + "..."
	}
	return s
}

// storedBytes sums the content bytes of a stored message slice.
func storedBytes(msgs []memory.StoredMessage) int {
	n := 0
	for _, m := range msgs {
		n += len(m.Content)
	}
	return n
}

// storedDateRange returns the earliest and latest CreatedAt across the slice.
func storedDateRange(msgs []memory.StoredMessage) (from, to time.Time) {
	for _, m := range msgs {
		if m.CreatedAt.IsZero() {
			continue
		}
		if from.IsZero() || m.CreatedAt.Before(from) {
			from = m.CreatedAt
		}
		if to.IsZero() || m.CreatedAt.After(to) {
			to = m.CreatedAt
		}
	}
	return from, to
}
