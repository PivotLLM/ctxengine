/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestLoadCompressionProfile_StripsComments verifies the human-facing comment in
// the default profile never reaches the prompt, while real role guidance does.
func TestLoadCompressionProfile_StripsComments(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "compression.md")

	if err := os.WriteFile(path, []byte("<!--\ndocumentation only\n-->\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := loadCompressionProfile(dir); got != "" {
		t.Errorf("comment-only profile = %q; want empty", got)
	}

	if err := os.WriteFile(path, []byte("<!-- doc -->\nPreserve PR numbers.\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := loadCompressionProfile(dir); got != "Preserve PR numbers." {
		t.Errorf("profile = %q; want %q", got, "Preserve PR numbers.")
	}
}

// TestCompactionReport_Content verifies the report records one entry per
// invocation with the right model labels and a success final line.
func TestCompactionReport_Content(t *testing.T) {
	store := &compressTestStore{history: makeConversation(10, 200)}

	rejecting := &mockLLM{model: "grok-4.3", responses: []string{invalidSummaryJSON("uncited")}}
	valid := &mockLLM{model: "claude-haiku-4-5", responses: []string{validSummaryJSON("goal")}}

	mgr := newCompressManager(t, store, []*mockLLM{rejecting, valid})
	mgr.msgCount = len(store.history)

	if err := mgr.doCompress(context.Background(), false); err != nil {
		t.Fatalf("doCompress error: %v", err)
	}
	rep := mgr.LastCompactionReport()
	if rep == nil {
		t.Fatal("expected a report")
	}
	if len(rep.Attempts) != 2 {
		t.Fatalf("expected 2 attempts; got %d: %+v", len(rep.Attempts), rep.Attempts)
	}
	if rep.Attempts[0].Model != "grok-4.3" || rep.Attempts[0].Status != "rejected" {
		t.Errorf("attempt[0] = %+v; want grok-4.3 rejected", rep.Attempts[0])
	}
	if rep.Attempts[1].Model != "claude-haiku-4-5" || rep.Attempts[1].Status != "ok" {
		t.Errorf("attempt[1] = %+v; want claude-haiku-4-5 ok", rep.Attempts[1])
	}
	if rep.Outcome != "success" {
		t.Errorf("outcome = %q; want success", rep.Outcome)
	}
	s := rep.String()
	for _, want := range []string{"Compaction:", "grok-4.3: rejected (missing citations)", "claude-haiku-4-5: ok", "Compacted to"} {
		if !strings.Contains(s, want) {
			t.Errorf("report string missing %q; got:\n%s", want, s)
		}
	}
}

// TestCompactionReport_DebugCapture verifies that enabling debug capture writes
// one JSONL record per invocation, including the verbatim request and response.
func TestCompactionReport_DebugCapture(t *testing.T) {
	dir := t.TempDir()
	store := &compressTestStore{history: makeConversation(10, 200)}
	llm := &mockLLM{model: "claude-haiku-4-5", responses: []string{validSummaryJSON("goal")}}

	mgr := newCompressManager(t, store, []*mockLLM{llm},
		WithCompressionProfileDir(dir),
		WithCompactDebug(true),
	)
	mgr.msgCount = len(store.history)

	if err := mgr.doCompress(context.Background(), false); err != nil {
		t.Fatalf("doCompress error: %v", err)
	}

	path := filepath.Join(dir, "compact.jsonl")
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("expected compact.jsonl: %v", err)
	}
	defer func() { noErr(t, f.Close()) }()

	lines := 0
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		lines++
		var rec map[string]any
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			t.Fatalf("invalid JSONL line: %v", err)
		}
		if rec["model"] != "claude-haiku-4-5" {
			t.Errorf("record model = %v; want claude-haiku-4-5", rec["model"])
		}
		if _, ok := rec["request"]; !ok {
			t.Error("record missing verbatim request")
		}
		if _, ok := rec["response"]; !ok {
			t.Error("record missing verbatim response")
		}
	}
	if lines == 0 {
		t.Error("expected at least one debug record")
	}
}

// invalidSummaryJSON produces a Summary that carries material (a goal) but no
// cited evidence (the goal has no refs), so it fails HasEvidence() and must be
// rejected by callModel.
func invalidSummaryJSON(goals string) string {
	return `{"version":2,"state":{"goals":[{"text":"` + goals + `"}]},"covered_seq_start":0,"covered_seq_end":0}`
}

// TestCompress_ValidationFallback verifies that when the first client returns a
// summary that fails validation (no cited material), the chain advances to the
// next client instead of re-trying the same one. This is the Mode-2 fix: before,
// validation ran outside the model call and a rejected summary never fell through.
func TestCompress_ValidationFallback(t *testing.T) {
	store := &compressTestStore{history: makeConversation(10, 200)}

	// Named models: the engine falls through by excluding the rejected model
	// on its next request, which needs a name to exclude.
	rejecting := &mockLLM{model: "rejecting", responses: []string{
		invalidSummaryJSON("uncited"),
		invalidSummaryJSON("uncited"),
		invalidSummaryJSON("uncited"),
	}}
	valid := &mockLLM{model: "valid", responses: []string{validSummaryJSON("fallback goal")}}

	mgr := newCompressManager(t, store, []*mockLLM{rejecting, valid})
	mgr.msgCount = len(store.history)

	err := mgr.doCompress(context.Background(), false)
	if err != nil {
		t.Fatalf("doCompress returned error: %v", err)
	}
	if rejecting.callCount == 0 {
		t.Error("expected the rejecting client to be tried")
	}
	if valid.callCount == 0 {
		t.Error("expected the chain to fall through to the valid client")
	}
	if !strings.Contains(store.summary, "fallback goal") {
		t.Errorf("expected summary from the valid client; got %q", store.summary)
	}
}

// TestCompress_NothingToCompress verifies that a conversation small enough that
// the retained tail already covers everything returns ErrNothingToCompress (a
// benign no-op) without invoking the LLM.
func TestCompress_NothingToCompress(t *testing.T) {
	store := &compressTestStore{history: makeConversation(1, 20)} // 2 short messages
	llm := &mockLLM{responses: []string{validSummaryJSON("unused")}}

	mgr := newCompressManager(t, store, []*mockLLM{llm})
	mgr.msgCount = len(store.history)

	err := mgr.doCompress(context.Background(), false)
	if !errors.Is(err, ErrNothingToCompress) {
		t.Fatalf("expected ErrNothingToCompress; got %v", err)
	}
	if llm.callCount != 0 {
		t.Errorf("expected no LLM calls; got %d", llm.callCount)
	}
}

// TestCompress_AllRejected_Failed verifies that when the LLM is invoked but every
// returned summary fails validation, ErrCompressionFailed is returned (distinct
// from the no-op ErrNothingToCompress).
func TestCompress_AllRejected_Failed(t *testing.T) {
	origHistory := makeConversation(10, 200)
	store := &compressTestStore{history: origHistory}

	rejecting := &mockLLM{responses: []string{
		invalidSummaryJSON("uncited"),
		invalidSummaryJSON("uncited"),
		invalidSummaryJSON("uncited"),
		invalidSummaryJSON("uncited"),
		invalidSummaryJSON("uncited"),
		invalidSummaryJSON("uncited"),
	}}

	mgr := newCompressManager(t, store, []*mockLLM{rejecting})
	mgr.msgCount = len(store.history)

	err := mgr.doCompress(context.Background(), false)
	if !errors.Is(err, ErrCompressionFailed) {
		t.Fatalf("expected ErrCompressionFailed; got %v", err)
	}
	if rejecting.callCount == 0 {
		t.Error("expected the LLM to be invoked")
	}
	if len(store.history) != len(origHistory) {
		t.Errorf("expected history unchanged (%d); got %d", len(origHistory), len(store.history))
	}
}

// TestBreaker_RecordOutcome unit-tests the failure circuit breaker accounting.
func TestBreaker_RecordOutcome(t *testing.T) {
	store := &compressTestStore{}
	mgr := newCompressManager(t, store, []*mockLLM{{}})

	// Failures accumulate and trip the breaker at the threshold.
	for i := range defaultMaxConsecutiveCompactFailures {
		if mgr.autoCompactionSuppressed() {
			t.Fatalf("breaker tripped early after %d failures", i)
		}
		mgr.recordCompactionOutcome(ErrCompressionFailed)
	}
	if !mgr.autoCompactionSuppressed() {
		t.Fatal("expected breaker to be tripped after threshold failures")
	}

	// A benign no-op clears the breaker.
	mgr.recordCompactionOutcome(ErrNothingToCompress)
	if mgr.autoCompactionSuppressed() {
		t.Fatal("expected ErrNothingToCompress to reset the breaker")
	}
	if mgr.consecutiveCompactFailures != 0 {
		t.Errorf("expected failure counter reset; got %d", mgr.consecutiveCompactFailures)
	}

	// ErrNothingToCompress must not count as a failure.
	for range defaultMaxConsecutiveCompactFailures + 2 {
		mgr.recordCompactionOutcome(ErrNothingToCompress)
	}
	if mgr.autoCompactionSuppressed() {
		t.Fatal("ErrNothingToCompress should never trip the breaker")
	}

	// Success resets too.
	mgr.recordCompactionOutcome(ErrCompressionFailed)
	mgr.recordCompactionOutcome(nil)
	if mgr.consecutiveCompactFailures != 0 {
		t.Errorf("expected success to reset failure counter; got %d", mgr.consecutiveCompactFailures)
	}
}

// TestBreaker_SuppressesAutoPath verifies that once tripped, the automatic
// compress() path stops invoking the LLM, while a manual Compact() still runs and
// resets the breaker.
func TestBreaker_SuppressesAutoPath(t *testing.T) {
	store := &compressTestStore{history: makeConversation(10, 200)}
	failing := &mockLLM{errors: []error{
		errors.New("f1"), errors.New("f2"), errors.New("f3"),
		errors.New("f4"), errors.New("f5"), errors.New("f6"),
		errors.New("f7"), errors.New("f8"), errors.New("f9"),
		errors.New("f10"), errors.New("f11"), errors.New("f12"),
		errors.New("f13"), errors.New("f14"), errors.New("f15"),
		errors.New("f16"), errors.New("f17"), errors.New("f18"),
	}}

	mgr := newCompressManager(t, store, []*mockLLM{failing})
	mgr.msgCount = len(store.history)

	// Trip the breaker with consecutive automatic-compaction failures.
	for range defaultMaxConsecutiveCompactFailures {
		if err := mgr.compress(context.Background(), false); err == nil {
			t.Fatal("compress against a failing model returned nil")
		}
	}
	if !mgr.autoCompactionSuppressed() {
		t.Fatal("expected breaker tripped after repeated auto failures")
	}

	callsBefore := failing.callCount
	// A further automatic attempt must be suppressed (no new LLM calls).
	noErr(t, mgr.compress(context.Background(), false))
	if failing.callCount != callsBefore {
		t.Errorf("expected auto path suppressed; LLM called %d more times", failing.callCount-callsBefore)
	}

	// A manual /compact bypasses the breaker; a successful one resets it.
	store2 := &compressTestStore{history: makeConversation(10, 200)}
	good := &mockLLM{responses: []string{validSummaryJSON("manual goal")}}
	// Reuse the tripped manager's state by pointing it at a fresh store+client.
	mgr.store = store2
	mgr.caller = chainOf([]*mockLLM{good})
	if err := mgr.Compact(context.Background()); err != nil {
		t.Fatalf("manual Compact returned error: %v", err)
	}
	if good.callCount == 0 {
		t.Error("expected manual Compact to bypass the breaker and call the LLM")
	}
	if mgr.autoCompactionSuppressed() {
		t.Error("expected a successful manual Compact to reset the breaker")
	}
}
