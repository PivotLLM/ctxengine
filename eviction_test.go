// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine/memory"
)

// --- test builders -------------------------------------------------------

// evToolCall builds a stored ToolCall (Function-form, as it is persisted) for a
// reader/writer tool with the given args.
func evToolCall(id, tool string, args map[string]any) spawnllm.ToolCall {
	b, err := json.Marshal(args)
	if err != nil {
		panic(fmt.Sprintf("evToolCall: marshal %v: %v", args, err))
	}
	return spawnllm.ToolCall{ID: id, Function: &spawnllm.FunctionCall{Name: tool, Arguments: string(b)}}
}

// turnSpec describes one turn group in a synthetic history (oldest first).
type turnSpec struct {
	// tool turn: assistant(ToolCalls=[tc]) + tool result with content.
	tool    string
	args    map[string]any
	id      string
	content string
	// text turn: a single assistant text message (tool == "").
	text string
}

// buildHistory turns specs (oldest→newest) into a stored slice with seqs. Each
// tool spec is two messages (one turn group); each text spec is one.
func buildHistory(specs ...turnSpec) []memory.StoredMessage {
	var msgs []spawnllm.Message
	for _, s := range specs {
		if s.tool == "" {
			msgs = append(msgs, spawnllm.Message{Role: "assistant", Content: s.text})
			continue
		}
		id := s.id
		if id == "" {
			id = s.tool + "-" + s.content[:min(4, len(s.content))]
		}
		msgs = append(msgs,
			spawnllm.Message{Role: "assistant", ToolCalls: []spawnllm.ToolCall{evToolCall(id, s.tool, s.args)}},
			spawnllm.Message{Role: "tool", ToolCallID: id, Content: s.content},
		)
	}
	out := make([]memory.StoredMessage, len(msgs))
	for i, m := range msgs {
		out[i] = memory.StoredMessage{Seq: int64(i + 1), Message: m}
	}
	return out
}

// markToolError flags the tool result for the given call id as an error (as the
// agent loop does for a failed tool), so the sweep treats its write as unapplied.
func markToolError(h []memory.StoredMessage, id string) {
	for i := range h {
		if h[i].Role == "tool" && h[i].ToolCallID == id {
			h[i].Type = MessageTypeToolError
		}
	}
}

func newEvictMgr(t *testing.T, store *seqStore, p EvictionPolicy) *Manager {
	t.Helper()
	// WithContextWindow(0) disables the derived budget so non-budget tests are
	// deterministic; budget tests set BudgetBytes explicitly.
	return asManager(t, New("sess", store, WithContextWindow(0), WithEvictionPolicy(p)))
}

// findToolResult returns the content of the tool result for the given call id.
func findToolResult(store *seqStore, id string) string {
	for _, sm := range store.GetHistoryWithSeqs("sess") {
		if sm.Role == "tool" && sm.ToolCallID == id {
			return sm.Content
		}
	}
	return ""
}

func basePolicy() EvictionPolicy {
	return EvictionPolicy{Enabled: true, ProtectTurns: 3, EvictTurns: 10, BudgetBytes: 0}
}

// --- tests ---------------------------------------------------------------

func TestSweep_Disabled(t *testing.T) {
	p := basePolicy()
	p.Enabled = false
	store := newSeqStore(buildHistory(
		turnSpec{tool: "file_read_bytes", id: "r", args: map[string]any{"path": "a.md"}, content: strings.Repeat("x", 500)},
		turnSpec{text: "1"}, turnSpec{text: "2"}, turnSpec{text: "3"},
		turnSpec{text: "4"}, turnSpec{text: "5"}, turnSpec{text: "6"},
	))
	events := newEvictMgr(t, store, p).SweepEvictions(context.Background())
	if events != nil {
		t.Fatalf("disabled policy must not evict; got %d events", len(events))
	}
	if got := findToolResult(store, "r"); isEvicted(got) {
		t.Fatalf("content evicted while disabled")
	}
}

func TestSweep_ProtectsRecent(t *testing.T) {
	// A large read in the last protect (3) turns must survive.
	store := newSeqStore(buildHistory(
		turnSpec{text: "old1"}, turnSpec{text: "old2"},
		turnSpec{tool: "file_read_bytes", id: "r", args: map[string]any{"path": "a.md"}, content: strings.Repeat("x", 500)}, // age 3
		turnSpec{text: "n1"}, turnSpec{text: "n2"},
	))
	events := newEvictMgr(t, store, basePolicy()).SweepEvictions(context.Background())
	if len(events) != 0 {
		t.Fatalf("protected read evicted: %+v", events)
	}
	if isEvicted(findToolResult(store, "r")) {
		t.Fatalf("protected read content evicted")
	}
}

func TestSweep_MidAgeReadKeptWithoutBudget(t *testing.T) {
	// A large, non-superseded read in the band between protect and evict is NOT
	// evicted on age alone (no "large" tier) — only the budget valve sheds it,
	// and budget is disabled here. It stays until the expiry age or real pressure.
	store := newSeqStore(buildHistory(
		turnSpec{text: "0"}, turnSpec{text: "1"},
		turnSpec{tool: "file_read_bytes", id: "r", args: map[string]any{"path": "ch17.md"}, content: strings.Repeat("x", 9000)}, // age 6
		turnSpec{text: "3"}, turnSpec{text: "4"}, turnSpec{text: "5"}, turnSpec{text: "6"}, turnSpec{text: "7"},
	))
	events := newEvictMgr(t, store, basePolicy()).SweepEvictions(context.Background())
	if len(events) != 0 {
		t.Fatalf("mid-age read evicted without budget pressure: %+v", events)
	}
	if isEvicted(findToolResult(store, "r")) {
		t.Fatalf("mid-age large read evicted on age alone")
	}
}

func TestSweep_StaleAnySize(t *testing.T) {
	// Small read at age 12 (> evict 10) → evicted "stale" regardless of size.
	specs := make([]turnSpec, 0, 1+11)
	specs = append(specs, turnSpec{tool: "file_read_bytes", id: "r", args: map[string]any{"path": "a.md"}, content: "tiny"})
	for range 11 { // 11 newer text turns ⇒ read age 12
		specs = append(specs, turnSpec{text: "t"})
	}
	store := newSeqStore(buildHistory(specs...))
	events := newEvictMgr(t, store, basePolicy()).SweepEvictions(context.Background())
	if len(events) != 1 || events[0].Reason != "stale" {
		t.Fatalf("want 1 stale eviction, got %+v", events)
	}
	if !isEvicted(findToolResult(store, "r")) {
		t.Fatalf("stale read content not evicted")
	}
}

func TestSweep_SupersededByReread(t *testing.T) {
	// Old small read of a.md, later re-read of a.md (outside protect) → old one
	// evicted "superseded" despite being small.
	store := newSeqStore(buildHistory(
		turnSpec{tool: "file_read_bytes", id: "old", args: map[string]any{"path": "a.md"}, content: "tiny"}, // age 5
		turnSpec{text: "1"}, turnSpec{text: "2"}, turnSpec{text: "3"},
		turnSpec{tool: "file_read_bytes", id: "new", args: map[string]any{"path": "a.md"}, content: "fresh"}, // age 1
	))
	events := newEvictMgr(t, store, basePolicy()).SweepEvictions(context.Background())
	// The evicted message is the tool *result* (seq 2: assistant=1, result=2).
	if len(events) != 1 || events[0].Reason != "superseded" || events[0].Seq != 2 {
		t.Fatalf("want old read superseded, got %+v", events)
	}
	if !isEvicted(findToolResult(store, "old")) {
		t.Fatalf("superseded read not evicted")
	}
	if isEvicted(findToolResult(store, "new")) {
		t.Fatalf("fresh read wrongly evicted")
	}
}

func TestSweep_SupersededByEdit(t *testing.T) {
	store := newSeqStore(buildHistory(
		turnSpec{tool: "file_read_bytes", id: "r", args: map[string]any{"path": "a.md"}, content: "tiny"}, // age 5
		turnSpec{text: "1"}, turnSpec{text: "2"}, turnSpec{text: "3"},
		turnSpec{tool: "file_edit", id: "e", args: map[string]any{"path": "a.md"}, content: "ok"}, // age 1, writer
	))
	events := newEvictMgr(t, store, basePolicy()).SweepEvictions(context.Background())
	if len(events) != 1 || events[0].Reason != "superseded" {
		t.Fatalf("want read superseded by edit, got %+v", events)
	}
}

func TestSweep_SupersededKeptInsideProtect(t *testing.T) {
	// Supersession now respects the protect window: a read at age 3 (within protect
	// 3) superseded by a later successful edit is KEPT, so the model can read a file
	// once and make several edits from it without the read being evicted underneath.
	store := newSeqStore(buildHistory(
		turnSpec{tool: "file_read_bytes", id: "r", args: map[string]any{"path": "a.md"}, content: strings.Repeat("x", 500)}, // age 3
		turnSpec{text: "1"},
		turnSpec{tool: "file_edit", id: "e", args: map[string]any{"path": "a.md"}, content: "ok"}, // age 1, successful write
	))
	events := newEvictMgr(t, store, basePolicy()).SweepEvictions(context.Background())
	if len(events) != 0 {
		t.Fatalf("recent superseded read should be kept, got %+v", events)
	}
	if isEvicted(findToolResult(store, "r")) {
		t.Fatalf("recent superseded read wrongly evicted")
	}
}

func TestSweep_SupersededEvictedPastProtect(t *testing.T) {
	// Past the protect window a superseded read is a pure stale duplicate and is
	// dropped: read at age 5 superseded by a later successful edit.
	store := newSeqStore(buildHistory(
		turnSpec{tool: "file_read_bytes", id: "r", args: map[string]any{"path": "a.md"}, content: strings.Repeat("x", 500)}, // age 5
		turnSpec{text: "1"}, turnSpec{text: "2"}, turnSpec{text: "3"},
		turnSpec{tool: "file_edit", id: "e", args: map[string]any{"path": "a.md"}, content: "ok"}, // age 1
	))
	events := newEvictMgr(t, store, basePolicy()).SweepEvictions(context.Background())
	if len(events) != 1 || events[0].Reason != "superseded" {
		t.Fatalf("want superseded read evicted past protect, got %+v", events)
	}
	if !isEvicted(findToolResult(store, "r")) {
		t.Fatalf("superseded read past protect not evicted")
	}
}

func TestSweep_RecentDuplicatesKeptOldDropped(t *testing.T) {
	// Re-reads of the same slice: duplicates PAST the protect window are dropped,
	// but recent ones (inside protect) are kept as the working set. Four reads of
	// the same page of a.md: only r1 (age 5) is evicted; r2/r3 (recent duplicates)
	// and r4 (latest) survive.
	store := newSeqStore(buildHistory(
		turnSpec{tool: "file_read_bytes", id: "r1", args: map[string]any{"path": "a.md"}, content: strings.Repeat("x", 500)}, // age 5 (past protect)
		turnSpec{text: "1"},
		turnSpec{tool: "file_read_bytes", id: "r2", args: map[string]any{"path": "a.md"}, content: strings.Repeat("y", 500)}, // age 3 (protected)
		turnSpec{tool: "file_read_bytes", id: "r3", args: map[string]any{"path": "a.md"}, content: strings.Repeat("z", 500)}, // age 2 (protected)
		turnSpec{tool: "file_read_bytes", id: "r4", args: map[string]any{"path": "a.md"}, content: strings.Repeat("w", 500)}, // age 1 (latest)
	))
	events := newEvictMgr(t, store, basePolicy()).SweepEvictions(context.Background())
	if len(events) != 1 || events[0].Reason != "superseded" {
		t.Fatalf("want only the past-protect duplicate evicted, got %d: %+v", len(events), events)
	}
	if !isEvicted(findToolResult(store, "r1")) {
		t.Fatalf("past-protect duplicate r1 not evicted")
	}
	for _, id := range []string{"r2", "r3", "r4"} {
		if isEvicted(findToolResult(store, id)) {
			t.Fatalf("recent read %s wrongly evicted", id)
		}
	}
}

func TestSweep_ProtectGuardsBudget(t *testing.T) {
	// Protect still shields recent reads from the budget valve. A protected read
	// (age 1, distinct path, not superseded) must survive even when the window is
	// over budget; only the older non-protected candidate is evicted.
	p := basePolicy()
	p.EvictTurns = 100 // disable stale tier so only budget acts
	p.BudgetBytes = 100
	store := newSeqStore(buildHistory(
		turnSpec{tool: "file_read_bytes", id: "old", args: map[string]any{"path": "b.md"}, content: strings.Repeat("x", 250)}, // age 5 candidate
		turnSpec{text: "1"}, turnSpec{text: "2"}, turnSpec{text: "3"},
		turnSpec{tool: "file_read_bytes", id: "recent", args: map[string]any{"path": "a.md"}, content: strings.Repeat("x", 250)}, // age 1 protected
	))
	events := newEvictMgr(t, store, p).SweepEvictions(context.Background())
	if len(events) != 1 || events[0].Reason != "budget" {
		t.Fatalf("want 1 budget eviction, got %+v", events)
	}
	if isEvicted(findToolResult(store, "recent")) {
		t.Fatalf("protected recent read evicted by budget")
	}
	if !isEvicted(findToolResult(store, "old")) {
		t.Fatalf("old candidate not evicted by budget")
	}
}

func TestSweep_Budget(t *testing.T) {
	// Three reads at ages 4–6, none large/stale/superseded → budget candidates.
	// Sizes 250+200+150=600 > budget 300 ⇒ evict largest-first (250, 200), keep 150.
	p := basePolicy()
	p.EvictTurns = 100 // disable the stale tier so only budget acts
	p.BudgetBytes = 300
	store := newSeqStore(buildHistory(
		turnSpec{tool: "file_read_bytes", id: "a", args: map[string]any{"path": "a.md"}, content: strings.Repeat("x", 250)}, // age 6
		turnSpec{tool: "file_read_bytes", id: "b", args: map[string]any{"path": "b.md"}, content: strings.Repeat("x", 200)}, // age 5
		turnSpec{tool: "file_read_bytes", id: "c", args: map[string]any{"path": "c.md"}, content: strings.Repeat("x", 150)}, // age 4
		turnSpec{text: "1"}, turnSpec{text: "2"}, turnSpec{text: "3"}, // ages 1-3 protected, no reader bytes
	))
	events := newEvictMgr(t, store, p).SweepEvictions(context.Background())
	if len(events) != 2 {
		t.Fatalf("want 2 budget evictions, got %+v", events)
	}
	for _, e := range events {
		if e.Reason != "budget" {
			t.Fatalf("non-budget reason: %+v", e)
		}
	}
	if !isEvicted(findToolResult(store, "a")) || !isEvicted(findToolResult(store, "b")) {
		t.Fatalf("largest reads not evicted")
	}
	if isEvicted(findToolResult(store, "c")) {
		t.Fatalf("smallest read evicted under budget")
	}
}

func TestSweep_NonReaderUntouched(t *testing.T) {
	// A non-reader tool result (msg_send) old + large is never evicted.
	store := newSeqStore(buildHistory(
		turnSpec{text: "0"}, turnSpec{text: "1"},
		turnSpec{tool: "msg_send", id: "m", args: map[string]any{"text": "hi"}, content: strings.Repeat("x", 5000)}, // age 6
		turnSpec{text: "3"}, turnSpec{text: "4"}, turnSpec{text: "5"}, turnSpec{text: "6"}, turnSpec{text: "7"},
	))
	events := newEvictMgr(t, store, basePolicy()).SweepEvictions(context.Background())
	if len(events) != 0 {
		t.Fatalf("non-reader evicted: %+v", events)
	}
}

func TestSweep_Idempotent(t *testing.T) {
	// Read at age 12 (> evict 10) → stale eviction on the first sweep, no-op after.
	specs := make([]turnSpec, 0, 1+11)
	specs = append(specs, turnSpec{tool: "file_read_bytes", id: "r", args: map[string]any{"path": "a.md"}, content: strings.Repeat("x", 200)})
	for range 11 {
		specs = append(specs, turnSpec{text: "t"})
	}
	store := newSeqStore(buildHistory(specs...))
	mgr := newEvictMgr(t, store, basePolicy())
	if got := mgr.SweepEvictions(context.Background()); len(got) != 1 {
		t.Fatalf("first sweep want 1, got %d", len(got))
	}
	if got := mgr.SweepEvictions(context.Background()); len(got) != 0 {
		t.Fatalf("second sweep must be no-op, got %+v", got)
	}
}

func TestSweep_EvictedStaysEvicted(t *testing.T) {
	// A read evicted young (superseded) stays evicted after it ages past
	// evict_turns: the content-based guard is tier/age independent, so it is
	// never re-evicted or re-reported.
	store := newSeqStore(buildHistory(
		turnSpec{tool: "file_read_bytes", id: "old", args: map[string]any{"path": "a.md"}, content: strings.Repeat("x", 500)}, // age 5, superseded
		turnSpec{text: "1"}, turnSpec{text: "2"}, turnSpec{text: "3"},
		turnSpec{tool: "file_read_bytes", id: "new", args: map[string]any{"path": "a.md"}, content: strings.Repeat("y", 500)}, // latest, kept
	))
	mgr := newEvictMgr(t, store, basePolicy())

	first := mgr.SweepEvictions(context.Background())
	if len(first) != 1 || first[0].Reason != "superseded" {
		t.Fatalf("first sweep: want 1 superseded eviction, got %+v", first)
	}
	oldSeq := first[0].Seq
	placeholder := findToolResult(store, "old")

	// Age both reads past evict_turns by appending more turns.
	aged := store.GetHistoryWithSeqs("sess")
	base := aged[len(aged)-1].Seq
	for i := range 12 {
		aged = append(aged, memory.StoredMessage{
			Seq:     base + int64(i+1),
			Message: spawnllm.Message{Role: "assistant", Content: "more"},
		})
	}
	noErr(t, store.SetHistoryWithSeqs("sess", aged))

	// The already-evicted "old" read must never reappear; only "new" (now stale)
	// may be newly evicted.
	for _, e := range mgr.SweepEvictions(context.Background()) {
		if e.Seq == oldSeq {
			t.Fatalf("already-evicted read re-evicted after aging: %+v", e)
		}
	}
	if got := findToolResult(store, "old"); got != placeholder {
		t.Fatalf("placeholder changed on re-sweep:\n  before: %q\n  after:  %q", placeholder, got)
	}
}

func TestSweep_FailedWriteDoesNotSupersede(t *testing.T) {
	// A (loop fix): a read past the protect window followed by a FAILED edit to the
	// same file must NOT be evicted — the failed edit left the file unchanged, so
	// the read is still the current view the model needs to build a correct edit.
	h := buildHistory(
		turnSpec{tool: "file_read_bytes", id: "r", args: map[string]any{"path": "a.md"}, content: strings.Repeat("x", 500)}, // age 5
		turnSpec{text: "1"}, turnSpec{text: "2"}, turnSpec{text: "3"},
		turnSpec{tool: "file_edit", id: "e", args: map[string]any{"path": "a.md"}, content: "old_text not found"}, // age 1, FAILED
	)
	markToolError(h, "e")
	store := newSeqStore(h)
	events := newEvictMgr(t, store, basePolicy()).SweepEvictions(context.Background())
	if len(events) != 0 {
		t.Fatalf("read evicted by a FAILED edit (the loop bug): %+v", events)
	}
	if isEvicted(findToolResult(store, "r")) {
		t.Fatalf("read wrongly evicted by a failed edit")
	}
}

func TestSweep_RecentSupersededEvictedUnderBudget(t *testing.T) {
	// B: a superseded read inside the protect window is kept normally but remains a
	// budget candidate, so a suddenly-bloated context still reclaims it.
	p := basePolicy()
	p.BudgetBytes = 100
	store := newSeqStore(buildHistory(
		turnSpec{tool: "file_read_bytes", id: "r", args: map[string]any{"path": "a.md"}, content: strings.Repeat("x", 500)}, // age 3, superseded, within protect
		turnSpec{text: "1"},
		turnSpec{tool: "file_edit", id: "e", args: map[string]any{"path": "a.md"}, content: "ok"}, // age 1
	))
	events := newEvictMgr(t, store, p).SweepEvictions(context.Background())
	if len(events) != 1 || events[0].Reason != "superseded" {
		t.Fatalf("want recent superseded read reclaimed under budget, got %+v", events)
	}
	if !isEvicted(findToolResult(store, "r")) {
		t.Fatalf("recent superseded read not reclaimed under budget pressure")
	}
}

func TestSweep_DifferentSlicesCoexist(t *testing.T) {
	// C: reading a different page (start_line) of the same file is not a duplicate;
	// both pages survive, so a large file read in chunks stays available to edit.
	store := newSeqStore(buildHistory(
		turnSpec{tool: "file_read_lines", id: "p1", args: map[string]any{"path": "big.md", "start_line": 1.0}, content: strings.Repeat("x", 500)}, // age 5, page 1
		turnSpec{text: "1"}, turnSpec{text: "2"}, turnSpec{text: "3"},
		turnSpec{tool: "file_read_lines", id: "p2", args: map[string]any{"path": "big.md", "start_line": 716.0}, content: strings.Repeat("y", 500)}, // age 1, page 2
	))
	events := newEvictMgr(t, store, basePolicy()).SweepEvictions(context.Background())
	if len(events) != 0 {
		t.Fatalf("distinct pages of one file evicted: %+v", events)
	}
	if isEvicted(findToolResult(store, "p1")) {
		t.Fatalf("page 1 evicted just because page 2 was read")
	}
}

func TestSweep_SameSliceRereadSupersedes(t *testing.T) {
	// C companion: re-reading the SAME page past the protect window supersedes the
	// earlier read of that page (a genuine refresh).
	store := newSeqStore(buildHistory(
		turnSpec{tool: "file_read_lines", id: "old", args: map[string]any{"path": "big.md", "start_line": 1.0}, content: strings.Repeat("x", 500)}, // age 5
		turnSpec{text: "1"}, turnSpec{text: "2"}, turnSpec{text: "3"},
		turnSpec{tool: "file_read_lines", id: "new", args: map[string]any{"path": "big.md", "start_line": 1.0}, content: strings.Repeat("y", 500)}, // age 1, same page
	))
	events := newEvictMgr(t, store, basePolicy()).SweepEvictions(context.Background())
	if len(events) != 1 || events[0].Reason != "superseded" {
		t.Fatalf("want same-page re-read to supersede the old read, got %+v", events)
	}
	if !isEvicted(findToolResult(store, "old")) || isEvicted(findToolResult(store, "new")) {
		t.Fatalf("wrong read evicted: old should go, new should stay")
	}
}

func TestSweep_WriteInvalidatesAllSlices(t *testing.T) {
	// C: a successful edit invalidates reads of ANY page of the file (line numbers
	// shift), not just the same slice. Page 1 read past protect is evicted by a
	// later successful edit even though the edit names no range.
	store := newSeqStore(buildHistory(
		turnSpec{tool: "file_read_lines", id: "p1", args: map[string]any{"path": "big.md", "start_line": 1.0}, content: strings.Repeat("x", 500)}, // age 5
		turnSpec{text: "1"}, turnSpec{text: "2"}, turnSpec{text: "3"},
		turnSpec{tool: "file_edit_lines", id: "e", args: map[string]any{"path": "big.md", "start": 10.0}, content: "Replaced lines 10-12"}, // age 1, writer
	))
	events := newEvictMgr(t, store, basePolicy()).SweepEvictions(context.Background())
	if len(events) != 1 || events[0].Reason != "superseded" {
		t.Fatalf("want page-1 read invalidated by a successful edit, got %+v", events)
	}
	if !isEvicted(findToolResult(store, "p1")) {
		t.Fatalf("read not invalidated by a successful edit to the file")
	}
}

func TestEvictionEvent_String(t *testing.T) {
	e := EvictionEvent{Tool: "file_read_bytes", Resource: "files/ch17.md", Bytes: 18432, AgeTurns: 6, Reason: "large"}
	want := "[Evicted 18432 bytes at 6 turns (large): file_read_bytes files/ch17.md]"
	if got := e.String(); got != want {
		t.Fatalf("String() = %q, want %q", got, want)
	}
}

func TestEvictionEvent_URLCap(t *testing.T) {
	longURL := "https://example.com/" + strings.Repeat("a", 200)
	e := EvictionEvent{Tool: "web_fetch", Resource: longURL, Bytes: 100, AgeTurns: 7, Reason: "stale"}
	got := e.String()
	if !strings.Contains(got, "…") {
		t.Fatalf("long URL not capped: %q", got)
	}
	if strings.Contains(got, strings.Repeat("a", 200)) {
		t.Fatalf("full long URL leaked into notice")
	}
}
