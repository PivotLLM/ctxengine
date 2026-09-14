// ctxengine
// License: MIT

package session_test

import (
	"testing"

	"github.com/PivotLLM/ctxengine/memory"
	"github.com/PivotLLM/spawnllm"
)

// These cover the JSONL persistence backend's previously-untested paths: the
// seq-aware history accessors, compaction-state round-trip, pending-turn
// bookkeeping, archive bounds, and per-session forget. A regression in any of
// these silently corrupts or loses durable conversation state.

func TestJSONLBackend_HistoryWithSeqs(t *testing.T) {
	b := newBackend(t)
	b.AddMessage("s1", "user", "one")
	b.AddMessage("s1", "assistant", "two")

	stored := b.GetHistoryWithSeqs("s1")
	if len(stored) != 2 {
		t.Fatalf("got %d stored messages, want 2", len(stored))
	}
	if stored[0].Seq <= 0 || stored[1].Seq <= stored[0].Seq {
		t.Errorf("seqs must be positive and increasing: %d, %d", stored[0].Seq, stored[1].Seq)
	}
	if stored[0].Content != "one" || stored[1].Content != "two" {
		t.Errorf("content not preserved: %+v", stored)
	}

	// Unknown session yields an empty slice, not nil-deref.
	if got := b.GetHistoryWithSeqs("nope"); len(got) != 0 {
		t.Errorf("unknown session should be empty, got %d", len(got))
	}
}

func TestJSONLBackend_SetHistoryWithSeqs_Replaces(t *testing.T) {
	b := newBackend(t)
	b.AddMessage("s1", "user", "old")

	replacement := []memory.StoredMessage{
		memory.NewStoredMessage(1, spawnllm.Message{Role: "user", Content: "fresh"}),
		memory.NewStoredMessage(2, spawnllm.Message{Role: "assistant", Content: "reply"}),
	}
	b.SetHistoryWithSeqs("s1", replacement)

	got := b.GetHistory("s1")
	if len(got) != 2 || got[0].Content != "fresh" || got[1].Content != "reply" {
		t.Errorf("history not replaced: %+v", got)
	}
}

func TestJSONLBackend_CompactionStateRoundTrip(t *testing.T) {
	b := newBackend(t)
	b.AddMessage("s1", "user", "hi") // ensure the session exists on disk

	want := memory.CompactionState{
		MeaningfulCount:             7,
		CompressedAtMeaningfulCount: 5,
		Cooling:                     true,
		ActiveModelIndex:            2,
		ExposeReasoning:             true,
	}
	if err := b.SetCompactionState("s1", want); err != nil {
		t.Fatalf("SetCompactionState: %v", err)
	}
	got, err := b.GetCompactionState("s1")
	if err != nil {
		t.Fatalf("GetCompactionState: %v", err)
	}
	if got.MeaningfulCount != want.MeaningfulCount || got.CompressedAtMeaningfulCount != want.CompressedAtMeaningfulCount ||
		got.Cooling != want.Cooling || got.ActiveModelIndex != want.ActiveModelIndex || got.ExposeReasoning != want.ExposeReasoning {
		t.Errorf("compaction state round-trip mismatch:\n got %+v\nwant %+v", got, want)
	}
}

func TestJSONLBackend_PendingTurnLifecycle(t *testing.T) {
	b := newBackend(t)
	b.AddMessage("s1", "user", "hi")

	if err := b.SetPendingTurn("s1"); err != nil {
		t.Fatalf("SetPendingTurn: %v", err)
	}
	pending, err := b.ListPendingSessions()
	if err != nil {
		t.Fatalf("ListPendingSessions: %v", err)
	}
	if !contains(pending, "s1") {
		t.Errorf("s1 should be pending, got %v", pending)
	}
	if err := b.ClearPendingTurn("s1"); err != nil {
		t.Fatalf("ClearPendingTurn: %v", err)
	}
	pending, _ = b.ListPendingSessions()
	if contains(pending, "s1") {
		t.Errorf("s1 should no longer be pending, got %v", pending)
	}
}

func TestJSONLBackend_ArchiveBoundsAndForget(t *testing.T) {
	b := newBackend(t)
	b.AddMessage("s1", "user", "a")
	b.AddMessage("s1", "assistant", "b")

	// Bounds are well-defined (min <= max) for a known session, and zero for an
	// unknown one — neither should error or panic.
	min, max := b.GetArchiveBounds("s1")
	if min < 0 || max < 0 || max < min {
		t.Errorf("unexpected archive bounds: min=%d max=%d", min, max)
	}
	if m, n := b.GetArchiveBounds("unknown"); m != 0 || n != 0 {
		t.Errorf("unknown session bounds should be 0,0; got %d,%d", m, n)
	}

	// ForgetSession drops in-memory caches without touching durable history.
	b.ForgetSession("s1")
	if got := b.GetHistory("s1"); len(got) != 2 {
		t.Errorf("durable history must survive ForgetSession, got %d", len(got))
	}
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}
