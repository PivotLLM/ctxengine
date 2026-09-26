// ctxengine
// License: MIT

package memory

import (
	"encoding/json"
	"testing"

	"github.com/PivotLLM/spawnllm"
)

// TestCommitCompaction_AllOrNothing checks that CommitCompaction writes the
// window, the session state and the summary checkpoint in one transaction: a
// commit that fails part-way (here on a message that cannot be marshalled,
// after the window was already cleared inside the transaction) leaves all
// three exactly as they were.
func TestCommitCompaction_AllOrNothing(t *testing.T) {
	a := openTestArchive(t)
	msg := func(seq int64, content string) StoredMessage {
		return NewStoredMessage(seq, spawnllm.Message{Role: "user", Content: content})
	}
	st := SessionState{Key: "k", NextSeq: 3}
	if err := a.ReplaceWindow([]StoredMessage{msg(1, "a"), msg(2, "b"), msg(3, "c")}, st); err != nil {
		t.Fatalf("ReplaceWindow: %v", err)
	}

	// A commit writes window, state and checkpoint together.
	st.Summary = "first"
	if err := a.CommitCompaction([]StoredMessage{msg(2, "b"), msg(3, "c")}, st,
		&SummaryRecord{Model: "m", SourceSeqStart: 1, SourceSeqEnd: 1, Summary: "first"}); err != nil {
		t.Fatalf("CommitCompaction: %v", err)
	}
	assertWindowSeqs(t, a, 2, 3)
	got, err := a.State()
	noErr(t, err)
	if got.Summary != "first" || got.NextSeq != 3 {
		t.Errorf("state after commit = %+v, want summary first, next_seq 3", got)
	}
	metas, err := a.ListSummaries()
	if err != nil || len(metas) != 1 || metas[0].Model != "m" {
		t.Fatalf("summaries after commit = %+v, %v; want one from model m", metas, err)
	}

	// A failing commit leaves all three as they were.
	bad := NewStoredMessage(4, spawnllm.Message{
		Role:               "user",
		ResponsesReasoning: []json.RawMessage{json.RawMessage("not json")},
	})
	st.Summary = "second"
	st.NextSeq = 4
	err = a.CommitCompaction([]StoredMessage{msg(3, "c"), bad}, st,
		&SummaryRecord{Model: "m", SourceSeqStart: 2, SourceSeqEnd: 2, Summary: "second"})
	if err == nil {
		t.Fatal("expected the commit to fail on an unmarshalable message")
	}
	assertWindowSeqs(t, a, 2, 3)
	got, err = a.State()
	noErr(t, err)
	if got.Summary != "first" || got.NextSeq != 3 {
		t.Errorf("state after failed commit = %+v, want the previous summary and next_seq", got)
	}
	if metas, err = a.ListSummaries(); err != nil || len(metas) != 1 {
		t.Errorf("summaries after failed commit = %d, %v; want still 1", len(metas), err)
	}
}

// TestCommitCompaction_NoCheckpoint pins that a nil checkpoint appends nothing
// while the window and state are still written.
func TestCommitCompaction_NoCheckpoint(t *testing.T) {
	a := openTestArchive(t)
	m := NewStoredMessage(1, spawnllm.Message{Role: "user", Content: "x"})
	if err := a.CommitCompaction([]StoredMessage{m}, SessionState{Key: "k", Summary: "s", NextSeq: 1}, nil); err != nil {
		t.Fatalf("CommitCompaction: %v", err)
	}
	assertWindowSeqs(t, a, 1)
	got, err := a.State()
	noErr(t, err)
	if got.Summary != "s" {
		t.Errorf("summary = %q, want s", got.Summary)
	}
	metas, err := a.ListSummaries()
	noErr(t, err)
	if len(metas) != 0 {
		t.Errorf("summaries = %d, want none", len(metas))
	}
}

func assertWindowSeqs(t *testing.T, a *ArchiveStore, want ...int64) {
	t.Helper()
	window, err := a.Window()
	if err != nil {
		t.Fatalf("Window: %v", err)
	}
	if len(window) != len(want) {
		t.Fatalf("window has %d rows, want %d: %+v", len(window), len(want), window)
	}
	for i, w := range want {
		if window[i].Seq != w {
			t.Errorf("window[%d].Seq = %d, want %d", i, window[i].Seq, w)
		}
	}
}
