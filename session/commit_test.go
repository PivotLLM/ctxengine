/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package session

import (
	"os"
	"testing"

	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine/memory"
)

// TestCommitCompaction writes a compaction result through the store and reads
// back every part of it: the window with its seqs preserved, the summary, the
// checkpoint, the compaction counters, and a seq counter that carries on.
func TestCommitCompaction(t *testing.T) {
	s := newStore(t)
	for i := range 5 {
		mustAdd(t, s, "c", spawnllm.Message{Role: "user", Content: string(rune('a' + i))})
	}
	tail := s.GetHistoryWithSeqs("c")[3:] // seqs 4, 5
	summary := "the summary"
	err := s.CommitCompaction("c", CompactionCommit{
		History: tail,
		Summary: &summary,
		Checkpoint: &memory.SummaryRecord{
			Model: "m", SourceSeqStart: 1, SourceSeqEnd: 3, CoveredSeqStart: 1, CoveredSeqEnd: 3, Summary: summary,
		},
		Compaction: func(st *memory.CompactionState) {
			st.CompressedAtMeaningfulCount = 5
			st.Cooling = true
		},
	})
	if err != nil {
		t.Fatalf("CommitCompaction: %v", err)
	}

	window := s.GetHistoryWithSeqs("c")
	if len(window) != 2 || window[0].Seq != 4 || window[1].Seq != 5 {
		t.Fatalf("window = %+v, want seqs 4, 5", window)
	}
	if got := s.GetSummary("c"); got != summary {
		t.Errorf("summary = %q, want %q", got, summary)
	}
	st, err := s.GetCompactionState("c")
	if err != nil {
		t.Fatalf("GetCompactionState: %v", err)
	}
	if st.MeaningfulCount != 5 || st.CompressedAtMeaningfulCount != 5 || !st.Cooling {
		t.Errorf("compaction state = %+v, want meaningful 5, compressed_at 5, cooling", st)
	}
	a, err := memory.OpenReadOnly(memory.ArchivePath(s.dir, "c"))
	if err != nil {
		t.Fatalf("OpenReadOnly: %v", err)
	}
	metas, err := a.ListSummaries()
	if err != nil || len(metas) != 1 || metas[0].Model != "m" || metas[0].SourceSeqEnd != 3 {
		t.Fatalf("checkpoints = %+v, %v; want one from model m covering #1-#3", metas, err)
	}
	if seq := mustAdd(t, s, "c", spawnllm.Message{Role: "user", Content: "f"}); seq != 6 {
		t.Errorf("seq after commit = %d, want 6", seq)
	}

	// A drop-only commit leaves the summary and the checkpoint log alone.
	if err = s.CommitCompaction("c", CompactionCommit{History: s.GetHistoryWithSeqs("c")[1:]}); err != nil {
		t.Fatalf("drop-only CommitCompaction: %v", err)
	}
	if got := s.GetSummary("c"); got != summary {
		t.Errorf("summary after drop-only commit = %q, want unchanged", got)
	}
	metas, err = a.ListSummaries()
	noErr(t, err)
	if len(metas) != 1 {
		t.Errorf("checkpoints after drop-only commit = %d, want still 1", len(metas))
	}
	if window := s.GetHistoryWithSeqs("c"); len(window) != 2 || window[0].Seq != 5 || window[1].Seq != 6 {
		t.Errorf("window after drop-only commit = %+v, want seqs 5, 6", window)
	}
}

// TestWriteErrorsAreReturned removes the store directory so every write
// fails, and checks each write method reports it instead of swallowing it.
func TestWriteErrorsAreReturned(t *testing.T) {
	dir := t.TempDir()
	s := openStore(t, dir)
	if err := os.RemoveAll(dir); err != nil {
		t.Fatal(err)
	}
	msg := spawnllm.Message{Role: "user", Content: "x"}
	if seq, err := s.AddFullMessage("w", msg); err == nil || seq != 0 {
		t.Errorf("AddFullMessage = %d, %v; want 0 and an error", seq, err)
	}
	if err := s.AddMessage("w", "user", "x"); err == nil {
		t.Error("AddMessage returned nil on a failed write")
	}
	if err := s.SetSummary("w", "s"); err == nil {
		t.Error("SetSummary returned nil on a failed write")
	}
	if err := s.SetHistory("w", []spawnllm.Message{msg}); err == nil {
		t.Error("SetHistory returned nil on a failed write")
	}
	if err := s.SetHistoryWithSeqs("w", []memory.StoredMessage{memory.NewStoredMessage(1, msg)}); err == nil {
		t.Error("SetHistoryWithSeqs returned nil on a failed write")
	}
	if err := s.TruncateHistory("w", 0); err == nil {
		t.Error("TruncateHistory returned nil on a failed write")
	}
	if err := s.CommitCompaction("w", CompactionCommit{}); err == nil {
		t.Error("CommitCompaction returned nil on a failed write")
	}
}
