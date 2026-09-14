// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"testing"
	"time"

	"github.com/PivotLLM/ctxengine/memory"
	"github.com/PivotLLM/spawnllm"
)

// TestPruneArchive_TriggeredOnArchiveOpen verifies the retention prune is WIRED
// into getOrOpenArchive: a fresh manager opening an existing over-cap archive
// prunes it on open (not just when pruneArchive is called directly).
func TestPruneArchive_TriggeredOnArchiveOpen(t *testing.T) {
	dir := t.TempDir()
	const key = "open-prune"

	// Populate an archive with 6 messages (no cap), then close it.
	mgr1 := newResetManager(newResetStore(key), key, WithArchiveDir(dir))
	for i := int64(1); i <= 6; i++ {
		mgr1.archiveAppend(i, spawnllm.Message{Role: "user", Content: "m"})
	}
	if a := mgr1.getOrOpenArchive(); a != nil {
		_ = a.Close()
	}

	// A fresh manager with a count cap of 3 opens the existing archive; the
	// open path must prune it down to the newest 3.
	mgr2 := newResetManager(newResetStore(key), key, WithArchiveDir(dir), WithArchiveMessageCount(3))
	minSeq, maxSeq, err := mgr2.getOrOpenArchive().Bounds()
	if err != nil {
		t.Fatalf("Bounds: %v", err)
	}
	if minSeq != 4 || maxSeq != 6 {
		t.Fatalf("after open bounds = [%d,%d], want [4,6] — pruning not wired into getOrOpenArchive", minSeq, maxSeq)
	}
}

// TestPruneArchive_TriggeredByCompaction verifies the retention prune is WIRED
// into persistStoredResult: a real compaction prunes the archive to the cap.
func TestPruneArchive_TriggeredByCompaction(t *testing.T) {
	dir := t.TempDir()
	store := &compressTestStore{history: makeConversation(10, 200)}
	// The summary must cite a seq inside the archive window [4,6] (count=3 over
	// archived seqs 1..6) so it survives StripOutOfRangeSeqRefs and the
	// compaction actually persists — only then does the wired prune run.
	const summaryInWindow = `{"version":2,"state":{"goals":[{"text":"g","refs":[{"seq_start":6,"seq_end":6}]}]},"covered_seq_start":0,"covered_seq_end":0}`
	llm := &mockLLM{responses: []string{summaryInWindow}}
	mgr := newCompressManager(store, []*mockLLM{llm}, WithArchiveDir(dir), WithArchiveMessageCount(3))
	mgr.msgCount = len(store.history)

	for i := int64(1); i <= 6; i++ {
		mgr.archiveAppend(i, spawnllm.Message{Role: "user", Content: "m"})
	}

	if err := mgr.Compact(context.Background()); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if store.summary == "" {
		t.Fatal("precondition: compaction did not persist a summary")
	}

	minSeq, maxSeq, err := mgr.getOrOpenArchive().Bounds()
	if err != nil {
		t.Fatalf("Bounds: %v", err)
	}
	if minSeq != 4 || maxSeq != 6 {
		t.Fatalf("after compaction bounds = [%d,%d], want [4,6] — pruning not wired into persistStoredResult", minSeq, maxSeq)
	}
}

// TestPruneArchive_HonorsMessageCount verifies pruneArchive keeps the newest N
// messages when archiveMessageCount > 0.
func TestPruneArchive_HonorsMessageCount(t *testing.T) {
	const key = "prune-session"
	store := newResetStore(key)
	mgr := newResetManager(store, key, WithArchiveDir(t.TempDir()), WithArchiveMessageCount(3))

	for i := 1; i <= 10; i++ {
		mgr.archiveAppend(int64(i), spawnllm.Message{Role: "user", Content: "m"})
	}

	mgr.pruneArchive(mgr.getOrOpenArchive())

	minSeq, maxSeq, err := mgr.getOrOpenArchive().Bounds()
	if err != nil {
		t.Fatalf("Bounds: %v", err)
	}
	if minSeq != 8 || maxSeq != 10 {
		t.Fatalf("bounds = [%d,%d], want [8,10]", minSeq, maxSeq)
	}
}

// TestPruneArchive_HonorsMessageDays verifies pruneArchive deletes messages
// older than archiveDays.
func TestPruneArchive_HonorsMessageDays(t *testing.T) {
	const key = "prune-session"
	store := newResetStore(key)
	mgr := newResetManager(store, key, WithArchiveDir(t.TempDir()), WithArchiveDays(2))

	a := mgr.getOrOpenArchive()
	now := time.Now()
	// seq 1: 5 days old (pruned). seq 2: today (kept).
	if err := a.Append(1, spawnllm.Message{Role: "user", Content: "old"}, now.AddDate(0, 0, -5)); err != nil {
		t.Fatalf("Append 1: %v", err)
	}
	if err := a.Append(2, spawnllm.Message{Role: "user", Content: "new"}, now); err != nil {
		t.Fatalf("Append 2: %v", err)
	}

	mgr.pruneArchive(a)

	minSeq, maxSeq, err := a.Bounds()
	if err != nil {
		t.Fatalf("Bounds: %v", err)
	}
	if minSeq != 2 || maxSeq != 2 {
		t.Fatalf("bounds = [%d,%d], want [2,2]", minSeq, maxSeq)
	}
}

// TestPruneArchive_HonorsSummaryCount verifies pruneArchive keeps the newest N
// summaries when summaryMaxCount > 0.
func TestPruneArchive_HonorsSummaryCount(t *testing.T) {
	const key = "prune-session"
	store := newResetStore(key)
	mgr := newResetManager(store, key, WithArchiveDir(t.TempDir()), WithSummaryMaxCount(2))

	a := mgr.getOrOpenArchive()
	for i := 0; i < 5; i++ {
		if _, err := a.AppendSummary(memory.SummaryRecord{GeneratedAt: time.Now(), Summary: "s"}); err != nil {
			t.Fatalf("AppendSummary %d: %v", i, err)
		}
	}

	mgr.pruneArchive(a)

	metas, err := a.ListSummaries()
	if err != nil {
		t.Fatalf("ListSummaries: %v", err)
	}
	if len(metas) != 2 {
		t.Fatalf("summaries = %d, want 2", len(metas))
	}
}

// TestPruneArchive_HonorsSummaryDays verifies pruneArchive deletes summaries
// older than summaryRetentionDays.
func TestPruneArchive_HonorsSummaryDays(t *testing.T) {
	const key = "prune-session"
	store := newResetStore(key)
	mgr := newResetManager(store, key, WithArchiveDir(t.TempDir()), WithSummaryRetentionDays(2))

	a := mgr.getOrOpenArchive()
	now := time.Now()
	if _, err := a.AppendSummary(memory.SummaryRecord{GeneratedAt: now.AddDate(0, 0, -10), Summary: "old"}); err != nil {
		t.Fatalf("AppendSummary old: %v", err)
	}
	if _, err := a.AppendSummary(memory.SummaryRecord{GeneratedAt: now, Summary: "new"}); err != nil {
		t.Fatalf("AppendSummary new: %v", err)
	}

	mgr.pruneArchive(a)

	metas, err := a.ListSummaries()
	if err != nil {
		t.Fatalf("ListSummaries: %v", err)
	}
	if len(metas) != 1 {
		t.Fatalf("summaries = %d, want 1", len(metas))
	}
}

// TestPruneArchive_ZeroConfigNoOp verifies that with all four caps at 0
// (defaults), pruneArchive deletes nothing.
func TestPruneArchive_ZeroConfigNoOp(t *testing.T) {
	const key = "prune-session"
	store := newResetStore(key)
	mgr := newResetManager(store, key, WithArchiveDir(t.TempDir()))

	a := mgr.getOrOpenArchive()
	now := time.Now()
	for i := 1; i <= 5; i++ {
		if err := a.Append(int64(i), spawnllm.Message{Role: "user", Content: "m"}, now.AddDate(0, 0, -100)); err != nil {
			t.Fatalf("Append %d: %v", i, err)
		}
	}
	for i := 0; i < 3; i++ {
		if _, err := a.AppendSummary(memory.SummaryRecord{GeneratedAt: now.AddDate(0, 0, -100), Summary: "s"}); err != nil {
			t.Fatalf("AppendSummary %d: %v", i, err)
		}
	}

	mgr.pruneArchive(a)

	_, maxSeq, err := a.Bounds()
	if err != nil {
		t.Fatalf("Bounds: %v", err)
	}
	if maxSeq != 5 {
		t.Fatalf("maxSeq = %d, want 5 (no prune)", maxSeq)
	}
	metas, err := a.ListSummaries()
	if err != nil {
		t.Fatalf("ListSummaries: %v", err)
	}
	if len(metas) != 3 {
		t.Fatalf("summaries = %d, want 3 (no prune)", len(metas))
	}
}
