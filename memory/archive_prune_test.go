package memory

import (
	"context"
	"testing"
	"time"
)

// countMessages returns the number of rows in the messages table.
func countMessages(t *testing.T, a *ArchiveStore) int {
	t.Helper()
	min, max, err := a.Bounds()
	if err != nil {
		t.Fatalf("Bounds: %v", err)
	}
	if max == 0 {
		return 0
	}
	rows, err := a.QueryRange(min, max)
	if err != nil {
		t.Fatalf("QueryRange: %v", err)
	}
	return len(rows)
}

// TestPruneMessagesToCount verifies the newest N messages are kept by seq.
func TestPruneMessagesToCount(t *testing.T) {
	a := openTestArchive(t)
	now := time.Now()
	for i := 1; i <= 10; i++ {
		if err := a.Append(int64(i), sampleMsg("user", "m"), now); err != nil {
			t.Fatalf("Append %d: %v", i, err)
		}
	}

	if err := a.PruneMessagesToCount(3); err != nil {
		t.Fatalf("PruneMessagesToCount: %v", err)
	}

	min, max, err := a.Bounds()
	if err != nil {
		t.Fatalf("Bounds: %v", err)
	}
	if min != 8 || max != 10 {
		t.Fatalf("bounds = [%d,%d], want [8,10]", min, max)
	}
	if got := countMessages(t, a); got != 3 {
		t.Fatalf("count = %d, want 3", got)
	}
}

// TestPruneMessagesToCount_NoOpAtZero verifies n <= 0 deletes nothing.
func TestPruneMessagesToCount_NoOpAtZero(t *testing.T) {
	a := openTestArchive(t)
	now := time.Now()
	for i := 1; i <= 5; i++ {
		if err := a.Append(int64(i), sampleMsg("user", "m"), now); err != nil {
			t.Fatalf("Append %d: %v", i, err)
		}
	}
	if err := a.PruneMessagesToCount(0); err != nil {
		t.Fatalf("PruneMessagesToCount(0): %v", err)
	}
	if got := countMessages(t, a); got != 5 {
		t.Fatalf("count = %d, want 5 (no-op)", got)
	}
}

// TestPruneMessagesBefore verifies messages older than the cutoff are deleted.
func TestPruneMessagesBefore(t *testing.T) {
	a := openTestArchive(t)
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	// seqs 1..5 at day 1..5
	for i := 1; i <= 5; i++ {
		ts := base.AddDate(0, 0, i-1)
		if err := a.Append(int64(i), sampleMsg("user", "m"), ts); err != nil {
			t.Fatalf("Append %d: %v", i, err)
		}
	}

	// Cutoff at day 3 (base+2 days): deletes seqs 1,2 (days 1,2).
	cutoff := base.AddDate(0, 0, 2)
	if err := a.PruneMessagesBefore(cutoff); err != nil {
		t.Fatalf("PruneMessagesBefore: %v", err)
	}

	min, max, err := a.Bounds()
	if err != nil {
		t.Fatalf("Bounds: %v", err)
	}
	if min != 3 || max != 5 {
		t.Fatalf("bounds = [%d,%d], want [3,5]", min, max)
	}
}

// TestPruneMessages_FTSConsistency verifies that after a message prune the FTS
// index no longer matches pruned rows but still finds surviving rows.
func TestPruneMessages_FTSConsistency(t *testing.T) {
	a := openTestArchive(t)
	now := time.Now()
	if err := a.Append(1, sampleMsg("user", "alpha unique_old_token"), now); err != nil {
		t.Fatalf("Append 1: %v", err)
	}
	if err := a.Append(2, sampleMsg("user", "beta unique_new_token"), now); err != nil {
		t.Fatalf("Append 2: %v", err)
	}

	// Prune to keep only the newest message (seq 2).
	if err := a.PruneMessagesToCount(1); err != nil {
		t.Fatalf("PruneMessagesToCount: %v", err)
	}

	ctx := context.Background()

	// The pruned row's token must no longer be searchable.
	oldRes, err := a.Search(ctx, "unique_old_token", "", 10)
	if err != nil {
		t.Fatalf("Search old: %v", err)
	}
	if len(oldRes) != 0 {
		t.Fatalf("pruned row still searchable: %d results", len(oldRes))
	}

	// The surviving row must still be searchable.
	newRes, err := a.Search(ctx, "unique_new_token", "", 10)
	if err != nil {
		t.Fatalf("Search new: %v", err)
	}
	if len(newRes) != 1 || newRes[0].Seq != 2 {
		t.Fatalf("surviving row search = %+v, want one result seq=2", newRes)
	}
}

// TestPruneSummariesToCount verifies the newest N summaries are kept by id.
func TestPruneSummariesToCount(t *testing.T) {
	a := openTestArchive(t)
	now := time.Now()
	for i := 0; i < 6; i++ {
		if _, err := a.AppendSummary(SummaryRecord{
			GeneratedAt: now,
			Summary:     "s",
		}); err != nil {
			t.Fatalf("AppendSummary %d: %v", i, err)
		}
	}

	if err := a.PruneSummariesToCount(2); err != nil {
		t.Fatalf("PruneSummariesToCount: %v", err)
	}

	metas, err := a.ListSummaries()
	if err != nil {
		t.Fatalf("ListSummaries: %v", err)
	}
	if len(metas) != 2 {
		t.Fatalf("summaries = %d, want 2", len(metas))
	}
	// Newest two ids are 5 and 6.
	if metas[0].ID != 5 || metas[1].ID != 6 {
		t.Fatalf("kept ids = [%d,%d], want [5,6]", metas[0].ID, metas[1].ID)
	}
}

// TestPruneSummariesToCount_NoOpAtZero verifies n <= 0 deletes nothing.
func TestPruneSummariesToCount_NoOpAtZero(t *testing.T) {
	a := openTestArchive(t)
	now := time.Now()
	for i := 0; i < 3; i++ {
		if _, err := a.AppendSummary(SummaryRecord{GeneratedAt: now, Summary: "s"}); err != nil {
			t.Fatalf("AppendSummary %d: %v", i, err)
		}
	}
	if err := a.PruneSummariesToCount(0); err != nil {
		t.Fatalf("PruneSummariesToCount(0): %v", err)
	}
	metas, err := a.ListSummaries()
	if err != nil {
		t.Fatalf("ListSummaries: %v", err)
	}
	if len(metas) != 3 {
		t.Fatalf("summaries = %d, want 3 (no-op)", len(metas))
	}
}

// TestPruneSummariesBefore verifies summaries older than the cutoff are deleted.
func TestPruneSummariesBefore(t *testing.T) {
	a := openTestArchive(t)
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	for i := 0; i < 5; i++ {
		if _, err := a.AppendSummary(SummaryRecord{
			GeneratedAt: base.AddDate(0, 0, i),
			Summary:     "s",
		}); err != nil {
			t.Fatalf("AppendSummary %d: %v", i, err)
		}
	}

	// Cutoff at day 2 (base+2): deletes ids generated at day 0 and day 1.
	cutoff := base.AddDate(0, 0, 2)
	if err := a.PruneSummariesBefore(cutoff); err != nil {
		t.Fatalf("PruneSummariesBefore: %v", err)
	}

	metas, err := a.ListSummaries()
	if err != nil {
		t.Fatalf("ListSummaries: %v", err)
	}
	if len(metas) != 3 {
		t.Fatalf("summaries = %d, want 3", len(metas))
	}
	if metas[0].ID != 3 {
		t.Fatalf("oldest surviving id = %d, want 3", metas[0].ID)
	}
}
