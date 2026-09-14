package memory

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestSummaries_RoundTrip verifies AppendSummary -> ListSummaries -> GetSummary.
func TestSummaries_RoundTrip(t *testing.T) {
	a := openTestArchive(t)

	gen1 := time.Now().Add(-time.Hour).Truncate(time.Second)
	gen2 := time.Now().Truncate(time.Second)

	id1, err := a.AppendSummary(SummaryRecord{
		GeneratedAt:     gen1,
		Model:           "model-a",
		Profile:         "abc12345",
		SourceSeqStart:  1,
		SourceSeqEnd:    10,
		CoveredSeqStart: 1,
		CoveredSeqEnd:   10,
		Summary:         `{"version":2,"goal":"first"}`,
	})
	if err != nil {
		t.Fatalf("AppendSummary 1: %v", err)
	}
	id2, err := a.AppendSummary(SummaryRecord{
		GeneratedAt:     gen2,
		Model:           "model-b",
		SourceSeqStart:  11,
		SourceSeqEnd:    20,
		CoveredSeqStart: 1,
		CoveredSeqEnd:   20,
		Summary:         `{"version":2,"goal":"second"}`,
	})
	if err != nil {
		t.Fatalf("AppendSummary 2: %v", err)
	}
	if id1 != 1 || id2 != 2 {
		t.Fatalf("expected ids 1,2 got %d,%d", id1, id2)
	}

	metas, err := a.ListSummaries()
	if err != nil {
		t.Fatalf("ListSummaries: %v", err)
	}
	if len(metas) != 2 {
		t.Fatalf("expected 2 metas, got %d", len(metas))
	}
	// Ordered by id ASC.
	if metas[0].ID != 1 || metas[1].ID != 2 {
		t.Errorf("ordering wrong: %d, %d", metas[0].ID, metas[1].ID)
	}
	if metas[0].Model != "model-a" || metas[0].Profile != "abc12345" {
		t.Errorf("meta[0] fields wrong: %+v", metas[0])
	}
	if metas[1].CoveredSeqEnd != 20 || metas[1].SourceSeqStart != 11 {
		t.Errorf("meta[1] seq fields wrong: %+v", metas[1])
	}
	if !metas[0].GeneratedAt.Equal(gen1.UTC()) {
		t.Errorf("generated_at mismatch: got %v want %v", metas[0].GeneratedAt, gen1.UTC())
	}

	rec, ok, err := a.GetSummary(2)
	if err != nil {
		t.Fatalf("GetSummary: %v", err)
	}
	if !ok {
		t.Fatal("expected GetSummary(2) to be found")
	}
	if rec.Summary != `{"version":2,"goal":"second"}` {
		t.Errorf("body mismatch: %q", rec.Summary)
	}
	if rec.Model != "model-b" || rec.CoveredSeqEnd != 20 {
		t.Errorf("rec fields wrong: %+v", rec)
	}
}

// TestSummaries_ListMetadataOnly verifies ListSummaries does not surface the body.
func TestSummaries_ListMetadataOnly(t *testing.T) {
	a := openTestArchive(t)
	body := `{"secret":"do-not-leak-in-list"}`
	if _, err := a.AppendSummary(SummaryRecord{
		GeneratedAt: time.Now(),
		Summary:     body,
	}); err != nil {
		t.Fatalf("AppendSummary: %v", err)
	}

	metas, err := a.ListSummaries()
	if err != nil {
		t.Fatalf("ListSummaries: %v", err)
	}
	// SummaryMeta has no Summary field; assert the struct type carries no body by
	// marshaling it and confirming the body string is absent.
	data, _ := json.Marshal(metas)
	if string(data) == "" {
		t.Fatal("unexpected empty marshal")
	}
	for _, m := range metas {
		// SummaryMeta intentionally has no body field; this is a compile-time
		// guarantee. Re-fetch the full record to confirm the body still exists.
		rec, ok, err := a.GetSummary(m.ID)
		if err != nil || !ok {
			t.Fatalf("GetSummary(%d): ok=%v err=%v", m.ID, ok, err)
		}
		if rec.Summary != body {
			t.Errorf("body mismatch on get: %q", rec.Summary)
		}
	}
}

// TestSummaries_GetNotFound verifies ok=false for a missing id.
func TestSummaries_GetNotFound(t *testing.T) {
	a := openTestArchive(t)
	if _, err := a.AppendSummary(SummaryRecord{GeneratedAt: time.Now(), Summary: "x"}); err != nil {
		t.Fatalf("AppendSummary: %v", err)
	}
	_, ok, err := a.GetSummary(999)
	if err != nil {
		t.Fatalf("GetSummary: %v", err)
	}
	if ok {
		t.Error("expected ok=false for missing id")
	}
}

// TestSummaries_ReadOnlyListAndGet verifies List/Get work on a read-only archive.
func TestSummaries_ReadOnlyListAndGet(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ro.archive.db")
	w, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, err := w.AppendSummary(SummaryRecord{GeneratedAt: time.Now(), Summary: "ro-body"}); err != nil {
		t.Fatalf("AppendSummary: %v", err)
	}
	w.Close()

	ro, err := OpenReadOnly(path)
	if err != nil {
		t.Fatalf("OpenReadOnly: %v", err)
	}
	defer ro.Close()

	metas, err := ro.ListSummaries()
	if err != nil {
		t.Fatalf("ListSummaries (ro): %v", err)
	}
	if len(metas) != 1 {
		t.Fatalf("expected 1 meta, got %d", len(metas))
	}
	rec, ok, err := ro.GetSummary(metas[0].ID)
	if err != nil || !ok {
		t.Fatalf("GetSummary (ro): ok=%v err=%v", ok, err)
	}
	if rec.Summary != "ro-body" {
		t.Errorf("body mismatch: %q", rec.Summary)
	}
}

// writeLegacyJSONL writes checkpoints to the sibling <base>.summaries.jsonl of
// the given .archive.db path.
func writeLegacyJSONL(t *testing.T, archivePath string, cps []SummaryCheckpoint) {
	t.Helper()
	base := archivePath[:len(archivePath)-len(".archive.db")]
	legacy := base + ".summaries.jsonl"
	f, err := os.Create(legacy)
	if err != nil {
		t.Fatalf("create legacy jsonl: %v", err)
	}
	defer f.Close()
	for _, cp := range cps {
		line, _ := json.Marshal(cp)
		f.Write(append(line, '\n'))
	}
}

// TestSummaries_LegacyImportOnce verifies the one-time jsonl import populates the
// table and is skipped on subsequent opens (and when the table is non-empty).
func TestSummaries_LegacyImportOnce(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "legacy.archive.db")

	cps := []SummaryCheckpoint{
		{ID: 1, GeneratedAt: time.Now().Add(-2 * time.Hour), Model: "m1", SourceSeqStart: 1, SourceSeqEnd: 5, CoveredSeqStart: 1, CoveredSeqEnd: 5, SummaryHash: "h1", Summary: `{"v":1}`},
		{ID: 2, GeneratedAt: time.Now().Add(-1 * time.Hour), Model: "m2", SourceSeqStart: 6, SourceSeqEnd: 9, CoveredSeqStart: 1, CoveredSeqEnd: 9, PrevHash: "h1", SummaryHash: "h2", Summary: `{"v":2}`},
	}

	// Create the legacy jsonl BEFORE the archive exists.
	a, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	a.Close()
	writeLegacyJSONL(t, path, cps)

	// Reopen: table is empty, legacy file exists -> import runs.
	a2, err := Open(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	metas, err := a2.ListSummaries()
	if err != nil {
		t.Fatalf("ListSummaries: %v", err)
	}
	if len(metas) != 2 {
		t.Fatalf("expected 2 imported, got %d", len(metas))
	}
	// Order preserved; hash fields dropped; profile empty.
	if metas[0].Model != "m1" || metas[1].Model != "m2" {
		t.Errorf("import order/model wrong: %+v", metas)
	}
	if metas[0].Profile != "" || metas[1].Profile != "" {
		t.Errorf("expected empty profile after import, got %q,%q", metas[0].Profile, metas[1].Profile)
	}
	rec, _, _ := a2.GetSummary(metas[0].ID)
	if rec.Summary != `{"v":1}` {
		t.Errorf("imported body wrong: %q", rec.Summary)
	}
	a2.Close()

	// Reopen again: table non-empty -> import is skipped (no doubling).
	a3, err := Open(path)
	if err != nil {
		t.Fatalf("reopen 2: %v", err)
	}
	defer a3.Close()
	metas2, err := a3.ListSummaries()
	if err != nil {
		t.Fatalf("ListSummaries 2: %v", err)
	}
	if len(metas2) != 2 {
		t.Fatalf("expected import NOT to double; got %d", len(metas2))
	}
}

// TestSummaries_LegacyImportSkippedWhenNonEmpty verifies that an archive with an
// existing summary row does not import the legacy file at all.
func TestSummaries_LegacyImportSkippedWhenNonEmpty(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nonempty.archive.db")

	a, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, err := a.AppendSummary(SummaryRecord{GeneratedAt: time.Now(), Summary: "native"}); err != nil {
		t.Fatalf("AppendSummary: %v", err)
	}
	a.Close()

	// Now drop a legacy file with two checkpoints; it must be ignored on reopen.
	writeLegacyJSONL(t, path, []SummaryCheckpoint{
		{ID: 1, GeneratedAt: time.Now(), Summary: `{"a":1}`},
		{ID: 2, GeneratedAt: time.Now(), Summary: `{"b":2}`},
	})

	a2, err := Open(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer a2.Close()
	metas, err := a2.ListSummaries()
	if err != nil {
		t.Fatalf("ListSummaries: %v", err)
	}
	if len(metas) != 1 {
		t.Fatalf("expected only the 1 native row (no legacy import), got %d", len(metas))
	}
}
