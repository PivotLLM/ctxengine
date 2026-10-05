/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/PivotLLM/ctxengine/memory"
)

// TestCompact_PersistsSummaryToArchive verifies that a compaction writes the
// generated summary into the per-session archive's summaries table, carrying the
// Profile fingerprint through, and that the body is the marshaled summary JSON.
func TestCompact_PersistsSummaryToArchive(t *testing.T) {
	profDir := t.TempDir()
	const profile = "preserve active branch names verbatim"
	if err := os.WriteFile(filepath.Join(profDir, "compression.md"), []byte(profile), 0o644); err != nil {
		t.Fatal(err)
	}

	archiveDir := t.TempDir()

	store := &compressTestStore{history: makeConversation(10, 200)}
	llm := &mockLLM{responses: []string{validSummaryJSON("active goal")}}
	mgr := newCompressManager(t, store, []*mockLLM{llm},
		WithCompressionProfileDir(profDir),
		WithCompressModel(ModelChain{Primary: "test-model"}),
		WithArchiveDir(archiveDir),
	)
	mgr.msgCount = len(store.history)

	if err := mgr.Compact(context.Background()); err != nil {
		t.Fatalf("Compact: %v", err)
	}

	// Open the archive and confirm exactly one summary row was persisted.
	archivePath := memory.ArchivePath(archiveDir, "sess")
	a, err := memory.OpenReadOnly(archivePath)
	if err != nil {
		t.Fatalf("OpenReadOnly archive: %v", err)
	}
	defer func() { noErr(t, a.Close()) }()

	metas, err := a.ListSummaries()
	if err != nil {
		t.Fatalf("ListSummaries: %v", err)
	}
	if len(metas) != 1 {
		t.Fatalf("expected 1 summary in archive, got %d", len(metas))
	}

	wantProfile := profileFingerprint(profile)
	if metas[0].Profile != wantProfile {
		t.Errorf("archive summary profile = %q, want %q", metas[0].Profile, wantProfile)
	}
	if metas[0].Model != "test-model" {
		t.Errorf("archive summary model = %q, want test-model", metas[0].Model)
	}

	rec, ok, err := a.GetSummary(metas[0].ID)
	if err != nil || !ok {
		t.Fatalf("GetSummary: ok=%v err=%v", ok, err)
	}
	// The stored body is the same marshaled summary JSON written to the store.
	if rec.Summary != store.summary {
		t.Errorf("archive body != store summary:\n archive: %s\n store:   %s", rec.Summary, store.summary)
	}
	s, err := unmarshalSummary(rec.Summary)
	if err != nil || s == nil {
		t.Fatalf("archive body is not valid summary JSON: %v", err)
	}
	if s.Profile != wantProfile {
		t.Errorf("body profile = %q, want %q", s.Profile, wantProfile)
	}
}
