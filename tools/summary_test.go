/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package tools

import (
	"strings"
	"testing"
	"time"

	"github.com/PivotLLM/ctxengine/memory"
)

// writeArchiveSummaries creates a .archive.db with the given summary records and
// returns the directory it lives in.
func writeArchiveSummaries(t *testing.T, sessionKey string, recs []memory.SummaryRecord) string {
	t.Helper()
	dir := t.TempDir()
	path := memory.ArchivePath(dir, sessionKey)
	a, err := memory.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { noErr(t, a.Close()) }()
	for _, r := range recs {
		if _, err := a.AppendSummary(r); err != nil {
			t.Fatalf("AppendSummary: %v", err)
		}
	}
	return dir
}

// TestSessionSummaryList_ResolvesAndLists verifies the list tool resolves the
// session via the call's key and returns a readable listing.
func TestSessionSummaryList_ResolvesAndLists(t *testing.T) {
	const key = "agent:main"
	dir := writeArchiveSummaries(t, key, []memory.SummaryRecord{
		{GeneratedAt: time.Now().Add(-time.Hour), Model: "m1", CoveredSeqStart: 1, CoveredSeqEnd: 10, Summary: `{"v":2}`},
		{GeneratedAt: time.Now(), Model: "m2", Profile: "abc12345", CoveredSeqStart: 1, CoveredSeqEnd: 20, Summary: `{"v":2}`},
	})

	res := run(t, archiveHost(dir), "summary_list", key, map[string]any{})
	if res.IsError {
		t.Fatalf("unexpected error: %s", res.ForLLM)
	}
	// Newest first: id 2 (model m2) should appear before id 1.
	if !strings.Contains(res.ForLLM, "id 2") || !strings.Contains(res.ForLLM, "id 1") {
		t.Errorf("expected both ids listed, got: %s", res.ForLLM)
	}
	if !strings.Contains(res.ForLLM, "#1-#20") {
		t.Errorf("expected covered range for id 2, got: %s", res.ForLLM)
	}
	if !strings.Contains(res.ForLLM, "profile abc12345") {
		t.Errorf("expected profile shown, got: %s", res.ForLLM)
	}
}

// TestSessionSummaryList_Empty returns a friendly message when none exist.
func TestSessionSummaryList_Empty(t *testing.T) {
	const key = "empty"
	dir := writeArchiveSummaries(t, key, nil)
	res := run(t, archiveHost(dir), "summary_list", key, map[string]any{})
	if res.IsError {
		t.Fatalf("unexpected error: %s", res.ForLLM)
	}
	if !strings.Contains(res.ForLLM, "no context summaries") {
		t.Errorf("expected empty message, got: %s", res.ForLLM)
	}
}

// TestSessionSummaryList_NoArchive returns unavailable when the file is missing.
func TestSessionSummaryList_NoArchive(t *testing.T) {
	res := run(t, archiveHost(t.TempDir()), "summary_list", "nope", map[string]any{})
	if res.IsError {
		t.Fatalf("unexpected error: %s", res.ForLLM)
	}
	if !strings.Contains(res.ForLLM, "unavailable") {
		t.Errorf("expected unavailable, got: %s", res.ForLLM)
	}
}

// TestSessionSummaryGet_ResolvesAndRenders verifies the get tool returns the
// rendered body of a checkpoint resolved via the call's session key.
func TestSessionSummaryGet_ResolvesAndRenders(t *testing.T) {
	const key = "getsess"
	body := `{"version":2,"state":{"goals":[{"text":"ship the feature","refs":[{"seq_start":1,"seq_end":3}]}]},"covered_seq_start":1,"covered_seq_end":10}`
	dir := writeArchiveSummaries(t, key, []memory.SummaryRecord{
		{GeneratedAt: time.Now(), Model: "m1", CoveredSeqStart: 1, CoveredSeqEnd: 10, Summary: body},
	})

	res := run(t, archiveHost(dir), "summary_get", key, map[string]any{"id": 1})
	if res.IsError {
		t.Fatalf("unexpected error: %s", res.ForLLM)
	}
	if !strings.Contains(res.ForLLM, "checkpoint id 1") {
		t.Errorf("expected header, got: %s", res.ForLLM)
	}
	// The rendered Markdown should surface the goal text.
	if !strings.Contains(res.ForLLM, "ship the feature") {
		t.Errorf("expected rendered goal text, got: %s", res.ForLLM)
	}
}

// TestSessionSummaryGet_NotFound handles a bad id gracefully.
func TestSessionSummaryGet_NotFound(t *testing.T) {
	const key = "getsess2"
	dir := writeArchiveSummaries(t, key, []memory.SummaryRecord{
		{GeneratedAt: time.Now(), Summary: `{"v":2}`},
	})
	res := run(t, archiveHost(dir), "summary_get", key, map[string]any{"id": 999})
	if res.IsError {
		t.Fatalf("unexpected error: %s", res.ForLLM)
	}
	if !strings.Contains(res.ForLLM, "no context summary with id 999") {
		t.Errorf("expected not-found message, got: %s", res.ForLLM)
	}
}

// TestSessionSummaryGet_MissingID errors when id is absent.
func TestSessionSummaryGet_MissingID(t *testing.T) {
	const key = "getsess3"
	dir := writeArchiveSummaries(t, key, []memory.SummaryRecord{
		{GeneratedAt: time.Now(), Summary: `{"v":2}`},
	})
	res := run(t, archiveHost(dir), "summary_get", key, map[string]any{})
	if !res.IsError {
		t.Errorf("expected error for missing id, got: %s", res.ForLLM)
	}
}

// TestSessionSummary_MissingSessionKey errors when the call carries no session key.
func TestSessionSummary_MissingSessionKey(t *testing.T) {
	h := archiveHost(t.TempDir())
	if listRes := run(t, h, "summary_list", "", map[string]any{}); !listRes.IsError {
		t.Errorf("list: expected error for missing session key")
	}
	if getRes := run(t, h, "summary_get", "", map[string]any{"id": 1}); !getRes.IsError {
		t.Errorf("get: expected error for missing session key")
	}
}
