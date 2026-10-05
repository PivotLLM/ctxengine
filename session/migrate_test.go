/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package session

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine/memory"
)

func writeFixture(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func exists(t *testing.T, dir, name string) bool {
	t.Helper()
	_, err := os.Stat(filepath.Join(dir, name))
	if err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}
	return err == nil
}

// The fixture mirrors a production session: a truncated window (skip > 0), a
// summary, every compaction field, next_seq ahead of the line count, lines
// with and without seq, a checkpoint log and an ancient .archive.jsonl.
const fixtureKey = "agent:alice:telegram:group:-100123/45"

const fixtureBase = "agent_alice_telegram_group_-100123_45"

func writeProductionFixture(t *testing.T, dir string) {
	t.Helper()
	meta := map[string]any{
		"key":                            fixtureKey,
		"summary":                        `{"version":2,"overview":"about disks"}`,
		"skip":                           2,
		"count":                          5,
		"created_at":                     "2026-01-01T10:00:00.5Z",
		"updated_at":                     "2026-01-02T11:00:00Z",
		"next_seq":                       7,
		"meaningful_count":               4,
		"compressed_at_meaningful_count": 2,
		"summary_generated_at":           "2026-01-02T10:30:00Z",
		"summary_model":                  "m1",
		"compression_cooling":            true,
		"cooling_since_count":            3,
		"active_model_index":             1,
		"expose_reasoning":               true,
		"show_tool_activity":             true,
		"pending_turn":                   true,
	}
	metaJSON, err := json.MarshalIndent(meta, "", "  ")
	noErr(t, err)
	writeFixture(t, dir, fixtureBase+".meta.json", string(metaJSON))

	lines := []string{
		`{"seq":1,"created_at":"2026-01-01T10:00:00Z","role":"user","content":"gone one"}`,
		`{"seq":2,"created_at":"2026-01-01T10:01:00Z","role":"assistant","content":"gone two"}`,
		`{"role":"user","content":"legacy without seq"}`,
		`not json at all`,
		`{"seq":6,"created_at":"2026-01-02T10:00:00Z","role":"assistant","content":"reply","tool_calls":[{"id":"tc1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"x\"}"}}]}`,
		`{"seq":7,"created_at":"2026-01-02T11:00:00Z","role":"tool","content":"ok","tool_call_id":"tc1"}`,
	}
	writeFixture(t, dir, fixtureBase+".jsonl", strings.Join(lines, "\n")+"\n")

	cp := memory.SummaryCheckpoint{
		ID: 1, GeneratedAt: time.Date(2026, 1, 2, 10, 30, 0, 0, time.UTC), Model: "m1",
		SourceSeqStart: 1, SourceSeqEnd: 4, CoveredSeqStart: 1, CoveredSeqEnd: 4,
		SummaryHash: "h1", Summary: `{"version":2,"overview":"about disks"}`,
	}
	cpJSON, err := json.Marshal(cp)
	noErr(t, err)
	writeFixture(t, dir, fixtureBase+".summaries.jsonl", string(cpJSON)+"\n")
	writeFixture(t, dir, fixtureBase+".archive.jsonl", `{"seq":1,"role":"user","content":"ancient"}`+"\n")
}

func TestMigrateJSONL_ProductionFixture(t *testing.T) {
	dir := t.TempDir()
	writeProductionFixture(t, dir)

	report, err := MigrateJSONL(dir)
	if err != nil {
		t.Fatalf("MigrateJSONL: %v", err)
	}
	if len(report.Errors) != 0 {
		t.Fatalf("errors: %v", report.Errors)
	}
	if len(report.Migrated) != 1 || report.Migrated[0] != fixtureKey {
		t.Fatalf("Migrated = %v", report.Migrated)
	}
	if len(report.Skipped) != 0 {
		t.Errorf("Skipped = %v", report.Skipped)
	}

	// Sources renamed; the checkpoint log and the ancient archive left alone.
	for _, name := range []string{
		fixtureBase + ".jsonl.migrated", fixtureBase + ".meta.json.migrated",
		fixtureBase + ".summaries.jsonl", fixtureBase + ".archive.jsonl", fixtureBase + ".archive.db",
	} {
		if !exists(t, dir, name) {
			t.Errorf("expected %s", name)
		}
	}
	for _, name := range []string{fixtureBase + ".jsonl", fixtureBase + ".meta.json"} {
		if exists(t, dir, name) {
			t.Errorf("%s should have been renamed", name)
		}
	}

	a, err := memory.OpenReadOnly(memory.ArchivePath(dir, fixtureKey))
	if err != nil {
		t.Fatalf("OpenReadOnly: %v", err)
	}

	// Window: the first two lines are skipped, the corrupt one dropped, the
	// legacy line gets skip+lineNo, the rest keep their seq and payload.
	w, err := a.Window()
	if err != nil {
		t.Fatalf("Window: %v", err)
	}
	if len(w) != 3 {
		t.Fatalf("window = %+v", w)
	}
	if w[0].Seq != 5 || w[0].Content != "legacy without seq" || !w[0].CreatedAt.IsZero() {
		t.Errorf("legacy line = %+v", w[0])
	}
	if w[1].Seq != 6 || len(w[1].ToolCalls) != 1 || w[1].ToolCalls[0].Function.Name != "read_file" {
		t.Errorf("tool call line = %+v", w[1])
	}
	if !w[1].CreatedAt.Equal(time.Date(2026, 1, 2, 10, 0, 0, 0, time.UTC)) {
		t.Errorf("created_at not carried: %v", w[1].CreatedAt)
	}
	if w[2].Seq != 7 || w[2].ToolCallID != "tc1" {
		t.Errorf("tool result line = %+v", w[2])
	}

	// State: every meta field, pending_turn forced off.
	st, err := a.State()
	if err != nil {
		t.Fatalf("State: %v", err)
	}
	want := memory.SessionState{
		Key:       fixtureKey,
		Summary:   `{"version":2,"overview":"about disks"}`,
		NextSeq:   7,
		CreatedAt: time.Date(2026, 1, 1, 10, 0, 0, 500_000_000, time.UTC),
		UpdatedAt: time.Date(2026, 1, 2, 11, 0, 0, 0, time.UTC),
		Compaction: memory.CompactionState{
			MeaningfulCount:             4,
			CompressedAtMeaningfulCount: 2,
			Cooling:                     true,
			CoolingSinceCount:           3,
			SummaryGeneratedAt:          time.Date(2026, 1, 2, 10, 30, 0, 0, time.UTC),
			SummaryModel:                "m1",
			ActiveModelIndex:            1,
			ExposeReasoning:             true,
			ShowToolActivity:            true,
		},
	}
	if st != want {
		t.Errorf("state mismatch:\n got %+v\nwant %+v", st, want)
	}

	// The checkpoint log was imported into the summaries table on open.
	sums, err := a.ListSummaries()
	if err != nil {
		t.Fatalf("ListSummaries: %v", err)
	}
	if len(sums) != 1 || sums[0].Model != "m1" || sums[0].CoveredSeqEnd != 4 {
		t.Errorf("summaries = %+v", sums)
	}

	// A second run finds the window in place and skips the session; the
	// renamed sources are not touched.
	writeFixture(t, dir, fixtureBase+".meta.json", `{"key":"`+fixtureKey+`","next_seq":99}`)
	report, err = MigrateJSONL(dir)
	if err != nil {
		t.Fatalf("second MigrateJSONL: %v", err)
	}
	if len(report.Skipped) != 1 || report.Skipped[0] != fixtureKey || len(report.Migrated) != 0 || len(report.Errors) != 0 {
		t.Errorf("second run report = %+v", report)
	}
	if !exists(t, dir, fixtureBase+".meta.json") {
		t.Error("a skipped session's sources must be left untouched")
	}
	st, err = a.State()
	noErr(t, err)
	if st.NextSeq != 7 {
		t.Errorf("skipped session was rewritten: NextSeq = %d", st.NextSeq)
	}

	// The store carries on from the migrated counter.
	s := openStore(t, dir)
	if seq := mustAdd(t, s, fixtureKey, spawnllm.Message{Role: "user", Content: "after"}); seq != 8 {
		t.Errorf("seq after migration = %d, want 8", seq)
	}
	if got := len(s.GetHistory(fixtureKey)); got != 4 {
		t.Errorf("window after append = %d, want 4", got)
	}
	cs, err := s.GetCompactionState(fixtureKey)
	noErr(t, err)
	if cs.MeaningfulCount != 5 {
		t.Errorf("MeaningfulCount = %d, want 5", cs.MeaningfulCount)
	}
}

func TestMigrateJSONL_MetaOnlyAndLegacySeqs(t *testing.T) {
	dir := t.TempDir()
	// A session that only ever had a summary set: meta, no window.
	writeFixture(t, dir, "agent_bob_main.meta.json",
		`{"key":"agent:bob:main","summary":"just a summary","created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z"}`)
	// A session from before seqs existed: no next_seq, no seq on any line.
	writeFixture(t, dir, "old.meta.json", `{"key":"old","skip":1,"count":3}`)
	writeFixture(t, dir, "old.jsonl",
		`{"role":"user","content":"one"}`+"\n"+`{"role":"assistant","content":"two"}`+"\n"+`{"role":"user","content":"three"}`+"\n")

	report, err := MigrateJSONL(dir)
	if err != nil {
		t.Fatalf("MigrateJSONL: %v", err)
	}
	if len(report.Errors) != 0 || len(report.Migrated) != 2 {
		t.Fatalf("report = %+v", report)
	}

	s := openStore(t, dir)
	if got := s.GetSummary("agent:bob:main"); got != "just a summary" {
		t.Errorf("summary = %q", got)
	}
	if got := len(s.GetHistory("agent:bob:main")); got != 0 {
		t.Errorf("meta-only window = %d", got)
	}
	if exists(t, dir, "agent_bob_main.meta.json") || !exists(t, dir, "agent_bob_main.meta.json.migrated") {
		t.Error("meta-only source not renamed")
	}

	// Legacy lines get skip+lineNo (the position the JSONL store reported).
	stored := s.GetHistoryWithSeqs("old")
	if len(stored) != 2 || stored[0].Seq != 3 || stored[1].Seq != 4 || stored[1].Content != "three" {
		t.Errorf("legacy seqs = %+v", stored)
	}
	// NextSeq follows the highest assigned seq, so new messages never collide.
	if seq := mustAdd(t, s, "old", spawnllm.Message{Role: "user", Content: "four"}); seq != 5 {
		t.Errorf("seq after legacy migration = %d, want 5", seq)
	}
}

func TestMigrateJSONL_Errors(t *testing.T) {
	dir := t.TempDir()
	// A window with no meta: the key is unknown and never guessed.
	writeFixture(t, dir, "orphan.jsonl", `{"seq":1,"role":"user","content":"x"}`+"\n")
	// A meta without a key.
	writeFixture(t, dir, "nokey.meta.json", `{"summary":"s"}`)
	// A corrupt meta.
	writeFixture(t, dir, "broken.meta.json", `{`)
	// A directory entry is ignored.
	if err := os.Mkdir(filepath.Join(dir, "sub.jsonl"), 0o755); err != nil {
		t.Fatal(err)
	}

	report, err := MigrateJSONL(dir)
	if err != nil {
		t.Fatalf("MigrateJSONL: %v", err)
	}
	if len(report.Migrated) != 0 || len(report.Skipped) != 0 {
		t.Errorf("report = %+v", report)
	}
	for _, name := range []string{"orphan.jsonl", "nokey.meta.json", "broken.meta.json"} {
		if report.Errors[name] == nil {
			t.Errorf("expected an error for %s, got %v", name, report.Errors)
		}
		if !exists(t, dir, name) {
			t.Errorf("%s must be left in place on error", name)
		}
	}
	if exists(t, dir, "orphan.archive.db") {
		t.Error("no archive may be created for a session without a key")
	}

	// A missing directory is not an error.
	report, err = MigrateJSONL(filepath.Join(dir, "missing"))
	if err != nil || len(report.Errors) != 0 {
		t.Errorf("missing dir: %+v, %v", report, err)
	}
}
