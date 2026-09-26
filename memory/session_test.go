// ctxengine
// License: MIT

package memory

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/PivotLLM/spawnllm"

	cronmsg "github.com/PivotLLM/ctxengine/internal/testcron"
)

func openSessionArchive(t *testing.T, dir, key string) *ArchiveStore {
	t.Helper()
	a, err := Open(ArchivePath(dir, key))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { a.Close() })
	return a
}

func TestState_ZeroWhenNoRow(t *testing.T) {
	a := openSessionArchive(t, t.TempDir(), "s1")
	st, err := a.State()
	if err != nil {
		t.Fatalf("State: %v", err)
	}
	if st != (SessionState{}) {
		t.Errorf("expected zero state, got %+v", st)
	}
	w, err := a.Window()
	if err != nil {
		t.Fatalf("Window: %v", err)
	}
	if w == nil || len(w) != 0 {
		t.Errorf("expected empty non-nil window, got %#v", w)
	}
}

func TestSetState_RoundTrip(t *testing.T) {
	a := openSessionArchive(t, t.TempDir(), "agent:alice:main")
	want := SessionState{
		Key:         "agent:alice:main",
		Summary:     `{"v":2}`,
		NextSeq:     42,
		CreatedAt:   time.Date(2026, 1, 2, 3, 4, 5, 678, time.UTC),
		UpdatedAt:   time.Date(2026, 1, 3, 3, 4, 5, 9, time.UTC),
		PendingTurn: true,
		Compaction: CompactionState{
			MeaningfulCount:             7,
			CompressedAtMeaningfulCount: 5,
			Cooling:                     true,
			CoolingSinceCount:           6,
			SummaryGeneratedAt:          time.Date(2026, 1, 2, 12, 0, 0, 0, time.UTC),
			SummaryModel:                "m1",
			ActiveModelIndex:            2,
			ExposeReasoning:             true,
			ShowToolActivity:            true,
		},
	}
	if err := a.SetState(want); err != nil {
		t.Fatalf("SetState: %v", err)
	}
	got, err := a.State()
	if err != nil {
		t.Fatalf("State: %v", err)
	}
	if got != want {
		t.Errorf("round-trip mismatch:\n got %+v\nwant %+v", got, want)
	}

	// Upsert: a second write replaces the single row.
	want.Summary = "changed"
	want.PendingTurn = false
	if err := a.SetState(want); err != nil {
		t.Fatalf("SetState again: %v", err)
	}
	got, _ = a.State()
	if got != want {
		t.Errorf("upsert mismatch:\n got %+v\nwant %+v", got, want)
	}
}

func TestWindow_AppendReplaceTruncate(t *testing.T) {
	a := openSessionArchive(t, t.TempDir(), "w")
	st := SessionState{Key: "w"}
	for i := int64(1); i <= 5; i++ {
		st.NextSeq = i
		m := NewStoredMessage(i, spawnllm.Message{Role: "user", Content: string(rune('a' + i - 1))})
		if err := a.AppendWindow(m, st); err != nil {
			t.Fatalf("AppendWindow(%d): %v", i, err)
		}
	}
	w, err := a.Window()
	if err != nil {
		t.Fatalf("Window: %v", err)
	}
	if len(w) != 5 || w[0].Seq != 1 || w[4].Seq != 5 || w[4].Content != "e" {
		t.Fatalf("window = %+v", w)
	}
	if w[0].CreatedAt.IsZero() {
		t.Error("CreatedAt not persisted")
	}
	if got, _ := a.State(); got.NextSeq != 5 {
		t.Errorf("NextSeq = %d, want 5", got.NextSeq)
	}

	st.NextSeq = 9
	if err := a.TruncateWindow(2, st); err != nil {
		t.Fatalf("TruncateWindow: %v", err)
	}
	w, _ = a.Window()
	if len(w) != 2 || w[0].Seq != 4 || w[1].Seq != 5 {
		t.Fatalf("after truncate: %+v", w)
	}
	if got, _ := a.State(); got.NextSeq != 9 {
		t.Errorf("state not written with truncate: NextSeq = %d", got.NextSeq)
	}

	repl := []StoredMessage{
		NewStoredMessage(10, spawnllm.Message{Role: "user", Content: "x"}),
		NewStoredMessage(11, spawnllm.Message{Role: "assistant", Content: "y"}),
	}
	st.NextSeq = 11
	if err := a.ReplaceWindow(repl, st); err != nil {
		t.Fatalf("ReplaceWindow: %v", err)
	}
	w, _ = a.Window()
	if len(w) != 2 || w[0].Seq != 10 || w[1].Content != "y" {
		t.Fatalf("after replace: %+v", w)
	}

	if err := a.TruncateWindow(0, st); err != nil {
		t.Fatalf("TruncateWindow(0): %v", err)
	}
	if w, _ = a.Window(); len(w) != 0 {
		t.Fatalf("after truncate(0): %+v", w)
	}
}

// TestWindow_PreservesZeroCreatedAt covers the migration case: a legacy message
// with no timestamp must read back zero, not be stamped on every read.
func TestWindow_PreservesZeroCreatedAt(t *testing.T) {
	a := openSessionArchive(t, t.TempDir(), "z")
	legacy := StoredMessage{Seq: 1, Message: spawnllm.Message{Role: "user", Content: "old"}}
	if err := a.ReplaceWindow([]StoredMessage{legacy}, SessionState{Key: "z", NextSeq: 1}); err != nil {
		t.Fatalf("ReplaceWindow: %v", err)
	}
	w, _ := a.Window()
	if len(w) != 1 || !w[0].CreatedAt.IsZero() {
		t.Errorf("legacy CreatedAt should stay zero: %+v", w)
	}
}

func TestOpenReadOnly_WindowAndState(t *testing.T) {
	dir := t.TempDir()
	a := openSessionArchive(t, dir, "ro")
	st := SessionState{Key: "ro", Summary: "sum", NextSeq: 1}
	if err := a.AppendWindow(NewStoredMessage(1, spawnllm.Message{Role: "user", Content: "hi"}), st); err != nil {
		t.Fatalf("AppendWindow: %v", err)
	}

	r, err := OpenReadOnly(ArchivePath(dir, "ro"))
	if err != nil {
		t.Fatalf("OpenReadOnly: %v", err)
	}
	got, err := r.State()
	if err != nil {
		t.Fatalf("State: %v", err)
	}
	if got.Key != "ro" || got.Summary != "sum" {
		t.Errorf("state = %+v", got)
	}
	w, err := r.Window()
	if err != nil {
		t.Fatalf("Window: %v", err)
	}
	if len(w) != 1 || w[0].Content != "hi" {
		t.Errorf("window = %+v", w)
	}

	if _, err := OpenReadOnly(ArchivePath(dir, "missing")); err == nil {
		t.Error("expected error for a missing archive")
	}
}

func TestListSessions(t *testing.T) {
	dir := t.TempDir()
	for _, key := range []string{"agent:bob:main", "telegram:1/2"} {
		a := openSessionArchive(t, dir, key)
		if err := a.SetState(SessionState{Key: key, NextSeq: 1}); err != nil {
			t.Fatalf("SetState: %v", err)
		}
	}
	// An archive with no state row (never written) is not a session.
	openSessionArchive(t, dir, "empty")
	// Unrelated files are ignored.
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	keys, err := ListSessions(dir)
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(keys) != 2 || keys[0] != "agent:bob:main" || keys[1] != "telegram:1/2" {
		t.Errorf("keys = %v", keys)
	}

	keys, err = ListSessions(filepath.Join(dir, "nope"))
	if err != nil || keys != nil {
		t.Errorf("missing dir: keys=%v err=%v", keys, err)
	}
}

func TestDeleteSession(t *testing.T) {
	dir := t.TempDir()
	const key = "agent:alice:main"
	a := openSessionArchive(t, dir, key)
	if err := a.SetState(SessionState{Key: key, NextSeq: 1}); err != nil {
		t.Fatalf("SetState: %v", err)
	}
	path := ArchivePath(dir, key)
	if _, err := os.Stat(path + "-wal"); err != nil {
		t.Fatalf("expected a WAL sidecar while open: %v", err)
	}
	a.Close()

	if err := DeleteSession(dir, key); err != nil {
		t.Fatalf("DeleteSession: %v", err)
	}
	for _, p := range []string{path, path + "-wal", path + "-shm"} {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Errorf("%s should be gone, stat err=%v", filepath.Base(p), err)
		}
	}
	if err := DeleteSession(dir, key); err != nil {
		t.Errorf("deleting a missing session should be nil, got %v", err)
	}
}

// testCronPrefix mirrors the cron-wrapper header for building noise-classifier
// fixtures. The shared marker parser lives in cronmsg.
const testCronPrefix = "The following message is from a cron job that fired at "

func TestNoiseClassifier_CronNoise(t *testing.T) {
	cache := NewNoiseCache()

	// First cron message with a given payload — not noise.
	payload := "check disk space: 42% used"
	content1 := testCronPrefix + "2026-01-01T00:00:00Z:\n\n" + payload
	msg1 := StoredMessage{Seq: 1, Message: spawnllm.Message{Role: "user", Content: content1}}
	if cache.IsNoise(msg1, cronmsg.CollapseKey) {
		t.Error("first cron message should not be noise")
	}
	cache.Record(msg1, cronmsg.CollapseKey)

	// Same payload at a different timestamp — noise.
	content2 := testCronPrefix + "2026-01-01T01:00:00Z:\n\n" + payload
	msg2 := StoredMessage{Seq: 2, Message: spawnllm.Message{Role: "user", Content: content2}}
	if !cache.IsNoise(msg2, cronmsg.CollapseKey) {
		t.Error("duplicate cron payload should be noise")
	}
	cache.Record(msg2, cronmsg.CollapseKey)

	// Different payload — not noise.
	content3 := testCronPrefix + "2026-01-01T02:00:00Z:\n\n" + "check disk space: 80% used"
	msg3 := StoredMessage{Seq: 3, Message: spawnllm.Message{Role: "user", Content: content3}}
	if cache.IsNoise(msg3, cronmsg.CollapseKey) {
		t.Error("different cron payload should not be noise")
	}
}

func TestNoiseClassifier_SameRole(t *testing.T) {
	cache := NewNoiseCache()

	msg1 := StoredMessage{Seq: 1, Message: spawnllm.Message{Role: "user", Content: "hello"}}
	if cache.IsNoise(msg1, cronmsg.CollapseKey) {
		t.Error("first message should not be noise")
	}
	cache.Record(msg1, cronmsg.CollapseKey)

	// Same role and content — noise.
	msg2 := StoredMessage{Seq: 2, Message: spawnllm.Message{Role: "user", Content: "hello"}}
	if !cache.IsNoise(msg2, cronmsg.CollapseKey) {
		t.Error("duplicate same-role content should be noise")
	}
}

func TestNoiseClassifier_DifferentContent(t *testing.T) {
	cache := NewNoiseCache()

	msg1 := StoredMessage{Seq: 1, Message: spawnllm.Message{Role: "user", Content: "hello"}}
	cache.Record(msg1, cronmsg.CollapseKey)

	msg2 := StoredMessage{Seq: 2, Message: spawnllm.Message{Role: "user", Content: "world"}}
	if cache.IsNoise(msg2, cronmsg.CollapseKey) {
		t.Error("different content should not be noise")
	}
}
