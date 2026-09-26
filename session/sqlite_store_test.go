// ctxengine
// License: MIT

package session

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/PivotLLM/spawnllm"

	cronmsg "github.com/PivotLLM/ctxengine/internal/testcron"
	"github.com/PivotLLM/ctxengine/memory"
)

// Compile-time interface satisfaction check.
var _ SessionStore = (*SQLiteStore)(nil)

func newStore(t *testing.T) *SQLiteStore {
	t.Helper()
	return openStore(t, t.TempDir())
}

// mustAdd appends msg and returns its seq, failing the test on a write error.
func mustAdd(t *testing.T, s *SQLiteStore, key string, msg spawnllm.Message) int64 {
	t.Helper()
	seq, err := s.AddFullMessage(key, msg)
	if err != nil {
		t.Fatalf("AddFullMessage(%s): %v", key, err)
	}
	return seq
}

func openStore(t *testing.T, dir string) *SQLiteStore {
	t.Helper()
	s, err := NewSQLiteStore(dir)
	if err != nil {
		t.Fatalf("NewSQLiteStore: %v", err)
	}
	t.Cleanup(func() { noErr(t, s.Close()) })
	return s
}

func TestNewSQLiteStore_CreatesDirectory(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "nested", "sessions")
	openStore(t, dir)
	info, err := os.Stat(dir)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if !info.IsDir() {
		t.Errorf("expected directory, got file")
	}
}

func TestAddAndGetHistory(t *testing.T) {
	s := newStore(t)
	noErr(t, s.AddMessage("s1", "user", "hello"))
	noErr(t, s.AddMessage("s1", "assistant", "hi"))

	history := s.GetHistory("s1")
	if len(history) != 2 {
		t.Fatalf("got %d messages, want 2", len(history))
	}
	if history[0].Role != "user" || history[0].Content != "hello" {
		t.Errorf("msg[0] = %+v", history[0])
	}
	if history[1].Role != "assistant" || history[1].Content != "hi" {
		t.Errorf("msg[1] = %+v", history[1])
	}

	// The database is the session's archive file.
	if _, err := os.Stat(memory.ArchivePath(s.dir, "s1")); err != nil {
		t.Errorf("expected s1.archive.db: %v", err)
	}
}

func TestAddFullMessage_ToolCallsRoundTrip(t *testing.T) {
	s := newStore(t)
	msg := spawnllm.Message{
		Role:    "assistant",
		Content: "done",
		ToolCalls: []spawnllm.ToolCall{
			{ID: "tc1", Function: &spawnllm.FunctionCall{Name: "read_file", Arguments: `{"path":"x"}`}},
		},
	}
	if seq, err := s.AddFullMessage("s1", msg); seq != 1 || err != nil {
		t.Errorf("seq = %d, err = %v, want 1", seq, err)
	}
	mustAdd(t, s, "s1", spawnllm.Message{Role: "tool", Content: "ok", ToolCallID: "tc1"})

	history := s.GetHistory("s1")
	if len(history) != 2 {
		t.Fatalf("got %d, want 2", len(history))
	}
	if len(history[0].ToolCalls) != 1 || history[0].ToolCalls[0].ID != "tc1" {
		t.Errorf("tool calls = %+v", history[0].ToolCalls)
	}
	if history[1].ToolCallID != "tc1" {
		t.Errorf("tool call id = %q", history[1].ToolCallID)
	}
}

func TestSeqMinting(t *testing.T) {
	s := newStore(t)
	for i := 1; i <= 5; i++ {
		if seq := mustAdd(t, s, "seq", spawnllm.Message{Role: "user", Content: "msg"}); seq != int64(i) {
			t.Errorf("AddFullMessage(%d) seq = %d", i, seq)
		}
	}
	stored := s.GetHistoryWithSeqs("seq")
	if len(stored) != 5 {
		t.Fatalf("expected 5 stored messages, got %d", len(stored))
	}
	for i, sm := range stored {
		if sm.Seq != int64(i+1) {
			t.Errorf("stored[%d].Seq = %d, want %d", i, sm.Seq, i+1)
		}
		if sm.CreatedAt.IsZero() {
			t.Errorf("stored[%d].CreatedAt is zero", i)
		}
	}

	// Truncation preserves the seqs of the survivors and the counter.
	noErr(t, s.TruncateHistory("seq", 3))
	stored = s.GetHistoryWithSeqs("seq")
	if len(stored) != 3 || stored[0].Seq != 3 || stored[2].Seq != 5 {
		t.Fatalf("after truncate: %+v", stored)
	}
	if seq := mustAdd(t, s, "seq", spawnllm.Message{Role: "user", Content: "after"}); seq != 6 {
		t.Errorf("seq after truncate = %d, want 6", seq)
	}
}

func TestAddMessage_StampsCreatedAt(t *testing.T) {
	s := newStore(t)
	before := time.Now().UTC()
	noErr(t, s.AddMessage("stamp", "user", "hello"))
	mustAdd(t, s, "stamp", spawnllm.Message{Role: "assistant", Content: "hi"})
	after := time.Now().UTC()

	stored := s.GetHistoryWithSeqs("stamp")
	if len(stored) != 2 {
		t.Fatalf("expected 2 stored messages, got %d", len(stored))
	}
	for i, m := range stored {
		if m.CreatedAt.Before(before) || m.CreatedAt.After(after) {
			t.Errorf("stored[%d].CreatedAt = %v, want in [%v, %v]", i, m.CreatedAt, before, after)
		}
	}
}

func TestGetHistory_EmptySession(t *testing.T) {
	s := newStore(t)
	history := s.GetHistory("nonexistent")
	if history == nil || len(history) != 0 {
		t.Errorf("got %#v, want empty slice", history)
	}
	if got := s.GetHistoryWithSeqs("nonexistent"); got == nil || len(got) != 0 {
		t.Errorf("got %#v, want empty slice", got)
	}
}

func TestSessionIsolation(t *testing.T) {
	s := newStore(t)
	noErr(t, s.AddMessage("s1", "user", "session1"))
	noErr(t, s.AddMessage("telegram:1/2", "user", "session2"))

	if h := s.GetHistory("s1"); len(h) != 1 || h[0].Content != "session1" {
		t.Errorf("s1: %+v", h)
	}
	if h := s.GetHistory("telegram:1/2"); len(h) != 1 || h[0].Content != "session2" {
		t.Errorf("telegram:1/2: %+v", h)
	}
	if _, err := os.Stat(filepath.Join(s.dir, "telegram_1_2.archive.db")); err != nil {
		t.Errorf("sanitized file name: %v", err)
	}
}

func TestSummary(t *testing.T) {
	s := newStore(t)
	if got := s.GetSummary("s1"); got != "" {
		t.Errorf("got %q, want empty", got)
	}
	noErr(t, s.SetSummary("s1", "test summary"))
	if got := s.GetSummary("s1"); got != "test summary" {
		t.Errorf("got %q, want %q", got, "test summary")
	}
	// SetSummary on a fresh session creates its state row with the key.
	st, err := s.GetCompactionState("s1")
	if err != nil {
		t.Fatalf("GetCompactionState: %v", err)
	}
	if st != (memory.CompactionState{}) {
		t.Errorf("fresh compaction state = %+v", st)
	}
	keys, err := memory.ListSessions(s.dir)
	if err != nil || len(keys) != 1 || keys[0] != "s1" {
		t.Errorf("ListSessions = %v, %v", keys, err)
	}
}

func TestTruncateHistory(t *testing.T) {
	s := newStore(t)
	for i := range 10 {
		noErr(t, s.AddMessage("trunc", "user", string(rune('a'+i))))
	}

	noErr(t, s.TruncateHistory("trunc", 4))
	history := s.GetHistory("trunc")
	if len(history) != 4 {
		t.Fatalf("expected 4, got %d", len(history))
	}
	if history[0].Content != "g" || history[3].Content != "j" {
		t.Errorf("kept = %+v", history)
	}

	// Keep more than exists keeps all.
	noErr(t, s.TruncateHistory("trunc", 100))
	if got := len(s.GetHistory("trunc")); got != 4 {
		t.Errorf("keep 100: got %d, want 4", got)
	}

	// Save is a no-op and the window survives it.
	if err := s.Save("trunc"); err != nil {
		t.Fatal(err)
	}
	if got := len(s.GetHistory("trunc")); got != 4 {
		t.Errorf("after save: got %d, want 4", got)
	}

	// keepLast <= 0 empties the window and resets the compression counters.
	st, err := s.GetCompactionState("trunc")
	noErr(t, err)
	st.CompressedAtMeaningfulCount = 3
	st.Cooling = true
	st.ActiveModelIndex = 2
	if err := s.SetCompactionState("trunc", st); err != nil {
		t.Fatal(err)
	}
	noErr(t, s.TruncateHistory("trunc", 0))
	if got := len(s.GetHistory("trunc")); got != 0 {
		t.Errorf("keep 0: got %d, want 0", got)
	}
	st, err = s.GetCompactionState("trunc")
	noErr(t, err)
	if st.MeaningfulCount != 0 || st.CompressedAtMeaningfulCount != 0 || st.Cooling {
		t.Errorf("counters not reset: %+v", st)
	}
	if st.ActiveModelIndex != 2 {
		t.Errorf("ActiveModelIndex should survive a reset, got %d", st.ActiveModelIndex)
	}
	// Seqs keep counting after a full reset.
	if seq := mustAdd(t, s, "trunc", spawnllm.Message{Role: "user", Content: "next"}); seq != 11 {
		t.Errorf("seq after reset = %d, want 11", seq)
	}
}

func TestSetHistory_ReplacesWithFreshSeqs(t *testing.T) {
	s := newStore(t)
	for range 5 {
		noErr(t, s.AddMessage("replace", "user", "old"))
	}
	noErr(t, s.TruncateHistory("replace", 2))

	before := time.Now().UTC()
	noErr(t, s.SetHistory("replace", []spawnllm.Message{
		{Role: "user", Content: "new1"},
		{Role: "assistant", Content: "new2"},
	}))
	after := time.Now().UTC()

	stored := s.GetHistoryWithSeqs("replace")
	if len(stored) != 2 || stored[0].Content != "new1" || stored[1].Content != "new2" {
		t.Fatalf("history = %+v", stored)
	}
	// Fresh seqs continue after the old counter (5), never reuse.
	if stored[0].Seq != 6 || stored[1].Seq != 7 {
		t.Errorf("seqs = %d,%d want 6,7", stored[0].Seq, stored[1].Seq)
	}
	for i, m := range stored {
		if m.CreatedAt.Before(before) || m.CreatedAt.After(after) {
			t.Errorf("stored[%d].CreatedAt = %v not stamped now", i, m.CreatedAt)
		}
	}
	if seq := mustAdd(t, s, "replace", spawnllm.Message{Role: "user", Content: "x"}); seq != 8 {
		t.Errorf("next seq = %d, want 8", seq)
	}
}

func TestSetHistoryWithSeqs_PreservesStableSeqs(t *testing.T) {
	s := newStore(t)
	for i := range 5 {
		noErr(t, s.AddMessage("preserve", "user", string(rune('a'+i))))
	}
	active := s.GetHistoryWithSeqs("preserve")
	tail := active[3:]
	noErr(t, s.SetHistoryWithSeqs("preserve", tail))

	stored := s.GetHistoryWithSeqs("preserve")
	if len(stored) != 2 || stored[0].Seq != 4 || stored[1].Seq != 5 {
		t.Fatalf("seqs not preserved: %+v", stored)
	}
	if !stored[0].CreatedAt.Equal(tail[0].CreatedAt) {
		t.Errorf("CreatedAt not carried over: %v vs %v", stored[0].CreatedAt, tail[0].CreatedAt)
	}
	if seq := mustAdd(t, s, "preserve", spawnllm.Message{Role: "assistant", Content: "next"}); seq != 6 {
		t.Errorf("next seq = %d, want 6", seq)
	}
}

// TestSetHistoryWithSeqs_NextSeqMonotonic covers the two ways NextSeq could
// go backwards: a rewrite with lower seqs than the counter, and messages with
// no seq that must be minted above everything seen.
func TestSetHistoryWithSeqs_NextSeqMonotonic(t *testing.T) {
	s := newStore(t)
	for range 5 {
		noErr(t, s.AddMessage("mono", "user", "m"))
	}
	// Only low seqs retained: the counter must stay at 5.
	noErr(t, s.SetHistoryWithSeqs("mono", []memory.StoredMessage{
		memory.NewStoredMessage(1, spawnllm.Message{Role: "user", Content: "one"}),
	}))
	if seq := mustAdd(t, s, "mono", spawnllm.Message{Role: "user", Content: "six"}); seq != 6 {
		t.Errorf("seq after low rewrite = %d, want 6", seq)
	}

	// Unnumbered messages are minted after the highest seen (here 9 > counter 6).
	noErr(t, s.SetHistoryWithSeqs("mono", []memory.StoredMessage{
		memory.NewStoredMessage(9, spawnllm.Message{Role: "user", Content: "nine"}),
		{Message: spawnllm.Message{Role: "assistant", Content: "unnumbered"}},
	}))
	stored := s.GetHistoryWithSeqs("mono")
	if len(stored) != 2 || stored[0].Seq != 9 || stored[1].Seq != 10 {
		t.Fatalf("stored = %+v", stored)
	}
	if stored[1].CreatedAt.IsZero() {
		t.Error("minted message should be stamped")
	}
	if seq := mustAdd(t, s, "mono", spawnllm.Message{Role: "user", Content: "x"}); seq != 11 {
		t.Errorf("seq after mint = %d, want 11", seq)
	}
}

func TestMeaningfulCount_ConsecutiveDuplicates(t *testing.T) {
	s := newStore(t)
	for i := range 3 {
		noErr(t, s.AddMessage("mc", "user", string(rune('a'+i))))
	}
	// Two duplicates of the last message (same role and content) are noise.
	noErr(t, s.AddMessage("mc", "user", "c"))
	noErr(t, s.AddMessage("mc", "user", "c"))
	// Same content from another role is not.
	noErr(t, s.AddMessage("mc", "assistant", "c"))

	if got := len(s.GetHistory("mc")); got != 6 {
		t.Errorf("window = %d, want 6", got)
	}
	st, err := s.GetCompactionState("mc")
	if err != nil {
		t.Fatal(err)
	}
	if st.MeaningfulCount != 4 {
		t.Errorf("MeaningfulCount = %d, want 4", st.MeaningfulCount)
	}
}

// TestSetNoiseKey covers the store-level switch: with no noise key two fires
// of one job (different timestamps) are both meaningful; with the host's key
// installed the second is noise.
func TestSetNoiseKey(t *testing.T) {
	const prefix = "The following message is from a cron job that fired at "
	fire := func(ts string) string { return prefix + ts + ":\n\ncheck disk" }
	fires := []string{"2026-01-01 00:00 UTC", "2026-01-01 01:00 UTC"}

	off := newStore(t)
	for _, ts := range fires {
		noErr(t, off.AddMessage("k", "user", fire(ts)))
	}
	st, err := off.GetCompactionState("k")
	noErr(t, err)
	if st.MeaningfulCount != 2 {
		t.Errorf("without a noise key MeaningfulCount = %d, want 2", st.MeaningfulCount)
	}

	on := newStore(t)
	on.SetNoiseKey(cronmsg.CollapseKey)
	for _, ts := range fires {
		noErr(t, on.AddMessage("k", "user", fire(ts)))
	}
	st, err = on.GetCompactionState("k")
	noErr(t, err)
	if st.MeaningfulCount != 1 {
		t.Errorf("with the noise key MeaningfulCount = %d, want 1", st.MeaningfulCount)
	}
}

func TestCompactionState_RoundTripAcrossReopen(t *testing.T) {
	dir := t.TempDir()
	const key = "agent:alice:main"
	s := openStore(t, dir)
	noErr(t, s.AddMessage(key, "user", "hi"))

	want := memory.CompactionState{
		MeaningfulCount:             7,
		CompressedAtMeaningfulCount: 5,
		Cooling:                     true,
		CoolingSinceCount:           6,
		SummaryGeneratedAt:          time.Now().Add(-time.Hour).UTC(),
		SummaryModel:                "m1",
		ActiveModelIndex:            2,
		ExposeReasoning:             true,
		ShowToolActivity:            true,
	}
	if err := s.SetCompactionState(key, want); err != nil {
		t.Fatalf("SetCompactionState: %v", err)
	}
	noErr(t, s.Close())

	s2 := openStore(t, dir)
	got, err := s2.GetCompactionState(key)
	if err != nil {
		t.Fatalf("GetCompactionState after reopen: %v", err)
	}
	if got != want {
		t.Errorf("compaction state round-trip mismatch:\n got %+v\nwant %+v", got, want)
	}
	// The counters persist alongside the window and summary.
	if h := s2.GetHistory(key); len(h) != 1 || h[0].Content != "hi" {
		t.Errorf("history after reopen = %+v", h)
	}
}

func TestPersistence_AcrossInstances(t *testing.T) {
	dir := t.TempDir()
	s1 := openStore(t, dir)
	seq := mustAdd(t, s1, "persist", spawnllm.Message{Role: "user", Content: "remember me"})
	if err := s1.SetSummary("persist", "a test session"); err != nil {
		t.Fatalf("SetSummary: %v", err)
	}
	noErr(t, s1.Close())

	s2 := openStore(t, dir)
	history := s2.GetHistoryWithSeqs("persist")
	if len(history) != 1 || history[0].Content != "remember me" || history[0].Seq != seq {
		t.Errorf("history = %+v", history)
	}
	if got := s2.GetSummary("persist"); got != "a test session" {
		t.Errorf("summary = %q", got)
	}
	if next := mustAdd(t, s2, "persist", spawnllm.Message{Role: "assistant", Content: "ok"}); next != seq+1 {
		t.Errorf("seq after reopen = %d, want %d", next, seq+1)
	}
}

func TestPendingTurn(t *testing.T) {
	s := newStore(t)

	keys, err := s.ListPendingSessions()
	if err != nil || len(keys) != 0 {
		t.Fatalf("empty store: keys=%v err=%v", keys, err)
	}

	// Session A: set and left set.
	noErr(t, s.AddMessage("session-a", "user", "hello"))
	if err := s.SetPendingTurn("session-a"); err != nil {
		t.Fatalf("SetPendingTurn: %v", err)
	}
	// Session B: set then cleared.
	noErr(t, s.AddMessage("session-b", "user", "hi"))
	if err := s.SetPendingTurn("session-b"); err != nil {
		t.Fatal(err)
	}
	if err := s.ClearPendingTurn("session-b"); err != nil {
		t.Fatal(err)
	}
	// Session C: never marked.
	noErr(t, s.AddMessage("session-c", "user", "hey"))
	// Session D: marked pending before any message exists.
	if err := s.SetPendingTurn("agent:bob:main"); err != nil {
		t.Fatal(err)
	}

	keys, err = s.ListPendingSessions()
	if err != nil {
		t.Fatalf("ListPendingSessions: %v", err)
	}
	if len(keys) != 2 || !contains(keys, "session-a") || !contains(keys, "agent:bob:main") {
		t.Errorf("pending = %v, want session-a and agent:bob:main", keys)
	}

	// The flag is durable: a fresh store sees it.
	noErr(t, s.Close())
	s2 := openStore(t, s.dir)
	keys, err = s2.ListPendingSessions()
	noErr(t, err)
	if len(keys) != 2 {
		t.Errorf("pending after reopen = %v", keys)
	}
}

func TestListPendingSessions_DirectoryNotExist(t *testing.T) {
	s := &SQLiteStore{dir: filepath.Join(t.TempDir(), "does-not-exist"), sessions: map[string]*sessionHandle{}}
	keys, err := s.ListPendingSessions()
	if err != nil || len(keys) != 0 {
		t.Errorf("missing dir: keys=%v err=%v", keys, err)
	}
}

func TestGetArchiveBounds(t *testing.T) {
	s := newStore(t)
	noErr(t, s.AddMessage("s1", "user", "a"))
	// The window is not the archive: nothing archived yet.
	if lo, hi := s.GetArchiveBounds("s1"); lo != 0 || hi != 0 {
		t.Errorf("bounds with empty archive = %d,%d want 0,0", lo, hi)
	}
	if lo, hi := s.GetArchiveBounds("unknown"); lo != 0 || hi != 0 {
		t.Errorf("unknown session bounds = %d,%d want 0,0", lo, hi)
	}

	// The engine archives through its own handle on the same file.
	a, err := memory.Open(memory.ArchivePath(s.dir, "s1"))
	if err != nil {
		t.Fatal(err)
	}
	for seq := int64(3); seq <= 5; seq++ {
		if err := a.Append(seq, spawnllm.Message{Role: "user", Content: "x"}, time.Now()); err != nil {
			t.Fatal(err)
		}
	}
	noErr(t, a.Close())
	if lo, hi := s.GetArchiveBounds("s1"); lo != 3 || hi != 5 {
		t.Errorf("bounds = %d,%d want 3,5", lo, hi)
	}
}

func TestForgetSession(t *testing.T) {
	s := newStore(t)
	const key = "telegram_42"
	noErr(t, s.AddMessage(key, "user", "hello"))

	s.mu.Lock()
	_, present := s.sessions[key]
	s.mu.Unlock()
	if !present {
		t.Fatalf("expected a cached handle for %q after a write", key)
	}

	s.ForgetSession(key)

	s.mu.Lock()
	_, present = s.sessions[key]
	s.mu.Unlock()
	if present {
		t.Fatalf("handle for %q should be gone after ForgetSession", key)
	}

	// Durable data is untouched and the database reopens on access; the
	// noise cache restarts empty, so the duplicate counts as meaningful.
	noErr(t, s.AddMessage(key, "user", "hello"))
	if hist := s.GetHistory(key); len(hist) != 2 {
		t.Fatalf("history altered by ForgetSession: %+v", hist)
	}
	st, err := s.GetCompactionState(key)
	noErr(t, err)
	if st.MeaningfulCount != 2 {
		t.Errorf("MeaningfulCount = %d, want 2 (cache dropped)", st.MeaningfulCount)
	}

	// Forgetting an unknown session is a harmless no-op, and a forgotten
	// session's database can be deleted.
	s.ForgetSession("does-not-exist")
	s.ForgetSession(key)
	if err := memory.DeleteSession(s.dir, key); err != nil {
		t.Fatalf("DeleteSession: %v", err)
	}
	if hist := s.GetHistory(key); len(hist) != 0 {
		t.Errorf("history after delete = %+v", hist)
	}
}

func TestConcurrent_AddAndRead(t *testing.T) {
	s := newStore(t)
	var wg sync.WaitGroup
	const goroutines, perGoroutine = 10, 20
	for range goroutines {
		wg.Go(func() {
			for i := range perGoroutine {
				noErr(t, s.AddMessage("concurrent", "user", fmt.Sprintf("msg %d", i)))
				_ = s.GetHistory("concurrent")
			}
		})
	}
	// A summariser truncating concurrently, as in the agent loop.
	wg.Go(func() {
		for range 10 {
			noErr(t, s.SetSummary("concurrent", "summary"))
			noErr(t, s.TruncateHistory("concurrent", 50))
			s.ForgetSession("concurrent")
		}
	})
	wg.Wait()

	stored := s.GetHistoryWithSeqs("concurrent")
	for i := 1; i < len(stored); i++ {
		if stored[i].Seq <= stored[i-1].Seq {
			t.Fatalf("seqs not increasing at %d: %+v", i, stored)
		}
	}
	st, err := s.GetCompactionState("concurrent")
	noErr(t, err)
	if st.MeaningfulCount == 0 {
		t.Error("expected meaningful messages")
	}
	if s.GetSummary("concurrent") != "summary" {
		t.Error("summary lost")
	}
}

func contains(s []string, v string) bool {
	return slices.Contains(s, v)
}
