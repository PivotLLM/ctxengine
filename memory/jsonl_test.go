package memory

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	cronmsg "github.com/PivotLLM/ctxengine/internal/testcron"
	"github.com/PivotLLM/spawnllm"
)

func newTestStore(t *testing.T) *JSONLStore {
	t.Helper()
	store, err := NewJSONLStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewJSONLStore: %v", err)
	}
	return store
}

func TestNewJSONLStore_CreatesDirectory(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "nested", "sessions")
	store, err := NewJSONLStore(dir)
	if err != nil {
		t.Fatalf("NewJSONLStore: %v", err)
	}
	defer store.Close()

	info, err := os.Stat(dir)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if !info.IsDir() {
		t.Errorf("expected directory, got file")
	}
}

func TestAddMessage_BasicRoundtrip(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	err := store.AddMessage(ctx, "s1", "user", "hello")
	if err != nil {
		t.Fatalf("AddMessage: %v", err)
	}
	err = store.AddMessage(ctx, "s1", "assistant", "hi there")
	if err != nil {
		t.Fatalf("AddMessage: %v", err)
	}

	history, err := store.GetHistory(ctx, "s1")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(history))
	}
	if history[0].Role != "user" || history[0].Content != "hello" {
		t.Errorf("msg[0] = %+v", history[0])
	}
	if history[1].Role != "assistant" || history[1].Content != "hi there" {
		t.Errorf("msg[1] = %+v", history[1])
	}
}

func TestAddMessage_AutoCreatesSession(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	// Adding a message to a non-existent session should work.
	err := store.AddMessage(ctx, "new-session", "user", "first message")
	if err != nil {
		t.Fatalf("AddMessage: %v", err)
	}

	history, err := store.GetHistory(ctx, "new-session")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 1 {
		t.Fatalf("expected 1 message, got %d", len(history))
	}
}

func TestAddFullMessage_WithToolCalls(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	msg := spawnllm.Message{
		Role:    "assistant",
		Content: "Let me search that.",
		ToolCalls: []spawnllm.ToolCall{
			{
				ID:   "call_abc",
				Type: "function",
				Function: &spawnllm.FunctionCall{
					Name:      "web_search",
					Arguments: `{"q":"golang jsonl"}`,
				},
			},
		},
	}

	_, err := store.AddFullMessage(ctx, "tc", msg)
	if err != nil {
		t.Fatalf("AddFullMessage: %v", err)
	}

	history, err := store.GetHistory(ctx, "tc")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 1 {
		t.Fatalf("expected 1, got %d", len(history))
	}
	if len(history[0].ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(history[0].ToolCalls))
	}
	tc := history[0].ToolCalls[0]
	if tc.ID != "call_abc" {
		t.Errorf("tool call ID = %q", tc.ID)
	}
	if tc.Function == nil || tc.Function.Name != "web_search" {
		t.Errorf("tool call function = %+v", tc.Function)
	}
}

func TestAddFullMessage_ToolCallID(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	msg := spawnllm.Message{
		Role:       "tool",
		Content:    "search results here",
		ToolCallID: "call_abc",
	}

	_, err := store.AddFullMessage(ctx, "tr", msg)
	if err != nil {
		t.Fatalf("AddFullMessage: %v", err)
	}

	history, err := store.GetHistory(ctx, "tr")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 1 {
		t.Fatalf("expected 1, got %d", len(history))
	}
	if history[0].ToolCallID != "call_abc" {
		t.Errorf("ToolCallID = %q", history[0].ToolCallID)
	}
}

func TestGetHistory_EmptySession(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	history, err := store.GetHistory(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if history == nil {
		t.Fatal("expected non-nil empty slice")
	}
	if len(history) != 0 {
		t.Errorf("expected 0 messages, got %d", len(history))
	}
}

func TestGetHistory_Ordering(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		err := store.AddMessage(
			ctx, "order",
			"user",
			string(rune('a'+i)),
		)
		if err != nil {
			t.Fatalf("AddMessage(%d): %v", i, err)
		}
	}

	history, err := store.GetHistory(ctx, "order")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 5 {
		t.Fatalf("expected 5, got %d", len(history))
	}
	for i := 0; i < 5; i++ {
		expected := string(rune('a' + i))
		if history[i].Content != expected {
			t.Errorf("msg[%d].Content = %q, want %q", i, history[i].Content, expected)
		}
	}
}

func TestSetSummary_GetSummary(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	// No summary yet.
	summary, err := store.GetSummary(ctx, "s1")
	if err != nil {
		t.Fatalf("GetSummary: %v", err)
	}
	if summary != "" {
		t.Errorf("expected empty, got %q", summary)
	}

	// Set a summary.
	err = store.SetSummary(ctx, "s1", "talked about Go")
	if err != nil {
		t.Fatalf("SetSummary: %v", err)
	}

	summary, err = store.GetSummary(ctx, "s1")
	if err != nil {
		t.Fatalf("GetSummary: %v", err)
	}
	if summary != "talked about Go" {
		t.Errorf("summary = %q", summary)
	}

	// Update summary.
	err = store.SetSummary(ctx, "s1", "updated summary")
	if err != nil {
		t.Fatalf("SetSummary: %v", err)
	}

	summary, err = store.GetSummary(ctx, "s1")
	if err != nil {
		t.Fatalf("GetSummary: %v", err)
	}
	if summary != "updated summary" {
		t.Errorf("summary = %q", summary)
	}
}

func TestTruncateHistory_KeepLast(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	for i := 0; i < 10; i++ {
		err := store.AddMessage(
			ctx, "trunc",
			"user",
			string(rune('a'+i)),
		)
		if err != nil {
			t.Fatalf("AddMessage: %v", err)
		}
	}

	err := store.TruncateHistory(ctx, "trunc", 4)
	if err != nil {
		t.Fatalf("TruncateHistory: %v", err)
	}

	history, err := store.GetHistory(ctx, "trunc")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 4 {
		t.Fatalf("expected 4, got %d", len(history))
	}
	// Should be the last 4: g, h, i, j
	if history[0].Content != "g" {
		t.Errorf("first kept = %q, want 'g'", history[0].Content)
	}
	if history[3].Content != "j" {
		t.Errorf("last kept = %q, want 'j'", history[3].Content)
	}
}

func TestTruncateHistory_KeepZero(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		err := store.AddMessage(ctx, "empty", "user", "msg")
		if err != nil {
			t.Fatalf("AddMessage: %v", err)
		}
	}

	err := store.TruncateHistory(ctx, "empty", 0)
	if err != nil {
		t.Fatalf("TruncateHistory: %v", err)
	}

	history, err := store.GetHistory(ctx, "empty")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 0 {
		t.Errorf("expected 0, got %d", len(history))
	}
}

func TestTruncateHistory_KeepMoreThanExists(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	for i := 0; i < 3; i++ {
		err := store.AddMessage(ctx, "few", "user", "msg")
		if err != nil {
			t.Fatalf("AddMessage: %v", err)
		}
	}

	// Keep 100, but only 3 exist — should keep all.
	err := store.TruncateHistory(ctx, "few", 100)
	if err != nil {
		t.Fatalf("TruncateHistory: %v", err)
	}

	history, err := store.GetHistory(ctx, "few")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 3 {
		t.Errorf("expected 3, got %d", len(history))
	}
}

func TestSetHistory_ReplacesAll(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	// Add some initial messages.
	for i := 0; i < 5; i++ {
		err := store.AddMessage(ctx, "replace", "user", "old")
		if err != nil {
			t.Fatalf("AddMessage: %v", err)
		}
	}

	// Replace with new history.
	newHistory := []spawnllm.Message{
		{Role: "user", Content: "new1"},
		{Role: "assistant", Content: "new2"},
	}
	err := store.SetHistory(ctx, "replace", newHistory)
	if err != nil {
		t.Fatalf("SetHistory: %v", err)
	}

	history, err := store.GetHistory(ctx, "replace")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 2 {
		t.Fatalf("expected 2, got %d", len(history))
	}
	if history[0].Content != "new1" || history[1].Content != "new2" {
		t.Errorf("history = %+v", history)
	}
}

func TestSetHistory_ResetsSkip(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	// Add messages and truncate.
	for i := 0; i < 10; i++ {
		err := store.AddMessage(ctx, "skip-reset", "user", "old")
		if err != nil {
			t.Fatalf("AddMessage: %v", err)
		}
	}
	err := store.TruncateHistory(ctx, "skip-reset", 3)
	if err != nil {
		t.Fatalf("TruncateHistory: %v", err)
	}

	// SetHistory should reset skip to 0.
	newHistory := []spawnllm.Message{
		{Role: "user", Content: "fresh"},
	}
	err = store.SetHistory(ctx, "skip-reset", newHistory)
	if err != nil {
		t.Fatalf("SetHistory: %v", err)
	}

	history, err := store.GetHistory(ctx, "skip-reset")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 1 {
		t.Fatalf("expected 1, got %d", len(history))
	}
	if history[0].Content != "fresh" {
		t.Errorf("content = %q", history[0].Content)
	}
}

func TestSetHistoryWithSeqs_PreservesStableSeqs(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		if err := store.AddMessage(ctx, "preserve-seq", "user", string(rune('a'+i))); err != nil {
			t.Fatalf("AddMessage: %v", err)
		}
	}

	active, err := store.GetHistoryWithSeqs(ctx, "preserve-seq")
	if err != nil {
		t.Fatalf("GetHistoryWithSeqs: %v", err)
	}
	tail := active[3:]
	if err := store.SetHistoryWithSeqs(ctx, "preserve-seq", tail); err != nil {
		t.Fatalf("SetHistoryWithSeqs: %v", err)
	}

	stored, err := store.GetHistoryWithSeqs(ctx, "preserve-seq")
	if err != nil {
		t.Fatalf("GetHistoryWithSeqs after rewrite: %v", err)
	}
	if len(stored) != 2 {
		t.Fatalf("expected 2 retained messages, got %d", len(stored))
	}
	if stored[0].Seq != 4 || stored[1].Seq != 5 {
		t.Fatalf("seqs not preserved: got %d,%d want 4,5", stored[0].Seq, stored[1].Seq)
	}

	if err := store.AddMessage(ctx, "preserve-seq", "assistant", "next"); err != nil {
		t.Fatalf("AddMessage after rewrite: %v", err)
	}
	stored, err = store.GetHistoryWithSeqs(ctx, "preserve-seq")
	if err != nil {
		t.Fatalf("GetHistoryWithSeqs final: %v", err)
	}
	if got := stored[len(stored)-1].Seq; got != 6 {
		t.Fatalf("next seq = %d, want 6", got)
	}
}

func TestAppendSummaryCheckpoint_WritesHashChain(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	first := SummaryCheckpoint{
		Model:           "model-a",
		SourceSeqStart:  1,
		SourceSeqEnd:    10,
		CoveredSeqStart: 1,
		CoveredSeqEnd:   10,
		Summary:         `{"version":2,"state":{}}`,
	}
	second := SummaryCheckpoint{
		Model:           "model-b",
		SourceSeqStart:  11,
		SourceSeqEnd:    20,
		CoveredSeqStart: 1,
		CoveredSeqEnd:   20,
		Summary:         `{"version":2,"state":{"goals":[]}}`,
	}
	if err := store.AppendSummaryCheckpoint(ctx, "summary-chain", first); err != nil {
		t.Fatalf("AppendSummaryCheckpoint first: %v", err)
	}
	if err := store.AppendSummaryCheckpoint(ctx, "summary-chain", second); err != nil {
		t.Fatalf("AppendSummaryCheckpoint second: %v", err)
	}

	data, err := os.ReadFile(store.summaryHistoryPath("summary-chain"))
	if err != nil {
		t.Fatalf("ReadFile summary history: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 checkpoint lines, got %d: %q", len(lines), string(data))
	}
	var gotFirst, gotSecond SummaryCheckpoint
	if err := json.Unmarshal([]byte(lines[0]), &gotFirst); err != nil {
		t.Fatalf("unmarshal first checkpoint: %v", err)
	}
	if err := json.Unmarshal([]byte(lines[1]), &gotSecond); err != nil {
		t.Fatalf("unmarshal second checkpoint: %v", err)
	}
	if gotFirst.ID != 1 || gotSecond.ID != 2 {
		t.Fatalf("checkpoint IDs = %d,%d; want 1,2", gotFirst.ID, gotSecond.ID)
	}
	if gotFirst.SummaryHash == "" || gotSecond.SummaryHash == "" {
		t.Fatalf("expected non-empty summary hashes: %#v %#v", gotFirst, gotSecond)
	}
	if gotSecond.PrevHash != gotFirst.SummaryHash {
		t.Fatalf("second PrevHash = %q, want first SummaryHash %q", gotSecond.PrevHash, gotFirst.SummaryHash)
	}
}

func TestColonInKey(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	err := store.AddMessage(ctx, "telegram:123", "user", "hi")
	if err != nil {
		t.Fatalf("AddMessage: %v", err)
	}

	history, err := store.GetHistory(ctx, "telegram:123")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 1 {
		t.Fatalf("expected 1, got %d", len(history))
	}

	// Verify the file is named with underscore.
	jsonlFile := filepath.Join(store.dir, "telegram_123.jsonl")
	if _, statErr := os.Stat(jsonlFile); statErr != nil {
		t.Errorf("expected file %s to exist: %v", jsonlFile, statErr)
	}
}

func TestCompact_RemovesSkippedMessages(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	// Write 10 messages, then truncate to keep last 3.
	for i := 0; i < 10; i++ {
		err := store.AddMessage(ctx, "compact", "user", string(rune('a'+i)))
		if err != nil {
			t.Fatalf("AddMessage: %v", err)
		}
	}
	err := store.TruncateHistory(ctx, "compact", 3)
	if err != nil {
		t.Fatalf("TruncateHistory: %v", err)
	}

	// Before compact: file still has 10 lines.
	allOnDisk, err := readMessages(store.jsonlPath("compact"), 0)
	if err != nil {
		t.Fatalf("readMessages: %v", err)
	}
	if len(allOnDisk) != 10 {
		t.Fatalf("before compact: expected 10 on disk, got %d", len(allOnDisk))
	}

	// Compact.
	err = store.Compact(ctx, "compact")
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}

	// After compact: file should have only 3 lines.
	allOnDisk, err = readMessages(store.jsonlPath("compact"), 0)
	if err != nil {
		t.Fatalf("readMessages: %v", err)
	}
	if len(allOnDisk) != 3 {
		t.Fatalf("after compact: expected 3 on disk, got %d", len(allOnDisk))
	}

	// GetHistory should still return the same 3 messages.
	history, err := store.GetHistory(ctx, "compact")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 3 {
		t.Fatalf("expected 3, got %d", len(history))
	}
	if history[0].Content != "h" || history[2].Content != "j" {
		t.Errorf("wrong content: %+v", history)
	}
}

func TestCompact_NoOpWhenNoSkip(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		err := store.AddMessage(ctx, "noop", "user", "msg")
		if err != nil {
			t.Fatalf("AddMessage: %v", err)
		}
	}

	// Compact without prior truncation — should be a no-op.
	err := store.Compact(ctx, "noop")
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}

	history, err := store.GetHistory(ctx, "noop")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 5 {
		t.Errorf("expected 5, got %d", len(history))
	}
}

func TestCompact_ThenAppend(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	for i := 0; i < 8; i++ {
		err := store.AddMessage(ctx, "cap", "user", string(rune('a'+i)))
		if err != nil {
			t.Fatalf("AddMessage: %v", err)
		}
	}

	err := store.TruncateHistory(ctx, "cap", 2)
	if err != nil {
		t.Fatalf("TruncateHistory: %v", err)
	}
	err = store.Compact(ctx, "cap")
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}

	// Append after compaction should work correctly.
	err = store.AddMessage(ctx, "cap", "user", "new")
	if err != nil {
		t.Fatalf("AddMessage after compact: %v", err)
	}

	history, err := store.GetHistory(ctx, "cap")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 3 {
		t.Fatalf("expected 3, got %d", len(history))
	}
	// g, h (kept from truncation), new (appended after compaction).
	if history[0].Content != "g" {
		t.Errorf("first = %q, want 'g'", history[0].Content)
	}
	if history[2].Content != "new" {
		t.Errorf("last = %q, want 'new'", history[2].Content)
	}
}

func TestTruncateHistory_StaleMetaCount(t *testing.T) {
	// Simulates a crash between JSONL append and meta update in addMsg:
	// file has N+1 lines but meta.Count is still N. TruncateHistory must
	// reconcile with the real line count so that keepLast is accurate.
	store := newTestStore(t)
	ctx := context.Background()

	// Write 10 messages normally (meta.Count = 10).
	for i := 0; i < 10; i++ {
		err := store.AddMessage(ctx, "stale", "user", string(rune('a'+i)))
		if err != nil {
			t.Fatalf("AddMessage: %v", err)
		}
	}

	// Simulate crash: append a line to JSONL but do NOT update meta.
	// This leaves meta.Count = 10 while the file has 11 lines.
	jsonlPath := store.jsonlPath("stale")
	f, err := os.OpenFile(jsonlPath, os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		t.Fatalf("open for append: %v", err)
	}
	_, err = f.WriteString(`{"role":"user","content":"orphan"}` + "\n")
	if err != nil {
		t.Fatalf("write orphan: %v", err)
	}
	f.Close()

	// TruncateHistory(keepLast=4) should keep the last 4 of 11 lines,
	// not the last 4 of 10.
	err = store.TruncateHistory(ctx, "stale", 4)
	if err != nil {
		t.Fatalf("TruncateHistory: %v", err)
	}

	history, err := store.GetHistory(ctx, "stale")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 4 {
		t.Fatalf("expected 4, got %d", len(history))
	}
	// Last 4 of [a,b,c,d,e,f,g,h,i,j,orphan] = [h,i,j,orphan]
	if history[0].Content != "h" {
		t.Errorf("first kept = %q, want 'h'", history[0].Content)
	}
	if history[3].Content != "orphan" {
		t.Errorf("last kept = %q, want 'orphan'", history[3].Content)
	}
}

func TestCrashRecovery_PartialLine(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	// Write a valid message first.
	err := store.AddMessage(ctx, "crash", "user", "valid")
	if err != nil {
		t.Fatalf("AddMessage: %v", err)
	}

	// Simulate a crash by appending a partial JSON line directly.
	jsonlPath := store.jsonlPath("crash")
	f, err := os.OpenFile(jsonlPath, os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		t.Fatalf("open for append: %v", err)
	}
	_, err = f.WriteString(`{"role":"user","content":"incomple`)
	if err != nil {
		t.Fatalf("write partial: %v", err)
	}
	f.Close()

	// GetHistory should return only the valid message.
	history, err := store.GetHistory(ctx, "crash")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 1 {
		t.Fatalf("expected 1 valid message, got %d", len(history))
	}
	if history[0].Content != "valid" {
		t.Errorf("content = %q", history[0].Content)
	}
}

func TestPersistence_AcrossInstances(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	// Write with first instance.
	store1, err := NewJSONLStore(dir)
	if err != nil {
		t.Fatalf("NewJSONLStore: %v", err)
	}
	err = store1.AddMessage(ctx, "persist", "user", "remember me")
	if err != nil {
		t.Fatalf("AddMessage: %v", err)
	}
	err = store1.SetSummary(ctx, "persist", "a test session")
	if err != nil {
		t.Fatalf("SetSummary: %v", err)
	}
	store1.Close()

	// Read with second instance.
	store2, err := NewJSONLStore(dir)
	if err != nil {
		t.Fatalf("NewJSONLStore: %v", err)
	}
	defer store2.Close()

	history, err := store2.GetHistory(ctx, "persist")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(history) != 1 || history[0].Content != "remember me" {
		t.Errorf("history = %+v", history)
	}

	summary, err := store2.GetSummary(ctx, "persist")
	if err != nil {
		t.Fatalf("GetSummary: %v", err)
	}
	if summary != "a test session" {
		t.Errorf("summary = %q", summary)
	}
}

func TestConcurrent_AddAndRead(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	var wg sync.WaitGroup
	const goroutines = 10
	const msgsPerGoroutine = 20

	// Concurrent writes.
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < msgsPerGoroutine; i++ {
				_ = store.AddMessage(ctx, "concurrent", "user", "msg")
			}
		}()
	}
	wg.Wait()

	history, err := store.GetHistory(ctx, "concurrent")
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	expected := goroutines * msgsPerGoroutine
	if len(history) != expected {
		t.Errorf("expected %d messages, got %d", expected, len(history))
	}
}

func TestConcurrent_SummarizeRace(t *testing.T) {
	// Simulates the #704 race: one goroutine adds messages while
	// another truncates + sets summary — like summarizeSession().
	store := newTestStore(t)
	ctx := context.Background()

	// Seed with some messages.
	for i := 0; i < 20; i++ {
		err := store.AddMessage(ctx, "race", "user", "seed")
		if err != nil {
			t.Fatalf("AddMessage: %v", err)
		}
	}

	var wg sync.WaitGroup

	// Writer goroutine (main agent loop).
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			_ = store.AddMessage(ctx, "race", "user", "new")
		}
	}()

	// Summarizer goroutine (background task).
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			_ = store.SetSummary(ctx, "race", "summary")
			_ = store.TruncateHistory(ctx, "race", 5)
		}
	}()

	wg.Wait()

	// Verify the store is still in a consistent state.
	_, err := store.GetHistory(ctx, "race")
	if err != nil {
		t.Fatalf("GetHistory after race: %v", err)
	}
	_, err = store.GetSummary(ctx, "race")
	if err != nil {
		t.Fatalf("GetSummary after race: %v", err)
	}
}

func TestMultipleSessions_Isolation(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	err := store.AddMessage(ctx, "s1", "user", "msg for s1")
	if err != nil {
		t.Fatalf("AddMessage: %v", err)
	}
	err = store.AddMessage(ctx, "s2", "user", "msg for s2")
	if err != nil {
		t.Fatalf("AddMessage: %v", err)
	}

	h1, err := store.GetHistory(ctx, "s1")
	if err != nil {
		t.Fatalf("GetHistory s1: %v", err)
	}
	h2, err := store.GetHistory(ctx, "s2")
	if err != nil {
		t.Fatalf("GetHistory s2: %v", err)
	}

	if len(h1) != 1 || h1[0].Content != "msg for s1" {
		t.Errorf("s1 history = %+v", h1)
	}
	if len(h2) != 1 || h2[0].Content != "msg for s2" {
		t.Errorf("s2 history = %+v", h2)
	}
}

func BenchmarkAddMessage(b *testing.B) {
	dir := b.TempDir()
	store, err := NewJSONLStore(dir)
	if err != nil {
		b.Fatalf("NewJSONLStore: %v", err)
	}
	defer store.Close()
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = store.AddMessage(ctx, "bench", "user", "benchmark message content")
	}
}

func BenchmarkGetHistory_100(b *testing.B) {
	dir := b.TempDir()
	store, err := NewJSONLStore(dir)
	if err != nil {
		b.Fatalf("NewJSONLStore: %v", err)
	}
	defer store.Close()
	ctx := context.Background()

	for i := 0; i < 100; i++ {
		_ = store.AddMessage(ctx, "bench", "user", "message content")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = store.GetHistory(ctx, "bench")
	}
}

func BenchmarkGetHistory_1000(b *testing.B) {
	dir := b.TempDir()
	store, err := NewJSONLStore(dir)
	if err != nil {
		b.Fatalf("NewJSONLStore: %v", err)
	}
	defer store.Close()
	ctx := context.Background()

	for i := 0; i < 1000; i++ {
		_ = store.AddMessage(ctx, "bench", "user", "message content")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = store.GetHistory(ctx, "bench")
	}
}

// --- Phase 2 tests ---

func TestStoredMessage_SeqAssignment(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		err := store.AddMessage(ctx, "seq", "user", "msg")
		if err != nil {
			t.Fatalf("AddMessage(%d): %v", i, err)
		}
	}

	// Read raw StoredMessages from disk to verify seq numbers.
	stored, err := readStoredMessages(store.jsonlPath("seq"), 0)
	if err != nil {
		t.Fatalf("readStoredMessages: %v", err)
	}
	if len(stored) != 5 {
		t.Fatalf("expected 5 stored messages, got %d", len(stored))
	}
	for i, sm := range stored {
		want := int64(i + 1)
		if sm.Seq != want {
			t.Errorf("stored[%d].Seq = %d, want %d", i, sm.Seq, want)
		}
	}

	// After Compact, seq numbers should be preserved (not reset).
	err = store.TruncateHistory(ctx, "seq", 3)
	if err != nil {
		t.Fatalf("TruncateHistory: %v", err)
	}
	err = store.Compact(ctx, "seq")
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}

	stored, err = readStoredMessages(store.jsonlPath("seq"), 0)
	if err != nil {
		t.Fatalf("readStoredMessages after compact: %v", err)
	}
	if len(stored) != 3 {
		t.Fatalf("after compact: expected 3, got %d", len(stored))
	}
	// seq numbers 3, 4, 5 should be preserved.
	for i, sm := range stored {
		want := int64(i + 3)
		if sm.Seq != want {
			t.Errorf("after compact stored[%d].Seq = %d, want %d", i, sm.Seq, want)
		}
	}

	// AddMessage after compact should continue with seq=6.
	err = store.AddMessage(ctx, "seq", "user", "after compact")
	if err != nil {
		t.Fatalf("AddMessage after compact: %v", err)
	}
	stored, err = readStoredMessages(store.jsonlPath("seq"), 0)
	if err != nil {
		t.Fatalf("readStoredMessages final: %v", err)
	}
	last := stored[len(stored)-1]
	if last.Seq != 6 {
		t.Errorf("last.Seq = %d, want 6", last.Seq)
	}
}

// testCronPrefix mirrors the cron-wrapper header for building noise-classifier
// fixtures. The shared marker parser lives in cronmsg.
const testCronPrefix = "The following message is from a cron job that fired at "

func TestNoiseClassifier_CronNoise(t *testing.T) {
	cache := newNoiseCache()

	// First cron message with a given payload — not noise.
	payload := "check disk space: 42% used"
	content1 := testCronPrefix + "2026-01-01T00:00:00Z:\n\n" + payload
	msg1 := StoredMessage{Seq: 1, Message: spawnllm.Message{Role: "user", Content: content1}}
	if isNoise(msg1, cache, cronmsg.CollapseKey) {
		t.Error("first cron message should not be noise")
	}
	updateNoiseCache(msg1, cache, cronmsg.CollapseKey)

	// Same payload at a different timestamp — noise.
	content2 := testCronPrefix + "2026-01-01T01:00:00Z:\n\n" + payload
	msg2 := StoredMessage{Seq: 2, Message: spawnllm.Message{Role: "user", Content: content2}}
	if !isNoise(msg2, cache, cronmsg.CollapseKey) {
		t.Error("duplicate cron payload should be noise")
	}
	updateNoiseCache(msg2, cache, cronmsg.CollapseKey)

	// Different payload — not noise.
	content3 := testCronPrefix + "2026-01-01T02:00:00Z:\n\n" + "check disk space: 80% used"
	msg3 := StoredMessage{Seq: 3, Message: spawnllm.Message{Role: "user", Content: content3}}
	if isNoise(msg3, cache, cronmsg.CollapseKey) {
		t.Error("different cron payload should not be noise")
	}
}

func TestNoiseClassifier_SameRole(t *testing.T) {
	cache := newNoiseCache()

	msg1 := StoredMessage{Seq: 1, Message: spawnllm.Message{Role: "user", Content: "hello"}}
	if isNoise(msg1, cache, cronmsg.CollapseKey) {
		t.Error("first message should not be noise")
	}
	updateNoiseCache(msg1, cache, cronmsg.CollapseKey)

	// Same role and content — noise.
	msg2 := StoredMessage{Seq: 2, Message: spawnllm.Message{Role: "user", Content: "hello"}}
	if !isNoise(msg2, cache, cronmsg.CollapseKey) {
		t.Error("duplicate same-role content should be noise")
	}
}

func TestNoiseClassifier_DifferentContent(t *testing.T) {
	cache := newNoiseCache()

	msg1 := StoredMessage{Seq: 1, Message: spawnllm.Message{Role: "user", Content: "hello"}}
	updateNoiseCache(msg1, cache, cronmsg.CollapseKey)

	msg2 := StoredMessage{Seq: 2, Message: spawnllm.Message{Role: "user", Content: "world"}}
	if isNoise(msg2, cache, cronmsg.CollapseKey) {
		t.Error("different content should not be noise")
	}
}

func TestMeaningfulCount_IncrementedForNonNoise(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	// Write 3 distinct messages.
	for i := 0; i < 3; i++ {
		content := string(rune('a' + i))
		err := store.AddMessage(ctx, "mc", "user", content)
		if err != nil {
			t.Fatalf("AddMessage(%d): %v", i, err)
		}
	}

	// Write 2 duplicate messages (same role+content as the last one).
	for i := 0; i < 2; i++ {
		err := store.AddMessage(ctx, "mc", "user", "c")
		if err != nil {
			t.Fatalf("AddMessage(dup %d): %v", i, err)
		}
	}

	meta, err := store.readMeta("mc")
	if err != nil {
		t.Fatalf("readMeta: %v", err)
	}

	if meta.Count != 5 {
		t.Errorf("Count = %d, want 5", meta.Count)
	}
	// Only the 3 distinct messages should be meaningful.
	if meta.MeaningfulCount != 3 {
		t.Errorf("MeaningfulCount = %d, want 3", meta.MeaningfulCount)
	}
}

// TestJSONLStore_NoArchiveJSONL verifies that JSONLStore no longer creates
// .archive.jsonl files. Archive writes are now owned by ArchiveStore (SQLite).
func TestJSONLStore_NoArchiveJSONL(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	err := store.AddMessage(ctx, "arch", "user", "hello")
	if err != nil {
		t.Fatalf("AddMessage: %v", err)
	}

	// The .archive.jsonl file must not be created.
	archJSONL := filepath.Join(store.dir, "arch.archive.jsonl")
	if _, err := os.Stat(archJSONL); !os.IsNotExist(err) {
		t.Errorf(".archive.jsonl should not be created, got err=%v", err)
	}
}

// TestTruncateHistory_ResetsTrackingFields verifies that a full reset (keepLast=0)
// clears MeaningfulCount and CompressionCooling in meta.
// Archive deletion is now handled by ContextManager.Reset() via ArchiveStore.Delete();
// JSONLStore no longer manages the archive file.
func TestTruncateHistory_ResetsTrackingFields(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		err := store.AddMessage(ctx, "ar", "user", "msg")
		if err != nil {
			t.Fatalf("AddMessage: %v", err)
		}
	}

	err := store.TruncateHistory(ctx, "ar", 0)
	if err != nil {
		t.Fatalf("TruncateHistory: %v", err)
	}

	meta, err := store.readMeta("ar")
	if err != nil {
		t.Fatalf("readMeta: %v", err)
	}
	if meta.MeaningfulCount != 0 {
		t.Errorf("MeaningfulCount = %d, want 0", meta.MeaningfulCount)
	}
	if meta.CompressionCooling {
		t.Error("CompressionCooling should be false after reset")
	}
}

// TestAddMessage_StampsCreatedAt is the regression test for the bug where
// the JSONL append path constructed StoredMessage literals without setting
// CreatedAt, persisting every message with "created_at":"0001-01-01T00:00:00Z"
// on disk and corrupting downstream covered_seq_*_at metadata on context
// summaries. The fix is the NewStoredMessage constructor; if the stamping
// is removed from that constructor (or a future caller bypasses it via a
// raw struct literal on the append path), this test must fail.
func TestAddMessage_StampsCreatedAt(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	before := time.Now().UTC()
	if err := store.AddMessage(ctx, "stamp-sess", "user", "hello"); err != nil {
		t.Fatalf("AddMessage: %v", err)
	}
	if _, err := store.AddFullMessage(ctx, "stamp-sess", spawnllm.Message{
		Role:    "assistant",
		Content: "hi",
	}); err != nil {
		t.Fatalf("AddFullMessage: %v", err)
	}
	after := time.Now().UTC()

	stored, err := store.GetHistoryWithSeqs(ctx, "stamp-sess")
	if err != nil {
		t.Fatalf("GetHistoryWithSeqs: %v", err)
	}
	if len(stored) != 2 {
		t.Fatalf("expected 2 stored messages, got %d", len(stored))
	}

	for i, m := range stored {
		if m.CreatedAt.IsZero() {
			t.Errorf("stored[%d].CreatedAt is zero — bug regression (jsonl append did not stamp)", i)
			continue
		}
		// The persisted value round-trips through json.Marshal, so the
		// monotonic clock reading is stripped. Compare with wall-clock
		// bounds.
		if m.CreatedAt.Before(before) || m.CreatedAt.After(after) {
			t.Errorf("stored[%d].CreatedAt = %v, want in [%v, %v]",
				i, m.CreatedAt, before, after)
		}
	}
}

// TestSetHistory_StampsCreatedAt covers the SetHistory rewrite path, which
// previously also constructed StoredMessage literals without CreatedAt.
func TestSetHistory_StampsCreatedAt(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	before := time.Now().UTC()
	err := store.SetHistory(ctx, "rewrite-sess", []spawnllm.Message{
		{Role: "user", Content: "a"},
		{Role: "assistant", Content: "b"},
	})
	if err != nil {
		t.Fatalf("SetHistory: %v", err)
	}
	after := time.Now().UTC()

	stored, err := store.GetHistoryWithSeqs(ctx, "rewrite-sess")
	if err != nil {
		t.Fatalf("GetHistoryWithSeqs: %v", err)
	}
	if len(stored) != 2 {
		t.Fatalf("expected 2 stored messages, got %d", len(stored))
	}
	for i, m := range stored {
		if m.CreatedAt.IsZero() {
			t.Errorf("stored[%d].CreatedAt is zero — SetHistory did not stamp", i)
			continue
		}
		if m.CreatedAt.Before(before) || m.CreatedAt.After(after) {
			t.Errorf("stored[%d].CreatedAt = %v, want in [%v, %v]",
				i, m.CreatedAt, before, after)
		}
	}
}

// TestCompactionState_PersistsToggles verifies the per-session /tools and
// /reasoning flags survive a store reopen (i.e. a process restart), round-tripping
// through the on-disk session meta.
func TestCompactionState_PersistsToggles(t *testing.T) {
	dir := t.TempDir()
	const key = "agent:wendy:main"

	store, err := NewJSONLStore(dir)
	if err != nil {
		t.Fatalf("NewJSONLStore: %v", err)
	}
	// Create the session so its meta exists.
	if _, err := store.AddFullMessage(context.Background(), key, spawnllm.Message{Role: "user", Content: "hi"}); err != nil {
		t.Fatalf("AddFullMessage: %v", err)
	}

	st, err := store.GetCompactionState(key)
	if err != nil {
		t.Fatalf("GetCompactionState: %v", err)
	}
	st.ShowToolActivity = true
	st.ExposeReasoning = true
	if err := store.SetCompactionState(key, st); err != nil {
		t.Fatalf("SetCompactionState: %v", err)
	}

	// Reopen the store (simulate a restart) and read back from disk.
	store2, err := NewJSONLStore(dir)
	if err != nil {
		t.Fatalf("reopen NewJSONLStore: %v", err)
	}
	got, err := store2.GetCompactionState(key)
	if err != nil {
		t.Fatalf("GetCompactionState after reopen: %v", err)
	}
	if !got.ShowToolActivity {
		t.Error("ShowToolActivity did not persist across restart")
	}
	if !got.ExposeReasoning {
		t.Error("ExposeReasoning did not persist across restart")
	}
}

// TestSetNoiseKey_StoreLevel covers the store-level switch: with no noise key
// two fires of one job (different timestamps) are both meaningful; with the
// host's key installed the second is noise.
func TestSetNoiseKey_StoreLevel(t *testing.T) {
	ctx := context.Background()
	fire := func(ts string) string { return testCronPrefix + ts + ":\n\ncheck disk" }

	off := newTestStore(t)
	for _, ts := range []string{"2026-01-01 00:00 UTC", "2026-01-01 01:00 UTC"} {
		if err := off.AddMessage(ctx, "k", "user", fire(ts)); err != nil {
			t.Fatal(err)
		}
	}
	if meta, _ := off.readMeta("k"); meta.MeaningfulCount != 2 {
		t.Errorf("without a noise key MeaningfulCount = %d, want 2", meta.MeaningfulCount)
	}

	on := newTestStore(t)
	on.SetNoiseKey(cronmsg.CollapseKey)
	for _, ts := range []string{"2026-01-01 00:00 UTC", "2026-01-01 01:00 UTC"} {
		if err := on.AddMessage(ctx, "k", "user", fire(ts)); err != nil {
			t.Fatal(err)
		}
	}
	if meta, _ := on.readMeta("k"); meta.MeaningfulCount != 1 {
		t.Errorf("with the noise key MeaningfulCount = %d, want 1", meta.MeaningfulCount)
	}
}
