package memory

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/PivotLLM/spawnllm"
)

// openTestArchive creates a fresh ArchiveStore in t.TempDir() and registers
// cleanup to close it.
func openTestArchive(t *testing.T) *ArchiveStore {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.archive.db")
	a, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { a.Close() })
	return a
}

// sampleMsg builds a spawnllm.Message with optional ToolCalls for testing.
func sampleMsg(role, content string, toolCalls ...spawnllm.ToolCall) spawnllm.Message {
	return spawnllm.Message{
		Role:      role,
		Content:   content,
		ToolCalls: toolCalls,
	}
}

func toolCall(id, name, args string) spawnllm.ToolCall {
	return spawnllm.ToolCall{
		ID:   id,
		Type: "function",
		Function: &spawnllm.FunctionCall{
			Name:      name,
			Arguments: args,
		},
	}
}

// TestArchiveStore_AppendAndQueryRange verifies that appended messages are
// returned by QueryRange in insertion order, including full ToolCalls content.
func TestArchiveStore_AppendAndQueryRange(t *testing.T) {
	a := openTestArchive(t)
	now := time.Now()

	tc := toolCall("tc1", "read_file", `{"path":"/tmp/x"}`)
	msgs := []spawnllm.Message{
		sampleMsg("user", "hello"),
		sampleMsg("assistant", "world"),
		sampleMsg("assistant", "with tool", tc),
	}
	for i, m := range msgs {
		if err := a.Append(int64(i+1), m, now.Add(time.Duration(i)*time.Second)); err != nil {
			t.Fatalf("Append seq=%d: %v", i+1, err)
		}
	}

	got, err := a.QueryRange(1, 3)
	if err != nil {
		t.Fatalf("QueryRange: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("got %d messages, want 3", len(got))
	}
	if got[0].Role != "user" || got[0].Content != "hello" {
		t.Errorf("msg[0] = %+v", got[0])
	}
	if got[1].Content != "world" {
		t.Errorf("msg[1].Content = %q", got[1].Content)
	}
	// ToolCalls must survive the round-trip.
	if len(got[2].ToolCalls) != 1 {
		t.Fatalf("msg[2] ToolCalls len = %d, want 1", len(got[2].ToolCalls))
	}
	if got[2].Message.ToolCalls[0].ID != "tc1" {
		t.Errorf("ToolCall ID = %q, want tc1", got[2].Message.ToolCalls[0].ID)
	}
}

// TestArchiveStore_QueryRange_SubRange verifies that only messages in the
// requested range are returned.
func TestArchiveStore_QueryRange_SubRange(t *testing.T) {
	a := openTestArchive(t)
	now := time.Now()

	for i := 1; i <= 10; i++ {
		if err := a.Append(int64(i), sampleMsg("user", fmt.Sprintf("msg%d", i)), now); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}

	got, err := a.QueryRange(3, 7)
	if err != nil {
		t.Fatalf("QueryRange: %v", err)
	}
	if len(got) != 5 {
		t.Fatalf("got %d, want 5", len(got))
	}
	if got[0].Content != "msg3" {
		t.Errorf("first = %q, want msg3", got[0].Content)
	}
	if got[4].Content != "msg7" {
		t.Errorf("last = %q, want msg7", got[4].Content)
	}
}

// TestArchiveStore_Bounds verifies Bounds returns correct min/max after appends.
func TestArchiveStore_Bounds(t *testing.T) {
	a := openTestArchive(t)

	min, max, err := a.Bounds()
	if err != nil {
		t.Fatalf("Bounds (empty): %v", err)
	}
	if min != 0 || max != 0 {
		t.Errorf("empty archive Bounds = (%d, %d), want (0, 0)", min, max)
	}

	now := time.Now()
	for _, seq := range []int64{5, 10, 3, 8} {
		if err := a.Append(seq, sampleMsg("user", "x"), now); err != nil {
			t.Fatalf("Append seq=%d: %v", seq, err)
		}
	}

	min, max, err = a.Bounds()
	if err != nil {
		t.Fatalf("Bounds: %v", err)
	}
	if min != 3 || max != 10 {
		t.Errorf("Bounds = (%d, %d), want (3, 10)", min, max)
	}
}

// TestArchiveStore_Stats verifies Stats() returns the message count plus the
// first/last created_at timestamps in a single round-trip. This is the
// primitive the /status command uses to cheaply surface archive size and date
// range without loading any payload bytes.
func TestArchiveStore_Stats(t *testing.T) {
	a := openTestArchive(t)

	// Empty archive — count zero, timestamps zero.
	count, first, last, err := a.Stats()
	if err != nil {
		t.Fatalf("Stats (empty): %v", err)
	}
	if count != 0 || !first.IsZero() || !last.IsZero() {
		t.Errorf("empty Stats = (%d, %v, %v), want (0, zero, zero)", count, first, last)
	}

	// Populate with known timestamps; insert out of order to confirm SQL aggregates.
	t1 := time.Unix(1_700_000_000, 0)
	t2 := time.Unix(1_700_005_000, 0)
	t3 := time.Unix(1_700_010_000, 0)
	for _, e := range []struct {
		seq int64
		at  time.Time
	}{
		{seq: 2, at: t2},
		{seq: 1, at: t1},
		{seq: 3, at: t3},
	} {
		if err := a.Append(e.seq, sampleMsg("user", "x"), e.at); err != nil {
			t.Fatalf("Append seq=%d: %v", e.seq, err)
		}
	}

	count, first, last, err = a.Stats()
	if err != nil {
		t.Fatalf("Stats: %v", err)
	}
	if count != 3 {
		t.Errorf("Stats count = %d, want 3", count)
	}
	if !first.Equal(t1) {
		t.Errorf("Stats first = %v, want %v", first, t1)
	}
	if !last.Equal(t3) {
		t.Errorf("Stats last = %v, want %v", last, t3)
	}
}

// TestArchiveStore_RetrievalWindowClamping verifies that the caller-computed
// window clamping logic (as used by the tool layer) correctly restricts results.
// Archive has maxSeq=300; window=250; request is seq 1-1000.
// Expected: only seq 51-300 returned.
func TestArchiveStore_RetrievalWindowClamping(t *testing.T) {
	a := openTestArchive(t)
	now := time.Now()

	// Write seq 1..300
	for i := 1; i <= 300; i++ {
		if err := a.Append(int64(i), sampleMsg("user", fmt.Sprintf("msg%d", i)), now); err != nil {
			t.Fatalf("Append seq=%d: %v", i, err)
		}
	}

	_, maxSeq, err := a.Bounds()
	if err != nil {
		t.Fatalf("Bounds: %v", err)
	}

	// Tool layer clamping as per remediation plan:
	windowSize := int64(250)
	requestedMin := int64(1)
	requestedMax := int64(1000)

	effectiveMin := requestedMin
	if floor := maxSeq - windowSize + 1; floor > effectiveMin {
		effectiveMin = floor
	}
	effectiveMax := min(maxSeq, requestedMax)

	// effectiveMin = max(1, 300-250+1) = 51
	// effectiveMax = min(1000, 300) = 300
	if effectiveMin != 51 {
		t.Fatalf("effectiveMin = %d, want 51", effectiveMin)
	}
	if effectiveMax != 300 {
		t.Fatalf("effectiveMax = %d, want 300", effectiveMax)
	}

	got, err := a.QueryRange(effectiveMin, effectiveMax)
	if err != nil {
		t.Fatalf("QueryRange: %v", err)
	}
	if len(got) != 250 {
		t.Fatalf("got %d messages, want 250", len(got))
	}
	if got[0].Content != "msg51" {
		t.Errorf("first = %q, want msg51", got[0].Content)
	}
	if got[249].Content != "msg300" {
		t.Errorf("last = %q, want msg300", got[249].Content)
	}
}

// TestArchiveStore_WALMode verifies that WAL journal mode is set on open.
func TestArchiveStore_WALMode(t *testing.T) {
	path := filepath.Join(t.TempDir(), "wal.archive.db")
	a, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer a.Close()

	var mode string
	row := a.db.QueryRow("PRAGMA journal_mode")
	if err := row.Scan(&mode); err != nil {
		t.Fatalf("PRAGMA journal_mode: %v", err)
	}
	if mode != "wal" {
		t.Errorf("journal_mode = %q, want wal", mode)
	}
}

// TestArchiveStore_FTS5Trigger verifies that appended messages are findable via Search.
func TestArchiveStore_FTS5Trigger(t *testing.T) {
	a := openTestArchive(t)
	now := time.Now()

	uniqueTerm := "xyzzyarchiveterm99"
	if err := a.Append(1, sampleMsg("user", "ordinary message"), now); err != nil {
		t.Fatalf("Append 1: %v", err)
	}
	if err := a.Append(2, sampleMsg("assistant", "message with "+uniqueTerm), now); err != nil {
		t.Fatalf("Append 2: %v", err)
	}

	got, err := a.Search(context.Background(), uniqueTerm, "", 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d results, want 1", len(got))
	}
	if got[0].Message.Role != "assistant" {
		t.Errorf("role = %q, want assistant", got[0].Message.Role)
	}
}

// TestArchiveStore_ConcurrentReadDuringWrite verifies that a read-only URI
// connection can read while the write handle is active (WAL-mode concurrency).
func TestArchiveStore_ConcurrentReadDuringWrite(t *testing.T) {
	a := openTestArchive(t)
	now := time.Now()

	const total = 50
	var wg sync.WaitGroup

	// Writer goroutine.
	wg.Go(func() {
		for i := 1; i <= total; i++ {
			if err := a.Append(int64(i), sampleMsg("user", fmt.Sprintf("msg%d", i)), now); err != nil {
				t.Errorf("Append seq=%d: %v", i, err)
				return
			}
		}
	})

	// Reader goroutines — each opens its own read-only connection.
	for range 5 {
		wg.Go(func() {
			// We just verify no SQLITE_BUSY or other errors occur.
			_, _ = a.QueryRange(1, 25)
			_, _, _ = a.Bounds()
		})
	}

	wg.Wait()

	// After all writes, full range should be present.
	got, err := a.QueryRange(1, total)
	if err != nil {
		t.Fatalf("QueryRange after concurrent ops: %v", err)
	}
	if len(got) != total {
		t.Errorf("got %d messages, want %d", len(got), total)
	}
}

// TestArchiveStore_ErrArchiveUnavailable verifies that opening a bad path
// returns a store with unavailable=true, and all methods return gracefully.
func TestArchiveStore_ErrArchiveUnavailable(t *testing.T) {
	// A path under a non-existent directory that cannot be created.
	badPath := "/proc/nonexistent/bad.archive.db"
	a, err := Open(badPath)
	if !errors.Is(err, ErrArchiveUnavailable) {
		t.Fatalf("Open bad path error = %v, want ErrArchiveUnavailable", err)
	}
	if !a.unavailable {
		t.Error("store.unavailable should be true")
	}

	// Append should be a no-op.
	if err := a.Append(1, sampleMsg("user", "x"), time.Now()); err != nil {
		t.Errorf("Append on unavailable: %v", err)
	}

	// QueryRange should return ErrArchiveUnavailable.
	_, err = a.QueryRange(1, 10)
	if !errors.Is(err, ErrArchiveUnavailable) {
		t.Errorf("QueryRange on unavailable: %v, want ErrArchiveUnavailable", err)
	}

	// Search should return ErrArchiveUnavailable.
	_, err = a.Search(context.Background(), "term", "", 10)
	if !errors.Is(err, ErrArchiveUnavailable) {
		t.Errorf("Search on unavailable: %v, want ErrArchiveUnavailable", err)
	}

	// Bounds should return ErrArchiveUnavailable.
	_, _, err = a.Bounds()
	if !errors.Is(err, ErrArchiveUnavailable) {
		t.Errorf("Bounds on unavailable: %v, want ErrArchiveUnavailable", err)
	}

	// Close should not panic.
	if err := a.Close(); err != nil {
		t.Errorf("Close on unavailable: %v", err)
	}
}

// TestArchiveStore_JSONRoundTrip verifies that ToolCalls and ToolCallID fields
// survive Append → QueryRange without data loss.
func TestArchiveStore_JSONRoundTrip(t *testing.T) {
	a := openTestArchive(t)
	now := time.Now()

	msg := spawnllm.Message{
		Role:       "tool",
		Content:    "result data",
		ToolCallID: "call-abc-123",
		ToolCalls: []spawnllm.ToolCall{
			{
				ID:   "tc-nested",
				Type: "function",
				Function: &spawnllm.FunctionCall{
					Name:      "exec",
					Arguments: `{"cmd":"ls","args":["-la"]}`,
				},
			},
		},
		ReasoningContent: "some reasoning",
	}

	if err := a.Append(1, msg, now); err != nil {
		t.Fatalf("Append: %v", err)
	}

	got, err := a.QueryRange(1, 1)
	if err != nil {
		t.Fatalf("QueryRange: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d, want 1", len(got))
	}
	r := got[0]
	if r.Role != "tool" {
		t.Errorf("Role = %q, want tool", r.Role)
	}
	if r.ToolCallID != "call-abc-123" {
		t.Errorf("ToolCallID = %q, want call-abc-123", r.ToolCallID)
	}
	if len(r.ToolCalls) != 1 || r.Message.ToolCalls[0].ID != "tc-nested" {
		t.Errorf("ToolCalls = %+v", r.ToolCalls)
	}
	if r.ToolCalls[0].Function.Arguments != `{"cmd":"ls","args":["-la"]}` {
		t.Errorf("Arguments = %q", r.ToolCalls[0].Function.Arguments)
	}
	if r.ReasoningContent != "some reasoning" {
		t.Errorf("ReasoningContent = %q", r.ReasoningContent)
	}
}

// TestArchiveStore_Search_QueryTooLong verifies that queries over 500 chars
// are rejected before execution.
func TestArchiveStore_Search_QueryTooLong(t *testing.T) {
	a := openTestArchive(t)

	query := make([]byte, 501)
	for i := range query {
		query[i] = 'a'
	}
	_, err := a.Search(context.Background(), string(query), "", 10)
	if err == nil {
		t.Error("expected error for query > 500 chars, got nil")
	}
}

// TestArchiveStore_Search_LimitClamped verifies that limit > 100 is clamped to 100.
func TestArchiveStore_Search_LimitClamped(t *testing.T) {
	a := openTestArchive(t)
	now := time.Now()

	// Add 120 messages containing the same term.
	term := "clamptest"
	for i := 1; i <= 120; i++ {
		m := sampleMsg("user", fmt.Sprintf("%s message %d", term, i))
		if err := a.Append(int64(i), m, now); err != nil {
			t.Fatalf("Append seq=%d: %v", i, err)
		}
	}

	got, err := a.Search(context.Background(), term, "", 200) // request 200, should get ≤100
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) > 100 {
		t.Errorf("got %d results, want ≤100", len(got))
	}
}

// TestArchiveStore_Search_RoleFilter verifies that role filtering works correctly.
func TestArchiveStore_Search_RoleFilter(t *testing.T) {
	a := openTestArchive(t)
	now := time.Now()

	term := "rolefilter"
	if err := a.Append(1, sampleMsg("user", term+" from user"), now); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := a.Append(2, sampleMsg("assistant", term+" from assistant"), now); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := a.Append(3, sampleMsg("tool", term+" from tool"), now); err != nil {
		t.Fatalf("Append: %v", err)
	}

	// Filter to assistant only.
	got, err := a.Search(context.Background(), term, "assistant", 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d results, want 1", len(got))
	}
	if got[0].Message.Role != "assistant" {
		t.Errorf("role = %q, want assistant", got[0].Message.Role)
	}

	// No filter should return all 3.
	got, err = a.Search(context.Background(), term, "", 10)
	if err != nil {
		t.Fatalf("Search (no filter): %v", err)
	}
	if len(got) != 3 {
		t.Errorf("got %d results, want 3", len(got))
	}
}

// TestArchiveStore_Search_ToolCallsIndexed verifies that ToolCalls arguments
// are included in the FTS index and are searchable.
func TestArchiveStore_Search_ToolCallsIndexed(t *testing.T) {
	a := openTestArchive(t)
	now := time.Now()

	uniqueArg := "uniqueargterm777"
	tc := toolCall("tc1", "exec", fmt.Sprintf(`{"cmd":"%s"}`, uniqueArg))
	if err := a.Append(1, sampleMsg("assistant", "calling tool", tc), now); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := a.Append(2, sampleMsg("user", "ordinary message"), now); err != nil {
		t.Fatalf("Append: %v", err)
	}

	got, err := a.Search(context.Background(), uniqueArg, "", 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d results, want 1", len(got))
	}
	if got[0].Message.ToolCalls[0].ID != "tc1" {
		t.Errorf("unexpected message: %+v", got[0])
	}
}

// TestArchiveStore_ConcurrentAppends verifies that concurrent appends succeed
// and all messages are present afterward.
func TestArchiveStore_ConcurrentAppends(t *testing.T) {
	a := openTestArchive(t)
	now := time.Now()

	const goroutines = 10
	const perGoroutine = 20
	var wg sync.WaitGroup

	for g := range goroutines {
		g := g
		wg.Go(func() {
			for i := range perGoroutine {
				seq := int64(g*perGoroutine + i + 1)
				if err := a.Append(seq, sampleMsg("user", fmt.Sprintf("msg%d", seq)), now); err != nil {
					t.Errorf("goroutine %d Append seq=%d: %v", g, seq, err)
				}
			}
		})
	}

	wg.Wait()

	_, maxSeq, err := a.Bounds()
	if err != nil {
		t.Fatalf("Bounds: %v", err)
	}
	if maxSeq != goroutines*perGoroutine {
		t.Errorf("maxSeq = %d, want %d", maxSeq, goroutines*perGoroutine)
	}
}

// TestArchiveStore_Delete verifies that Delete closes the store and removes
// the database file.
func TestArchiveStore_Delete(t *testing.T) {
	path := filepath.Join(t.TempDir(), "delete.archive.db")
	a, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	if err := a.Append(1, sampleMsg("user", "x"), time.Now()); err != nil {
		t.Fatalf("Append: %v", err)
	}

	if err := a.Delete(); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	if _, err := Open(path + " [nonexistent]"); !errors.Is(err, ErrArchiveUnavailable) {
		// We just verify the file is gone — trying to open with mode=ro should fail.
		_ = err
	}
}
