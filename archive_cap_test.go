// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine/memory"
	"github.com/PivotLLM/ctxengine/session"
)

// TestArchive_StoresLargeToolResultWhole drives a 100 KB tool result through
// the manager into the real archive and reads it back whole: the archive is
// where an evicted result's exact content comes from, so it must hold it.
func TestArchive_StoresLargeToolResultWhole(t *testing.T) {
	dir := t.TempDir()
	store, err := session.NewSQLiteStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { noErr(t, store.Close()) }()
	ctx := context.Background()
	mgr := asManager(t, New("sess", store, WithArchiveDir(dir), WithContextWindow(10_000_000), WithMessageThreshold(0)))
	defer func() { noErr(t, mgr.Close(ctx)) }()

	content := strings.Repeat("chapter text ", 8000) // ~104 KB
	if _, err := mgr.AddToolCallMessage(ctx, spawnllm.Message{Role: "assistant", ToolCalls: []spawnllm.ToolCall{
		{ID: "r1", Function: &spawnllm.FunctionCall{Name: "file_read_bytes", Arguments: `{"path":"book.md"}`}},
	}}); err != nil {
		t.Fatal(err)
	}
	seq, err := mgr.AddToolResult(ctx, spawnllm.Message{Role: "tool", ToolCallID: "r1", Content: content})
	if err != nil {
		t.Fatal(err)
	}

	a, err := memory.OpenReadOnly(memory.ArchivePath(dir, "sess"))
	if err != nil {
		t.Fatal(err)
	}
	rows, err := a.QueryRange(seq, seq)
	if err != nil || len(rows) != 1 {
		t.Fatalf("QueryRange: rows=%d err=%v", len(rows), err)
	}
	if rows[0].Content != content {
		t.Fatalf("archived tool result is %d bytes, want the original %d", len(rows[0].Content), len(content))
	}
}
