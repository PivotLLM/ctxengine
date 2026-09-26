// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// TestEvictionPlaceholder_NamesSeqAndTool: the placeholder carries the archive
// seq and the session messages tool, so the model can pull the exact content
// back rather than only re-run the tool.
func TestEvictionPlaceholder_NamesSeqAndTool(t *testing.T) {
	got := evictionPlaceholder("file_read_bytes", "notes/ch1.md", 5000, 123)
	for _, want := range []string{"seq #123", sessionMessagesTool, "file_read_bytes", "notes/ch1.md", "5000 bytes"} {
		if !strings.Contains(got, want) {
			t.Errorf("placeholder %q lacks %q", got, want)
		}
	}
	if !isEvicted(got) {
		t.Error("placeholder must keep the eviction marker so the sweep stays idempotent")
	}
}

// TestSweep_PlaceholderCarriesStoredSeq: through the sweep the placeholder
// names the seq of the row it replaced.
func TestSweep_PlaceholderCarriesStoredSeq(t *testing.T) {
	store := newSeqStore(staleReads(1, 500))
	mgr := newEvictMgr(store, basePolicy())
	events := mgr.SweepEvictions(context.Background())
	if len(events) != 1 {
		t.Fatalf("evictions = %d, want 1", len(events))
	}
	if got := findToolResult(store, "ra"); !strings.Contains(got, fmt.Sprintf("seq #%d", events[0].Seq)) {
		t.Errorf("placeholder %q does not name seq #%d", got, events[0].Seq)
	}
}

// TestEvictionRoles_HostDeclared: a host declares its own reader and writer
// tools; the sweep evicts the declared reader's stale result, and a declared
// writer supersedes an earlier read of the same resource.
func TestEvictionRoles_HostDeclared(t *testing.T) {
	roles := EvictionRoles{
		Readers: map[string]ReaderRole{"doc_read": {ResourceArg: "doc_id"}},
		Writers: map[string]WriterRole{"doc_update": {ResourceArg: "doc_id"}},
	}
	p := basePolicy()

	// Stale: a doc_read older than EvictTurns goes; file_read_bytes, no longer
	// declared, stays whatever its age.
	store := newSeqStore(buildHistory(
		turnSpec{tool: "doc_read", id: "d", args: map[string]any{"doc_id": "D1"}, content: strings.Repeat("x", 500)},
		turnSpec{tool: "file_read_bytes", id: "f", args: map[string]any{"path": "a.md"}, content: strings.Repeat("y", 500)},
		turnSpec{text: "1"}, turnSpec{text: "2"}, turnSpec{text: "3"}, turnSpec{text: "4"},
		turnSpec{text: "5"}, turnSpec{text: "6"}, turnSpec{text: "7"}, turnSpec{text: "8"},
		turnSpec{text: "9"}, turnSpec{text: "10"},
	))
	mgr := New("sess", store, WithContextWindow(0), WithEvictionPolicy(p), WithEvictionRoles(roles)).(*Manager)
	events := mgr.SweepEvictions(context.Background())
	if len(events) != 1 || events[0].Tool != "doc_read" || events[0].Resource != "D1" {
		t.Fatalf("events = %+v, want one doc_read eviction of D1", events)
	}
	if isEvicted(findToolResult(store, "f")) {
		t.Error("file_read_bytes was evicted although the host did not declare it a reader")
	}

	// Superseded: a later doc_update of D1 evicts the earlier doc_read past the
	// protect window, even though the read itself is not yet stale.
	store = newSeqStore(buildHistory(
		turnSpec{tool: "doc_read", id: "d", args: map[string]any{"doc_id": "D1"}, content: strings.Repeat("x", 500)},
		turnSpec{tool: "doc_update", id: "w", args: map[string]any{"doc_id": "D1", "body": "new"}, content: "ok"},
		turnSpec{text: "1"}, turnSpec{text: "2"}, turnSpec{text: "3"},
	))
	mgr = New("sess", store, WithContextWindow(0), WithEvictionPolicy(p), WithEvictionRoles(roles)).(*Manager)
	events = mgr.SweepEvictions(context.Background())
	if len(events) != 1 || events[0].Reason != "superseded" {
		t.Fatalf("events = %+v, want the read superseded by the declared writer", events)
	}
}

// TestEvictionRoles_DefaultsUnchanged pins the built-in roles the sweep used
// before hosts could declare their own, so behaviour without WithEvictionRoles
// is what it was.
func TestEvictionRoles_DefaultsUnchanged(t *testing.T) {
	d := DefaultEvictionRoles()
	readers := []string{"file_read_bytes", "file_read_lines", "file_list", "file_search_lines", "file_search_bytes", "web_fetch"}
	for _, name := range readers {
		if _, ok := d.Readers[name]; !ok {
			t.Errorf("default readers lack %s", name)
		}
	}
	if len(d.Readers) != len(readers) {
		t.Errorf("default readers = %d, want %d", len(d.Readers), len(readers))
	}
	if d.Readers["file_read_lines"].RangeArg != "start_line" || d.Readers["file_read_lines"].RangeDefault != 1 {
		t.Errorf("file_read_lines paging role = %+v", d.Readers["file_read_lines"])
	}
	if d.Readers["file_search_lines"].KeyArg != "query" || d.Readers["file_search_lines"].ResourceDefault != "." {
		t.Errorf("file_search_lines role = %+v", d.Readers["file_search_lines"])
	}
	if d.Writers["file_move"].ResourceArg != "source_path" || d.Writers["file_copy"].ResourceArg != "destination_path" || len(d.Writers) != 12 {
		t.Errorf("default writers = %+v", d.Writers)
	}
	// Resolution ignores an MCP namespace prefix.
	if res, ok := d.readerResource(normToolName("mcp__claw__web_fetch"), map[string]any{"url": "http://x"}); !ok || res != "http://x" {
		t.Errorf("namespaced reader not resolved: %q %v", res, ok)
	}
}
