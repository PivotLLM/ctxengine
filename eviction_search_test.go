/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

import (
	"context"
	"strings"
	"testing"
)

// searchTurns builds a history of search turns followed by enough filler to age
// them past EvictTurns.
func searchTurns(specs []turnSpec, filler int) []turnSpec {
	for i := range filler {
		specs = append(specs, turnSpec{text: string(rune('a' + i))})
	}
	return specs
}

// TestSweep_SearchResultsEvictable closes a coverage gap: the search tools are
// as re-retrievable as the read tools (re-run the search) and their results are
// often larger, but they were absent from the reader map.
func TestSweep_SearchResultsEvictable(t *testing.T) {
	store := newSeqStore(buildHistory(searchTurns([]turnSpec{{
		tool: "file_search_lines", id: "s1",
		args:    map[string]any{"query": "chapter", "path": "novels/"},
		content: strings.Repeat("m", 5_000),
	}}, 12)...))

	events := newEvictMgr(t, store, basePolicy()).SweepEvictions(context.Background())

	if len(events) != 1 || events[0].Tool != "file_search_lines" {
		t.Fatalf("expected the aged search result to be evicted, got %+v", events)
	}
	if !isEvicted(findToolResult(store, "s1")) {
		t.Error("search result content not replaced by a placeholder")
	}
}

// TestSweep_SearchWithoutPathEvictable covers the implied resource: path is
// optional and defaults to the workspace root, so a repository-wide search —
// typically the largest result — would otherwise be the one never evicted.
func TestSweep_SearchWithoutPathEvictable(t *testing.T) {
	store := newSeqStore(buildHistory(searchTurns([]turnSpec{{
		tool: "file_search_lines", id: "s1",
		args:    map[string]any{"query": "chapter"},
		content: strings.Repeat("m", 5_000),
	}}, 12)...))

	events := newEvictMgr(t, store, basePolicy()).SweepEvictions(context.Background())
	if len(events) != 1 {
		t.Fatalf("a search with no explicit path must still be evictable, got %+v", events)
	}
}

// TestSweep_DifferentQueriesDoNotSupersede is the string analogue of the
// paginated-read rule: two searches of one tree return different content, so
// keying on the bare path would let the second silently evict the first while
// the model still needs it.
func TestSweep_DifferentQueriesDoNotSupersede(t *testing.T) {
	p := basePolicy()
	p.ProtectTurns = 0 // remove the protect window so only supersession matters
	store := newSeqStore(buildHistory(
		turnSpec{
			tool: "file_search_lines", id: "q1",
			args: map[string]any{"query": "alice", "path": "novels/"}, content: strings.Repeat("a", 500),
		},
		turnSpec{
			tool: "file_search_lines", id: "q2",
			args: map[string]any{"query": "bob", "path": "novels/"}, content: strings.Repeat("b", 500),
		},
	))

	events := newEvictMgr(t, store, p).SweepEvictions(context.Background())

	for _, e := range events {
		if e.Reason == "superseded" {
			t.Errorf("a different query must not supersede an earlier search: %+v", e)
		}
	}
	if isEvicted(findToolResult(store, "q1")) {
		t.Error("the earlier search for a different query was evicted")
	}
}

// TestSweep_SameQuerySupersedes is the matching on path: re-running the same
// search does make the earlier copy a stale duplicate.
func TestSweep_SameQuerySupersedes(t *testing.T) {
	p := basePolicy()
	p.ProtectTurns = 0
	store := newSeqStore(buildHistory(
		turnSpec{
			tool: "file_search_lines", id: "q1",
			args: map[string]any{"query": "alice", "path": "novels/"}, content: strings.Repeat("a", 500),
		},
		turnSpec{
			tool: "file_search_lines", id: "q2",
			args: map[string]any{"query": "alice", "path": "novels/"}, content: strings.Repeat("a", 500),
		},
	))

	newEvictMgr(t, store, p).SweepEvictions(context.Background())

	if !isEvicted(findToolResult(store, "q1")) {
		t.Error("re-running the same search should supersede the earlier result")
	}
	if isEvicted(findToolResult(store, "q2")) {
		t.Error("the newest result must survive")
	}
}

// TestSweep_WriteInvalidatesSearch verifies write-supersession still keys on the
// bare path, so editing a file invalidates searches over it.
func TestSweep_WriteInvalidatesSearch(t *testing.T) {
	p := basePolicy()
	p.ProtectTurns = 0
	store := newSeqStore(buildHistory(
		turnSpec{
			tool: "file_search_lines", id: "s1",
			args: map[string]any{"query": "alice", "path": "novels/ch1.md"}, content: strings.Repeat("a", 500),
		},
		turnSpec{
			tool: "file_write", id: "w1",
			args: map[string]any{"path": "novels/ch1.md", "content": "new"}, content: "ok",
		},
	))

	newEvictMgr(t, store, p).SweepEvictions(context.Background())

	if !isEvicted(findToolResult(store, "s1")) {
		t.Error("a successful write must invalidate an earlier search of that file")
	}
}
