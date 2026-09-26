// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/PivotLLM/ctxengine/memory"
	"github.com/PivotLLM/spawnllm"
)

// argPolicy is basePolicy with argument eviction switched on.
func argPolicy() EvictionPolicy {
	p := basePolicy()
	p.ArgBytes = 1024
	return p
}

// storedArgs returns the persisted (Function-form) arguments of the call with
// the given id, decoded. Reading the persisted form rather than the runtime map
// is the point: that is what a provider replays.
func storedArgs(t *testing.T, store *seqStore, id string) map[string]any {
	t.Helper()
	for _, sm := range store.GetHistoryWithSeqs("sess") {
		for _, tc := range sm.ToolCalls {
			if tc.ID != id {
				continue
			}
			if tc.Function == nil {
				t.Fatalf("call %s has no persisted Function form", id)
			}
			var out map[string]any
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &out); err != nil {
				t.Fatalf("call %s arguments are not valid JSON after eviction: %v\nraw: %s",
					id, err, tc.Function.Arguments)
			}
			return out
		}
	}
	t.Fatalf("call %s not found", id)
	return nil
}

// bigWriteHistory builds a history whose first turn is a large file_write,
// followed by enough text turns to age it past EvictTurns.
func bigWriteHistory(payload string, filler int) []memory.StoredMessage {
	specs := []turnSpec{{
		tool:    "file_write",
		id:      "w1",
		args:    map[string]any{"path": "novels/ch17.md", "content": payload},
		content: "wrote 47185 bytes",
	}}
	for i := 0; i < filler; i++ {
		specs = append(specs, turnSpec{text: string(rune('a' + i))})
	}
	return buildHistory(specs...)
}

// TestSweepArgs_EvictsAgedWritePayload is the core gap this change closes: a
// file_write body lives in the assistant message's ToolCalls, is counted in full
// by the token estimator, and had no eviction path at all. In production this
// was 45% of one agent's live window.
func TestSweepArgs_EvictsAgedWritePayload(t *testing.T) {
	payload := strings.Repeat("p", 40_000)
	store := newSeqStore(bigWriteHistory(payload, 12))

	events := newEvictMgr(store, argPolicy()).SweepEvictions(context.Background())

	var found *EvictionEvent
	for i := range events {
		if events[i].Reason == "args" {
			found = &events[i]
		}
	}
	if found == nil {
		t.Fatalf("expected an args eviction; got %d events: %+v", len(events), events)
	}
	if found.Bytes != len(payload) {
		t.Errorf("event Bytes = %d, want %d", found.Bytes, len(payload))
	}
	if found.Tool != "file_write" || found.Resource != "novels/ch17.md" {
		t.Errorf("event should name the tool and the file it wrote: %+v", found)
	}

	args := storedArgs(t, store, "w1")
	content, _ := args["content"].(string)
	if !isEvicted(content) {
		t.Errorf("payload not replaced by a placeholder: %.80q", content)
	}
	if len(content) >= len(payload) {
		t.Errorf("placeholder (%d bytes) did not shrink the payload (%d bytes)", len(content), len(payload))
	}
	if got := args["path"]; got != "novels/ch17.md" {
		t.Errorf("the identifying path argument must survive; got %v", got)
	}
}

// TestSweepArgs_ProtectsRecentWrite is the off path for age: a payload the model
// may still be working against is left alone.
func TestSweepArgs_ProtectsRecentWrite(t *testing.T) {
	payload := strings.Repeat("p", 40_000)
	store := newSeqStore(bigWriteHistory(payload, 2)) // well inside EvictTurns

	newEvictMgr(store, argPolicy()).SweepEvictions(context.Background())

	if content, _ := storedArgs(t, store, "w1")["content"].(string); isEvicted(content) {
		t.Error("a recent write payload must not be evicted")
	}
}

// TestSweepArgs_LeavesSmallArgsAlone verifies the size rule is what selects
// payloads: paths, queries and flags fall far below the threshold, which is why
// the placeholder can still name the resource.
func TestSweepArgs_LeavesSmallArgsAlone(t *testing.T) {
	store := newSeqStore(bigWriteHistory(strings.Repeat("p", 100), 12))

	newEvictMgr(store, argPolicy()).SweepEvictions(context.Background())

	args := storedArgs(t, store, "w1")
	if content, _ := args["content"].(string); isEvicted(content) {
		t.Error("a 100-byte argument is not a payload and must not be evicted")
	}
}

// TestSweepArgs_Disabled covers ArgBytes=0 as the off switch.
func TestSweepArgs_Disabled(t *testing.T) {
	p := argPolicy()
	p.ArgBytes = 0
	store := newSeqStore(bigWriteHistory(strings.Repeat("p", 40_000), 12))

	newEvictMgr(store, p).SweepEvictions(context.Background())

	if content, _ := storedArgs(t, store, "w1")["content"].(string); isEvicted(content) {
		t.Error("ArgBytes=0 must disable argument eviction")
	}
}

// TestSweepArgs_Idempotent guards the placeholder marker: a second sweep must
// not re-evict an already-evicted argument, which would emit an event every
// turn forever and shrink the placeholder into nonsense.
func TestSweepArgs_Idempotent(t *testing.T) {
	store := newSeqStore(bigWriteHistory(strings.Repeat("p", 40_000), 12))
	mgr := newEvictMgr(store, argPolicy())

	first := mgr.SweepEvictions(context.Background())
	firstAfter := storedArgs(t, store, "w1")["content"]

	second := mgr.SweepEvictions(context.Background())
	for _, e := range second {
		if e.Reason == "args" {
			t.Errorf("second sweep re-evicted an argument: %+v", e)
		}
	}
	if got := storedArgs(t, store, "w1")["content"]; got != firstAfter {
		t.Errorf("second sweep rewrote the placeholder:\n first: %v\nsecond: %v", firstAfter, got)
	}
	if len(first) == 0 {
		t.Error("expected the first sweep to evict something")
	}
}

// TestSweepArgs_MCPToolCovered is why the rule is size-based rather than a map
// of native writer tools: an MCP document tool's body costs exactly as much as
// file_write's, and a name registry would silently miss it.
func TestSweepArgs_MCPToolCovered(t *testing.T) {
	specs := []turnSpec{{
		tool:    "mcp__claw__simpledoc_item_add",
		id:      "m1",
		args:    map[string]any{"section_id": "s1", "text": strings.Repeat("d", 20_000)},
		content: "added",
	}}
	for i := 0; i < 12; i++ {
		specs = append(specs, turnSpec{text: string(rune('a' + i))})
	}
	store := newSeqStore(buildHistory(specs...))

	events := newEvictMgr(store, argPolicy()).SweepEvictions(context.Background())

	var sawArgs bool
	for _, e := range events {
		if e.Reason == "args" {
			sawArgs = true
			if e.Tool != "simpledoc_item_add" {
				t.Errorf("tool name should be normalized past the mcp__ prefix, got %q", e.Tool)
			}
		}
	}
	if !sawArgs {
		t.Fatalf("an MCP tool's oversized argument must be evicted; got %+v", events)
	}
	if text, _ := storedArgs(t, store, "m1")["text"].(string); !isEvicted(text) {
		t.Error("MCP payload not evicted")
	}
	if got := storedArgs(t, store, "m1")["section_id"]; got != "s1" {
		t.Errorf("small identifying argument must survive; got %v", got)
	}
}

// TestEvictLargeArgs_KeepsRuntimeMapInStep covers the dual representation: the
// runtime Arguments map is json:"-" but normToolCall prefers it, so leaving it
// stale would make every subsequent sweep see the original size.
func TestEvictLargeArgs_KeepsRuntimeMapInStep(t *testing.T) {
	payload := strings.Repeat("p", 4_000)
	tc := spawnllm.ToolCall{
		ID:        "w1",
		Name:      "file_write",
		Arguments: map[string]any{"path": "a.md", "content": payload},
		Function:  &spawnllm.FunctionCall{Name: "file_write", Arguments: `{"path":"a.md","content":"` + payload + `"}`},
	}

	if got := evictLargeArgs(&tc, 1024, DefaultEvictionRoles()); len(got) != 1 {
		t.Fatalf("expected 1 argument evicted, got %d", len(got))
	}
	runtime, _ := tc.Arguments["content"].(string)
	if !isEvicted(runtime) {
		t.Errorf("runtime Arguments map left stale: %.60q", runtime)
	}
	var persisted map[string]any
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &persisted); err != nil {
		t.Fatalf("persisted arguments are not valid JSON: %v", err)
	}
	if persisted["content"] != runtime {
		t.Error("runtime map and persisted form disagree after eviction")
	}
}

// TestEvictLargeArgs_NoFunctionForm is a defensive path: a call with no
// persisted Function form has nothing to rewrite.
func TestEvictLargeArgs_NoFunctionForm(t *testing.T) {
	tc := spawnllm.ToolCall{
		ID: "x", Name: "file_write",
		Arguments: map[string]any{"content": strings.Repeat("p", 4_000)},
	}
	if got := evictLargeArgs(&tc, 1024, DefaultEvictionRoles()); got != nil {
		t.Errorf("expected no evictions without a Function form, got %+v", got)
	}
}
