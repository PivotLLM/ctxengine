// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"
)

// systemLayers is the one-layer system prompt these tests assemble around,
// which is all they need to see where blocks land.
var systemLayers = []Layer{{Name: "static", Text: "SYSTEM"}}

// memMgr builds a Manager over a store; assemble places the two memory blocks
// the way the agent loop does, stable in the system message and routed on the
// current turn.
type memMgr struct {
	*Manager
	stable, routed string
}

func newMemMgr(t *testing.T, store *mockStore, stable, routed string) memMgr {
	t.Helper()
	m := New("test-session", store, WithContextWindow(100_000)).(*Manager)
	return memMgr{Manager: m, stable: stable, routed: routed}
}

func (m memMgr) Build(ctx context.Context) ([]spawnllm.Message, error) {
	asm, err := m.Assemble(ctx, AssembleRequest{Layers: systemLayers, Injections: []Injection{
		{Placement: PlaceSystemStable, Text: m.stable},
		{Placement: PlaceCurrentUser, Text: m.routed},
	}})
	return asm.Messages, err
}

// TestRoutedMemory_RidesOnTheCurrentTurn is the placement this change exists
// for. ROUTED is selected per turn, so it must not sit in the system message
// (that precedes the entire history, and anything volatile there invalidates
// the cached prefix for all of it) and it must not be folded into a stored
// message (that rewrites the message on every dispatch). It is the trailing
// user message, after the whole history.
func TestRoutedMemory_RidesOnTheCurrentTurn(t *testing.T) {
	store := newMockStore()
	store.SetHistory("test-session", []spawnllm.Message{
		{Role: "user", Content: "older question"},
		{Role: "assistant", Content: "older answer"},
		{Role: "user", Content: "current question"},
	})

	msgs, err := newMemMgr(t, store, "STABLEBLOCK", "ROUTEDBLOCK").Build(context.Background())
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	if !strings.Contains(msgs[0].Content, "STABLEBLOCK") {
		t.Error("stable memory belongs in the system message (cached prefix)")
	}
	if strings.Contains(msgs[0].Content, "ROUTEDBLOCK") {
		t.Error("routed memory must NOT be in the system message — it breaks the cached prefix for the whole history")
	}

	last := msgs[len(msgs)-1]
	if last.Role != "user" || last.Content != "ROUTEDBLOCK" {
		t.Fatalf("expected the routed block as the trailing user message, got %+v", last)
	}
	// Every stored message is rendered exactly as stored.
	if got := msgs[len(msgs)-2]; got.Role != "user" || got.Content != "current question" {
		t.Errorf("the current user turn was altered: %+v", got)
	}
	for _, m := range msgs[1 : len(msgs)-1] {
		if strings.Contains(m.Content, "ROUTEDBLOCK") {
			t.Errorf("routed memory folded into a stored message: %+v", m)
		}
	}
}

// TestRoutedMemory_NeverPersisted is the load-bearing safety property: the
// injected block lives only in the built slice. If it reached the store, history
// would gain one stale memory dump per turn, silently and cumulatively.
func TestRoutedMemory_NeverPersisted(t *testing.T) {
	store := newMockStore()
	store.SetHistory("test-session", []spawnllm.Message{{Role: "user", Content: "question"}})

	mgr := newMemMgr(t, store, "STABLEBLOCK", "ROUTEDBLOCK")
	for i := 0; i < 3; i++ {
		if _, err := mgr.Build(context.Background()); err != nil {
			t.Fatalf("Build: %v", err)
		}
	}

	for _, sm := range store.GetHistoryWithSeqs("test-session") {
		if strings.Contains(sm.Content, "ROUTEDBLOCK") || strings.Contains(sm.Content, "STABLEBLOCK") {
			t.Fatalf("memory block leaked into stored history: %q", sm.Content)
		}
	}
	if got := len(store.GetHistory("test-session")); got != 1 {
		t.Fatalf("stored history grew to %d messages", got)
	}
}

// TestRoutedMemory_StableAcrossRepeatedBuilds guards the cache property: the
// same inputs must produce a byte-identical slice every time, or the prefix
// breaks on every dispatch of the turn.
func TestRoutedMemory_StableAcrossRepeatedBuilds(t *testing.T) {
	store := newMockStore()
	store.SetHistory("test-session", []spawnllm.Message{{Role: "user", Content: "question"}})
	mgr := newMemMgr(t, store, "STABLEBLOCK", "ROUTEDBLOCK")

	first, _ := mgr.Build(context.Background())
	second, _ := mgr.Build(context.Background())

	if len(first) != len(second) {
		t.Fatalf("builds differ in length: %d vs %d", len(first), len(second))
	}
	for i := range first {
		if first[i].Role != second[i].Role || first[i].Content != second[i].Content {
			t.Errorf("message %d differs between builds:\n%q\nvs\n%q", i, first[i].Content, second[i].Content)
		}
	}
	if got := strings.Count(second[len(second)-1].Content, "ROUTEDBLOCK"); got != 1 {
		t.Errorf("routed block appears %d times in the trailing message, want 1", got)
	}
}

// TestRoutedMemory_AfterToolPlumbing covers a turn that opens on tool
// plumbing: the block still goes at the end, as a user turn after the tool
// result, a shape every provider accepts.
func TestRoutedMemory_AfterToolPlumbing(t *testing.T) {
	store := newMockStore()
	store.SetHistory("test-session", []spawnllm.Message{
		{Role: "user", Content: "go"},
		{Role: "assistant", ToolCalls: []spawnllm.ToolCall{{ID: "t1"}}},
		{Role: "tool", ToolCallID: "t1", Content: "tool output"},
	})

	msgs, err := newMemMgr(t, store, "STABLEBLOCK", "ROUTEDBLOCK").Build(context.Background())
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	assertRoles(t, msgs, "system", "user", "assistant", "tool", "user")
	if msgs[4].Content != "ROUTEDBLOCK" || msgs[3].Content != "tool output" {
		t.Errorf("routed block not appended after the tool result: %+v", msgs[3:])
	}
}

// TestRoutedInjection_EmptyIsNoop keeps a non-cognitive agent's slice
// untouched: no PlaceCurrentUser injection, no trailing message.
func TestRoutedInjection_EmptyIsNoop(t *testing.T) {
	if got := routedInjection([]Injection{{Placement: PlaceCurrentUser, Text: ""}, {Placement: PlaceSystemStable, Text: "S"}}); got != "" {
		t.Errorf("routedInjection = %q, want empty", got)
	}
	store := newMockStore()
	store.SetHistory("test-session", []spawnllm.Message{{Role: "user", Content: "question"}})
	msgs, err := newMemMgr(t, store, "STABLEBLOCK", "").Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	assertRoles(t, msgs, "system", "user")
}
