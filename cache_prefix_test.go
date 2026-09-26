// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"
)

// assertSamePrefix checks that a and b are identical over the first n
// messages, role and content.
func assertSamePrefix(t *testing.T, a, b []spawnllm.Message, n int) {
	t.Helper()
	if len(a) < n || len(b) < n {
		t.Fatalf("slices shorter than the %d-message prefix: %d and %d", n, len(a), len(b))
	}
	for i := range n {
		if a[i].Role != b[i].Role || a[i].Content != b[i].Content {
			t.Fatalf("prefix differs at message %d:\n%s %q\nvs\n%s %q", i, a[i].Role, a[i].Content, b[i].Role, b[i].Content)
		}
	}
}

func prefixRequest(injection string) AssembleRequest {
	return AssembleRequest{
		Layers:     []Layer{{Text: "RULES"}},
		Injections: []Injection{{Placement: PlaceCurrentUser, Text: injection}},
	}
}

// TestAssemble_PrefixStableAcrossTurns: with no compaction between them, two
// Assemble calls on consecutive turns render every message that existed at
// the first call identically — the per-turn injection changes and moves, but
// it never rewrites an earlier message.
func TestAssemble_PrefixStableAcrossTurns(t *testing.T) {
	ctx := context.Background()
	store := newMockStore()
	mgr := asManager(t, New("s", store, WithContextWindow(1_000_000)))
	if _, err := mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: "first question"}); err != nil {
		t.Fatal(err)
	}
	first, err := mgr.Assemble(ctx, prefixRequest("MEMORY-A"))
	if err != nil {
		t.Fatal(err)
	}
	if first.Changed() {
		t.Fatalf("first Assemble rewrote history: %+v", first)
	}
	// [system, user, injection]
	if len(first.Messages) != 3 || first.Messages[2].Content != "MEMORY-A" {
		t.Fatalf("first build = %v", roles(first.Messages))
	}

	if _, err = mgr.AddAssistantMessage(ctx, spawnllm.Message{Role: "assistant", Content: "first answer"}); err != nil {
		t.Fatal(err)
	}
	if _, err = mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: "second question"}); err != nil {
		t.Fatal(err)
	}
	second, err := mgr.Assemble(ctx, prefixRequest("MEMORY-B"))
	if err != nil {
		t.Fatal(err)
	}
	if second.Changed() {
		t.Fatalf("second Assemble rewrote history: %+v", second)
	}

	// Everything before the first call's injection is unchanged in the second.
	assertSamePrefix(t, first.Messages, second.Messages, len(first.Messages)-1)
	if tail := second.Messages[len(second.Messages)-1]; tail.Content != "MEMORY-B" {
		t.Errorf("second build does not end with its own injection: %+v", tail)
	}
	for _, m := range second.Messages[:len(second.Messages)-1] {
		if strings.Contains(m.Content, "MEMORY-") {
			t.Errorf("an earlier message carries an injection: %+v", m)
		}
	}
}

// TestAssemble_PrefixStableWithinTurn: across the iterations of one tool-using
// turn the user message that opened the turn renders identically, whether or
// not the host's per-turn injection changed between iterations.
func TestAssemble_PrefixStableWithinTurn(t *testing.T) {
	ctx := context.Background()
	store := newMockStore()
	mgr := asManager(t, New("s", store, WithContextWindow(1_000_000)))
	if _, err := mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: "look it up"}); err != nil {
		t.Fatal(err)
	}
	first, err := mgr.Assemble(ctx, prefixRequest("MEMORY-A"))
	if err != nil {
		t.Fatal(err)
	}

	if _, err = mgr.AddToolCallMessage(ctx, spawnllm.Message{Role: "assistant", ToolCalls: []spawnllm.ToolCall{{ID: "c1"}}}); err != nil {
		t.Fatal(err)
	}
	if _, err = mgr.AddToolResult(ctx, spawnllm.Message{Role: "tool", ToolCallID: "c1", Content: "42"}); err != nil {
		t.Fatal(err)
	}
	second, err := mgr.Assemble(ctx, prefixRequest("MEMORY-A-REVISED"))
	if err != nil {
		t.Fatal(err)
	}

	assertSamePrefix(t, first.Messages, second.Messages, len(first.Messages)-1)
	assertRoles(t, second.Messages, "system", "user", "assistant", "tool", "user")
	if second.Messages[1].Content != "look it up" {
		t.Errorf("the turn's user message was rewritten: %q", second.Messages[1].Content)
	}
}
