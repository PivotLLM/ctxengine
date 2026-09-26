// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"
)

// TestAssemble_NoChangeLeavesFlagsClear: a quiet assembly reports nothing
// changed, so a caller holding an earlier slice keeps it.
func TestAssemble_NoChangeLeavesFlagsClear(t *testing.T) {
	store := newMockStore()
	store.SetHistory("test-session", []spawnllm.Message{{Role: "user", Content: "hi"}})
	m := New("test-session", store, WithContextWindow(100_000), WithOverheadTokens(0)).(*Manager)

	asm, err := m.Assemble(context.Background(), AssembleRequest{ToolDefinitionTokens: 123, Layers: systemLayers})
	if err != nil {
		t.Fatal(err)
	}
	if asm.Changed() || asm.Compacted || len(asm.Evictions) != 0 {
		t.Fatalf("quiet assembly reported a change: %+v", asm)
	}
	if len(asm.Messages) != 2 || asm.Messages[0].Role != "system" {
		t.Fatalf("messages = %+v", asm.Messages)
	}
	if m.toolDefTokens != 123 {
		t.Fatalf("tool definition tokens not recorded: %d", m.toolDefTokens)
	}
}

// TestAssemble_HistoryPastSafetyCompacts: stored history over the safety line
// triggers the pre-build safety-net pass once, and the post-build check does
// not fire a second pass in the same call.
func TestAssemble_HistoryPastSafetyCompacts(t *testing.T) {
	store := newMockStore()
	// 10000-token window, safety at 80% → 8000 tokens; 40000 chars ≈ 10000 tokens.
	store.SetHistory("test-session", []spawnllm.Message{{Role: "user", Content: strings.Repeat("a", 40_000)}})
	m := New("test-session", store,
		WithContextWindow(10_000), WithSafetyPercent(80), WithOverheadTokens(0)).(*Manager)
	fired := 0
	m.SetTestCompressHook(func(safetyNet bool) {
		fired++
		if !safetyNet {
			t.Error("Assemble must only run the safety-net pass")
		}
	})

	asm, err := m.Assemble(context.Background(), AssembleRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if !asm.Compacted || !asm.Changed() {
		t.Fatalf("expected a compaction to be reported: %+v", asm)
	}
	if fired != 1 {
		t.Fatalf("compress fired %d times, want exactly 1 (post-build check skipped after pre-build pass)", fired)
	}
}

// TestAssemble_BuiltRequestPastSafetyCompacts: history alone is under the line
// but the built request (tool schemas + reserve) is not — the post-build
// check catches it.
func TestAssemble_BuiltRequestPastSafetyCompacts(t *testing.T) {
	store := newMockStore()
	// ≈2500 tokens of history in a 10000 window: well under 80% on its own.
	store.SetHistory("test-session", []spawnllm.Message{{Role: "user", Content: strings.Repeat("a", 10_000)}})
	m := New("test-session", store,
		WithContextWindow(10_000), WithSafetyPercent(80), WithOverheadTokens(0)).(*Manager)
	fired := 0
	m.SetTestCompressHook(func(bool) { fired++ })

	// 6000 tokens of tool schemas push the request to ≈8500 > 8000.
	asm, err := m.Assemble(context.Background(), AssembleRequest{ToolDefinitionTokens: 6_000})
	if err != nil {
		t.Fatal(err)
	}
	if !asm.Compacted || fired != 1 {
		t.Fatalf("post-build check did not compact: compacted=%v fired=%d", asm.Compacted, fired)
	}
}

// TestAssemble_InjectionsPlacedAndNeverPersisted: stable rides in the system
// message, per-turn as the trailing user message after the stored history,
// and neither reaches the store.
func TestAssemble_InjectionsPlacedAndNeverPersisted(t *testing.T) {
	store := newMockStore()
	store.SetHistory("test-session", []spawnllm.Message{
		{Role: "user", Content: "older"},
		{Role: "assistant", Content: "reply"},
		{Role: "user", Content: "current"},
	})
	m := New("test-session", store, WithContextWindow(100_000)).(*Manager)
	asm, err := m.Assemble(context.Background(), AssembleRequest{Injections: []Injection{
		{Placement: PlaceSystemStable, Text: "STABLE"},
		{Placement: PlaceCurrentUser, Text: "ROUTED"},
		{Placement: PlaceCurrentUser, Text: ""}, // empty is skipped, not rendered
	}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(asm.Messages[0].Content, "STABLE") || strings.Contains(asm.Messages[0].Content, "ROUTED") {
		t.Fatalf("system message = %q", asm.Messages[0].Content)
	}
	last := asm.Messages[len(asm.Messages)-1]
	if last.Role != "user" || last.Content != "ROUTED" {
		t.Fatalf("last message = %+v, want the routed block alone", last)
	}
	if current := asm.Messages[len(asm.Messages)-2]; current.Content != "current" {
		t.Fatalf("the stored user turn was altered: %+v", current)
	}
	for _, h := range store.GetHistory("test-session") {
		if strings.Contains(h.Content, "STABLE") || strings.Contains(h.Content, "ROUTED") {
			t.Fatalf("injection leaked into stored history: %q", h.Content)
		}
	}
}

// TestAdd_ReturnsTranscriptSeq: every Add hands back the seq the store
// assigned, in order, which is what memory evidence and session tools cite.
func TestAdd_ReturnsTranscriptSeq(t *testing.T) {
	store := newMockStore()
	m := New("test-session", store, WithContextWindow(100_000)).(*Manager)
	ctx := context.Background()
	s1, _ := m.AddUserMessage(ctx, msgWithContent("a"))
	s2, _ := m.AddToolCallMessage(ctx, spawnllm.Message{Role: "assistant"})
	s3, _ := m.AddToolResult(ctx, spawnllm.Message{Role: "tool"})
	s4, _ := m.AddAssistantMessage(ctx, spawnllm.Message{Role: "assistant", Content: "b"})
	if !(s1 < s2 && s2 < s3 && s3 < s4) {
		t.Fatalf("seqs not increasing: %d %d %d %d", s1, s2, s3, s4)
	}
}
