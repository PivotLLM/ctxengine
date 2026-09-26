// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"slices"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"
)

// newToolTurnsManager builds a Manager suitable for tool-turn tests.
// It uses the mockStore from trigger_test.go and the mockLLM + helpers from
// compress_test.go (all in the same package).
func newToolTurnsManager(t *testing.T, store *mockStore, clients []*mockLLM, opts ...Option) *Manager {
	t.Helper()
	baseOpts := slices.Concat([]Option{
		WithContextWindow(10000),
		// Tests below reason in exact token terms against a small window; the
		// real per-request reserve would swamp it. TestTriggers_CountReserve
		// covers the reserve itself.
		WithOverheadTokens(0),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithRetainTokenPercent(20),
		WithRetainMinMessages(2),
		WithModelCaller(chainOf(clients)),
	}, opts)
	cm := New("sess", store, baseOpts...)
	return asManager(t, cm)
}

// TestAddToolCallMessage_IncrementsMsgCount verifies that AddToolCallMessage
// writes the message to the store and increments msgCount by 1.
func TestAddToolCallMessage_IncrementsMsgCount(t *testing.T) {
	store := newMockStore()
	mgr := newToolTurnsManager(t, store, nil)

	before := mgr.msgCount
	msg := spawnllm.Message{
		Role:    "assistant",
		Content: "calling a tool",
		ToolCalls: []spawnllm.ToolCall{
			{ID: "tc1", Name: "test_tool"},
		},
	}

	if _, err := mgr.AddToolCallMessage(context.Background(), msg); err != nil {
		t.Fatalf("AddToolCallMessage returned error: %v", err)
	}

	if mgr.msgCount != before+1 {
		t.Errorf("msgCount: got %d, want %d", mgr.msgCount, before+1)
	}

	history := store.GetHistory("sess")
	if len(history) != 1 {
		t.Fatalf("expected 1 message in store, got %d", len(history))
	}
	if len(history[0].ToolCalls) != 1 {
		t.Error("expected ToolCalls to be preserved in stored message")
	}
}

// TestAddToolResult_IncrementsMsgCount verifies that AddToolResult writes the
// message to the store and increments msgCount by 1.
func TestAddToolResult_IncrementsMsgCount(t *testing.T) {
	store := newMockStore()
	mgr := newToolTurnsManager(t, store, nil)

	before := mgr.msgCount
	msg := spawnllm.Message{
		Role:       "tool",
		Content:    "tool output",
		ToolCallID: "tc1",
	}

	if _, err := mgr.AddToolResult(context.Background(), msg); err != nil {
		t.Fatalf("AddToolResult returned error: %v", err)
	}

	if mgr.msgCount != before+1 {
		t.Errorf("msgCount: got %d, want %d", mgr.msgCount, before+1)
	}

	history := store.GetHistory("sess")
	if len(history) != 1 {
		t.Fatalf("expected 1 message in store, got %d", len(history))
	}
	if history[0].ToolCallID != "tc1" {
		t.Error("expected ToolCallID to be preserved in stored message")
	}
}

// TestToolTurnPair_MsgCountIncrementsByTwo verifies that a complete tool turn
// (AddToolCallMessage + AddToolResult) increments msgCount by exactly 2.
func TestToolTurnPair_MsgCountIncrementsByTwo(t *testing.T) {
	store := newMockStore()
	mgr := newToolTurnsManager(t, store, nil)

	before := mgr.msgCount

	toolCallMsg := spawnllm.Message{
		Role:    "assistant",
		Content: "",
		ToolCalls: []spawnllm.ToolCall{
			{ID: "tc1", Name: "test_tool"},
		},
	}
	toolResultMsg := spawnllm.Message{
		Role:       "tool",
		Content:    "result content",
		ToolCallID: "tc1",
	}

	if _, err := mgr.AddToolCallMessage(context.Background(), toolCallMsg); err != nil {
		t.Fatalf("AddToolCallMessage: %v", err)
	}
	if _, err := mgr.AddToolResult(context.Background(), toolResultMsg); err != nil {
		t.Fatalf("AddToolResult: %v", err)
	}

	if mgr.msgCount != before+2 {
		t.Errorf("msgCount: got %d, want %d", mgr.msgCount, before+2)
	}
}

// TestPreDispatchCheck_BelowThreshold verifies that PreDispatchCheck returns the
// input slice unchanged and fires no compression when history is below all thresholds.
func TestPreDispatchCheck_BelowThreshold(t *testing.T) {
	store := newMockStore()
	mgr := newToolTurnsManager(t, store, nil)

	// Seed history well below any threshold (~400 chars → ~100 tokens, 1% of 10000).
	small := []spawnllm.Message{
		{Role: "user", Content: strings.Repeat("u", 200)},
		{Role: "assistant", Content: strings.Repeat("a", 200)},
	}
	noErr(t, store.SetHistory("sess", small))
	mgr.msgCount = len(small)

	compressed := false
	mgr.compressHook = func(_ bool) { compressed = true }

	input := []spawnllm.Message{
		{Role: "user", Content: "test"},
	}
	got, err := mgr.PreDispatchCheck(context.Background(), input)
	if err != nil {
		t.Fatalf("PreDispatchCheck returned error: %v", err)
	}
	if &got[0] != &input[0] {
		// Verify the returned slice is the same slice, not a rebuilt copy.
		// Since both are small, compare length and content.
		if len(got) != len(input) || got[0].Content != input[0].Content {
			t.Error("expected input slice to be returned unchanged")
		}
	}
	if compressed {
		t.Error("expected no compression to fire")
	}
}

// TestPreDispatchCheck_TriggersCompressionAndRebuilds verifies that when history
// exceeds the normal threshold, PreDispatchCheck fires compression and returns a
// freshly built slice (via Build), not the stale input.
func TestPreDispatchCheck_TriggersCompressionAndRebuilds(t *testing.T) {
	store := newMockStore()

	// Seed history above the 50% normal threshold:
	// 10 pairs × 2 msgs × 300 chars = 6000 chars → ~1500 tokens = 15% of 10000.
	// Use larger content: 10 pairs × 2 msgs × 2000 chars = 40000 chars → ~10000 tokens = 100% of 10000.
	history := makeConversation(10, 2000)
	noErr(t, store.SetHistory("sess", history))

	llm := &mockLLM{
		responses: []string{validSummaryJSON("compressed goals")},
	}
	mgr := newToolTurnsManager(t, store, []*mockLLM{llm})
	mgr.msgCount = len(history)

	// Use a larger input slice to contrast with the rebuilt (compressed) slice.
	input := make([]spawnllm.Message, len(history))
	copy(input, history)

	result, err := mgr.PreDispatchCheck(context.Background(), input)
	if err != nil {
		t.Fatalf("PreDispatchCheck returned error: %v", err)
	}

	// After compression the store should have fewer messages; Build() returns them.
	if len(result) >= len(input) {
		t.Errorf("expected rebuilt slice to be shorter than input (%d), got %d", len(input), len(result))
	}
}

// TestPreDispatchCheck_NormalBand_DoesNotFire verifies that PreDispatchCheck is
// emergency-only: in the normal band (normalPercent <= context < safetyPercent)
// it does NOT compress mid-turn. First-stage compaction is deferred to the turn
// boundary (triggerCheck). The input slice is returned unchanged with no error.
func TestPreDispatchCheck_NormalBand_DoesNotFire(t *testing.T) {
	store := newMockStore()

	// History at ~62.5% of context window: between normalPercent (50%) and
	// safetyPercent (80%). Pre-change this fired the normal path; now it must not.
	// 10 pairs × 2 msgs × 1250 chars = 25000 chars → ~6250 tokens = 62.5% of 10000.
	history := makeConversation(10, 1250)
	noErr(t, store.SetHistory("sess", history))

	compressed := false
	mgr := newToolTurnsManager(t, store, nil)
	mgr.compressHook = func(_ bool) { compressed = true }
	mgr.msgCount = len(history)

	input := make([]spawnllm.Message, len(history))
	copy(input, history)

	result, err := mgr.PreDispatchCheck(context.Background(), input)
	if err != nil {
		t.Fatalf("expected no error in normal band, got: %v", err)
	}
	if compressed {
		t.Error("expected no mid-turn compression in the normal band (deferred to turn boundary)")
	}
	if len(result) != len(input) {
		t.Errorf("expected input slice unchanged (len %d), got len %d", len(input), len(result))
	}
}

// TestPreDispatchCheck_CountTrigger_DoesNotFireMidTurn verifies that the
// message-count trigger no longer fires mid-turn: even with the count threshold
// far exceeded, PreDispatchCheck does not compress while context is below the
// safety-net level.
func TestPreDispatchCheck_CountTrigger_DoesNotFireMidTurn(t *testing.T) {
	store := newMockStore()

	// Small history (~2% of context) but a large message count since last compaction.
	history := makeConversation(2, 200)
	noErr(t, store.SetHistory("sess", history))

	compressed := false
	mgr := newToolTurnsManager(t, store, nil)
	mgr.compressHook = func(_ bool) { compressed = true }
	mgr.compressedAtCount = 0
	mgr.msgCount = defaultMessageThreshold + 50 // well past the count threshold

	input := make([]spawnllm.Message, len(history))
	copy(input, history)

	if _, err := mgr.PreDispatchCheck(context.Background(), input); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if compressed {
		t.Error("count trigger must not fire mid-turn; it is deferred to the turn boundary")
	}
}

// TestPreDispatchCheck_ReturnsFreshBuiltSlice verifies that when compression fires
// and succeeds, the returned slice is the freshly built output from Build(), which
// reflects the post-compression store state, not the pre-compression input.
func TestPreDispatchCheck_ReturnsFreshBuiltSlice(t *testing.T) {
	store := newMockStore()

	// History at 100% of context window.
	history := makeConversation(10, 2000)
	noErr(t, store.SetHistory("sess", history))

	llm := &mockLLM{
		responses: []string{validSummaryJSON("goals after compression")},
	}
	mgr := newToolTurnsManager(t, store, []*mockLLM{llm})
	mgr.msgCount = len(history)

	input := make([]spawnllm.Message, len(history))
	copy(input, history)

	result, err := mgr.PreDispatchCheck(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// The result must not be the same slice as input and must be shorter,
	// proving Build() was called after compression, not the input returned.
	if len(result) >= len(input) {
		t.Errorf("expected fresh built slice (shorter than input %d), got len %d", len(input), len(result))
	}

	// The store should have been updated by compression.
	if store.GetSummary("sess") == "" {
		t.Error("expected summary to be stored after compression")
	}
}
