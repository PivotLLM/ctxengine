// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"
)

// newCheckCompressManager builds a Manager for CheckAndCompress tests.
func newCheckCompressManager(store *mockStore, clients []*mockLLM, opts ...Option) *Manager {
	baseOpts := []Option{
		WithContextWindow(10000),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithRetainTokenPercent(20),
		WithRetainMinMessages(2),
		WithModelCaller(chainOf(clients)),
	}
	baseOpts = append(baseOpts, opts...)
	cm := New("sess", store, baseOpts...)
	return cm.(*Manager)
}

// TestCheckAndCompress_BelowThreshold verifies that CheckAndCompress returns the
// input slice unchanged when tokens (including overhead) are below the normal threshold.
func TestCheckAndCompress_BelowThreshold(t *testing.T) {
	store := newMockStore()
	// 4 chars × 100 = 400 chars → 100 tokens of content; overheadTokens = 200 → 300 total.
	// contextWindow = 10000 with normalPercent = 50 → threshold at 5000 tokens.
	// 300 tokens is well below 5000, so no compression.
	small := []spawnllm.Message{
		{Role: "user", Content: strings.Repeat("a", 400)},
	}
	store.SetHistory("sess", small)

	compressed := false
	mgr := newCheckCompressManager(store, nil,
		WithContextWindow(10000),
		WithNormalPercent(50),
		WithOverheadTokens(200),
	)
	mgr.compressHook = func(_ bool) { compressed = true }
	mgr.msgCount = 1

	built := []spawnllm.Message{{Role: "user", Content: "hello"}}
	result, err := mgr.CheckAndCompress(context.Background(), built)
	if err != nil {
		t.Fatalf("CheckAndCompress returned error: %v", err)
	}
	if compressed {
		t.Error("expected no compression to fire")
	}
	if len(result) != len(built) {
		t.Errorf("expected input slice unchanged (len %d), got len %d", len(built), len(result))
	}
}

// TestCheckAndCompress_OverheadPushesOverSafety verifies that when stored history
// is within the safety threshold but adding overheadTokens pushes the rendered
// request over the safety-net level, the emergency compression fires at the
// CheckAndCompress call. (CheckAndCompress is emergency-only; the normal band is
// handled at the turn boundary.)
func TestCheckAndCompress_OverheadPushesOverSafety(t *testing.T) {
	store := newMockStore()

	// contextWindow = 1000. safetyPercent = 80 → trigger at 800 tokens.
	// Built messages: 1600 chars → 400 tokens from content.
	// overheadTokens = 500 → total = 900 tokens = 90% of 1000 → above safetyPercent.
	built := []spawnllm.Message{
		{Role: "user", Content: strings.Repeat("b", 1600)},
	}
	store.SetHistory("sess", built)

	var hookCalls []bool
	mgr := newCheckCompressManager(store, nil,
		WithContextWindow(1000),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithOverheadTokens(500),
	)
	mgr.compressHook = func(safetyNet bool) { hookCalls = append(hookCalls, safetyNet) }
	mgr.msgCount = 1

	_, err := mgr.CheckAndCompress(context.Background(), built)
	// err may be nil (hook short-circuits) or ErrCompressionFailed — the important
	// thing is that the hook was called, proving compression was triggered.
	_ = err

	if len(hookCalls) == 0 {
		t.Fatal("expected emergency compression to fire when overhead pushes total over safety threshold")
	}
	if !hookCalls[0] {
		t.Error("expected the fired compression to be the safety-net path (safetyNet=true)")
	}
}

// TestCheckAndCompress_NormalBand_DoesNotFire verifies that CheckAndCompress is
// emergency-only: a rendered request in the normal band (normalPercent <= total <
// safetyPercent) does NOT compress at dispatch time. First-stage compaction is
// deferred to the turn boundary.
func TestCheckAndCompress_NormalBand_DoesNotFire(t *testing.T) {
	store := newMockStore()

	// 2400 chars → 600 tokens content; overhead 0 → 600 tokens = 60% of 1000:
	// above normalPercent (50%), below safetyPercent (80%).
	history := makeConversation(6, 200)
	store.SetHistory("sess", history)

	compressed := false
	mgr := newCheckCompressManager(store, nil,
		WithContextWindow(1000),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithOverheadTokens(0),
	)
	mgr.compressHook = func(_ bool) { compressed = true }
	mgr.msgCount = len(history)

	built := make([]spawnllm.Message, len(history))
	copy(built, history)

	result, err := mgr.CheckAndCompress(context.Background(), built)
	if err != nil {
		t.Fatalf("expected no error in normal band, got: %v", err)
	}
	if compressed {
		t.Error("expected no compression in the normal band (deferred to turn boundary)")
	}
	if len(result) != len(built) {
		t.Errorf("expected input slice unchanged (len %d), got len %d", len(built), len(result))
	}
}

// TestCheckAndCompress_NoContextWindow returns input unchanged when contextWindow == 0.
func TestCheckAndCompress_NoContextWindow(t *testing.T) {
	store := newMockStore()
	mgr := newCheckCompressManager(store, nil,
		WithContextWindow(0),
	)
	built := makeConversation(5, 500)
	result, err := mgr.CheckAndCompress(context.Background(), built)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != len(built) {
		t.Errorf("expected input unchanged (len %d); got %d", len(built), len(result))
	}
}

// TestCheckAndCompress_CompressionFiredReturnsFreshSlice verifies that when
// compression fires and succeeds (via a real LLM client, not hook), the returned
// slice is the fresh Build() output rather than the stale input.
func TestCheckAndCompress_CompressionFiredReturnsFreshSlice(t *testing.T) {
	store := newMockStore()

	// contextWindow = 1000. safetyPercent = 80 → trigger at 800 tokens.
	// Built: 2000 chars → 500 tokens. overheadTokens = 400 → 900 tokens → 90% → triggers.
	history := makeConversation(5, 200)
	store.SetHistory("sess", history)

	llm := &mockLLM{
		responses: []string{validSummaryJSON("fresh slice goal")},
	}

	mgr := newCheckCompressManager(store, []*mockLLM{llm},
		WithContextWindow(1000),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithOverheadTokens(400),
	)
	mgr.msgCount = len(history)

	built := make([]spawnllm.Message, len(history))
	copy(built, history)

	result, err := mgr.CheckAndCompress(context.Background(), built)
	if err != nil {
		t.Fatalf("CheckAndCompress returned unexpected error: %v", err)
	}

	// After compression the store has fewer messages; Build() produces a shorter slice.
	if len(result) >= len(built) {
		t.Errorf("expected fresh (shorter) slice after compression; got len %d (input len %d)",
			len(result), len(built))
	}
}

// TestCheckAndCompress_ToolCallsCountedInTokens verifies that ToolCalls argument
// content is included in the token estimate, not just Message.Content.
func TestCheckAndCompress_ToolCallsCountedInTokens(t *testing.T) {
	// A message with empty Content but large ToolCalls arguments.
	// 4000 chars in Arguments → ~1000 tokens. contextWindow = 1000, normalPercent = 50
	// → threshold at 500 tokens. 1000 tokens > 500 → should trigger compression.
	largeArgs := strings.Repeat("x", 4000)
	msgs := []spawnllm.Message{
		{
			Role:    "assistant",
			Content: "",
			ToolCalls: []spawnllm.ToolCall{
				{ID: "tc1", Function: &spawnllm.FunctionCall{
					Name:      "big_tool",
					Arguments: largeArgs,
				}},
			},
		},
	}

	tokenCount := estimateTokens(msgs)
	// ~1000 tokens from tool call arguments + overhead from JSON structure.
	if tokenCount < 900 {
		t.Errorf("expected ToolCalls arguments to be counted; got only %d tokens", tokenCount)
	}
}

// TestCheckAndCompress_NormalBandWithCooldown_DoesNotFire verifies that a request
// in the normal band does not compress at the CheckAndCompress call regardless of
// cooldown state — CheckAndCompress is emergency-only, so the normal band never
// fires here whether cooling or not.
func TestCheckAndCompress_NormalBandWithCooldown_DoesNotFire(t *testing.T) {
	store := newMockStore()
	// Above normalPercent (50%) at 60% with no overhead — still below safety (80%).
	history := makeConversation(6, 200) // 2400 chars → 600 tokens → 60% of 1000
	store.SetHistory("sess", history)

	compressed := false
	mgr := newCheckCompressManager(store, nil,
		WithContextWindow(1000),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithOverheadTokens(0),
	)
	mgr.compressHook = func(_ bool) { compressed = true }
	mgr.msgCount = len(history)

	mgr.cooling = true
	mgr.coolingSinceCount = mgr.msgCount

	built := make([]spawnllm.Message, len(history))
	copy(built, history)

	_, err := mgr.CheckAndCompress(context.Background(), built)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if compressed {
		t.Error("expected no compression in the normal band (emergency-only)")
	}
}
