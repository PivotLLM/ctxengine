// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"
)

// failSaveStore wraps compressTestStore and makes Save() return an error.
type failSaveStore struct {
	compressTestStore
}

func (s *failSaveStore) Save(_ string) error {
	return errors.New("disk full")
}

// TestItem1_LargeMsgRemovedBeforePersist is a regression test for Item 1.
//
// Scenario: the session has one oversized assistant message that keeps the
// token count above safetyPercent even after group drops. Before the fix,
// applyLargeMsgChecks ran AFTER persistResult, so the oversized content was
// written to durable storage. After the fix, applyLargeMsgChecks runs before
// persistResult, so the persisted history must not contain the oversized content.
func TestItem1_LargeMsgRemovedBeforePersist(t *testing.T) {
	// contextWindow = 1000 tokens, safetyPercent = 80 → threshold = 800 tokens.
	// We need: groups dropped but large message still keeps us over 800 tokens.
	//
	// History layout (conversation, no system msg):
	//   [0] user  "u1"      (tiny)
	//   [1] asst  <huge>    (≈900 tokens = 3600 chars) — this is the problematic message
	//   [2] user  "u2"      (tiny — current turn)
	//
	// After LLM clients all fail, safety-net drops oldest groups:
	//   dropOldestGroups: drops group ending at 0 (user "u1"), then 1 (asst huge).
	//   After dropping both, only [user "u2"] remains → well under threshold.
	//   applyLargeMsgChecks then runs on the remaining slice; the huge message
	//   is no longer present. persistResult stores the clean slice.
	//
	// The key regression: before Item 1, persistResult ran with the huge message
	// still present in currentConversation (applyLargeMsgChecks mutated in-place
	// but was called after Save). After Item 1, the huge content is truncated
	// before Save is called.

	hugePad := strings.Repeat("X", 3600) // ~900 tokens

	store := &compressTestStore{
		history: []spawnllm.Message{
			{Role: "user", Content: "u1"},
			{Role: "assistant", Content: hugePad},
			{Role: "user", Content: "u2"},
		},
	}

	// All LLM calls fail → safety-net fallback.
	errList := make([]error, 20)
	for i := range errList {
		errList[i] = errors.New("fail")
	}
	llm := &mockLLM{errors: errList}

	mgr := newCompressManager(t, store, []*mockLLM{llm},
		WithContextWindow(1000),
		WithSafetyPercent(80),
		WithRetainMinMessages(0),
	)
	mgr.msgCount = len(store.history)

	err := mgr.doCompress(context.Background(), true)
	// May return nil, ErrCompressionPartial, or ErrCompressionFailed — any is
	// acceptable as long as the persisted history does not contain the huge message.
	_ = err

	// Verify that the persisted history does not contain the oversized content.
	for _, msg := range store.history {
		if strings.Contains(msg.Content, hugePad) {
			t.Errorf("persisted history still contains oversized content in role=%q message", msg.Role)
		}
	}
}

// TestItem1_PersistResultReturnsErrorOnSaveFailure verifies that persistResult
// propagates Save() failures as ErrCompressionFailed.
func TestItem1_PersistResultReturnsErrorOnSaveFailure(t *testing.T) {
	store := &failSaveStore{}
	// Distinct and over the retain budget, so the pass has something to
	// summarize: identical turns used to be collapsed into the summary, which
	// was the only reason a 400-token history compacted here.
	store.history = distinctConversation(20, 300)

	llm := &mockLLM{
		responses: []string{validSummaryJSON("goal")},
	}

	mgr := newCompressManager(t, &store.compressTestStore, []*mockLLM{llm},
		WithContextWindow(10000),
		WithNormalPercent(50),
		WithSafetyPercent(80),
	)
	// Wire the failing store directly.
	mgr.store = store
	mgr.msgCount = len(store.history)

	err := mgr.doCompress(context.Background(), false)
	if !errors.Is(err, ErrCompressionFailed) {
		t.Fatalf("expected ErrCompressionFailed when Save() fails; got %v", err)
	}
}

// TestItem3_ForceCompress_ToolLoopNoOrphanedMessages verifies that ForceCompress
// produces no orphaned tool results when the last message is a tool result.
//
// History layout:
//
//	[0] user "question"
//	[1] asst <tool-call: id="tc1">  (padded to consume tokens)
//	[2] tool <result for tc1>
//	[3] asst "final answer"
//	[4] user "follow-up"            (current trigger)
//	[5] asst <tool-call: id="tc2">  (padded)
//	[6] tool <result for tc2>       ← last message (in-progress tool loop)
//
// ForceCompress should keep the current turn group intact. The group ending at
// [6] (tool result for tc2) extends back through [5] (assistant with ToolCalls
// containing tc2). Both [5] and [6] must be present; no orphaned tool result.
func TestItem3_ForceCompress_ToolLoopNoOrphanedMessages(t *testing.T) {
	// contextWindow=200 tokens, safetyPercent=80 → threshold=160 tokens.
	// Pad old messages to ~80 tokens (320 chars) each so groups [0-4] push
	// total well above the threshold, forcing drops of older groups.
	pad := strings.Repeat("p", 320)

	history := []spawnllm.Message{
		{Role: "user", Content: "question " + pad},
		{
			Role:    "assistant",
			Content: "thinking " + pad,
			ToolCalls: []spawnllm.ToolCall{
				{ID: "tc1", Function: &spawnllm.FunctionCall{Name: "tool_a", Arguments: `{}`}},
			},
		},
		{Role: "tool", Content: "result1", ToolCallID: "tc1"},
		{Role: "assistant", Content: "final answer " + pad},
		{Role: "user", Content: "follow-up " + pad},
		{
			Role:    "assistant",
			Content: "more thinking",
			ToolCalls: []spawnllm.ToolCall{
				{ID: "tc2", Function: &spawnllm.FunctionCall{Name: "tool_b", Arguments: `{}`}},
			},
		},
		{Role: "tool", Content: "result2", ToolCallID: "tc2"},
	}

	store := &compressTestStore{history: history}

	mgr := newCompressManager(t, store, nil,
		WithContextWindow(200),
		WithSafetyPercent(80),
	)

	err := mgr.ForceCompress(context.Background())
	if err != nil {
		t.Fatalf("ForceCompress returned unexpected error: %v", err)
	}

	got := store.history

	// History must be shorter than the original.
	if len(got) >= len(history) {
		t.Fatalf("expected history to be reduced; got %d (original %d)", len(got), len(history))
	}

	// Verify no orphaned tool results: every message with a ToolCallID must be
	// preceded (somewhere) by an assistant message whose ToolCalls contains that ID.
	assistantToolCallIDs := make(map[string]bool)
	for _, msg := range got {
		for _, tc := range msg.ToolCalls {
			assistantToolCallIDs[tc.ID] = true
		}
	}
	for _, msg := range got {
		if msg.ToolCallID != "" && !assistantToolCallIDs[msg.ToolCallID] {
			t.Errorf("orphaned tool result: ToolCallID=%q has no matching assistant ToolCall in history", msg.ToolCallID)
		}
	}

	// The current in-progress tool loop (tc2) must be preserved intact.
	foundTC2Call := false
	foundTC2Result := false
	for _, msg := range got {
		for _, tc := range msg.ToolCalls {
			if tc.ID == "tc2" {
				foundTC2Call = true
			}
		}
		if msg.ToolCallID == "tc2" {
			foundTC2Result = true
		}
	}
	if !foundTC2Call {
		t.Error("current turn assistant tool-call (tc2) not found in compressed history")
	}
	if !foundTC2Result {
		t.Error("current turn tool result (tc2) not found in compressed history")
	}
}

// TestItem3_ForceCompress_OversizedResultInCurrentTurnIsTruncated verifies
// that when the current turn group alone is past the safety line because of
// one oversized tool result, ForceCompress truncates that result (the same
// large-message rule the safety-net compaction applies) rather than failing,
// so the retry can go out.
func TestItem3_ForceCompress_OversizedResultInCurrentTurnIsTruncated(t *testing.T) {
	// contextWindow=100 tokens, safetyPercent=80 → threshold=80 tokens.
	// Current turn group = one tool result with ~100 tokens (400 chars).
	hugePad := strings.Repeat("Z", 400) // ~100 tokens

	history := []spawnllm.Message{
		{Role: "user", Content: "old message"},
		{
			Role:    "assistant",
			Content: "calling tool",
			ToolCalls: []spawnllm.ToolCall{
				{ID: "tc1", Function: &spawnllm.FunctionCall{Name: "big_tool", Arguments: `{}`}},
			},
		},
		{Role: "tool", Content: hugePad, ToolCallID: "tc1"},
	}

	store := &compressTestStore{history: history}

	mgr := newCompressManager(t, store, nil,
		WithContextWindow(100),
		WithSafetyPercent(80),
	)

	if err := mgr.ForceCompress(context.Background()); err != nil {
		t.Fatalf("ForceCompress: %v", err)
	}
	got := store.GetHistory("test")
	if len(got) != 2 || got[0].Role != "assistant" || got[1].ToolCallID != "tc1" {
		t.Fatalf("window after ForceCompress = %v, want the current tool-call turn", got)
	}
	if !strings.Contains(got[1].Content, "[**TRUNCATED DUE TO SIZE**]") {
		t.Errorf("oversized tool result was not truncated: %d chars", len(got[1].Content))
	}
	if mgr.contextPercent(got) >= 80 {
		t.Errorf("window still past the safety line after ForceCompress: %.0f%%", mgr.contextPercent(got))
	}
}

// TestItem3_ForceCompress_OversizedCurrentTurnReturnsError verifies that when
// the current turn group cannot be made to fit — several results each under
// the per-message truncation threshold, together past the safety line —
// ForceCompress keeps the group whole and returns ErrCompressionFailed rather
// than emptying the window.
func TestItem3_ForceCompress_OversizedCurrentTurnReturnsError(t *testing.T) {
	// contextWindow=100 tokens, safetyPercent=80 → threshold=80 tokens; the
	// truncation threshold is 60 tokens per message. Three results of ~55
	// tokens (220 chars) each stay under it but total ~165.
	pad := strings.Repeat("Z", 220)

	history := []spawnllm.Message{
		{Role: "user", Content: "old message"},
		{
			Role: "assistant",
			ToolCalls: []spawnllm.ToolCall{
				{ID: "tc1", Function: &spawnllm.FunctionCall{Name: "t", Arguments: `{}`}},
				{ID: "tc2", Function: &spawnllm.FunctionCall{Name: "t", Arguments: `{}`}},
				{ID: "tc3", Function: &spawnllm.FunctionCall{Name: "t", Arguments: `{}`}},
			},
		},
		{Role: "tool", Content: pad, ToolCallID: "tc1"},
		{Role: "tool", Content: pad, ToolCallID: "tc2"},
		{Role: "tool", Content: pad, ToolCallID: "tc3"},
	}

	store := &compressTestStore{history: history}

	mgr := newCompressManager(t, store, nil,
		WithContextWindow(100),
		WithSafetyPercent(80),
	)

	err := mgr.ForceCompress(context.Background())
	if !errors.Is(err, ErrCompressionFailed) {
		t.Fatalf("expected ErrCompressionFailed for oversized current turn; got %v", err)
	}
	got := store.GetHistory("test")
	if len(got) != 4 {
		t.Fatalf("window after failed ForceCompress = %d messages, want the whole current group (4)", len(got))
	}
}
