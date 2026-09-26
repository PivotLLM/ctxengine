// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"testing"

	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine/memory"
)

func sanAssistantWithTools(toolIDs ...string) spawnllm.Message {
	calls := make([]spawnllm.ToolCall, len(toolIDs))
	for i, id := range toolIDs {
		calls[i] = spawnllm.ToolCall{ID: id, Type: "function"}
	}
	return spawnllm.Message{Role: "assistant", ToolCalls: calls}
}

func sanToolResult(id string) spawnllm.Message {
	return spawnllm.Message{Role: "tool", Content: "result", ToolCallID: id}
}

func TestSanitizeHistoryForProvider_EmptyHistory(t *testing.T) {
	result := sanitizeHistoryForProvider(nil)
	if len(result) != 0 {
		t.Fatalf("expected empty, got %d messages", len(result))
	}

	result = sanitizeHistoryForProvider([]spawnllm.Message{})
	if len(result) != 0 {
		t.Fatalf("expected empty, got %d messages", len(result))
	}
}

func TestSanitizeHistoryForProvider_SingleToolCall(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "hello"),
		sanAssistantWithTools("A"),
		sanToolResult("A"),
		msg("assistant", "done"),
	}

	result := sanitizeHistoryForProvider(history)
	if len(result) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(result))
	}
	assertRoles(t, result, "user", "assistant", "tool", "assistant")
}

func TestSanitizeHistoryForProvider_MultiToolCalls(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "do two things"),
		sanAssistantWithTools("A", "B"),
		sanToolResult("A"),
		sanToolResult("B"),
		msg("assistant", "both done"),
	}

	result := sanitizeHistoryForProvider(history)
	if len(result) != 5 {
		t.Fatalf("expected 5 messages, got %d: %+v", len(result), roles(result))
	}
	assertRoles(t, result, "user", "assistant", "tool", "tool", "assistant")
}

func TestSanitizeHistoryForProvider_AssistantToolCallAfterPlainAssistant(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "hi"),
		msg("assistant", "thinking"),
		sanAssistantWithTools("A"),
		sanToolResult("A"),
	}

	result := sanitizeHistoryForProvider(history)
	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d: %+v", len(result), roles(result))
	}
	assertRoles(t, result, "user", "assistant")
}

func TestSanitizeHistoryForProvider_OrphanedLeadingTool(t *testing.T) {
	history := []spawnllm.Message{
		sanToolResult("A"),
		msg("user", "hello"),
	}

	result := sanitizeHistoryForProvider(history)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d: %+v", len(result), roles(result))
	}
	assertRoles(t, result, "user")
}

func TestSanitizeHistoryForProvider_ToolAfterUserDropped(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "hello"),
		sanToolResult("A"),
	}

	result := sanitizeHistoryForProvider(history)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d: %+v", len(result), roles(result))
	}
	assertRoles(t, result, "user")
}

func TestSanitizeHistoryForProvider_ToolAfterAssistantNoToolCalls(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "hello"),
		msg("assistant", "hi"),
		sanToolResult("A"),
	}

	result := sanitizeHistoryForProvider(history)
	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d: %+v", len(result), roles(result))
	}
	assertRoles(t, result, "user", "assistant")
}

func TestSanitizeHistoryForProvider_AssistantToolCallAtStart(t *testing.T) {
	history := []spawnllm.Message{
		sanAssistantWithTools("A"),
		sanToolResult("A"),
		msg("user", "hello"),
	}

	result := sanitizeHistoryForProvider(history)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d: %+v", len(result), roles(result))
	}
	assertRoles(t, result, "user")
}

func TestSanitizeHistoryForProvider_MultiToolCallsThenNewRound(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "do two things"),
		sanAssistantWithTools("A", "B"),
		sanToolResult("A"),
		sanToolResult("B"),
		msg("assistant", "done"),
		msg("user", "hi"),
		sanAssistantWithTools("C"),
		sanToolResult("C"),
		msg("assistant", "done again"),
	}

	result := sanitizeHistoryForProvider(history)
	if len(result) != 9 {
		t.Fatalf("expected 9 messages, got %d: %+v", len(result), roles(result))
	}
	assertRoles(t, result, "user", "assistant", "tool", "tool", "assistant", "user", "assistant", "tool", "assistant")
}

func TestSanitizeHistoryForProvider_ConsecutiveMultiToolRounds(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "start"),
		sanAssistantWithTools("A", "B"),
		sanToolResult("A"),
		sanToolResult("B"),
		sanAssistantWithTools("C", "D"),
		sanToolResult("C"),
		sanToolResult("D"),
		msg("assistant", "all done"),
	}

	result := sanitizeHistoryForProvider(history)
	if len(result) != 8 {
		t.Fatalf("expected 8 messages, got %d: %+v", len(result), roles(result))
	}
	assertRoles(t, result, "user", "assistant", "tool", "tool", "assistant", "tool", "tool", "assistant")
}

func TestSanitizeHistoryForProvider_PlainConversation(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "hello"),
		msg("assistant", "hi"),
		msg("user", "how are you"),
		msg("assistant", "fine"),
	}

	result := sanitizeHistoryForProvider(history)
	if len(result) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(result))
	}
	assertRoles(t, result, "user", "assistant", "user", "assistant")
}

func roles(msgs []spawnllm.Message) []string {
	r := make([]string, len(msgs))
	for i, m := range msgs {
		r[i] = m.Role
	}
	return r
}

func assertRoles(t *testing.T, msgs []spawnllm.Message, expected ...string) {
	t.Helper()
	if len(msgs) != len(expected) {
		t.Fatalf("role count mismatch: got %v, want %v", roles(msgs), expected)
	}
	for i, exp := range expected {
		if msgs[i].Role != exp {
			t.Errorf("message[%d]: got role %q, want %q", i, msgs[i].Role, exp)
		}
	}
}

// assertInterrupted checks that m is the synthesised result for call id.
func assertInterrupted(t *testing.T, m spawnllm.Message, id string) {
	t.Helper()
	if m.Role != "tool" || m.ToolCallID != id {
		t.Fatalf("expected synthesised result for %q, got role=%q tool_call_id=%q", id, m.Role, m.ToolCallID)
	}
	if m.Content != interruptedToolResult {
		t.Errorf("synthesised result content = %q, want the interrupted marker", m.Content)
	}
}

// TestSanitizeHistoryForProvider_IncompleteToolResults covers a turn whose
// calls are only partly answered — the shape a crash between the assistant
// write and the tool-result writes leaves behind. The group is kept and the
// missing result is synthesised, so a strict provider ("An assistant message
// with 'tool_calls' must be followed by tool messages responding to each
// 'tool_call_id'") accepts it and the model is told the call was made rather
// than left to run it again.
func TestSanitizeHistoryForProvider_IncompleteToolResults(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "do two things"),
		sanAssistantWithTools("A", "B"),
		sanToolResult("A"),
		// sanToolResult("B") is missing.
		msg("user", "next question"),
		msg("assistant", "answer"),
	}

	result := sanitizeHistoryForProvider(history)
	assertRoles(t, result, "user", "assistant", "tool", "tool", "user", "assistant")
	if result[2].ToolCallID != "A" || result[2].Content != "result" {
		t.Errorf("the real result for A was altered: %+v", result[2])
	}
	assertInterrupted(t, result[3], "B")
}

// TestSanitizeHistoryForProvider_MissingAllToolResults covers a turn with no
// results at all: every call gets a synthesised result, in call order,
// directly after the turn.
func TestSanitizeHistoryForProvider_MissingAllToolResults(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "do something"),
		sanAssistantWithTools("A", "B"),
		msg("user", "hello"),
		msg("assistant", "hi"),
	}

	result := sanitizeHistoryForProvider(history)
	assertRoles(t, result, "user", "assistant", "tool", "tool", "user", "assistant")
	assertInterrupted(t, result[2], "A")
	assertInterrupted(t, result[3], "B")
}

// TestSanitizeHistoryForProvider_PartialToolResultsInMiddle checks that an
// interrupted group in the middle of a conversation is completed in place and
// the groups around it are untouched.
func TestSanitizeHistoryForProvider_PartialToolResultsInMiddle(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "first"),
		sanAssistantWithTools("A"),
		sanToolResult("A"),
		msg("assistant", "done"),
		msg("user", "second"),
		sanAssistantWithTools("B", "C"),
		sanToolResult("B"),
		// sanToolResult("C") is missing
		msg("user", "third"),
		sanAssistantWithTools("D"),
		sanToolResult("D"),
		msg("assistant", "all done"),
	}

	result := sanitizeHistoryForProvider(history)
	assertRoles(t, result,
		"user", "assistant", "tool", "assistant",
		"user", "assistant", "tool", "tool",
		"user", "assistant", "tool", "assistant")
	assertInterrupted(t, result[7], "C")
	if result[6].ToolCallID != "B" {
		t.Errorf("result[6] answers %q, want B", result[6].ToolCallID)
	}
}

// TestSanitizeHistoryForProvider_InterruptedGroupStaysProviderValid pins the
// ordering rule the synthesis must respect: every result of a group, real or
// synthesised, sits contiguously after its turn and answers a call that turn
// made. A stray result in the same group is still dropped.
func TestSanitizeHistoryForProvider_InterruptedGroupStaysProviderValid(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "go"),
		sanAssistantWithTools("A", "B", "C"),
		sanToolResult("C"),
		sanToolResult("stray"),
		msg("assistant", "carrying on"),
	}

	result := sanitizeHistoryForProvider(history)
	assertRoles(t, result, "user", "assistant", "tool", "tool", "tool", "assistant")

	declared := map[string]bool{}
	for _, tc := range result[1].ToolCalls {
		declared[tc.ID] = true
	}
	seen := map[string]bool{}
	for _, m := range result[2:5] {
		if !declared[m.ToolCallID] {
			t.Errorf("result for undeclared call %q survived", m.ToolCallID)
		}
		seen[m.ToolCallID] = true
	}
	for _, id := range []string{"A", "B", "C"} {
		if !seen[id] {
			t.Errorf("call %q has no result after sanitising", id)
		}
	}
	if result[2].ToolCallID != "C" || result[2].Content != "result" {
		t.Errorf("the real result must precede the synthesised ones: %+v", result[2])
	}
	assertInterrupted(t, result[3], "A")
	assertInterrupted(t, result[4], "B")
}

// TestSanitizeHistoryForProvider_BadPredecessorStillDropped pins that
// synthesis applies only to a missing result: a group whose turn follows a
// plain assistant message is malformed in another way and is still dropped
// whole, unanswered calls included.
func TestSanitizeHistoryForProvider_BadPredecessorStillDropped(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "hi"),
		msg("assistant", "thinking"),
		sanAssistantWithTools("A", "B"),
		sanToolResult("A"),
	}

	result := sanitizeHistoryForProvider(history)
	assertRoles(t, result, "user", "assistant")
}

// TestBuild_InterruptedGroupLeavesStoreUntouched drives the synthesis through
// Build: the built slice carries the synthesised result, while the stored
// history — count, messages and seqs — is exactly what it was. The synthesised
// result exists only in the request.
func TestBuild_InterruptedGroupLeavesStoreUntouched(t *testing.T) {
	stored := []memory.StoredMessage{
		{Seq: 40, Message: msg("user", "go")},
		{Seq: 41, Message: sanAssistantWithTools("A", "B")},
		{Seq: 42, Message: sanToolResult("A")},
	}
	store := newSeqStore(stored)
	mgr := New("sess", store, WithContextWindow(100_000)).(*Manager)

	built, err := mgr.Build(context.Background())
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	assertRoles(t, built, "user", "assistant", "tool", "tool")
	assertInterrupted(t, built[3], "B")

	after := store.GetHistoryWithSeqs("sess")
	if len(after) != len(stored) {
		t.Fatalf("stored history changed: %d messages, want %d", len(after), len(stored))
	}
	for i := range stored {
		if after[i].Seq != stored[i].Seq {
			t.Errorf("seq[%d] = %d, want %d", i, after[i].Seq, stored[i].Seq)
		}
		if after[i].Role != stored[i].Role || after[i].ToolCallID != stored[i].ToolCallID || after[i].Content != stored[i].Content {
			t.Errorf("stored[%d] changed: %+v", i, after[i].Message)
		}
	}
}
