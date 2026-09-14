// ctxengine
// License: MIT

package ctxengine

import (
	"testing"

	"github.com/PivotLLM/spawnllm"
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

// TestSanitizeHistoryForProvider_IncompleteToolResults tests the forward validation
// that ensures assistant messages with tool_calls have ALL matching tool results.
// This fixes the DeepSeek error: "An assistant message with 'tool_calls' must be
// followed by tool messages responding to each 'tool_call_id'."
func TestSanitizeHistoryForProvider_IncompleteToolResults(t *testing.T) {
	// Assistant expects tool results for both A and B, but only A is present
	history := []spawnllm.Message{
		msg("user", "do two things"),
		sanAssistantWithTools("A", "B"),
		sanToolResult("A"),
		// sanToolResult("B") is missing - this would cause DeepSeek to fail
		msg("user", "next question"),
		msg("assistant", "answer"),
	}

	result := sanitizeHistoryForProvider(history)
	// The assistant message with incomplete tool results should be dropped,
	// along with its partial tool result. The remaining messages are:
	// user ("do two things"), user ("next question"), assistant ("answer")
	if len(result) != 3 {
		t.Fatalf("expected 3 messages, got %d: %+v", len(result), roles(result))
	}
	assertRoles(t, result, "user", "user", "assistant")
}

// TestSanitizeHistoryForProvider_MissingAllToolResults tests the case where
// an assistant message has tool_calls but no tool results follow at all.
func TestSanitizeHistoryForProvider_MissingAllToolResults(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "do something"),
		sanAssistantWithTools("A"),
		// No tool results at all
		msg("user", "hello"),
		msg("assistant", "hi"),
	}

	result := sanitizeHistoryForProvider(history)
	// The assistant message with no tool results should be dropped.
	// Remaining: user ("do something"), user ("hello"), assistant ("hi")
	if len(result) != 3 {
		t.Fatalf("expected 3 messages, got %d: %+v", len(result), roles(result))
	}
	assertRoles(t, result, "user", "user", "assistant")
}

// TestSanitizeHistoryForProvider_PartialToolResultsInMiddle tests that
// incomplete tool results in the middle of a conversation are properly handled.
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
	// First round is complete (user, assistant+tools, tool, assistant),
	// second round is incomplete and dropped (assistant+tools, partial tool),
	// third round is complete (user, assistant+tools, tool, assistant).
	// Remaining: user, assistant, tool, assistant, user, user, assistant, tool, assistant
	if len(result) != 9 {
		t.Fatalf("expected 9 messages, got %d: %+v", len(result), roles(result))
	}
	assertRoles(t, result, "user", "assistant", "tool", "assistant", "user", "user", "assistant", "tool", "assistant")
}
