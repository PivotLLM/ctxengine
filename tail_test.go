// ctxengine
// License: MIT

package ctxengine

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/PivotLLM/spawnllm"
	"github.com/PivotLLM/spawnllm/protocoltypes"

	cronmsg "github.com/PivotLLM/ctxengine/internal/testcron"
	"github.com/PivotLLM/ctxengine/memory"
)

// testNow is the fixed clock every age-sensitive tail test measures against, so
// results never depend on when the suite runs.
var testNow = time.Date(2026, 8, 21, 12, 0, 0, 0, time.UTC)

// storedAt wraps plain messages as StoredMessage with per-message ages in days
// before testNow. A missing age entry means "just now". Passing no ages at all
// makes every message current, which is what the non-age tests want.
func storedAt(msgs []spawnllm.Message, agesDays ...int) []memory.StoredMessage {
	out := make([]memory.StoredMessage, len(msgs))
	for i, m := range msgs {
		age := 0
		if i < len(agesDays) {
			age = agesDays[i]
		}
		out[i] = memory.StoredMessage{
			Seq:       int64(i + 1),
			CreatedAt: testNow.AddDate(0, 0, -age),
			Message:   m,
		}
	}
	return out
}

// selectTailMsgs runs selectTail with no age cap over messages that are all
// current, returning just the retained messages. Most tail behaviour is
// age-independent, so this keeps those tests focused on budget and floor.
func selectTailMsgs(history []spawnllm.Message, budget, minMessages int) []spawnllm.Message {
	tail, _ := selectTail(storedAt(history), budget, minMessages, 0, testNow, estimateTokens, cronmsg.CollapseKey)
	if len(tail) == 0 {
		return nil
	}
	return storedToPlain(tail)
}

func msg(role, content string) spawnllm.Message {
	return spawnllm.Message{Role: role, Content: content}
}

func toolCallMsg(content, tcID string) spawnllm.Message {
	return spawnllm.Message{
		Role:    "assistant",
		Content: content,
		ToolCalls: []protocoltypes.ToolCall{
			{ID: tcID},
		},
	}
}

func toolResultMsg(content, callID string) spawnllm.Message {
	return spawnllm.Message{
		Role:       "tool",
		Content:    content,
		ToolCallID: callID,
	}
}

// TestSelectTail_BudgetRetention verifies that messages are dropped when the
// token budget is exhausted and they don't need to be kept for the floor.
func TestSelectTail_BudgetRetention(t *testing.T) {
	// Build a history where only the last 2 messages fit in the budget.
	// estimateTokens: 2 chars/5 tokens, so 100 runes → 40 tokens.
	longContent := fmt.Sprintf("%0100d", 0) // 100 chars → ~40 tokens
	history := []spawnllm.Message{
		msg("user", longContent),      // too old, won't fit
		msg("assistant", longContent), // too old, won't fit
		msg("user", "hi"),             // fits
		msg("assistant", "ok"),        // fits
	}
	// Budget of 10 tokens; only "hi"+"ok" fit (each ~1 token).
	got := selectTailMsgs(history, 10, 0)
	if len(got) != 2 {
		t.Fatalf("want 2 messages, got %d", len(got))
	}
	if got[0].Content != "hi" || got[1].Content != "ok" {
		t.Errorf("unexpected contents: %v", got)
	}
}

// TestSelectTail_MinFloorOverride verifies that minMessages overrides the
// budget, keeping at least that many meaningful messages.
func TestSelectTail_MinFloorOverride(t *testing.T) {
	// Each message has distinct content so noise collapse does not shrink them.
	history := []spawnllm.Message{
		msg("user", fmt.Sprintf("%0200d", 1)),      // 200 chars → ~80 tokens
		msg("assistant", fmt.Sprintf("%0200d", 2)), // distinct
		msg("user", fmt.Sprintf("%0200d", 3)),      // distinct
		msg("assistant", fmt.Sprintf("%0200d", 4)), // distinct
	}
	// Budget of 1 token — nothing fits — but floor is 3.
	got := selectTailMsgs(history, 1, 3)
	if len(got) < 3 {
		t.Fatalf("want at least 3 messages due to floor, got %d", len(got))
	}
}

// TestSelectTail_ToolGroupKeptWhole verifies that a tool-call group (assistant
// with ToolCalls + tool result) is kept as an atomic unit.
func TestSelectTail_ToolGroupKeptWhole(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "start"),
		toolCallMsg("calling tool", "tc1"),
		toolResultMsg("tool output", "tc1"),
		msg("user", "follow up"),
	}
	// Budget large enough to fit everything.
	got := selectTailMsgs(history, 10000, 0)
	if len(got) != 4 {
		t.Fatalf("want 4 messages, got %d", len(got))
	}
}

// TestSelectTail_ToolGroupDroppedWhole verifies that when a tool-call group
// cannot fit in the budget, all of its messages are dropped together.
func TestSelectTail_ToolGroupDroppedWhole(t *testing.T) {
	// Tool group has ~160 chars = ~64 tokens; follow-up has ~2 tokens.
	// Budget of 10 fits follow-up but NOT the tool group.
	longTool := fmt.Sprintf("%0200d", 0)
	history := []spawnllm.Message{
		msg("user", "before"),
		toolCallMsg(longTool, "tc1"),
		toolResultMsg(longTool, "tc1"),
		msg("user", "after"),
	}
	got := selectTailMsgs(history, 10, 0)
	// Only "after" fits; the tool group (tc1 + result) must not be partially included.
	for _, m := range got {
		if m.ToolCallID == "tc1" || (len(m.ToolCalls) > 0 && m.ToolCalls[0].ID == "tc1") {
			t.Errorf("tool group message leaked into tail: %+v", m)
		}
	}
}

// TestSelectTail_LeadingToolGroupTrimmed verifies the retained tail never starts
// on a partial tool group: when the budget cut lands inside a tool-call sequence,
// the leading assistant tool-call + tool results are trimmed (handed to the
// summary) so the tail begins on a clean boundary the provider sanitizer accepts.
func TestSelectTail_LeadingToolGroupTrimmed(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "old question"),
		toolCallMsg("calling tool", "tc1"),
		toolResultMsg("tool output", "tc1"),
		msg("user", "recent question"),
	}
	// Budget fits everything by tokens, but force the floor so all groups are
	// collected — then the leading tool group must still be trimmed.
	got := selectTailMsgs(history, 10000, 0)
	if len(got) == 0 {
		t.Fatal("expected a non-empty tail")
	}
	first := got[0]
	if first.Role == "tool" || (first.Role == "assistant" && len(first.ToolCalls) > 0) {
		t.Fatalf("tail starts on a partial tool group: %+v", first)
	}

	// Now make the cut land mid-group: only the tool result + final user fit by
	// budget, so resolveGroup pulls in the assistant — the whole leading group
	// must be trimmed, leaving just the clean trailing user message.
	got2 := selectTailMsgs(history[1:], 10000, 0) // [toolcall, toolresult, user]
	if len(got2) != 1 || got2[0].Content != "recent question" {
		t.Fatalf("expected only the trailing user message, got %+v", got2)
	}
}

// TestSelectTail_NoiseCollapsed verifies that consecutive identical same-role
// messages are collapsed to at most one in the retained tail.
func TestSelectTail_NoiseCollapsed(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "hello"),
		msg("assistant", "hi"),
		msg("user", "hello"),   // duplicate of history[0]
		msg("assistant", "hi"), // duplicate of history[1]
		msg("user", "different"),
	}
	got := selectTailMsgs(history, 10000, 0)
	// Collapsed: only one "hello" user and one "hi" assistant should survive each.
	userCount := 0
	for _, m := range got {
		if m.Role == "user" && m.Content == "hello" {
			userCount++
		}
	}
	if userCount > 1 {
		t.Errorf("expected at most 1 'hello' user message, got %d", userCount)
	}
	assistantCount := 0
	for _, m := range got {
		if m.Role == "assistant" && m.Content == "hi" {
			assistantCount++
		}
	}
	if assistantCount > 1 {
		t.Errorf("expected at most 1 'hi' assistant message, got %d", assistantCount)
	}
}

// TestSelectTail_CronNoiseCollapsed verifies cron-wrapper messages with the
// same payload are treated as noise and collapsed.
func TestSelectTail_CronNoiseCollapsed(t *testing.T) {
	wrap := func(ts, payload string) spawnllm.Message {
		return msg("user", testCronPrefix+"2026-01-01 "+ts+":\n"+payload)
	}
	history := []spawnllm.Message{
		wrap("10:00", "run backup"),
		msg("assistant", "done"),
		wrap("11:00", "run backup"), // same payload — noise
		msg("assistant", "done"),
	}
	got := selectTailMsgs(history, 10000, 0)
	cronCount := 0
	for _, m := range got {
		if strings.HasPrefix(m.Content, testCronPrefix) {
			cronCount++
		}
	}
	if cronCount > 1 {
		t.Errorf("expected cron noise collapsed to 1, got %d cron messages", cronCount)
	}
}

// TestSelectTail_EmptyHistory returns nil for an empty input.
func TestSelectTail_EmptyHistory(t *testing.T) {
	got := selectTailMsgs(nil, 1000, 2)
	if got != nil {
		t.Errorf("want nil, got %v", got)
	}
}

// TestSelectTail_ZeroBudget treats zero budget as unlimited.
func TestSelectTail_ZeroBudget(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "a"),
		msg("assistant", "b"),
		msg("user", "c"),
	}
	got := selectTailMsgs(history, 0, 0)
	if len(got) != 3 {
		t.Errorf("zero budget should keep all messages, got %d", len(got))
	}
}

// TestResolveGroup_NoToolCall returns single-message group.
func TestResolveGroup_NoToolCall(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "hello"),
		msg("assistant", "world"),
	}
	g := resolveGroup(history, 1)
	if g.start != 1 || g.end != 1 {
		t.Errorf("want {1,1}, got %+v", g)
	}
}

// TestResolveGroup_ToolCall returns group from matching assistant to result.
func TestResolveGroup_ToolCall(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "go"),
		toolCallMsg("calling", "tc42"),
		toolResultMsg("result", "tc42"),
	}
	g := resolveGroup(history, 2)
	if g.start != 1 || g.end != 2 {
		t.Errorf("want {1,2}, got %+v", g)
	}
}

// TestResolveGroup_UnmatchedToolCall falls back to single-message group.
func TestResolveGroup_UnmatchedToolCall(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "hi"),
		toolResultMsg("orphan result", "unknown-id"),
	}
	g := resolveGroup(history, 1)
	if g.start != 1 || g.end != 1 {
		t.Errorf("want {1,1} for unmatched tool call, got %+v", g)
	}
}

// TestSelectTail_ToolPlumbingIsNeverNoise is the regression guard for a
// production 400. Assistant messages that make a tool call carry empty Content,
// so a run of them looked like a run of identical messages to the noise
// comparison. Collapsing one dropped the tool_calls it declared and orphaned the
// tool results that followed, and DeepSeek rejected the whole request:
// "Messages with role 'tool' must be a response to a preceding message with
// 'tool_calls'". Every turn failed until the message aged out of the window.
func TestSelectTail_ToolPlumbingIsNeverNoise(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "store three facts"),
		toolCallMsg("", "tc1"), // empty content — the shape that looked like noise
		toolResultMsg("stored fact one", "tc1"),
		toolCallMsg("", "tc2"),
		toolResultMsg("stored fact two", "tc2"),
		toolCallMsg("", "tc3"),
		toolResultMsg("stored fact three", "tc3"),
	}

	got := selectTailMsgs(history, 0, 0)

	if len(got) != len(history) {
		t.Fatalf("expected all %d messages retained, got %d — tool plumbing was collapsed", len(history), len(got))
	}
	// Every tool result must still have its declaring assistant present.
	declared := map[string]bool{}
	for _, m := range got {
		for _, tc := range m.ToolCalls {
			declared[tc.ID] = true
		}
	}
	for _, m := range got {
		if m.Role == "tool" && !declared[m.ToolCallID] {
			t.Errorf("tool result %q orphaned — its assistant turn was collapsed away", m.ToolCallID)
		}
	}
}

// TestSelectTail_IdenticalToolResultsBothKept is the mirror case: two calls to
// one tool can legitimately return the same text, and dropping the second
// breaks the assistant message that expects a result for it.
func TestSelectTail_IdenticalToolResultsBothKept(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "check twice"),
		toolCallMsg("", "a"),
		toolResultMsg("same output", "a"),
		toolCallMsg("", "b"),
		toolResultMsg("same output", "b"),
	}

	got := selectTailMsgs(history, 0, 0)

	results := 0
	for _, m := range got {
		if m.Role == "tool" {
			results++
		}
	}
	if results != 2 {
		t.Errorf("expected both identical tool results kept, got %d", results)
	}
}

// TestSelectTail_ConversationalNoiseStillCollapses confirms the narrowing did
// not disable noise collapse for what it is actually for.
func TestSelectTail_ConversationalNoiseStillCollapses(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "same thing"),
		msg("user", "same thing"),
		msg("assistant", "ok"),
	}
	got := selectTailMsgs(history, 0, 0)
	if len(got) != 2 {
		t.Errorf("expected the duplicate user message collapsed, got %d messages", len(got))
	}
}
