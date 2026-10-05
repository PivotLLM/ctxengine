/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

import (
	"testing"

	"github.com/PivotLLM/spawnllm"
)

func asstCall(ids ...string) spawnllm.Message {
	m := spawnllm.Message{Role: "assistant"}
	for _, id := range ids {
		m.ToolCalls = append(m.ToolCalls, spawnllm.ToolCall{ID: id})
	}
	return m
}

func orphanToolResult(id, content string) spawnllm.Message {
	return spawnllm.Message{Role: "tool", ToolCallID: id, Content: content}
}

// TestSanitize_DropsToolResultWithUnmatchedID is the production failure this
// fixes. A tool result whose owning assistant turn is gone used to be accepted
// on the strength of an unrelated neighbour that happened to make some call, and
// DeepSeek answered 400 — "Messages with role 'tool' must be a response to a
// preceding message with 'tool_calls'" — on every turn until it aged out.
func TestSanitize_DropsToolResultWithUnmatchedID(t *testing.T) {
	out := sanitizeHistoryForProvider([]spawnllm.Message{
		{Role: "user", Content: "go"},
		asstCall("keep-me"),
		orphanToolResult("keep-me", "ok"),
		orphanToolResult("orphan", "result of a dropped assistant turn"),
	})

	for _, m := range out {
		if m.Role == "tool" && m.ToolCallID == "orphan" {
			t.Fatal("orphaned tool result survived — strict providers reject the whole request")
		}
	}
	var kept bool
	for _, m := range out {
		if m.Role == "tool" && m.ToolCallID == "keep-me" {
			kept = true
		}
	}
	if !kept {
		t.Error("the matching tool result must survive")
	}
}

// TestSanitize_KeepsParallelToolResults guards the case the id check could
// easily break: one assistant turn making several calls, answered by a run of
// tool results. All of them must survive.
func TestSanitize_KeepsParallelToolResults(t *testing.T) {
	out := sanitizeHistoryForProvider([]spawnllm.Message{
		{Role: "user", Content: "go"},
		asstCall("a", "b", "c"),
		orphanToolResult("a", "ra"),
		orphanToolResult("b", "rb"),
		orphanToolResult("c", "rc"),
	})

	got := map[string]bool{}
	for _, m := range out {
		if m.Role == "tool" {
			got[m.ToolCallID] = true
		}
	}
	for _, id := range []string{"a", "b", "c"} {
		if !got[id] {
			t.Errorf("parallel tool result %q was dropped", id)
		}
	}
}

// TestSanitize_ConsecutiveToolCallTurns covers back-to-back assistant tool-call
// turns — the shape that produced the production orphans. Each result must be
// matched against its OWN assistant turn, not merely the nearest one.
func TestSanitize_ConsecutiveToolCallTurns(t *testing.T) {
	out := sanitizeHistoryForProvider([]spawnllm.Message{
		{Role: "user", Content: "go"},
		asstCall("one"),
		orphanToolResult("one", "r1"),
		asstCall("two"),
		orphanToolResult("two", "r2"),
	})

	tools := 0
	for _, m := range out {
		if m.Role == "tool" {
			tools++
		}
	}
	if tools != 2 {
		t.Errorf("expected both tool results kept, got %d", tools)
	}
}

// TestSanitize_RecoversWendysBrokenHistory reproduces the exact shape found in
// the production session that 400'd on every turn: consecutive assistant
// tool-call turns where the middle ones were collapsed away, leaving their
// results behind.
//
// Two fixes cover this — noise collapse no longer eats tool plumbing, so it
// stops happening; and this sanitiser drops what is already broken, so a session
// corrupted before the fix recovers on the next dispatch instead of needing a
// /clear. That second half matters: other providers accept orphans, so a history
// can be quietly malformed for a long time before a strict one rejects it.
func TestSanitize_RecoversWendysBrokenHistory(t *testing.T) {
	out := sanitizeHistoryForProvider([]spawnllm.Message{
		{Role: "user", Content: "capture these facts"},
		asstCall("VmGcXf"),
		orphanToolResult("VmGcXf", "Stored memory h77B5H"),
		// seqs 10884 and 10886 (the owning assistant turns) are gone
		orphanToolResult("quHJrX", "Stored memory hMJVAM"),
		orphanToolResult("ETRQGe", "Updated domain dB5YQ8."),
		{Role: "user", Content: "carry on"},
	})

	declared := map[string]bool{}
	for _, m := range out {
		for _, tc := range m.ToolCalls {
			declared[tc.ID] = true
		}
	}
	for _, m := range out {
		if m.Role == "tool" && !declared[m.ToolCallID] {
			t.Errorf("orphan %q survived sanitisation — DeepSeek would reject the request", m.ToolCallID)
		}
	}
	// The healthy pair and the surrounding conversation must be untouched.
	if len(out) != 4 {
		t.Errorf("expected user + assistant + its result + user = 4 messages, got %d", len(out))
	}
	if out[len(out)-1].Content != "carry on" {
		t.Errorf("conversation after the orphans was disturbed: %+v", out)
	}
}
