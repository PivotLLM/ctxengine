/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"
	"github.com/PivotLLM/spawnllm/protocoltypes"
)

// TestEstimate_CountsReasoningContent is the regression guard for the estimator
// undercount: reasoning_content is replayed to the provider on every historical
// assistant message, and ignoring it understated reasoning-model sessions by
// roughly half.
func TestEstimate_CountsReasoningContent(t *testing.T) {
	plain := []spawnllm.Message{{Role: "assistant", Content: strings.Repeat("a", 400)}}
	withReasoning := []spawnllm.Message{{
		Role:             "assistant",
		Content:          strings.Repeat("a", 400),
		ReasoningContent: strings.Repeat("r", 800),
	}}

	base := estimateTokens(plain)
	got := estimateTokens(withReasoning)
	if want := base + 200; got != want {
		t.Errorf("reasoning content not counted: got %d, want %d (base %d + 800 runes/4)", got, want, base)
	}
}

// TestEstimate_CountsResponsesReasoning covers the OpenAI Responses variant,
// which is replayed as opaque items before the turn's function calls.
func TestEstimate_CountsResponsesReasoning(t *testing.T) {
	item := json.RawMessage(`"` + strings.Repeat("x", 398) + `"`) // 400 runes with quotes
	msgs := []spawnllm.Message{{Role: "assistant", ResponsesReasoning: []json.RawMessage{item}}}
	if got, want := estimateTokens(msgs), 100; got != want {
		t.Errorf("responses reasoning not counted: got %d, want %d", got, want)
	}
}

// TestEstimate_IgnoresSystemPartsAndAttachments pins the two fields that must
// NOT be counted. Attachments are archive-side metadata populated by the media
// path and never sent to the LLM. SystemParts is no longer populated by ClawEh
// at all — every wired adapter strips it — but the field still exists on the
// wire type, so this guards against a future caller reintroducing it as a
// mirror of Content and silently double-counting the system prompt.
func TestEstimate_IgnoresSystemPartsAndAttachments(t *testing.T) {
	body := strings.Repeat("s", 400)
	msgs := []spawnllm.Message{{
		Role:        "system",
		Content:     body, // what is actually sent
		SystemParts: []protocoltypes.ContentBlock{{Type: "text", Text: body}},
		Attachments: []protocoltypes.MessageAttachment{{Filename: strings.Repeat("f", 400), Size: 1}},
	}}
	if got, want := estimateTokens(msgs), 100; got != want {
		t.Errorf("got %d, want %d (Content only — neither SystemParts nor attachments reach the LLM)", got, want)
	}
}

// TestEstimate_MediaChargedPerItem verifies media costs a flat per-item figure
// rather than the rune length of its data: URI. A 1MB base64 image would
// otherwise estimate at ~250k tokens, dwarfing the window on its own.
func TestEstimate_MediaChargedPerItem(t *testing.T) {
	huge := "data:image/png;base64," + strings.Repeat("Q", 1<<20)
	msgs := []spawnllm.Message{{Role: "user", Media: []string{huge, huge}}}
	if got, want := estimateTokens(msgs), 2*mediaTokensPerItem; got != want {
		t.Errorf("got %d, want %d (flat per-item, not %d runes)", got, want, 2*len(huge))
	}
}

// TestEstimate_CountsToolCallArguments guards the pre-existing behaviour that
// makes writer-argument eviction worth doing: a file_write payload is counted in
// full even though it never appears in Content.
func TestEstimate_CountsToolCallArguments(t *testing.T) {
	payload := strings.Repeat("w", 4000)
	msgs := []spawnllm.Message{{
		Role: "assistant",
		ToolCalls: []protocoltypes.ToolCall{{
			ID:       "tc1",
			Function: &protocoltypes.FunctionCall{Name: "file_write", Arguments: `{"content":"` + payload + `"}`},
		}},
	}}
	if got := estimateTokens(msgs); got < 1000 {
		t.Errorf("tool-call arguments undercounted: got %d tokens for a %d-byte payload", got, len(payload))
	}
}

// TestEstimateToolDefinitionTokens covers the tool-schema cost the Manager
// cannot see for itself, including the empty case.
func TestEstimateToolDefinitionTokens(t *testing.T) {
	if got := EstimateToolDefinitionTokens(nil); got != 0 {
		t.Errorf("no tools should cost 0 tokens, got %d", got)
	}
	defs := []spawnllm.ToolDefinition{{
		Type: "function",
		Function: protocoltypes.ToolFunctionDefinition{
			Name:        "file_write",
			Description: strings.Repeat("d", 400),
		},
	}}
	if got := EstimateToolDefinitionTokens(defs); got < 100 {
		t.Errorf("tool definitions undercounted: got %d", got)
	}
}
