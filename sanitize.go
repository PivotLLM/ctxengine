/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

import (
	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine/logger"
)

// interruptedToolResult is the content of the result the sanitiser synthesises
// for a tool call that has no stored result. It tells the model the call was
// issued and may well have run, so it does not repeat a side-effecting call
// on the strength of a missing answer.
const interruptedToolResult = "[interrupted — the gateway restarted before this tool call finished; " +
	"outcome unknown. Do not assume it did not run.]"

// sanitizeHistoryForProvider repairs what a strict provider would reject from
// a stored history before it is sent. The unit of validity is the tool group:
// an assistant turn that makes tool calls, followed by the results that answer
// them. A group is kept when the turn follows a user or tool message; a call
// in it with no result gets a synthesised interruptedToolResult so the group
// is complete and the model knows the call was made. A result that answers no
// call in its group, or that sits outside any group, is dropped on its own; a
// group whose turn follows a plain assistant message is dropped whole; and
// stored system messages are always dropped because build composes the single
// system message itself.
//
// An unanswered call is what a crash between the assistant write and the
// tool-result writes leaves behind. Dropping the group would hide from the
// model that the tools already ran, and it would run them again — the very
// thing to avoid for a side-effecting tool — so the group stays and the gap
// is labelled instead. The other shapes come from compaction and eviction (a
// boundary that cuts a group in half, a collapsed turn whose results remain);
// tolerant providers accept them silently, so a history can be malformed for
// a long time before a strict one answers 400 on every turn. Sanitising on
// every dispatch means such a session recovers by itself. The result is a new
// slice; the stored history and its seqs are never touched.
func sanitizeHistoryForProvider(history []spawnllm.Message) []spawnllm.Message {
	if len(history) == 0 {
		return history
	}

	var drops sanitizeDrops
	out := make([]spawnllm.Message, 0, len(history))
	for i := 0; i < len(history); {
		msg := history[i]
		switch {
		case msg.Role == "system":
			drops.system++
			i++
		case msg.Role == "tool":
			// Not consumed by a group above, so nothing it could answer is
			// visible to the provider.
			drops.orphanResult++
			i++
		case msg.Role == "assistant" && len(msg.ToolCalls) > 0:
			g := collectToolGroup(msg, history[i+1:])
			i += g.span
			drops.strayResult += g.strays
			if len(out) == 0 || (out[len(out)-1].Role != "user" && out[len(out)-1].Role != "tool") {
				drops.badPredecessor++
				continue
			}
			out = append(out, g.msgs...)
			for _, id := range g.unanswered {
				out = append(out, spawnllm.Message{Role: "tool", ToolCallID: id, Content: interruptedToolResult})
			}
			drops.synthesized += len(g.unanswered)
		default:
			out = append(out, msg)
			i++
		}
	}

	drops.log(len(out))
	return out
}

// toolGroup is one assistant tool-call turn with the results that answer it.
type toolGroup struct {
	msgs       []spawnllm.Message // the turn, then the results that answer one of its calls
	span       int                // history entries the group covers, strays included
	strays     int                // results in the group that answer none of its calls
	unanswered []string           // call IDs with no result, in the order the turn made them
}

// collectToolGroup gathers the group started by turn, which must be an
// assistant turn with tool calls; following holds the history entries after
// it. The group extends over the contiguous tool results at the head of
// following; the first non-result ends it.
func collectToolGroup(turn spawnllm.Message, following []spawnllm.Message) toolGroup {
	answered := make(map[string]bool, len(turn.ToolCalls))
	for _, tc := range turn.ToolCalls {
		answered[tc.ID] = false
	}

	g := toolGroup{msgs: []spawnllm.Message{turn}, span: 1}
	for _, m := range following {
		if m.Role != "tool" {
			break
		}
		g.span++
		if _, declared := answered[m.ToolCallID]; !declared {
			g.strays++
			continue
		}
		answered[m.ToolCallID] = true
		g.msgs = append(g.msgs, m)
	}

	for _, tc := range turn.ToolCalls {
		if !answered[tc.ID] {
			g.unanswered = append(g.unanswered, tc.ID)
			answered[tc.ID] = true // a turn that repeats an ID gets one result for it
		}
	}
	return g
}

// sanitizeDrops counts what was removed, by reason, and what was synthesised,
// so one summary line is logged per dispatch: a single compaction boundary can
// orphan several leading turns, and a line per message would repeat on every
// dispatch.
type sanitizeDrops struct {
	system, orphanResult, strayResult, badPredecessor int
	synthesized                                       int // interrupted results added to complete a group
}

func (d sanitizeDrops) log(kept int) {
	n := d.system + d.orphanResult + d.strayResult + d.badPredecessor
	if n == 0 && d.synthesized == 0 {
		return
	}
	logger.DebugCF("llmcontext", "sanitized history for provider", map[string]any{
		"dropped_total":       n,
		"system":              d.system,
		"orphan_result":       d.orphanResult,
		"stray_result":        d.strayResult,
		"bad_predecessor":     d.badPredecessor,
		"synthesized_results": d.synthesized,
		"kept":                kept,
	})
}
