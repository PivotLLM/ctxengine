// ctxengine
// License: MIT

package ctxengine

import (
	"github.com/PivotLLM/ctxengine/logger"
	"github.com/PivotLLM/spawnllm"
)

// sanitizeHistoryForProvider drops what a strict provider would reject from a
// stored history before it is sent. The unit of validity is the tool group: an
// assistant turn that makes tool calls, followed by the results that answer
// them. A group is kept whole when every call is answered and the turn follows
// a user or tool message; otherwise the turn and its results go together. A
// result that answers no call in its group, or that sits outside any group, is
// dropped on its own, and stored system messages are always dropped because
// build composes the single system message itself.
//
// Compaction and eviction can leave any of these shapes behind (a boundary
// that cuts a group in half, a collapsed turn whose results remain), and
// tolerant providers accept them silently, so a history can be malformed for a
// long time before a strict one answers 400 on every turn. Sanitising on every
// dispatch means such a session recovers by itself.
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
			g := collectToolGroup(history[i:])
			i += g.span
			drops.strayResult += g.strays
			switch {
			case len(out) == 0 || (out[len(out)-1].Role != "user" && out[len(out)-1].Role != "tool"):
				drops.badPredecessor++
			case !g.complete:
				drops.incompleteGroup++
			default:
				out = append(out, g.msgs...)
			}
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
	msgs     []spawnllm.Message // the turn, then the results that answer one of its calls
	span     int                // history entries the group covers, strays included
	strays   int                // results in the group that answer none of its calls
	complete bool               // every call has at least one result
}

// collectToolGroup gathers the group starting at history[0], which must be an
// assistant turn with tool calls. The group extends over the contiguous tool
// results that follow it; the first non-result ends it.
func collectToolGroup(history []spawnllm.Message) toolGroup {
	turn := history[0]
	answered := make(map[string]bool, len(turn.ToolCalls))
	for _, tc := range turn.ToolCalls {
		answered[tc.ID] = false
	}

	g := toolGroup{msgs: []spawnllm.Message{turn}, span: 1}
	for _, m := range history[1:] {
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

	g.complete = true
	for _, ok := range answered {
		if !ok {
			g.complete = false
			break
		}
	}
	return g
}

// sanitizeDrops counts what was removed, by reason, so one summary line is
// logged per dispatch: a single compaction boundary can orphan several
// leading turns, and a line per message would repeat on every dispatch.
type sanitizeDrops struct {
	system, orphanResult, strayResult, badPredecessor, incompleteGroup int
}

func (d sanitizeDrops) log(kept int) {
	n := d.system + d.orphanResult + d.strayResult + d.badPredecessor + d.incompleteGroup
	if n == 0 {
		return
	}
	logger.DebugCF("llmcontext", "sanitized history for provider", map[string]any{
		"dropped_total":         n,
		"system":                d.system,
		"orphan_result":         d.orphanResult,
		"stray_result":          d.strayResult,
		"bad_predecessor":       d.badPredecessor,
		"incomplete_tool_group": d.incompleteGroup,
		"kept":                  kept,
	})
}
