// ctxengine
// License: MIT

package ctxengine

import (
	"github.com/PivotLLM/ctxengine/logger"
	"github.com/PivotLLM/spawnllm"
)

// sanitizeHistoryForProvider drops the messages a strict provider would reject
// from a stored history before it is sent: stale system messages, tool results
// that answer no visible tool call, tool-call turns without a valid predecessor,
// and tool-call turns whose results are incomplete.
func sanitizeHistoryForProvider(history []spawnllm.Message) []spawnllm.Message {
	if len(history) == 0 {
		return history
	}

	// Drop reasons are counted and logged once at the end rather than per message,
	// because a single post-compaction boundary can orphan several leading
	// tool-call turns and would otherwise spam one DBG line each, every dispatch.
	var dropSystem, dropLeadingTool, dropOrphanTool, dropAsstStart, dropAsstBadPred, dropIncompleteGroup int

	sanitized := make([]spawnllm.Message, 0, len(history))
	for _, msg := range history {
		switch msg.Role {
		case "system":
			// Drop system messages from history. build always constructs
			// its own single system message (layers + summary + injections);
			// extra system messages would break providers that only accept
			// one (Anthropic, Codex).
			dropSystem++
			continue

		case "tool":
			if len(sanitized) == 0 {
				dropLeadingTool++
				continue
			}
			// Walk backwards to the nearest assistant message, skipping over any
			// preceding tool results (the parallel-tool-call case), and require
			// that THIS result answers one of the calls that assistant actually
			// declared.
			//
			// Matching the id matters, not merely finding an assistant that made
			// some call: a result whose id belongs to a dropped assistant turn
			// would otherwise be accepted on the strength of an unrelated
			// neighbour, and strict providers reject it — DeepSeek answers 400
			// with "Messages with role 'tool' must be a response to a preceding
			// message with 'tool_calls'", which kills every turn until the
			// message ages out of the window.
			open := map[string]bool{}
			for i := len(sanitized) - 1; i >= 0; i-- {
				if sanitized[i].Role == "tool" {
					continue
				}
				if sanitized[i].Role == "assistant" {
					for _, tc := range sanitized[i].ToolCalls {
						open[tc.ID] = true
					}
				}
				break
			}
			if !open[msg.ToolCallID] {
				dropOrphanTool++
				continue
			}
			sanitized = append(sanitized, msg)

		case "assistant":
			if len(msg.ToolCalls) > 0 {
				if len(sanitized) == 0 {
					dropAsstStart++
					continue
				}
				prev := sanitized[len(sanitized)-1]
				if prev.Role != "user" && prev.Role != "tool" {
					dropAsstBadPred++
					continue
				}
			}
			sanitized = append(sanitized, msg)

		default:
			sanitized = append(sanitized, msg)
		}
	}

	// Second pass: ensure every assistant message with tool_calls has matching
	// tool result messages following it. This is required by strict providers
	// like DeepSeek that enforce: "An assistant message with 'tool_calls' must
	// be followed by tool messages responding to each 'tool_call_id'."
	final := make([]spawnllm.Message, 0, len(sanitized))
	for i := 0; i < len(sanitized); i++ {
		msg := sanitized[i]
		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			// Collect expected tool_call IDs
			expected := make(map[string]bool, len(msg.ToolCalls))
			for _, tc := range msg.ToolCalls {
				expected[tc.ID] = false
			}

			// Check following messages for matching tool results
			toolMsgCount := 0
			for j := i + 1; j < len(sanitized); j++ {
				if sanitized[j].Role != "tool" {
					break
				}
				toolMsgCount++
				if _, exists := expected[sanitized[j].ToolCallID]; exists {
					expected[sanitized[j].ToolCallID] = true
				}
			}

			// If any tool_call_id is missing, drop this assistant message and its partial tool messages
			allFound := true
			for _, found := range expected {
				if !found {
					allFound = false
					dropIncompleteGroup++
					break
				}
			}

			if !allFound {
				// Skip this assistant message and its tool messages
				i += toolMsgCount
				continue
			}
		}
		final = append(final, msg)
	}

	if n := dropSystem + dropLeadingTool + dropOrphanTool + dropAsstStart + dropAsstBadPred + dropIncompleteGroup; n > 0 {
		logger.DebugCF("llmcontext", "sanitized history for provider", map[string]any{
			"dropped_total":         n,
			"system":                dropSystem,
			"leading_tool_orphans":  dropLeadingTool,
			"orphan_tool":           dropOrphanTool,
			"assistant_at_start":    dropAsstStart,
			"assistant_bad_pred":    dropAsstBadPred,
			"incomplete_tool_group": dropIncompleteGroup,
			"kept":                  len(final),
		})
	}

	return final
}
