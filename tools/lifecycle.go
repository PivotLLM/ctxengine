// ctxengine
// License: MIT

package tools

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/PivotLLM/toolspec"

	"github.com/PivotLLM/ctxengine"
)

// The lifecycle tools (compact, info, clear) act on the host's live agent loop
// through the closures on Host rather than on the archive.

const compactDescription = "Trigger an immediate context compaction for the current session. " +
	"Use after completing a major task to summarise prior messages and free context window space. " +
	"The result returns the compaction report and the new summary now in your context; " +
	"pass an optional `message` to append a note to yourself (e.g. what to do next)."

func compactSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"message": map[string]any{
				"type":        "string",
				"description": "Optional note appended to the result — e.g. a reminder of what to do next after compaction.",
			},
		},
	}
}

// compact implements the session_compact tool.
func (h Host) compact(call *toolspec.ToolCall) (*toolspec.Result, error) {
	key, r := sessionKey(call)
	if r != nil {
		return r, nil
	}
	if h.Compact == nil {
		return errResult("compact function not configured"), nil
	}
	message, _ := call.Args["message"].(string)

	h.log("info", "session compaction requested by tool", map[string]any{"session": key})
	report, summary, err := h.Compact(call.Ctx, key)

	var b strings.Builder
	switch {
	case report != "":
		// The report already describes the outcome (attempts + final line).
		b.WriteString(report)
	case errors.Is(err, ctxengine.ErrNothingToCompress):
		b.WriteString("Session is already compact — nothing to summarize.")
	case err != nil:
		h.log("warn", "session compaction failed", map[string]any{"session": key, "error": err.Error()})
		return errResult(fmt.Sprintf("compaction failed: %v", err)), nil
	default:
		b.WriteString("Session compacted successfully.")
	}

	if s := strings.TrimSpace(summary); s != "" {
		b.WriteString("\n\n--- Current summary (now in your context) ---\n")
		b.WriteString(s)
	}
	if m := strings.TrimSpace(message); m != "" {
		b.WriteString("\n\n--- Your note ---\n")
		b.WriteString(m)
	}
	return textResult(b.String()), nil
}

const infoDescription = "Return metadata about the current session: session key, start time, channel, " +
	"message count, archive sequence range, and the seq range covered by the current summary. " +
	"Use to orient yourself after a context compression or to understand how much history is available."

func infoSchema() map[string]any {
	return map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	}
}

// info implements the session_info tool.
func (h Host) info(call *toolspec.ToolCall) (*toolspec.Result, error) {
	key, r := sessionKey(call)
	if r != nil {
		return r, nil
	}
	if h.SessionInfo == nil {
		return errResult("session info function not configured"), nil
	}
	info, err := h.SessionInfo(call.Ctx, key)
	if err != nil {
		h.log("warn", "session info lookup failed", map[string]any{"session": key, "error": err.Error()})
		return errResult("session info error: " + err.Error()), nil
	}
	if info == nil {
		return errResult("session info error: no session information returned"), nil
	}
	out, _ := json.Marshal(info)
	return textResult(string(out)), nil
}

const clearDescription = "Clear your active conversation and start fresh. Long-term memory is preserved — " +
	"past messages stay retrievable via session_messages / session_search and past summaries " +
	"via session_summary_get. Use between unrelated short tasks to reset working context. " +
	"Pass an optional `message` as a handoff note to your fresh self (e.g. the next task to do). " +
	"After calling this, stop and end your turn: the clear happens at the turn boundary and a new " +
	"turn begins with your handoff."

func clearSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"message": map[string]any{
				"type":        "string",
				"description": "Optional handoff note delivered to you on the fresh turn after the clear — e.g. the next task or relevant context.",
			},
		},
	}
}

// clear implements the session_clear tool. It clears the agent's active
// conversation (preserving the durable archive and summary log) and then hands
// the agent a fresh turn. An optional message is delivered to the agent as a
// self-authored handoff note on that fresh turn.
//
// The clear itself is deferred to a clean turn boundary by the host's agent
// loop; it never wipes history mid-turn. Off by default — enable only for
// agents intended to run autonomous task loops.
func (h Host) clear(call *toolspec.ToolCall) (*toolspec.Result, error) {
	key, r := sessionKey(call)
	if r != nil {
		return r, nil
	}
	if h.Clear == nil {
		return errResult("clear function not configured"), nil
	}
	message, _ := call.Args["message"].(string)
	if err := h.Clear(call.Ctx, key, message); err != nil {
		h.log("warn", "session clear refused", map[string]any{"session": key, "error": err.Error()})
		return errResult(err.Error()), nil
	}
	h.log("info", "session clear queued by tool", map[string]any{"session": key, "has_note": message != ""})
	text := "Context clear queued. End your turn now — your active conversation will be reset " +
		"and a new turn will begin"
	if message != "" {
		text += " with your handoff note."
	} else {
		text += "."
	}
	return textResult(text), nil
}
