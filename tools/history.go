// ctxengine
// License: MIT

package tools

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/PivotLLM/toolspec"

	"github.com/PivotLLM/ctxengine/internal/iox"
)

const messagesDescription = "Retrieve historical messages from the current session archive by sequence number. " +
	"Returns messages in the requested seq range, capped to the most recent 5000. " +
	"Request smaller ranges for efficiency. " +
	"Use when the context summary references a seq number and you need the full message content."

func messagesSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"seq": map[string]any{
				"type":        "integer",
				"description": "Single message sequence number. Takes precedence over seq_start/seq_end.",
			},
			"seq_start": map[string]any{
				"type":        "integer",
				"description": "Start of sequence range (inclusive). Used when seq is absent.",
			},
			"seq_end": map[string]any{
				"type":        "integer",
				"description": "End of sequence range (inclusive). Used when seq is absent.",
			},
		},
	}
}

// messages implements the session_messages tool: it reads messages from the
// session archive by sequence number.
func (h Host) messages(call *toolspec.ToolCall) (*toolspec.Result, error) {
	seqStart, seqEnd, err := parseSeqArgs(call.Args)
	if err != nil {
		return errResult(err.Error()), nil
	}

	a, key, r := h.openArchive(call)
	if r != nil {
		return r, nil
	}
	defer iox.CloseQuietly("tools", a)

	const windowSize = 5000
	_, maxSeq, boundsErr := a.Bounds()
	if boundsErr != nil {
		return h.archiveError("archive bounds error", key, boundsErr), nil
	}

	if maxSeq == 0 {
		return textResult("not available in the current archive window"), nil
	}

	effectiveMin := seqStart
	if floor := maxSeq - windowSize + 1; floor > effectiveMin {
		effectiveMin = floor
	}
	effectiveMax := min(maxSeq, seqEnd)

	msgs, readErr := a.QueryRange(effectiveMin, effectiveMax)
	if readErr != nil {
		return h.archiveError("archive read error", key, readErr), nil
	}

	if len(msgs) == 0 {
		return textResult("not available in the current archive window"), nil
	}

	type toolResultEntry struct {
		seq     int64
		content string
		status  string
	}
	toolResults := make(map[string]toolResultEntry)
	for _, m := range msgs {
		if m.Role == "tool" && m.ToolCallID != "" {
			toolResults[m.ToolCallID] = toolResultEntry{
				seq:     m.Seq,
				content: m.Content,
				status:  "success",
			}
		}
	}

	type toolCallEntry struct {
		Name   string `json:"name"`
		Input  string `json:"input"`
		Output string `json:"output,omitempty"`
		Status string `json:"status"`
		Seq    int64  `json:"result_seq,omitempty"`
	}
	type msgEntry struct {
		Seq       int64           `json:"seq"`
		Role      string          `json:"role"`
		Source    string          `json:"source,omitempty"`
		Content   string          `json:"content"`
		CreatedAt time.Time       `json:"created_at"`
		ToolCalls []toolCallEntry `json:"tool_calls,omitempty"`
	}
	entries := make([]msgEntry, len(msgs))
	for i, m := range msgs {
		e := msgEntry{
			Seq:       m.Seq,
			Role:      m.Role,
			Source:    m.Source,
			Content:   m.Content,
			CreatedAt: m.CreatedAt,
		}
		for _, tc := range m.ToolCalls {
			tce := toolCallEntry{
				Name:   tc.Name,
				Input:  "",
				Status: "pending",
			}
			if tc.Function != nil {
				tce.Input = tc.Function.Arguments
			}
			if res, ok := toolResults[tc.ID]; ok {
				tce.Output = res.content
				tce.Status = res.status
				tce.Seq = res.seq
			}
			e.ToolCalls = append(e.ToolCalls, tce)
		}
		entries[i] = e
	}
	out, err := json.Marshal(entries)
	if err != nil {
		return errResult("encode error: " + err.Error()), nil
	}
	return textResult(string(out)), nil
}

// parseSeqArgs returns the seq range from args.
func parseSeqArgs(args map[string]any) (seqStart, seqEnd int64, err error) {
	if seq, ok := intArg(args, "seq"); ok {
		return seq, seq, nil
	}
	start, hasStart := intArg(args, "seq_start")
	end, hasEnd := intArg(args, "seq_end")
	if !hasStart || !hasEnd {
		return 0, 0, errors.New("seq or seq_start+seq_end required")
	}
	if start > end {
		return 0, 0, fmt.Errorf("seq_start (%d) > seq_end (%d)", start, end)
	}
	return start, end, nil
}
