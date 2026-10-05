/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package tools

import (
	"fmt"
	"strings"

	"github.com/PivotLLM/toolspec"

	"github.com/PivotLLM/ctxengine"
	"github.com/PivotLLM/ctxengine/internal/iox"
	"github.com/PivotLLM/ctxengine/memory"
)

const summaryListDescription = "List the context-summary checkpoints recorded for the current session. " +
	"Each entry shows its id, the covered message sequence range, when it was generated, " +
	"and the model that produced it. Use session_summary_get with an id to read the full summary. " +
	"Newest checkpoints have the highest ids."

func summaryListSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"limit": map[string]any{
				"type":        "integer",
				"description": "Maximum number of checkpoints to return (most recent first). Omit for all.",
			},
			"offset": map[string]any{
				"type":        "integer",
				"description": "Number of most-recent checkpoints to skip before listing. Default 0.",
			},
		},
	}
}

// summaryList implements the session_summary_list tool: it lists the
// context-summary checkpoints stored in the session archive.
func (h Host) summaryList(call *toolspec.ToolCall) (*toolspec.Result, error) {
	a, key, r := h.openArchive(call)
	if r != nil {
		return r, nil
	}
	defer iox.CloseQuietly("tools", a)

	metas, listErr := a.ListSummaries()
	if listErr != nil {
		return h.archiveError("summary list error", key, listErr), nil
	}

	if len(metas) == 0 {
		return textResult("no context summaries recorded for this session yet"), nil
	}

	// Present newest first for readability.
	reversed := make([]memory.SummaryMeta, len(metas))
	for i, m := range metas {
		reversed[len(metas)-1-i] = m
	}

	offset := 0
	if o, ok := intArg(call.Args, "offset"); ok && o > 0 {
		offset = int(o)
	}
	if offset >= len(reversed) {
		return textResult("no context summaries in the requested range"), nil
	}
	reversed = reversed[offset:]

	if l, ok := intArg(call.Args, "limit"); ok && l > 0 && int(l) < len(reversed) {
		reversed = reversed[:l]
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "%d context summary checkpoint(s) (newest first):\n", len(reversed))
	for _, m := range reversed {
		fmt.Fprintf(&sb, "- id %d: covers #%d-#%d, generated %s",
			m.ID, m.CoveredSeqStart, m.CoveredSeqEnd,
			m.GeneratedAt.Format("2006-01-02 15:04 UTC"))
		if m.Model != "" {
			fmt.Fprintf(&sb, ", model %s", m.Model)
		}
		if m.Profile != "" {
			fmt.Fprintf(&sb, ", profile %s", m.Profile)
		}
		sb.WriteString("\n")
	}
	return textResult(strings.TrimRight(sb.String(), "\n")), nil
}

const summaryGetDescription = "Retrieve the full text of one context-summary checkpoint for the current session by id. " +
	"Use session_summary_list to discover available ids. " +
	"Returns the summary rendered as Markdown (goals, progress, pending, constraints, key moments) " +
	"with the message sequence references it cites."

func summaryGetSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"id": map[string]any{
				"type":        "integer",
				"description": "The checkpoint id to retrieve (from session_summary_list).",
			},
		},
		"required": []string{"id"},
	}
}

// summaryGet implements the session_summary_get tool: it returns the full
// rendered body of one stored context-summary checkpoint.
func (h Host) summaryGet(call *toolspec.ToolCall) (*toolspec.Result, error) {
	if _, r := sessionKey(call); r != nil {
		return r, nil
	}

	id, ok := intArg(call.Args, "id")
	if !ok {
		return errResult("id parameter is required"), nil
	}

	a, key, r := h.openArchive(call)
	if r != nil {
		return r, nil
	}
	defer iox.CloseQuietly("tools", a)

	rec, found, getErr := a.GetSummary(id)
	if getErr != nil {
		return h.archiveError("summary get error", key, getErr), nil
	}
	if !found {
		return textResult(fmt.Sprintf("no context summary with id %d", id)), nil
	}

	// Render the stored JSON body as readable Markdown. The covered seq range
	// bounds the rendered references; an unparseable body falls back to raw text.
	rendered := ctxengine.RenderSummaryFromRaw(rec.Summary, rec.CoveredSeqStart, rec.CoveredSeqEnd)

	var sb strings.Builder
	fmt.Fprintf(&sb, "Context summary checkpoint id %d (covers #%d-#%d, generated %s",
		rec.ID, rec.CoveredSeqStart, rec.CoveredSeqEnd,
		rec.GeneratedAt.Format("2006-01-02 15:04 UTC"))
	if rec.Model != "" {
		fmt.Fprintf(&sb, ", model %s", rec.Model)
	}
	if rec.Profile != "" {
		fmt.Fprintf(&sb, ", profile %s", rec.Profile)
	}
	sb.WriteString(")\n\n")
	sb.WriteString(rendered)
	return textResult(sb.String()), nil
}
