// ctxengine
// License: MIT

package tools

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/PivotLLM/toolspec"

	"github.com/PivotLLM/ctxengine/internal/iox"
)

const searchDescription = "Search archived session messages using FTS5 full-text search. " +
	"FTS5 syntax examples: cat AND dog, \"exact phrase\", cat OR dog, NOT cat, cat*, NEAR(cat dog, 5). " +
	"Default 20 results, maximum 100. " +
	"Role filter values: \"user\", \"assistant\", \"tool\". Omit for all roles. " +
	"Use when you need to find past messages by topic rather than by sequence number. " +
	"Maximum query length: 500 characters."

func searchSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{
				"type":        "string",
				"description": "FTS5 search expression. Supports AND, OR, NOT, phrase queries (\"exact phrase\"), prefix (word*), and NEAR(cat dog, 5). Maximum 500 characters.",
			},
			"role": map[string]any{
				"type":        "string",
				"description": "Filter results by role: \"user\", \"assistant\", or \"tool\". Omit to return all roles.",
				"enum":        []string{"user", "assistant", "tool"},
			},
			"limit": map[string]any{
				"type":        "integer",
				"description": "Maximum number of results to return. Default 20, maximum 100.",
			},
		},
		"required": []string{"query"},
	}
}

// search implements the session_search tool: FTS5 full-text search over the
// session archive.
func (h Host) search(call *toolspec.ToolCall) (*toolspec.Result, error) {
	if _, r := sessionKey(call); r != nil {
		return r, nil
	}

	query, ok := call.Args["query"].(string)
	if !ok || strings.TrimSpace(query) == "" {
		return errResult("query parameter is required"), nil
	}

	role := stringArg(call.Args, "role")

	limit := 20
	if l, ok := intArg(call.Args, "limit"); ok {
		limit = int(l)
	}
	if limit < 1 {
		limit = 1
	}
	if limit > 100 {
		limit = 100
	}

	a, key, r := h.openArchive(call)
	if r != nil {
		return r, nil
	}
	defer iox.CloseQuietly("tools", a)

	results, searchErr := a.Search(call.Ctx, query, role, limit)
	if searchErr != nil {
		return h.archiveError("search error", key, searchErr), nil
	}

	if len(results) == 0 {
		return textResult("no matching messages found"), nil
	}

	type msgEntry struct {
		Seq       int64     `json:"seq"`
		Role      string    `json:"role"`
		Source    string    `json:"source,omitempty"`
		Content   string    `json:"content"`
		CreatedAt time.Time `json:"created_at"`
	}
	entries := make([]msgEntry, len(results))
	for i, res := range results {
		entries[i] = msgEntry{
			Seq:       res.Seq,
			Role:      res.Message.Role,
			Source:    res.Message.Source,
			Content:   res.Message.Content,
			CreatedAt: res.CreatedAt,
		}
	}
	out, err := json.Marshal(entries)
	if err != nil {
		return errResult("encode error: " + err.Error()), nil
	}
	return textResult(string(out)), nil
}
