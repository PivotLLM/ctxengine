// ctxengine
// License: MIT
//
// Copyright (c) 2026 Tenebris Technologies Inc.

// Package sessiontools exposes the session history and lifecycle tools as
// toolspec definitions with BARE names ("messages", "search", "compact",
// "info", "summary_list", "summary_get", "clear"). A host mounts them under its
// own namespace (ClawEh publishes "session_messages" and so on).
//
// The archive-backed tools (messages, search, summary_list, summary_get) read
// the per-session archive at <Host.SessionsDir>/<key>.archive.db directly. The
// lifecycle tools (compact, info, clear) call back into the host through the
// closures on Host, because they act on the host's live agent loop.
//
// Every tool is session-scoped: the handlers read the caller's session key
// from toolspec.ToolCall.Session and refuse to run without one.
package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/PivotLLM/toolspec"

	"github.com/PivotLLM/ctxengine/memory"
)

// SessionInfo holds the structured data returned by the session_info tool.
type SessionInfo struct {
	Server              string           `json:"server,omitempty"` // e.g. "ClawEh 0.4.8"
	OS                  string           `json:"os,omitempty"`     // e.g. "linux/amd64"
	SessionKey          string           `json:"session_key"`
	StartedAt           *time.Time       `json:"started_at,omitempty"`
	Channel             string           `json:"channel,omitempty"`
	ContextMessageCount int              `json:"context_message_count"`
	ArchiveMinSeq       int64            `json:"archive_min_seq"`
	ArchiveMaxSeq       int64            `json:"archive_max_seq"`
	TotalArchived       int64            `json:"total_archived"`
	SummaryCovers       *SummaryCoverage `json:"summary_covers,omitempty"`
	LastCompressedAt    *time.Time       `json:"last_compressed_at,omitempty"`
}

// SummaryCoverage describes the seq range covered by the current summary.
type SummaryCoverage struct {
	SeqStart    int64      `json:"seq_start"`
	SeqEnd      int64      `json:"seq_end"`
	GeneratedAt *time.Time `json:"generated_at,omitempty"`
}

// Host is what the session tools need from the application embedding them.
type Host struct {
	// SessionsDir is the directory holding the <key>.archive.db files. Empty
	// (as during a deps-free catalogue enumeration) means the archive-backed
	// tools are unavailable and refuse to touch disk.
	SessionsDir string
	// Compact triggers an immediate context compaction for a session and
	// returns the compaction report and the resulting rendered summary. Nil
	// means the compact tool is unavailable.
	Compact func(ctx context.Context, sessionKey string) (report, summary string, err error)
	// Clear queues a clear of the session's active conversation, delivering
	// message as a handoff note on the fresh turn. Nil means the clear tool is
	// unavailable.
	Clear func(ctx context.Context, sessionKey, message string) error
	// SessionInfo reports the live state of a session. Nil means the info tool
	// is unavailable.
	SessionInfo func(ctx context.Context, sessionKey string) (*SessionInfo, error)
	// Log receives diagnostics; level is one of "debug", "info", "warn",
	// "error". Nil means silent.
	Log func(level, message string, fields map[string]any)
}

// Definitions returns the session tool definitions bound to h.
func Definitions(h Host) []toolspec.ToolDefinition {
	def := func(name, desc string, schema map[string]any, allow *bool, handler toolspec.ToolHandler) toolspec.ToolDefinition {
		return toolspec.ToolDefinition{
			Name:          name,
			Description:   desc,
			RawSchema:     schema,
			SessionScoped: true,
			DefaultAllow:  allow,
			Category:      "context",
			Handler:       handler,
		}
	}

	return []toolspec.ToolDefinition{
		def("messages", messagesDescription, messagesSchema(), toolspec.Allow(true), h.messages),
		def("search", searchDescription, searchSchema(), toolspec.Allow(true), h.search),
		def("summary_list", summaryListDescription, summaryListSchema(), toolspec.Allow(true), h.summaryList),
		def("summary_get", summaryGetDescription, summaryGetSchema(), toolspec.Allow(true), h.summaryGet),
		def("compact", compactDescription, compactSchema(), toolspec.Allow(true), h.compact),
		def("info", infoDescription, infoSchema(), toolspec.Allow(true), h.info),
		// Denied by default — opt-in only.
		def("clear", clearDescription, clearSchema(), nil, h.clear),
	}
}

func (h Host) log(level, message string, fields map[string]any) {
	if h.Log != nil {
		h.Log(level, message, fields)
	}
}

func errResult(message string) *toolspec.Result {
	return &toolspec.Result{IsError: true, ForLLM: message}
}

func textResult(text string) *toolspec.Result {
	return &toolspec.Result{ForLLM: text}
}

// sessionKey returns the caller's session key, or an error Result when the
// call carries none.
func sessionKey(call *toolspec.ToolCall) (string, *toolspec.Result) {
	if call.Session == "" {
		return "", errResult("session key not available")
	}
	return call.Session, nil
}

const archiveUnavailableText = "archive unavailable — see server logs"

// openArchive opens the caller's session archive read-only. A non-nil Result
// is the tool's answer: the host has no sessions directory, the call has no
// session key, or the archive could not be opened.
func (h Host) openArchive(call *toolspec.ToolCall) (*memory.ArchiveStore, string, *toolspec.Result) {
	if h.SessionsDir == "" {
		return nil, "", errResult("session archive tools are not available (no sessions directory configured)")
	}
	key, r := sessionKey(call)
	if r != nil {
		return nil, "", r
	}
	a, err := memory.OpenReadOnly(memory.ArchivePath(h.SessionsDir, key))
	if err != nil {
		return nil, "", h.archiveError("archive open error", key, err)
	}
	return a, key, nil
}

// archiveError maps an archive failure to the tool result. An unavailable
// archive is reported as plain text pointing at the server logs (the memory
// package has already logged the cause); anything else is an error result
// prefixed with the failing stage.
func (h Host) archiveError(stage, key string, err error) *toolspec.Result {
	if errors.Is(err, memory.ErrArchiveUnavailable) {
		h.log("debug", "session archive unavailable", map[string]any{"session": key, "stage": stage})
		return textResult(archiveUnavailableText)
	}
	h.log("warn", "session archive error", map[string]any{"session": key, "stage": stage, "error": err.Error()})
	return errResult(fmt.Sprintf("%s: %v", stage, err))
}

func intArg(args map[string]any, key string) (int64, bool) {
	v, ok := args[key]
	if !ok || v == nil {
		return 0, false
	}
	switch n := v.(type) {
	case int:
		return int64(n), true
	case int64:
		return n, true
	case float64:
		return int64(n), true
	case json.Number:
		i, e := n.Int64()
		return i, e == nil
	}
	return 0, false
}
