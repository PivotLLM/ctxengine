/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package session

import (
	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine/memory"
)

// SessionStore defines the persistence operations the engine uses.
// SQLiteStore is the implementation; the interface lets tests substitute an
// in-memory store without touching the engine.
//
// Write methods return the store's error so a failed write is never taken for
// a successful one: the engine hands it back to the caller (Add*) or fails the
// pass that needed it (compaction, Reset). Read methods return the empty value
// on failure and log it.
type SessionStore interface { //nolint:interfacebloat // the engine's public storage contract; splitting it is a breaking change
	// AddMessage appends a simple role/content message to the session.
	AddMessage(sessionKey, role, content string) error
	// AddFullMessage appends a complete message including tool calls. It returns
	// the monotonic sequence number assigned to the written message so callers
	// can key durable side-stores (e.g. the archive) under the same seq, or 0
	// and the error when nothing was written.
	AddFullMessage(sessionKey string, msg spawnllm.Message) (int64, error)
	// GetHistory returns the full message history for the session.
	GetHistory(key string) []spawnllm.Message
	// GetHistoryWithSeqs returns the full message history with seq numbers intact.
	// Implementations that have no durable seq counter (an in-memory test store)
	// synthesize seq as i+1 for the i-th message.
	GetHistoryWithSeqs(key string) []memory.StoredMessage
	// GetSummary returns the conversation summary, or "" if none.
	GetSummary(key string) string
	// SetSummary replaces the conversation summary.
	SetSummary(key, summary string) error
	// SetHistory replaces the full message history.
	SetHistory(key string, history []spawnllm.Message) error
	// TruncateHistory keeps only the last keepLast messages.
	TruncateHistory(key string, keepLast int) error
	// SetPendingTurn marks a session as having an LLM turn in flight.
	SetPendingTurn(sessionKey string) error
	// ClearPendingTurn marks a session's turn as complete.
	ClearPendingTurn(sessionKey string) error
	// GetArchiveBounds returns the inclusive seq range of messages stored in
	// the session archive. Returns (0, 0) if no archive exists yet.
	GetArchiveBounds(sessionKey string) (minSeq, maxSeq int64)
	// ListPendingSessions returns session keys where PendingTurn is true.
	ListPendingSessions() ([]string, error)
	// Save persists any pending state to durable storage.
	Save(key string) error
	// Close releases resources held by the store.
	Close() error
}

// CompactionCommit is what one compaction pass writes: the retained window,
// the new current summary and its checkpoint when the pass produced one, and
// the compaction counters. SQLiteStore.CommitCompaction writes all of it in
// one transaction; the engine uses that through ctxengine.CompactionCommitter
// and falls back to separate writes on a store without it.
type CompactionCommit struct {
	// History replaces the window. Seqs are preserved; a message without one
	// is minted a seq after the highest seen, as SetHistoryWithSeqs does.
	History []memory.StoredMessage
	// Summary, when non-nil, becomes the session's current summary. Nil leaves
	// the current summary as it is (a drop-only pass).
	Summary *string
	// Checkpoint, when non-nil, is appended to the session's summary log.
	Checkpoint *memory.SummaryRecord
	// Compaction, when non-nil, is applied to the durable compaction counters
	// before they are written with the rest.
	Compaction func(*memory.CompactionState)
}
