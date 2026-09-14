package session

import (
	"github.com/PivotLLM/ctxengine/memory"
	"github.com/PivotLLM/spawnllm"
)

// SessionStore defines the persistence operations used by the agent loop.
// SQLiteStore is the implementation; the interface lets tests substitute an
// in-memory store without touching the agent loop code.
//
// Write methods (Add*, Set*, Truncate*) are fire-and-forget: they do not
// return errors. Implementations should log failures internally. This is the
// contract the agent loop relies on.
type SessionStore interface {
	// AddMessage appends a simple role/content message to the session.
	AddMessage(sessionKey, role, content string)
	// AddFullMessage appends a complete message including tool calls. It returns
	// the monotonic sequence number assigned to the written message so callers
	// can key durable side-stores (e.g. the archive) under the same seq.
	AddFullMessage(sessionKey string, msg spawnllm.Message) int64
	// GetHistory returns the full message history for the session.
	GetHistory(key string) []spawnllm.Message
	// GetHistoryWithSeqs returns the full message history with seq numbers intact.
	// Implementations that have no durable seq counter (an in-memory test store)
	// synthesize seq as i+1 for the i-th message.
	GetHistoryWithSeqs(key string) []memory.StoredMessage
	// GetSummary returns the conversation summary, or "" if none.
	GetSummary(key string) string
	// SetSummary replaces the conversation summary.
	SetSummary(key, summary string)
	// SetHistory replaces the full message history.
	SetHistory(key string, history []spawnllm.Message)
	// TruncateHistory keeps only the last keepLast messages.
	TruncateHistory(key string, keepLast int)
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
