package memory

import (
	"context"

	"github.com/PivotLLM/spawnllm"
)

// Store defines an interface for persistent session storage.
// Each method is an atomic operation — there is no separate Save() call.
type Store interface {
	// AddMessage appends a simple text message to a session.
	AddMessage(ctx context.Context, sessionKey, role, content string) error

	// AddFullMessage appends a complete message (with tool calls, etc.) to a
	// session. It returns the monotonic sequence number assigned to the written
	// message so callers can key durable side-stores (e.g. the archive) under
	// the same seq, keeping one sequence space across memory and archive.
	AddFullMessage(ctx context.Context, sessionKey string, msg spawnllm.Message) (int64, error)

	// GetHistory returns all messages for a session in insertion order.
	// Returns an empty slice (not nil) if the session does not exist.
	GetHistory(ctx context.Context, sessionKey string) ([]spawnllm.Message, error)

	// GetHistoryWithSeqs returns the active history window with seq numbers
	// intact. Each StoredMessage has the monotonically increasing seq number
	// assigned at write time. Seq numbers are preserved for seq-aware
	// summarization. Returns an empty slice if the session does not exist.
	GetHistoryWithSeqs(ctx context.Context, sessionKey string) ([]StoredMessage, error)

	// GetSummary returns the conversation summary for a session.
	// Returns an empty string if no summary exists.
	GetSummary(ctx context.Context, sessionKey string) (string, error)

	// SetSummary updates the conversation summary for a session.
	SetSummary(ctx context.Context, sessionKey, summary string) error

	// TruncateHistory removes all but the last keepLast messages from a session.
	// If keepLast <= 0, all messages are removed.
	TruncateHistory(ctx context.Context, sessionKey string, keepLast int) error

	// SetHistory replaces all messages in a session with the provided history.
	SetHistory(ctx context.Context, sessionKey string, history []spawnllm.Message) error

	// SetHistoryWithSeqs replaces all messages in a session while preserving the
	// caller-supplied stable sequence numbers. This is used by context compaction:
	// retained tail messages must keep the same IDs that were written to the
	// archive so summaries can cite retrievable messages reliably.
	SetHistoryWithSeqs(ctx context.Context, sessionKey string, history []StoredMessage) error

	// Compact reclaims storage by physically removing logically truncated
	// data. Backends that do not accumulate dead data may return nil.
	Compact(ctx context.Context, sessionKey string) error

	// SetPendingTurn marks a session as having an LLM turn in flight.
	// A true value on startup indicates the process was interrupted mid-turn.
	SetPendingTurn(ctx context.Context, sessionKey string) error

	// ClearPendingTurn marks a session's turn as complete.
	ClearPendingTurn(ctx context.Context, sessionKey string) error

	// GetArchiveBounds returns the inclusive seq range of messages stored in the
	// session archive. Returns (0, 0) if no archive exists yet.
	GetArchiveBounds(ctx context.Context, sessionKey string) (minSeq, maxSeq int64, err error)

	// ListPendingSessions returns session keys for all sessions where PendingTurn is true.
	ListPendingSessions(ctx context.Context) ([]string, error)

	// Close releases any resources held by the store.
	Close() error
}
