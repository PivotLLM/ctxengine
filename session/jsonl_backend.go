package session

import (
	"context"

	"github.com/PivotLLM/ctxengine/logger"
	"github.com/PivotLLM/ctxengine/memory"
	"github.com/PivotLLM/spawnllm"
)

// JSONLBackend adapts a memory.Store into the SessionStore interface.
// Write errors are logged rather than returned, matching the fire-and-forget
// contract of SessionManager that the agent loop relies on.
type JSONLBackend struct {
	store memory.Store
}

// NewJSONLBackend wraps a memory.Store for use as a SessionStore.
func NewJSONLBackend(store memory.Store) *JSONLBackend {
	return &JSONLBackend{store: store}
}

func (b *JSONLBackend) AddMessage(sessionKey, role, content string) {
	if err := b.store.AddMessage(context.Background(), sessionKey, role, content); err != nil {
		logger.WarnCF("session", "add message", map[string]any{"error": err.Error()})
	}
}

func (b *JSONLBackend) AddFullMessage(sessionKey string, msg spawnllm.Message) int64 {
	seq, err := b.store.AddFullMessage(context.Background(), sessionKey, msg)
	if err != nil {
		logger.WarnCF("session", "add full message", map[string]any{"error": err.Error()})
		return 0
	}
	return seq
}

func (b *JSONLBackend) GetHistory(key string) []spawnllm.Message {
	msgs, err := b.store.GetHistory(context.Background(), key)
	if err != nil {
		logger.WarnCF("session", "get history", map[string]any{"error": err.Error()})
		return []spawnllm.Message{}
	}
	return msgs
}

func (b *JSONLBackend) GetHistoryWithSeqs(key string) []memory.StoredMessage {
	stored, err := b.store.GetHistoryWithSeqs(context.Background(), key)
	if err != nil {
		logger.WarnCF("session", "get history with seqs", map[string]any{"error": err.Error()})
		return []memory.StoredMessage{}
	}
	return stored
}

func (b *JSONLBackend) GetSummary(key string) string {
	summary, err := b.store.GetSummary(context.Background(), key)
	if err != nil {
		logger.WarnCF("session", "get summary", map[string]any{"error": err.Error()})
		return ""
	}
	return summary
}

func (b *JSONLBackend) SetSummary(key, summary string) {
	if err := b.store.SetSummary(context.Background(), key, summary); err != nil {
		logger.WarnCF("session", "set summary", map[string]any{"error": err.Error()})
	}
}

func (b *JSONLBackend) SetHistory(key string, history []spawnllm.Message) {
	if err := b.store.SetHistory(context.Background(), key, history); err != nil {
		logger.WarnCF("session", "set history", map[string]any{"error": err.Error()})
	}
}

// SetHistoryWithSeqs replaces the session history while preserving stable
// sequence numbers. Context compaction uses this so summary citations remain
// linked to archive message IDs.
func (b *JSONLBackend) SetHistoryWithSeqs(key string, history []memory.StoredMessage) {
	if err := b.store.SetHistoryWithSeqs(context.Background(), key, history); err != nil {
		logger.WarnCF("session", "set history with seqs", map[string]any{"error": err.Error()})
	}
}

// AppendSummaryCheckpoint appends one summary checkpoint when the underlying
// store supports checkpoint history.
func (b *JSONLBackend) AppendSummaryCheckpoint(sessionKey string, checkpoint memory.SummaryCheckpoint) error {
	type checkpointAppender interface {
		AppendSummaryCheckpoint(context.Context, string, memory.SummaryCheckpoint) error
	}
	if a, ok := b.store.(checkpointAppender); ok {
		return a.AppendSummaryCheckpoint(context.Background(), sessionKey, checkpoint)
	}
	return nil
}

func (b *JSONLBackend) TruncateHistory(key string, keepLast int) {
	if err := b.store.TruncateHistory(context.Background(), key, keepLast); err != nil {
		logger.WarnCF("session", "truncate history", map[string]any{"error": err.Error()})
	}
}

func (b *JSONLBackend) SetPendingTurn(sessionKey string) error {
	return b.store.SetPendingTurn(context.Background(), sessionKey)
}

func (b *JSONLBackend) ClearPendingTurn(sessionKey string) error {
	return b.store.ClearPendingTurn(context.Background(), sessionKey)
}

func (b *JSONLBackend) ListPendingSessions() ([]string, error) {
	return b.store.ListPendingSessions(context.Background())
}

func (b *JSONLBackend) GetArchiveBounds(sessionKey string) (minSeq, maxSeq int64) {
	min, max, err := b.store.GetArchiveBounds(context.Background(), sessionKey)
	if err != nil {
		logger.WarnCF("session", "get archive bounds", map[string]any{"error": err.Error()})
		return 0, 0
	}
	return min, max
}

// Save persists session state. Since the JSONL store fsyncs every write
// immediately, the data is already durable. Save runs compaction to reclaim
// space from logically truncated messages (no-op when there are none).
func (b *JSONLBackend) Save(key string) error {
	return b.store.Compact(context.Background(), key)
}

// GetCompactionState retrieves the durable compression state for sessionKey.
// Implements the ctxengine.CompactionStateStore interface via structural typing.
// Returns a zero CompactionState if the underlying store does not support it.
func (b *JSONLBackend) GetCompactionState(sessionKey string) (memory.CompactionState, error) {
	type compactionGetter interface {
		GetCompactionState(sessionKey string) (memory.CompactionState, error)
	}
	if g, ok := b.store.(compactionGetter); ok {
		return g.GetCompactionState(sessionKey)
	}
	return memory.CompactionState{}, nil
}

// SetCompactionState persists the compression state for sessionKey.
// Implements the ctxengine.CompactionStateStore interface via structural typing.
// No-ops if the underlying store does not support persistent compaction state.
func (b *JSONLBackend) SetCompactionState(sessionKey string, state memory.CompactionState) error {
	type compactionSetter interface {
		SetCompactionState(sessionKey string, state memory.CompactionState) error
	}
	if s, ok := b.store.(compactionSetter); ok {
		return s.SetCompactionState(sessionKey, state)
	}
	return nil
}

// ForgetSession drops in-memory per-session state held by the underlying store
// (e.g. the noise-dedup cache). Durable data on disk is untouched. No-ops if
// the underlying store does not support it. Called when a session's context
// manager is evicted so per-session caches do not grow unbounded.
func (b *JSONLBackend) ForgetSession(key string) {
	type sessionForgetter interface {
		ForgetSession(key string)
	}
	if f, ok := b.store.(sessionForgetter); ok {
		f.ForgetSession(key)
	}
}

// Close releases resources held by the underlying store.
func (b *JSONLBackend) Close() error {
	return b.store.Close()
}
