// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/PivotLLM/ctxengine/memory"
	"github.com/PivotLLM/spawnllm"
)

// newResetStore returns a mockStore pre-populated with a few messages and a
// summary for the given sessionKey, ready for Reset tests.
func newResetStore(sessionKey string) *mockStore {
	s := newMockStore()
	s.history[sessionKey] = []spawnllm.Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
	}
	s.summary[sessionKey] = `{"version":2,"state":{"goals":[{"text":"some goal","refs":[{"seq_start":1}]}]}}`
	return s
}

// newResetManager returns a *Manager suitable for reset tests.
// opts are applied after the base options so callers can override.
func newResetManager(store *mockStore, sessionKey string, opts ...Option) *Manager {
	baseOpts := []Option{
		WithContextWindow(10000),
		WithNormalPercent(90), // high threshold — compression won't fire in these tests
		WithSafetyPercent(95),
		WithMessageThreshold(100),
	}
	baseOpts = append(baseOpts, opts...)
	cm := New(sessionKey, store, baseOpts...)
	return cm.(*Manager)
}

// TestReset_ClearsHistory verifies that after Reset, GetHistory returns nil/empty.
func TestReset_ClearsHistory(t *testing.T) {
	const key = "test-session"
	store := newResetStore(key)
	mgr := newResetManager(store, key)

	if err := mgr.Reset(context.Background()); err != nil {
		t.Fatalf("Reset returned error: %v", err)
	}

	history := store.GetHistory(key)
	if len(history) != 0 {
		t.Errorf("expected empty history after Reset; got %d messages", len(history))
	}
}

// TestReset_ClearsSummary verifies that after Reset, GetSummary returns "".
func TestReset_ClearsSummary(t *testing.T) {
	const key = "test-session"
	store := newResetStore(key)
	mgr := newResetManager(store, key)

	// Confirm summary is non-empty before reset.
	if store.GetSummary(key) == "" {
		t.Fatal("precondition: summary should be non-empty before Reset")
	}

	if err := mgr.Reset(context.Background()); err != nil {
		t.Fatalf("Reset returned error: %v", err)
	}

	if got := store.GetSummary(key); got != "" {
		t.Errorf("expected empty summary after Reset; got %q", got)
	}
}

// TestReset_ClearsMsgCount verifies that after Reset, the in-memory msgCount is 0.
func TestReset_ClearsMsgCount(t *testing.T) {
	const key = "test-session"
	store := newResetStore(key)
	mgr := newResetManager(store, key)

	// Simulate some message activity.
	mgr.msgCount = 7
	mgr.compressedAtCount = 3
	mgr.cooling = true
	mgr.coolingSinceCount = 5

	if err := mgr.Reset(context.Background()); err != nil {
		t.Fatalf("Reset returned error: %v", err)
	}

	if mgr.msgCount != 0 {
		t.Errorf("expected msgCount=0 after Reset; got %d", mgr.msgCount)
	}
	if mgr.compressedAtCount != 0 {
		t.Errorf("expected compressedAtCount=0 after Reset; got %d", mgr.compressedAtCount)
	}
	if mgr.cooling {
		t.Error("expected cooling=false after Reset")
	}
	if mgr.coolingSinceCount != 0 {
		t.Errorf("expected coolingSinceCount=0 after Reset; got %d", mgr.coolingSinceCount)
	}
	if !mgr.lastCompressedAt.IsZero() {
		t.Errorf("expected lastCompressedAt to be zero after Reset; got %v", mgr.lastCompressedAt)
	}
	if mgr.lastCompressionGain != 0 {
		t.Errorf("expected lastCompressionGain=0 after Reset; got %v", mgr.lastCompressionGain)
	}
}

// TestReset_PreservesArchiveFile verifies that Reset preserves the archive .db
// (long-term memory) — clearing the conversation must not wipe the archive.
func TestReset_PreservesArchiveFile(t *testing.T) {
	const key = "test-session"
	archiveDir := t.TempDir()
	store := newResetStore(key)
	mgr := newResetManager(store, key, WithArchiveDir(archiveDir))

	// Trigger lazy archive creation by appending a message.
	mgr.archiveAppend(1, spawnllm.Message{Role: "user", Content: "hello"})

	// Confirm the file exists.
	sanitized := memory.SanitizeSessionKey(key)
	archivePath := filepath.Join(archiveDir, sanitized+".archive.db")
	if _, err := os.Stat(archivePath); os.IsNotExist(err) {
		t.Fatal("precondition: archive file should exist before Reset")
	}

	if err := mgr.Reset(context.Background()); err != nil {
		t.Fatalf("Reset returned error: %v", err)
	}

	// The archive file must still exist (long-term memory preserved).
	if _, err := os.Stat(archivePath); os.IsNotExist(err) {
		t.Errorf("expected archive file to be PRESERVED after Reset, but it was deleted")
	}

	// The archived row must still be retrievable.
	archive := mgr.getOrOpenArchive()
	if archive == nil {
		t.Fatal("expected archive to remain available after Reset")
	}
	minSeq, maxSeq, err := archive.Bounds()
	if err != nil {
		t.Fatalf("Bounds() returned error: %v", err)
	}
	if minSeq != 1 || maxSeq != 1 {
		t.Errorf("expected preserved bounds [1,1] after Reset; got [%d,%d]", minSeq, maxSeq)
	}
}

// TestReset_ArchiveContinuesAfterNewMessages verifies that after Reset, the
// archive is preserved and a subsequent write is appended alongside the old
// rows (memory continues; it does not start fresh).
func TestReset_ArchiveContinuesAfterNewMessages(t *testing.T) {
	const key = "test-session"
	archiveDir := t.TempDir()
	store := newResetStore(key)
	mgr := newResetManager(store, key, WithArchiveDir(archiveDir))

	// Write a few messages to the archive before reset, keyed by memory seq.
	for i := range 3 {
		content := strings.Repeat("x", 20)
		mgr.archiveAppend(int64(i+1), spawnllm.Message{Role: "user", Content: content})
	}

	// Reset preserves the archive.
	if err := mgr.Reset(context.Background()); err != nil {
		t.Fatalf("Reset returned error: %v", err)
	}

	// A new message continues under the next memory seq; the old rows remain.
	mgr.archiveAppend(4, spawnllm.Message{Role: "user", Content: "after clear"})

	archive := mgr.getOrOpenArchive()
	if archive == nil {
		t.Fatal("expected archive to remain available after Reset + new write")
	}
	minSeq, maxSeq, err := archive.Bounds()
	if err != nil {
		t.Fatalf("Bounds() returned error: %v", err)
	}
	if minSeq != 1 {
		t.Errorf("expected oldest preserved seq=1 after clear; got %d", minSeq)
	}
	if maxSeq != 4 {
		t.Errorf("expected newest seq=4 after continued write; got %d", maxSeq)
	}
}

// TestReset_ClearsPendingTurn verifies that ClearPendingTurn is called during Reset.
// We test this via the mockStore's side-effect tracking.
func TestReset_ClearsPendingTurn(t *testing.T) {
	const key = "test-session"
	store := newResetStore(key)

	// Track whether ClearPendingTurn was called by embedding a custom store.
	called := false
	trackingStore := &clearPendingTrackingStore{
		mockStore: store,
		onClear:   func() { called = true },
	}

	mgr := newResetManagerWithStore(trackingStore, key)

	if err := mgr.Reset(context.Background()); err != nil {
		t.Fatalf("Reset returned error: %v", err)
	}

	if !called {
		t.Error("expected ClearPendingTurn to be called during Reset")
	}
}

// clearPendingTrackingStore wraps mockStore and calls onClear when
// ClearPendingTurn is invoked.
type clearPendingTrackingStore struct {
	*mockStore
	onClear func()
}

func (s *clearPendingTrackingStore) ClearPendingTurn(_ string) error {
	s.onClear()
	return nil
}

// newResetManagerWithStore constructs a Manager using any SessionStore.
func newResetManagerWithStore(store interface {
	AddMessage(sessionKey, role, content string)
	AddFullMessage(sessionKey string, msg spawnllm.Message) int64
	GetHistory(key string) []spawnllm.Message
	GetHistoryWithSeqs(key string) []memory.StoredMessage
	GetSummary(key string) string
	SetSummary(key, summary string)
	SetHistory(key string, history []spawnllm.Message)
	TruncateHistory(key string, keepLast int)
	SetPendingTurn(sessionKey string) error
	ClearPendingTurn(sessionKey string) error
	GetArchiveBounds(sessionKey string) (int64, int64)
	ListPendingSessions() ([]string, error)
	Save(key string) error
	Close() error
}, sessionKey string, opts ...Option,
) *Manager {
	baseOpts := []Option{
		WithContextWindow(10000),
		WithNormalPercent(90),
		WithSafetyPercent(95),
		WithMessageThreshold(100),
	}
	baseOpts = append(baseOpts, opts...)
	cm := New(sessionKey, store, baseOpts...)
	return cm.(*Manager)
}
