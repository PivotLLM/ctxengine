/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package session

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine/internal/iox"
	"github.com/PivotLLM/ctxengine/logger"
	"github.com/PivotLLM/ctxengine/memory"
)

// SQLiteStore implements SessionStore on top of the per-session archive
// databases (<dir>/<sanitized key>.archive.db): the live window and the
// session state share the file with the all-time archive. One
// *memory.ArchiveStore handle per session, opened lazily on first use, cached,
// and closed by ForgetSession or Close. Every operation on a session runs
// under that session's mutex. Write methods return their error; read methods
// log it and return the empty value.
type SQLiteStore struct {
	dir      string
	mu       sync.Mutex // protects sessions and noiseKey
	sessions map[string]*sessionHandle
	// noiseKey recognises repeated-source messages for the noise classifier.
	// Nil (the default) means only identical same-role content is noise.
	noiseKey memory.NoiseKeyFunc
	// folderPerm and filePerm are the modes of the directory and archive
	// files the store creates.
	folderPerm os.FileMode
	filePerm   os.FileMode
}

// StoreOption configures NewSQLiteStore.
type StoreOption func(*SQLiteStore)

// WithFolderPermissions sets the mode of the sessions directory when
// NewSQLiteStore creates it (default memory.DefaultFolderPermissions). An
// existing directory is left as it is. Only the permission bits are used; 0
// keeps the default.
func WithFolderPermissions(perm os.FileMode) StoreOption {
	return func(s *SQLiteStore) {
		if perm &= os.ModePerm; perm != 0 {
			s.folderPerm = perm
		}
	}
}

// WithFilePermissions sets the mode of the archive databases and their -wal
// and -shm files (default memory.DefaultFilePermissions). Existing files whose
// mode differs are changed to it when opened. Only the permission bits are
// used; 0 keeps the default.
func WithFilePermissions(perm os.FileMode) StoreOption {
	return func(s *SQLiteStore) {
		if perm &= os.ModePerm; perm != 0 {
			s.filePerm = perm
		}
	}
}

// sessionHandle is the cached per-session state: the archive handle and the
// noise cache, both guarded by mu. closed is set when the handle is dropped
// from the cache so an operation that raced the drop reopens instead of
// using a closed database.
type sessionHandle struct {
	mu      sync.Mutex
	archive *memory.ArchiveStore
	noise   *memory.NoiseCache
	closed  bool
}

// NewSQLiteStore creates a store rooted at dir. A missing dir is created and
// set to the folder mode; an existing one is left as it is. No database is
// opened until a session is used.
func NewSQLiteStore(dir string, opts ...StoreOption) (*SQLiteStore, error) {
	s := &SQLiteStore{
		dir:        dir,
		sessions:   make(map[string]*sessionHandle),
		folderPerm: memory.DefaultFolderPermissions,
		filePerm:   memory.DefaultFilePermissions,
	}
	for _, o := range opts {
		o(s)
	}
	if err := iox.EnsureDir(dir, s.folderPerm); err != nil {
		return nil, fmt.Errorf("session: %w", err)
	}
	return s, nil
}

// openOptions passes the store's modes to memory.Open.
func (s *SQLiteStore) openOptions() []memory.OpenOption {
	return []memory.OpenOption{
		memory.WithFolderPermissions(s.folderPerm),
		memory.WithFilePermissions(s.filePerm),
	}
}

// SetNoiseKey installs the function that recognises repeated fires of one
// source (the host's scheduled-job wrapper, say) so they count as noise even
// though each carries a different timestamp. Without it only identical
// same-role content is noise.
func (s *SQLiteStore) SetNoiseKey(fn memory.NoiseKeyFunc) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.noiseKey = fn
}

// with runs fn on key's archive under the session lock, opening the database
// on first use. The handle is looked up again if it was dropped between the
// lookup and the lock.
func (s *SQLiteStore) with(key string, fn func(a *memory.ArchiveStore, h *sessionHandle, noise memory.NoiseKeyFunc) error) error {
	for {
		s.mu.Lock()
		h, ok := s.sessions[key]
		if !ok {
			h = &sessionHandle{noise: memory.NewNoiseCache()}
			s.sessions[key] = h
		}
		noise := s.noiseKey
		s.mu.Unlock()

		h.mu.Lock()
		if h.closed {
			h.mu.Unlock()
			continue
		}
		if h.archive == nil {
			// Open logs the cause of a failure itself; the handle stays
			// unopened so the next call retries.
			a, err := memory.Open(memory.ArchivePath(s.dir, key), s.openOptions()...)
			if err != nil {
				h.mu.Unlock()
				return err
			}
			h.archive = a
		}
		err := fn(h.archive, h, noise)
		h.mu.Unlock()
		return err
	}
}

// state reads the session_state row with Key filled in for a session that
// has no row yet.
func state(a *memory.ArchiveStore, key string) (memory.SessionState, error) {
	st, err := a.State()
	if err != nil {
		return memory.SessionState{}, err
	}
	st.Key = key
	return st, nil
}

func warn(op, key string, err error) {
	logger.WarnCF("session", op, map[string]any{"session": key, "error": err.Error()})
}

func (s *SQLiteStore) AddMessage(sessionKey, role, content string) error {
	_, err := s.AddFullMessage(sessionKey, spawnllm.Message{Role: role, Content: content})
	return err
}

// AddFullMessage appends msg to the window under the next sequence number and
// returns it, or 0 and the error when nothing was written. The message counts
// towards meaningful_count unless the noise classifier finds it a duplicate
// of the previous one.
func (s *SQLiteStore) AddFullMessage(sessionKey string, msg spawnllm.Message) (int64, error) {
	var seq int64
	err := s.with(sessionKey, func(a *memory.ArchiveStore, h *sessionHandle, noise memory.NoiseKeyFunc) error {
		st, err := state(a, sessionKey)
		if err != nil {
			return err
		}

		// Assign monotonically increasing sequence number.
		seq = st.NextSeq + 1
		if seq <= 0 {
			seq = 1
		}
		stored := memory.NewStoredMessage(seq, msg)

		// Determine if this message is noise (no new information).
		isNoisy := h.noise.IsNoise(stored, noise)

		now := time.Now()
		if st.CreatedAt.IsZero() {
			st.CreatedAt = now
		}
		st.NextSeq = seq
		if !isNoisy {
			st.Compaction.MeaningfulCount++
		}
		st.UpdatedAt = now

		if err := a.AppendWindow(stored, st); err != nil {
			return err
		}

		// Update noise cache after computing isNoisy.
		h.noise.Record(stored, noise)

		logger.DebugCF("memory", "message_stored",
			map[string]any{
				"seq":     seq,
				"length":  len(msg.Content),
				"counted": !isNoisy,
				"role":    msg.Role,
				"session": sessionKey,
			})
		return nil
	})
	if err != nil {
		return 0, err
	}
	return seq, nil
}

func (s *SQLiteStore) GetHistory(key string) []spawnllm.Message {
	stored := s.GetHistoryWithSeqs(key)
	msgs := make([]spawnllm.Message, len(stored))
	for i, sm := range stored {
		msgs[i] = sm.Message
	}
	return msgs
}

func (s *SQLiteStore) GetHistoryWithSeqs(key string) []memory.StoredMessage {
	var stored []memory.StoredMessage
	err := s.with(key, func(a *memory.ArchiveStore, _ *sessionHandle, _ memory.NoiseKeyFunc) error {
		var err error
		stored, err = a.Window()
		return err
	})
	if err != nil {
		warn("get history", key, err)
		return []memory.StoredMessage{}
	}
	return stored
}

func (s *SQLiteStore) GetSummary(key string) string {
	var summary string
	err := s.with(key, func(a *memory.ArchiveStore, _ *sessionHandle, _ memory.NoiseKeyFunc) error {
		st, err := a.State()
		summary = st.Summary
		return err
	})
	if err != nil {
		warn("get summary", key, err)
		return ""
	}
	return summary
}

func (s *SQLiteStore) SetSummary(key, summary string) error {
	return s.with(key, func(a *memory.ArchiveStore, _ *sessionHandle, _ memory.NoiseKeyFunc) error {
		st, err := state(a, key)
		if err != nil {
			return err
		}
		now := time.Now()
		if st.CreatedAt.IsZero() {
			st.CreatedAt = now
		}
		st.Summary = summary
		st.UpdatedAt = now
		return a.SetState(st)
	})
}

// SetHistory replaces the window. The new messages take fresh sequence
// numbers after the current NextSeq so seqs stay monotonic across rewrites;
// SetHistory takes []spawnllm.Message, so there is no prior CreatedAt to carry
// over and each message is stamped now.
func (s *SQLiteStore) SetHistory(key string, history []spawnllm.Message) error {
	return s.with(key, func(a *memory.ArchiveStore, _ *sessionHandle, _ memory.NoiseKeyFunc) error {
		st, err := state(a, key)
		if err != nil {
			return err
		}
		now := time.Now()
		if st.CreatedAt.IsZero() {
			st.CreatedAt = now
		}
		st.UpdatedAt = now

		seq := st.NextSeq
		stored := make([]memory.StoredMessage, len(history))
		for i, msg := range history {
			seq++
			stored[i] = memory.NewStoredMessage(seq, msg)
		}
		st.NextSeq = seq
		return a.ReplaceWindow(stored, st)
	})
}

// withStableSeqs prepares history for a seq-preserving window rewrite: a
// message without a seq is minted one after the highest seen, and the
// returned counter never goes below nextSeq, so seqs stay monotonic.
func withStableSeqs(history []memory.StoredMessage, nextSeq int64) ([]memory.StoredMessage, int64) {
	maxSeq := nextSeq
	stored := make([]memory.StoredMessage, len(history))
	for i, sm := range history {
		if sm.Seq <= 0 {
			maxSeq++
			sm.Seq = maxSeq
		}
		if sm.Seq > maxSeq {
			maxSeq = sm.Seq
		}
		stored[i] = memory.NewStoredMessageAt(sm.Seq, sm.Message, sm.CreatedAt)
	}
	return stored, maxSeq
}

// SetHistoryWithSeqs replaces the window while preserving stable seq numbers.
// It is intended for eviction and for compaction on a store without
// CommitCompaction: retained tail messages keep the IDs already advertised in
// summaries and written to the archive. Messages without a seq are minted one
// after the highest seen; NextSeq never goes backwards.
func (s *SQLiteStore) SetHistoryWithSeqs(key string, history []memory.StoredMessage) error {
	return s.with(key, func(a *memory.ArchiveStore, _ *sessionHandle, _ memory.NoiseKeyFunc) error {
		st, err := state(a, key)
		if err != nil {
			return err
		}
		now := time.Now()
		if st.CreatedAt.IsZero() {
			st.CreatedAt = now
		}
		st.UpdatedAt = now

		stored, next := withStableSeqs(history, st.NextSeq)
		st.NextSeq = next
		return a.ReplaceWindow(stored, st)
	})
}

// CommitCompaction writes one compaction result in a single transaction: the
// window (seqs preserved, as SetHistoryWithSeqs), the current summary when
// c.Summary is set, the summary checkpoint when c.Checkpoint is set, and the
// compaction counters after c.Compaction. A crash cannot leave the truncated
// window beside a stale summary, or a checkpoint without the window it
// describes.
func (s *SQLiteStore) CommitCompaction(key string, c CompactionCommit) error {
	return s.with(key, func(a *memory.ArchiveStore, _ *sessionHandle, _ memory.NoiseKeyFunc) error {
		st, err := state(a, key)
		if err != nil {
			return err
		}
		now := time.Now()
		if st.CreatedAt.IsZero() {
			st.CreatedAt = now
		}
		st.UpdatedAt = now

		stored, next := withStableSeqs(c.History, st.NextSeq)
		st.NextSeq = next
		if c.Summary != nil {
			st.Summary = *c.Summary
		}
		if c.Compaction != nil {
			c.Compaction(&st.Compaction)
		}
		return a.CommitCompaction(stored, st, c.Checkpoint)
	})
}

// TruncateHistory keeps only the last keepLast window rows. keepLast <= 0 is a
// full reset: the window is emptied and the compression counters cleared
// (the archive is untouched — Manager.Reset preserves it by design).
func (s *SQLiteStore) TruncateHistory(key string, keepLast int) error {
	return s.with(key, func(a *memory.ArchiveStore, _ *sessionHandle, _ memory.NoiseKeyFunc) error {
		st, err := state(a, key)
		if err != nil {
			return err
		}
		if keepLast <= 0 {
			st.Compaction.MeaningfulCount = 0
			st.Compaction.CompressedAtMeaningfulCount = 0
			st.Compaction.Cooling = false
		}
		st.UpdatedAt = time.Now()
		return a.TruncateWindow(keepLast, st)
	})
}

func (s *SQLiteStore) setPendingTurn(key string, pending bool) error {
	return s.with(key, func(a *memory.ArchiveStore, _ *sessionHandle, _ memory.NoiseKeyFunc) error {
		st, err := state(a, key)
		if err != nil {
			return err
		}
		st.PendingTurn = pending
		return a.SetState(st)
	})
}

func (s *SQLiteStore) SetPendingTurn(sessionKey string) error {
	return s.setPendingTurn(sessionKey, true)
}

func (s *SQLiteStore) ClearPendingTurn(sessionKey string) error {
	return s.setPendingTurn(sessionKey, false)
}

// ListPendingSessions opens every *.archive.db in the store directory
// read-only and returns the key of each session whose state has pending_turn
// set. Databases that cannot be read are skipped. Returns nil, nil if the
// directory does not exist.
func (s *SQLiteStore) ListPendingSessions() ([]string, error) {
	entries, err := os.ReadDir(s.dir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("session: list pending sessions: %w", err)
	}

	var keys []string
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".archive.db") {
			continue
		}
		a, err := memory.OpenReadOnly(filepath.Join(s.dir, entry.Name()))
		if err != nil {
			continue
		}
		st, err := a.State()
		if err != nil {
			continue
		}
		if st.PendingTurn && st.Key != "" {
			keys = append(keys, st.Key)
		}
	}
	return keys, nil
}

// GetArchiveBounds returns the inclusive seq range of the session's archived
// messages, (0, 0) when the archive is empty.
func (s *SQLiteStore) GetArchiveBounds(sessionKey string) (minSeq, maxSeq int64) {
	err := s.with(sessionKey, func(a *memory.ArchiveStore, _ *sessionHandle, _ memory.NoiseKeyFunc) error {
		var err error
		minSeq, maxSeq, err = a.Bounds()
		return err
	})
	if err != nil {
		warn("get archive bounds", sessionKey, err)
		return 0, 0
	}
	return minSeq, maxSeq
}

// Save is a no-op: every write is its own committed SQLite transaction.
func (s *SQLiteStore) Save(string) error { return nil }

// GetCompactionState reads the compaction counters from the session state.
// Implements the ctxengine.CompactionStateStore interface via structural typing.
func (s *SQLiteStore) GetCompactionState(sessionKey string) (memory.CompactionState, error) {
	var cs memory.CompactionState
	err := s.with(sessionKey, func(a *memory.ArchiveStore, _ *sessionHandle, _ memory.NoiseKeyFunc) error {
		st, err := a.State()
		cs = st.Compaction
		return err
	})
	if err != nil {
		return memory.CompactionState{}, err
	}
	return cs, nil
}

// SetCompactionState writes the compaction counters back to the session state.
// Implements the ctxengine.CompactionStateStore interface via structural typing.
func (s *SQLiteStore) SetCompactionState(sessionKey string, cs memory.CompactionState) error {
	return s.with(sessionKey, func(a *memory.ArchiveStore, _ *sessionHandle, _ memory.NoiseKeyFunc) error {
		st, err := state(a, sessionKey)
		if err != nil {
			return err
		}
		st.Compaction = cs
		st.UpdatedAt = time.Now()
		return a.SetState(st)
	})
}

// ForgetSession closes key's database handle and drops it, together with the
// noise cache, from the store. Durable data is untouched: a subsequent access
// reopens the database. Called when a session's context manager is evicted so
// the cache does not grow unbounded over the lifetime of the process, and
// before a session's database is deleted.
func (s *SQLiteStore) ForgetSession(key string) {
	s.mu.Lock()
	h, ok := s.sessions[key]
	delete(s.sessions, key)
	s.mu.Unlock()
	if !ok {
		return
	}
	closeHandle(key, h)
}

// closeHandle marks h dropped and closes its database once any in-flight
// operation has finished.
func closeHandle(key string, h *sessionHandle) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.closed = true
	if h.archive == nil {
		return
	}
	if err := h.archive.Close(); err != nil {
		warn("close", key, err)
	}
	h.archive = nil
}

// Close closes every cached database handle.
func (s *SQLiteStore) Close() error {
	s.mu.Lock()
	handles := s.sessions
	s.sessions = make(map[string]*sessionHandle)
	s.mu.Unlock()
	for key, h := range handles {
		closeHandle(key, h)
	}
	return nil
}
