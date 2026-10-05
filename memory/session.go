/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package memory

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine/internal/iox"
	"github.com/PivotLLM/ctxengine/logger"
)

// SessionState is the per-session record that used to live in <key>.meta.json.
// It is the single session_state row of the session's archive database.
type SessionState struct {
	Key         string
	Summary     string
	NextSeq     int64
	CreatedAt   time.Time
	UpdatedAt   time.Time
	PendingTurn bool
	Compaction  CompactionState
}

// CompactionState holds the durable compression counters for a session.
// Defined here so the session store can return it without importing the
// engine package (which would create a circular import).
type CompactionState struct {
	MeaningfulCount             int       `json:"meaningful_count"`
	CompressedAtMeaningfulCount int       `json:"compressed_at_meaningful_count"`
	Cooling                     bool      `json:"cooling"`
	CoolingSinceCount           int       `json:"cooling_since_count"`
	SummaryGeneratedAt          time.Time `json:"summary_generated_at,omitempty"`
	SummaryModel                string    `json:"summary_model,omitempty"`
	ActiveModelIndex            int       `json:"active_model_index,omitempty"`
	// ExposeReasoning, when true, delivers the model's reasoning to the user for
	// this session (toggled via /reasoning). Default false.
	ExposeReasoning bool `json:"expose_reasoning,omitempty"`
	// ShowToolActivity, when true, posts a one-line breadcrumb for each tool call
	// to the user for this session (toggled via /tools). Default false.
	ShowToolActivity bool `json:"show_tool_activity,omitempty"`
}

// sanitizeKey converts a session key to a safe filename component.
// Replaces ':' with '_' (session key separator) and '/' and '\' with '_'
// so composite IDs (e.g. Telegram forum "chatID/threadID", Slack "channel/thread_ts")
// do not create subdirectories or break on Windows.
func sanitizeKey(key string) string {
	s := strings.ReplaceAll(key, ":", "_")
	s = strings.ReplaceAll(s, "/", "_")
	s = strings.ReplaceAll(s, "\\", "_")
	return s
}

// SanitizeSessionKey is the single filename rule for every per-session file:
// ':' '/' '\' become '_'. Every package that names a session file must use it
// rather than carry a copy.
func SanitizeSessionKey(key string) string { return sanitizeKey(key) }

// archiveSuffix is the file extension of a per-session database.
const archiveSuffix = ".archive.db"

// ArchivePath returns the SQLite archive path for a session under dir
// (normally <workspace>/sessions): <dir>/<sanitized-key>.archive.db.
func ArchivePath(dir, sessionKey string) string {
	return filepath.Join(dir, sanitizeKey(sessionKey)+archiveSuffix)
}

// ListSessions returns the session keys of every *.archive.db in dir that has
// a session_state row, sorted. Keys are read from the databases, not
// reconstructed from file names. A database that cannot be opened or has no
// state row (an archive that predates the window fold, or an empty one) is
// skipped. Returns nil, nil if dir does not exist.
func ListSessions(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("memory: list sessions: %w", err)
	}
	var keys []string
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), archiveSuffix) {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		a, err := OpenReadOnly(path)
		if err != nil {
			continue
		}
		st, err := a.State()
		if err != nil {
			logger.DebugCF("memory", "list sessions: skipping unreadable archive",
				map[string]any{"path": entry.Name(), "error": err.Error()})
			continue
		}
		if st.Key != "" {
			keys = append(keys, st.Key)
		}
	}
	sort.Strings(keys)
	return keys, nil
}

// DeleteSession removes <key>.archive.db and its -wal/-shm sidecars. A
// missing file is not an error. Any open handle on the database must be
// closed first (session.SQLiteStore.ForgetSession).
func DeleteSession(dir, key string) error {
	path := ArchivePath(dir, key)
	for _, p := range []string{path, path + "-wal", path + "-shm"} {
		if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("memory: delete session: %w", err)
		}
	}
	return nil
}

// unixNano encodes t for the window and session_state tables; the zero time
// is stored as 0.
func unixNano(t time.Time) int64 {
	if t.IsZero() {
		return 0
	}
	return t.UnixNano()
}

// fromUnixNano decodes a window/session_state timestamp; 0 is the zero time.
func fromUnixNano(n int64) time.Time {
	if n == 0 {
		return time.Time{}
	}
	return time.Unix(0, n).UTC()
}

// reader returns the connection to read window and state rows from: the write
// connection when the store owns one (so a read that follows a write on the
// same handle never sees a WAL visibility gap), otherwise a short-lived
// read-only connection that done closes.
func (a *ArchiveStore) reader() (db *sql.DB, done func(), err error) {
	if a.unavailable {
		return nil, nil, ErrArchiveUnavailable
	}
	if a.db != nil {
		a.mu.Lock()
		return a.db, a.mu.Unlock, nil
	}
	db, err = openReadOnly(a.path)
	if err != nil {
		return nil, nil, err
	}
	return db, func() { iox.CloseQuietly("memory", db) }, nil
}

// Window returns the live history window ordered by seq. Returns an empty
// slice (not nil) when the window is empty.
func (a *ArchiveStore) Window() ([]StoredMessage, error) {
	db, done, err := a.reader()
	if err != nil {
		return nil, err
	}
	defer done()

	rows, err := db.QueryContext(context.Background(),
		`SELECT seq, payload, created_at FROM window ORDER BY seq`)
	if err != nil {
		return nil, err
	}
	defer func() {
		if closeErr := rows.Close(); closeErr != nil {
			logger.DebugCF("memory", "rows close failed", map[string]any{"error": closeErr.Error()})
		}
	}()

	msgs := []StoredMessage{}
	for rows.Next() {
		var (
			seq       int64
			payload   string
			createdAt int64
		)
		if err := rows.Scan(&seq, &payload, &createdAt); err != nil {
			return nil, err
		}
		var msg spawnllm.Message
		if err := json.Unmarshal([]byte(payload), &msg); err != nil {
			return nil, err
		}
		// A literal rather than NewStoredMessageAt: a legacy message migrated
		// without a timestamp stays zero instead of being stamped on every read.
		msgs = append(msgs, StoredMessage{Seq: seq, CreatedAt: fromUnixNano(createdAt), Message: msg})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return msgs, nil
}

// State returns the session_state row, or the zero value and nil if there is
// none yet.
func (a *ArchiveStore) State() (SessionState, error) {
	db, done, err := a.reader()
	if err != nil {
		return SessionState{}, err
	}
	defer done()

	var (
		st                                                      SessionState
		createdAt, updatedAt, summaryGeneratedAt                int64
		cooling, exposeReasoning, showToolActivity, pendingTurn bool
	)
	err = db.QueryRowContext(context.Background(),
		`SELECT key, summary, next_seq, created_at, updated_at,
		        meaningful_count, compressed_at_meaningful_count, cooling, cooling_since_count,
		        summary_generated_at, summary_model, active_model_index,
		        expose_reasoning, show_tool_activity, pending_turn
		 FROM session_state WHERE id = 1`).Scan(
		&st.Key, &st.Summary, &st.NextSeq, &createdAt, &updatedAt,
		&st.Compaction.MeaningfulCount, &st.Compaction.CompressedAtMeaningfulCount, &cooling, &st.Compaction.CoolingSinceCount,
		&summaryGeneratedAt, &st.Compaction.SummaryModel, &st.Compaction.ActiveModelIndex,
		&exposeReasoning, &showToolActivity, &pendingTurn,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return SessionState{}, nil
	}
	if err != nil {
		return SessionState{}, err
	}
	st.CreatedAt = fromUnixNano(createdAt)
	st.UpdatedAt = fromUnixNano(updatedAt)
	st.PendingTurn = pendingTurn
	st.Compaction.Cooling = cooling
	st.Compaction.SummaryGeneratedAt = fromUnixNano(summaryGeneratedAt)
	st.Compaction.ExposeReasoning = exposeReasoning
	st.Compaction.ShowToolActivity = showToolActivity
	return st, nil
}

// execer is the subset of *sql.DB and *sql.Tx the state and window writers use.
type execer interface {
	Exec(query string, args ...any) (sql.Result, error)
}

// writeState upserts the single session_state row.
func writeState(ex execer, st SessionState) error {
	_, err := ex.Exec(
		`INSERT OR REPLACE INTO session_state
		    (id, key, summary, next_seq, created_at, updated_at,
		     meaningful_count, compressed_at_meaningful_count, cooling, cooling_since_count,
		     summary_generated_at, summary_model, active_model_index,
		     expose_reasoning, show_tool_activity, pending_turn)
		 VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		st.Key, st.Summary, st.NextSeq, unixNano(st.CreatedAt), unixNano(st.UpdatedAt),
		st.Compaction.MeaningfulCount, st.Compaction.CompressedAtMeaningfulCount, st.Compaction.Cooling, st.Compaction.CoolingSinceCount,
		unixNano(st.Compaction.SummaryGeneratedAt), st.Compaction.SummaryModel, st.Compaction.ActiveModelIndex,
		st.Compaction.ExposeReasoning, st.Compaction.ShowToolActivity, st.PendingTurn,
	)
	return err
}

// insertWindow inserts one window row.
func insertWindow(ex execer, msg StoredMessage) error {
	payload, err := json.Marshal(msg.Message)
	if err != nil {
		return fmt.Errorf("memory: marshal message: %w", err)
	}
	_, err = ex.Exec(
		`INSERT OR REPLACE INTO window (seq, role, payload, created_at) VALUES (?, ?, ?, ?)`,
		msg.Seq, msg.Role, string(payload), unixNano(msg.CreatedAt),
	)
	return err
}

// SetState upserts the session_state row (id = 1).
func (a *ArchiveStore) SetState(st SessionState) error {
	if a.unavailable || a.db == nil {
		return ErrArchiveUnavailable
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return writeState(a.db, st)
}

// withTx runs fn inside one write transaction.
func (a *ArchiveStore) withTx(fn func(tx *sql.Tx) error) error {
	if a.unavailable || a.db == nil {
		return ErrArchiveUnavailable
	}
	a.mu.Lock()
	defer a.mu.Unlock()

	tx, err := a.db.Begin()
	if err != nil {
		return err
	}
	if err := fn(tx); err != nil {
		if rbErr := tx.Rollback(); rbErr != nil {
			logger.WarnCF("memory", "archive rollback failed",
				map[string]any{"path": a.path, "error": rbErr.Error()})
		}
		return err
	}
	return tx.Commit()
}

// AppendWindow inserts one window row and writes st, in one transaction.
func (a *ArchiveStore) AppendWindow(msg StoredMessage, st SessionState) error {
	return a.withTx(func(tx *sql.Tx) error {
		if err := insertWindow(tx, msg); err != nil {
			return err
		}
		return writeState(tx, st)
	})
}

// replaceWindow deletes every window row and inserts msgs.
func replaceWindow(ex execer, msgs []StoredMessage) error {
	if _, err := ex.Exec(`DELETE FROM window`); err != nil {
		return err
	}
	for _, msg := range msgs {
		if err := insertWindow(ex, msg); err != nil {
			return err
		}
	}
	return nil
}

// ReplaceWindow replaces every window row with msgs and writes st, in one
// transaction.
func (a *ArchiveStore) ReplaceWindow(msgs []StoredMessage, st SessionState) error {
	return a.withTx(func(tx *sql.Tx) error {
		if err := replaceWindow(tx, msgs); err != nil {
			return err
		}
		return writeState(tx, st)
	})
}

// CommitCompaction writes one compaction result in a single transaction: the
// window is replaced with msgs, st (carrying the new summary and counters) is
// written, and checkpoint, when non-nil, is appended to summaries. Either all
// of it is durable or none of it is, so a crash cannot leave a truncated
// window beside a stale summary, or a checkpoint for a window that was never
// written.
func (a *ArchiveStore) CommitCompaction(msgs []StoredMessage, st SessionState, checkpoint *SummaryRecord) error {
	return a.withTx(func(tx *sql.Tx) error {
		if err := replaceWindow(tx, msgs); err != nil {
			return err
		}
		if err := writeState(tx, st); err != nil {
			return err
		}
		if checkpoint != nil {
			if _, err := insertSummary(tx, *checkpoint); err != nil {
				return err
			}
		}
		return nil
	})
}

// TruncateWindow deletes all but the newest keepLast window rows (every row
// when keepLast <= 0) and writes st, in one transaction.
func (a *ArchiveStore) TruncateWindow(keepLast int, st SessionState) error {
	return a.withTx(func(tx *sql.Tx) error {
		var err error
		if keepLast <= 0 {
			_, err = tx.Exec(`DELETE FROM window`)
		} else {
			_, err = tx.Exec(
				`DELETE FROM window WHERE seq NOT IN (SELECT seq FROM window ORDER BY seq DESC LIMIT ?)`,
				keepLast)
		}
		if err != nil {
			return err
		}
		return writeState(tx, st)
	})
}
