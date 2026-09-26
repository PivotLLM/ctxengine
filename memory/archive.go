package memory

import (
	"bufio"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/PivotLLM/spawnllm"
	_ "modernc.org/sqlite"

	"github.com/PivotLLM/ctxengine/internal/iox"
	"github.com/PivotLLM/ctxengine/logger"
)

// ErrArchiveUnavailable is returned when the ArchiveStore failed to open and
// all subsequent operations are no-ops or empty returns. Exported so callers
// can distinguish graceful-degradation failures from real errors.
var ErrArchiveUnavailable = errors.New("memory: archive unavailable")

// archiveSchema is the whole per-session database: the all-time message
// archive with its FTS index, the summary checkpoints, the live window the
// model is shown (window.seq shares the messages.seq space; created_at is unix
// nanoseconds, see StoredMessage) and the single session_state row that used
// to live in <key>.meta.json (timestamps in unix nanoseconds). Truncating the
// window deletes rows; there is no skip offset.
const archiveSchema = `
CREATE TABLE IF NOT EXISTS messages (
    seq          INTEGER PRIMARY KEY,
    role         TEXT    NOT NULL,
    payload      TEXT    NOT NULL,
    text         TEXT    NOT NULL,
    created_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_created_at ON messages(created_at);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    text,
    content='messages',
    content_rowid='seq'
);
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, text) VALUES (new.seq, new.text);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text) VALUES ('delete', old.seq, old.text);
END;
CREATE TABLE IF NOT EXISTS summaries (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at      INTEGER NOT NULL,
    model             TEXT,
    profile           TEXT,
    source_seq_start  INTEGER,
    source_seq_end    INTEGER,
    covered_seq_start INTEGER,
    covered_seq_end   INTEGER,
    summary           TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS window (
    seq        INTEGER PRIMARY KEY,
    role       TEXT    NOT NULL,
    payload    TEXT    NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS session_state (
    id                             INTEGER PRIMARY KEY CHECK (id = 1),
    key                            TEXT    NOT NULL,
    summary                        TEXT    NOT NULL DEFAULT '',
    next_seq                       INTEGER NOT NULL DEFAULT 0,
    created_at                     INTEGER NOT NULL DEFAULT 0,
    updated_at                     INTEGER NOT NULL DEFAULT 0,
    meaningful_count               INTEGER NOT NULL DEFAULT 0,
    compressed_at_meaningful_count INTEGER NOT NULL DEFAULT 0,
    cooling                        INTEGER NOT NULL DEFAULT 0,
    cooling_since_count            INTEGER NOT NULL DEFAULT 0,
    summary_generated_at           INTEGER NOT NULL DEFAULT 0,
    summary_model                  TEXT    NOT NULL DEFAULT '',
    active_model_index             INTEGER NOT NULL DEFAULT 0,
    expose_reasoning               INTEGER NOT NULL DEFAULT 0,
    show_tool_activity             INTEGER NOT NULL DEFAULT 0,
    pending_turn                   INTEGER NOT NULL DEFAULT 0
);
`

// SummaryCheckpoint is one record of the legacy <base>.summaries.jsonl
// checkpoint log, imported into the summaries table when an archive is opened.
type SummaryCheckpoint struct {
	ID              int       `json:"id"`
	GeneratedAt     time.Time `json:"generated_at"`
	Model           string    `json:"model,omitempty"`
	SourceSeqStart  int64     `json:"source_seq_start"`
	SourceSeqEnd    int64     `json:"source_seq_end"`
	CoveredSeqStart int64     `json:"covered_seq_start"`
	CoveredSeqEnd   int64     `json:"covered_seq_end"`
	PrevHash        string    `json:"prev_hash,omitempty"`
	SummaryHash     string    `json:"summary_hash"`
	Summary         string    `json:"summary"`
}

// SearchResult holds one result from an ArchiveStore.Search call.
// It pairs the archive sequence number and timestamp with the deserialized message.
type SearchResult struct {
	Seq       int64
	CreatedAt time.Time
	Message   spawnllm.Message
}

// SummaryRecord is the full stored representation of a context-summary
// checkpoint, including the marshaled summary body.
type SummaryRecord struct {
	ID              int64
	GeneratedAt     time.Time
	Model           string
	Profile         string
	SourceSeqStart  int64
	SourceSeqEnd    int64
	CoveredSeqStart int64
	CoveredSeqEnd   int64
	Summary         string
}

// SummaryMeta is the metadata-only view of a stored summary (no body). Returned
// by ListSummaries so callers can enumerate checkpoints cheaply.
type SummaryMeta struct {
	ID              int64
	GeneratedAt     time.Time
	Model           string
	Profile         string
	SourceSeqStart  int64
	SourceSeqEnd    int64
	CoveredSeqStart int64
	CoveredSeqEnd   int64
}

// ArchiveStore is an append-only SQLite store for session message archives.
// It supports FTS5 full-text search and efficient seq-range retrieval.
//
// The caller (ContextManager) is the sole writer. Readers open separate
// read-only connections (openReadOnly); WAL mode lets them read while the
// writer writes, and the busy timeout in their DSN makes them wait out the
// locks WAL does not exempt them from rather than fail at once.
type ArchiveStore struct {
	db          *sql.DB
	path        string
	mu          sync.Mutex // held only for writes (Append, Delete)
	unavailable bool       // set true when Open fails; all ops are no-ops
}

// OpenReadOnly creates a read-only ArchiveStore for an existing archive at path.
// Only QueryRange, Search, and Bounds are valid on the returned store; each
// opens its own short-lived read-only SQLite connection. No write connection,
// WAL pragma, or schema creation is performed.
// Returns ErrArchiveUnavailable if the file does not exist or is unreadable.
func OpenReadOnly(path string) (*ArchiveStore, error) {
	// Verify the file exists before returning the store.
	if _, err := os.Stat(path); err != nil {
		return &ArchiveStore{path: path, unavailable: true}, ErrArchiveUnavailable
	}
	return &ArchiveStore{path: path}, nil
}

// busyTimeoutMillis is how long a connection waits for a lock before failing
// with SQLITE_BUSY. The writer sets it with a PRAGMA; readers carry it in
// their DSN.
const busyTimeoutMillis = 2000

// openReadOnly opens a short-lived read-only connection to the database at
// path. The busy timeout rides in the DSN so every connection the pool opens
// has it, not only one that ran a PRAGMA. Without it a reader that meets a
// lock — the writer switching journal mode on open, WAL recovery after a
// crash, a checkpoint that needs the file, or a database still in rollback
// mode — fails at once with "database is locked" instead of waiting for it
// to clear. journal_mode is a property of the file the writer sets once, so
// a reader does not set it.
func openReadOnly(path string) (*sql.DB, error) {
	return sql.Open("sqlite", fmt.Sprintf("file:%s?mode=ro&_pragma=busy_timeout(%d)", path, busyTimeoutMillis))
}

// Open opens (or creates) an archive database at path.
// Enables WAL mode, creates schema, and runs an FTS5 integrity check.
// On failure it returns a store with unavailable=true plus ErrArchiveUnavailable.
func Open(path string) (*ArchiveStore, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		logger.WarnCF("memory", "archive open failed",
			map[string]any{"path": path, "error": err.Error()})
		return &ArchiveStore{path: path, unavailable: true}, ErrArchiveUnavailable
	}

	// Single writer connection; no connection pooling needed.
	db.SetMaxOpenConns(1)

	// Enable WAL mode for concurrent reader support.
	if _, err := db.Exec("PRAGMA journal_mode=WAL"); err != nil {
		iox.CloseQuietly("memory", db)
		logger.WarnCF("memory", "archive WAL failed",
			map[string]any{"path": path, "error": err.Error()})
		return &ArchiveStore{path: path, unavailable: true}, ErrArchiveUnavailable
	}
	if _, err := db.Exec(fmt.Sprintf("PRAGMA busy_timeout=%d", busyTimeoutMillis)); err != nil {
		iox.CloseQuietly("memory", db)
		logger.WarnCF("memory", "archive busy_timeout failed",
			map[string]any{"path": path, "error": err.Error()})
		return &ArchiveStore{path: path, unavailable: true}, ErrArchiveUnavailable
	}

	// Create schema.
	if _, err := db.Exec(archiveSchema); err != nil {
		iox.CloseQuietly("memory", db)
		logger.WarnCF("memory", "archive schema failed",
			map[string]any{"path": path, "error": err.Error()})
		return &ArchiveStore{path: path, unavailable: true}, ErrArchiveUnavailable
	}

	// FTS5 integrity check; rebuild if needed.
	if _, err := db.Exec("INSERT INTO messages_fts(messages_fts) VALUES ('integrity-check')"); err != nil {
		logger.WarnCF("memory", "archive FTS5 integrity check failed, rebuilding",
			map[string]any{"path": path, "error": err.Error()})
		if _, rebuildErr := db.Exec("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')"); rebuildErr != nil {
			iox.CloseQuietly("memory", db)
			logger.WarnCF("memory", "archive FTS5 rebuild failed",
				map[string]any{"path": path, "error": rebuildErr.Error()})
			return &ArchiveStore{path: path, unavailable: true}, ErrArchiveUnavailable
		}
	}

	store := &ArchiveStore{db: db, path: path}

	// One-time import of legacy <base>.summaries.jsonl checkpoints into the
	// summaries table. Best-effort: a failure logs and is ignored so it never
	// blocks archive open.
	store.importLegacySummaries()

	return store, nil
}

// legacySummariesPath derives the sibling <base>.summaries.jsonl path from the
// archive's own .archive.db path. Returns "" if the path does not end in the
// expected suffix.
func (a *ArchiveStore) legacySummariesPath() string {
	const suffix = ".archive.db"
	if !strings.HasSuffix(a.path, suffix) {
		return ""
	}
	base := strings.TrimSuffix(a.path, suffix)
	return base + ".summaries.jsonl"
}

// importLegacySummaries performs a one-time import of the legacy
// <base>.summaries.jsonl checkpoint log into the summaries table. It runs only
// when the summaries table is empty (the idempotency guard) and the legacy file
// exists. Each JSONL line is parsed as a SummaryCheckpoint; the hash fields are
// dropped and profile is set to "" (the legacy log carried no profile column).
// Order is preserved. Any failure is logged and otherwise ignored so it never
// breaks archive open.
func (a *ArchiveStore) importLegacySummaries() {
	if a.unavailable || a.db == nil {
		return
	}

	// Idempotency guard: only import when the table is empty.
	var count int
	if err := a.db.QueryRow(`SELECT COUNT(*) FROM summaries`).Scan(&count); err != nil {
		logger.WarnCF("memory", "archive summaries count failed",
			map[string]any{"path": a.path, "error": err.Error()})
		return
	}
	if count > 0 {
		return
	}

	legacyPath := a.legacySummariesPath()
	if legacyPath == "" {
		return
	}
	f, err := os.Open(legacyPath) //nolint:gosec // G304: derived from the archive's own path
	if os.IsNotExist(err) {
		return
	}
	if err != nil {
		logger.WarnCF("memory", "archive legacy summaries open failed",
			map[string]any{"path": legacyPath, "error": err.Error()})
		return
	}
	defer iox.CloseQuietly("memory", f)

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)
	imported := 0
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var cp SummaryCheckpoint
		if err := json.Unmarshal(line, &cp); err != nil {
			logger.WarnCF("memory", "archive legacy summaries skip corrupt line",
				map[string]any{"path": legacyPath, "error": err.Error()})
			continue
		}
		if _, err := a.AppendSummary(SummaryRecord{
			GeneratedAt:     cp.GeneratedAt,
			Model:           cp.Model,
			Profile:         "",
			SourceSeqStart:  cp.SourceSeqStart,
			SourceSeqEnd:    cp.SourceSeqEnd,
			CoveredSeqStart: cp.CoveredSeqStart,
			CoveredSeqEnd:   cp.CoveredSeqEnd,
			Summary:         cp.Summary,
		}); err != nil {
			logger.WarnCF("memory", "archive legacy summaries import failed",
				map[string]any{"path": legacyPath, "error": err.Error()})
			return
		}
		imported++
	}
	if err := scanner.Err(); err != nil {
		logger.WarnCF("memory", "archive legacy summaries scan failed",
			map[string]any{"path": legacyPath, "error": err.Error()})
	}
	if imported > 0 {
		logger.InfoCF("memory", "archive imported legacy summary checkpoints",
			map[string]any{"count": imported, "path": filepath.Base(legacyPath)})
	}
}

// Append writes one message to the archive.
// msg is serialized as JSON payload; searchable text is derived from
// msg.Content plus any ToolCalls arguments.
// Acquires the write mutex; no-op if unavailable.
func (a *ArchiveStore) Append(seq int64, msg spawnllm.Message, createdAt time.Time) error {
	if a.unavailable {
		return nil
	}

	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}

	text := deriveArchiveText(msg)

	a.mu.Lock()
	defer a.mu.Unlock()

	_, err = a.db.Exec(
		`INSERT OR REPLACE INTO messages (seq, role, payload, text, created_at) VALUES (?, ?, ?, ?, ?)`,
		seq, msg.Role, string(payload), text, createdAt.Unix(),
	)
	return err
}

// PruneMessagesToCount deletes all but the newest n messages by seq. The
// messages_ad AFTER DELETE trigger keeps the external-content messages_fts table
// in sync, so search results never reference pruned rows. No-op when n <= 0 or
// the store is unavailable. Acquires the write mutex.
func (a *ArchiveStore) PruneMessagesToCount(n int) error {
	if a.unavailable || a.db == nil || n <= 0 {
		return nil
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	_, err := a.db.Exec(
		`DELETE FROM messages WHERE seq <= (SELECT COALESCE(MAX(seq), 0) FROM messages) - ?`,
		n,
	)
	return err
}

// PruneMessagesBefore deletes messages older than cutoff (created_at <
// cutoff.Unix()). The messages_ad AFTER DELETE trigger keeps messages_fts in
// sync. No-op when the store is unavailable. Acquires the write mutex.
func (a *ArchiveStore) PruneMessagesBefore(cutoff time.Time) error {
	if a.unavailable || a.db == nil {
		return nil
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	_, err := a.db.Exec(
		`DELETE FROM messages WHERE created_at < ?`,
		cutoff.Unix(),
	)
	return err
}

// PruneSummariesToCount deletes all but the newest n summaries by id. No-op when
// n <= 0 or the store is unavailable. Acquires the write mutex.
func (a *ArchiveStore) PruneSummariesToCount(n int) error {
	if a.unavailable || a.db == nil || n <= 0 {
		return nil
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	_, err := a.db.Exec(
		`DELETE FROM summaries WHERE id <= (SELECT COALESCE(MAX(id), 0) FROM summaries) - ?`,
		n,
	)
	return err
}

// PruneSummariesBefore deletes summaries older than cutoff (generated_at <
// cutoff.Unix()). No-op when the store is unavailable. Acquires the write mutex.
func (a *ArchiveStore) PruneSummariesBefore(cutoff time.Time) error {
	if a.unavailable || a.db == nil {
		return nil
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	_, err := a.db.Exec(
		`DELETE FROM summaries WHERE generated_at < ?`,
		cutoff.Unix(),
	)
	return err
}

// AppendSummary inserts one context-summary checkpoint and returns its new id.
// generated_at is stored as a unix timestamp, matching messages.created_at.
// Acquires the write mutex; no-op (returns 0) if unavailable.
func (a *ArchiveStore) AppendSummary(rec SummaryRecord) (int64, error) {
	if a.unavailable || a.db == nil {
		return 0, nil
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	return insertSummary(a.db, rec)
}

// insertSummary inserts one summaries row and returns its id. A zero
// GeneratedAt is stamped now.
func insertSummary(ex execer, rec SummaryRecord) (int64, error) {
	genAt := rec.GeneratedAt
	if genAt.IsZero() {
		genAt = time.Now()
	}
	res, err := ex.Exec(
		`INSERT INTO summaries
		    (generated_at, model, profile, source_seq_start, source_seq_end, covered_seq_start, covered_seq_end, summary)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		genAt.Unix(), rec.Model, rec.Profile,
		rec.SourceSeqStart, rec.SourceSeqEnd,
		rec.CoveredSeqStart, rec.CoveredSeqEnd, rec.Summary,
	)
	if err != nil {
		return 0, err
	}
	return res.LastInsertId()
}

// ListSummaries returns metadata for all stored summaries (no body), ordered by
// id ascending. Uses a read-only connection opened and closed per call so it
// works against a read-only-opened archive.
// Returns (nil, ErrArchiveUnavailable) if the store is unavailable.
func (a *ArchiveStore) ListSummaries() ([]SummaryMeta, error) {
	if a.unavailable {
		return nil, ErrArchiveUnavailable
	}

	db, err := openReadOnly(a.path)
	if err != nil {
		return nil, err
	}
	defer iox.CloseQuietly("memory", db)

	rows, err := db.QueryContext(context.Background(),
		`SELECT id, generated_at, model, profile, source_seq_start, source_seq_end, covered_seq_start, covered_seq_end
		 FROM summaries ORDER BY id ASC`,
	)
	if err != nil {
		return nil, err
	}
	defer func() {
		if closeErr := rows.Close(); closeErr != nil {
			logger.DebugCF("memory", "rows close failed", map[string]any{"error": closeErr.Error()})
		}
	}()

	var metas []SummaryMeta
	for rows.Next() {
		var (
			m       SummaryMeta
			genUnix int64
			model   sql.NullString
			profile sql.NullString
		)
		if err := rows.Scan(&m.ID, &genUnix, &model, &profile,
			&m.SourceSeqStart, &m.SourceSeqEnd, &m.CoveredSeqStart, &m.CoveredSeqEnd); err != nil {
			return nil, err
		}
		m.GeneratedAt = time.Unix(genUnix, 0).UTC()
		m.Model = model.String
		m.Profile = profile.String
		metas = append(metas, m)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return metas, nil
}

// GetSummary returns the full summary record for id, including the body.
// ok is false if no row exists with that id. Uses a read-only connection opened
// and closed per call so it works against a read-only-opened archive.
// Returns (SummaryRecord{}, false, ErrArchiveUnavailable) if the store is unavailable.
func (a *ArchiveStore) GetSummary(id int64) (SummaryRecord, bool, error) {
	if a.unavailable {
		return SummaryRecord{}, false, ErrArchiveUnavailable
	}

	db, err := openReadOnly(a.path)
	if err != nil {
		return SummaryRecord{}, false, err
	}
	defer iox.CloseQuietly("memory", db)

	var (
		rec     SummaryRecord
		genUnix int64
		model   sql.NullString
		profile sql.NullString
	)
	row := db.QueryRowContext(context.Background(),
		`SELECT id, generated_at, model, profile, source_seq_start, source_seq_end, covered_seq_start, covered_seq_end, summary
		 FROM summaries WHERE id = ?`,
		id,
	)
	err = row.Scan(&rec.ID, &genUnix, &model, &profile,
		&rec.SourceSeqStart, &rec.SourceSeqEnd, &rec.CoveredSeqStart, &rec.CoveredSeqEnd, &rec.Summary)
	if errors.Is(err, sql.ErrNoRows) {
		return SummaryRecord{}, false, nil
	}
	if err != nil {
		return SummaryRecord{}, false, err
	}
	rec.GeneratedAt = time.Unix(genUnix, 0).UTC()
	rec.Model = model.String
	rec.Profile = profile.String
	return rec, true, nil
}

// MaxSeq returns the current maximum sequence number using the write connection.
// Avoids the WAL visibility gap that affects read-only connections opened with a
// separate URI. Returns 0 if no messages exist or the store is unavailable.
func (a *ArchiveStore) MaxSeq() int64 {
	if a.unavailable || a.db == nil {
		return 0
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	var maxSeq int64
	row := a.db.QueryRowContext(context.Background(), `SELECT COALESCE(MAX(seq), 0) FROM messages`)
	if err := row.Scan(&maxSeq); err != nil {
		return 0
	}
	return maxSeq
}

// QueryRange returns messages with seq in [minSeq, maxSeq] inclusive.
// Uses a read-only connection opened and closed per call.
// Returns (nil, ErrArchiveUnavailable) if the store is unavailable.
func (a *ArchiveStore) QueryRange(minSeq, maxSeq int64) ([]StoredMessage, error) {
	if a.unavailable {
		return nil, ErrArchiveUnavailable
	}

	db, err := openReadOnly(a.path)
	if err != nil {
		return nil, err
	}
	defer iox.CloseQuietly("memory", db)

	rows, err := db.QueryContext(context.Background(),
		`SELECT seq, payload, created_at FROM messages WHERE seq BETWEEN ? AND ? ORDER BY seq`,
		minSeq, maxSeq,
	)
	if err != nil {
		return nil, err
	}
	defer func() {
		if closeErr := rows.Close(); closeErr != nil {
			logger.DebugCF("memory", "rows close failed", map[string]any{"error": closeErr.Error()})
		}
	}()

	return scanStoredMessages(rows)
}

// Search runs an FTS5 full-text query and returns matching messages with their
// archive sequence numbers.
// query must not exceed 500 characters.
// role is an optional filter ("user"/"assistant"/"tool"); empty = all roles.
// limit is clamped to [1, 100].
// Uses a read-only connection opened and closed per call.
// Returns (nil, ErrArchiveUnavailable) if the store is unavailable.
// FTS5 parse errors are returned as regular errors (not panics); callers should
// surface them as tool errors.
func (a *ArchiveStore) Search(ctx context.Context, query, role string, limit int) ([]SearchResult, error) {
	if a.unavailable {
		return nil, ErrArchiveUnavailable
	}
	if len(query) > 500 {
		return nil, errors.New("memory: archive search query exceeds 500 characters")
	}
	if limit < 1 {
		limit = 1
	}
	if limit > 100 {
		limit = 100
	}

	db, err := openReadOnly(a.path)
	if err != nil {
		return nil, err
	}
	defer iox.CloseQuietly("memory", db)

	rows, err := db.QueryContext(ctx,
		`SELECT m.seq, m.payload, m.created_at
		 FROM   messages_fts
		 JOIN   messages m ON m.seq = messages_fts.rowid
		 WHERE  messages_fts MATCH ?
		   AND  (? = '' OR m.role = ?)
		 ORDER  BY rank
		 LIMIT  ?`,
		query, role, role, limit,
	)
	if err != nil {
		return nil, err
	}
	defer func() {
		if closeErr := rows.Close(); closeErr != nil {
			logger.DebugCF("memory", "rows close failed", map[string]any{"error": closeErr.Error()})
		}
	}()

	return scanSearchResults(rows)
}

// Bounds returns the inclusive [minSeq, maxSeq] of stored messages.
// Returns (0, 0, nil) if no messages exist.
// Uses a read-only connection opened and closed per call.
func (a *ArchiveStore) Bounds() (minSeq, maxSeq int64, err error) {
	if a.unavailable {
		return 0, 0, ErrArchiveUnavailable
	}

	db, err := openReadOnly(a.path)
	if err != nil {
		return 0, 0, err
	}
	defer iox.CloseQuietly("memory", db)

	row := db.QueryRowContext(context.Background(),
		`SELECT COALESCE(MIN(seq), 0), COALESCE(MAX(seq), 0) FROM messages`,
	)
	err = row.Scan(&minSeq, &maxSeq)
	return minSeq, maxSeq, err
}

// Stats returns the total message count and the first/last created_at
// timestamps stored in the archive in one round-trip.
// Returns (0, zero, zero, nil) if no messages exist.
// Uses a read-only connection opened and closed per call.
func (a *ArchiveStore) Stats() (count int, first, last time.Time, err error) {
	if a.unavailable {
		return 0, time.Time{}, time.Time{}, ErrArchiveUnavailable
	}

	db, err := openReadOnly(a.path)
	if err != nil {
		return 0, time.Time{}, time.Time{}, err
	}
	defer iox.CloseQuietly("memory", db)

	var firstUnix, lastUnix int64
	row := db.QueryRowContext(context.Background(),
		`SELECT COUNT(*), COALESCE(MIN(created_at), 0), COALESCE(MAX(created_at), 0) FROM messages`,
	)
	if err = row.Scan(&count, &firstUnix, &lastUnix); err != nil {
		return 0, time.Time{}, time.Time{}, err
	}
	if firstUnix > 0 {
		first = time.Unix(firstUnix, 0)
	}
	if lastUnix > 0 {
		last = time.Unix(lastUnix, 0)
	}
	return count, first, last, nil
}

// MinSeqAfter returns the smallest seq with created_at >= t.Unix().
// Returns (0, nil) if no messages exist at or after that time or the store is unavailable.
// Uses a read-only connection opened and closed per call.
func (a *ArchiveStore) MinSeqAfter(t time.Time) (int64, error) {
	if a.unavailable {
		return 0, ErrArchiveUnavailable
	}

	db, err := openReadOnly(a.path)
	if err != nil {
		return 0, err
	}
	defer iox.CloseQuietly("memory", db)

	var minSeq int64
	row := db.QueryRowContext(context.Background(),
		`SELECT COALESCE(MIN(seq), 0) FROM messages WHERE created_at >= ?`,
		t.Unix(),
	)
	if err := row.Scan(&minSeq); err != nil {
		return 0, err
	}
	return minSeq, nil
}

// Close closes the write connection. Safe to call multiple times.
func (a *ArchiveStore) Close() error {
	if a.unavailable || a.db == nil {
		return nil
	}
	err := a.db.Close()
	a.db = nil
	return err
}

// Delete closes the connection and removes the .db file.
func (a *ArchiveStore) Delete() error {
	if err := a.Close(); err != nil {
		return err
	}
	if a.path == "" {
		return nil
	}
	err := os.Remove(a.path)
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

// deriveArchiveText produces the FTS-indexed text for a message:
// msg.Content followed by any ToolCalls arguments serialized as JSON.
func deriveArchiveText(msg spawnllm.Message) string {
	text := msg.Content
	if len(msg.ToolCalls) > 0 {
		if b, err := json.Marshal(msg.ToolCalls); err == nil {
			if text != "" {
				text += " "
			}
			text += string(b)
		}
	}
	return text
}

// scanStoredMessages reads (seq, payload, created_at) rows and deserializes each into a StoredMessage.
func scanStoredMessages(rows *sql.Rows) ([]StoredMessage, error) {
	var msgs []StoredMessage
	for rows.Next() {
		var seq int64
		var payload string
		var createdAtUnix int64
		if err := rows.Scan(&seq, &payload, &createdAtUnix); err != nil {
			return nil, err
		}
		var msg spawnllm.Message
		if err := json.Unmarshal([]byte(payload), &msg); err != nil {
			return nil, err
		}
		msgs = append(msgs, NewStoredMessageAt(seq, msg, time.Unix(createdAtUnix, 0).UTC()))
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return msgs, nil
}

// scanSearchResults reads (seq, payload, created_at) rows and deserializes each into a SearchResult.
func scanSearchResults(rows *sql.Rows) ([]SearchResult, error) {
	var results []SearchResult
	for rows.Next() {
		var seq int64
		var payload string
		var createdAtUnix int64
		if err := rows.Scan(&seq, &payload, &createdAtUnix); err != nil {
			return nil, err
		}
		var msg spawnllm.Message
		if err := json.Unmarshal([]byte(payload), &msg); err != nil {
			return nil, err
		}
		results = append(results, SearchResult{
			Seq:       seq,
			CreatedAt: time.Unix(createdAtUnix, 0).UTC(),
			Message:   msg,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return results, nil
}
