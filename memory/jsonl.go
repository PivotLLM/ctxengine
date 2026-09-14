package memory

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/PivotLLM/ctxengine/logger"
	"github.com/PivotLLM/spawnllm"
)

const (
	// numLockShards is the fixed number of mutexes used to serialize
	// per-session access. Using a sharded array instead of a map keeps
	// memory bounded regardless of how many sessions are created over
	// the lifetime of the process — important for a long-running daemon.
	numLockShards = 64

	// maxLineSize is the maximum size of a single JSON line in a .jsonl
	// file. Tool results (read_file, web search, etc.) can be large, so
	// we set a generous limit. The scanner starts at 64 KB and grows
	// only as needed up to this cap.
	maxLineSize = 10 * 1024 * 1024 // 10 MB
)

// sessionMeta holds per-session metadata stored in a .meta.json file.
type sessionMeta struct {
	Key       string    `json:"key"`
	Summary   string    `json:"summary"`
	Skip      int       `json:"skip"`
	Count     int       `json:"count"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`

	// Phase 2 fields
	NextSeq                     int64     `json:"next_seq,omitempty"`
	MeaningfulCount             int       `json:"meaningful_count,omitempty"`
	CompressedAtMeaningfulCount int       `json:"compressed_at_meaningful_count,omitempty"`
	ArchiveMinSeq               int64     `json:"archive_min_seq,omitempty"`
	ArchiveMaxSeq               int64     `json:"archive_max_seq,omitempty"`
	SummaryGeneratedAt          time.Time `json:"summary_generated_at,omitempty"`
	SummaryModel                string    `json:"summary_model,omitempty"`
	CompressionCooling          bool      `json:"compression_cooling,omitempty"`
	CoolingSinceCount           int       `json:"cooling_since_count,omitempty"`
	ActiveModelIndex            int       `json:"active_model_index,omitempty"`
	ExposeReasoning             bool      `json:"expose_reasoning,omitempty"`
	ShowToolActivity            bool      `json:"show_tool_activity,omitempty"`

	// PendingTurn is true while an LLM turn is in flight for this session.
	// Set before the LLM call; cleared when the final response is persisted.
	// A true value on startup indicates the process was interrupted mid-turn.
	PendingTurn bool `json:"pending_turn,omitempty"`
}

// SummaryCheckpoint is one append-only record of a generated context summary.
// The latest summary still lives in sessionMeta.Summary for fast context
// injection; checkpoints are for audit, recovery, and debugging.
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

// NoiseKeyFunc identifies messages that are repeated fires of one source (a
// scheduled job, for instance) by a key: two consecutive messages with the
// same key are duplicates whatever their timestamps say. ok is false for
// content that is not such a message. See JSONLStore.SetNoiseKey.
type NoiseKeyFunc func(content string) (key string, ok bool)

// noiseCache tracks the last written message per role and the last noise key
// so that duplicate messages can be classified as noise.
type noiseCache struct {
	lastByRole map[string]string // role -> last written Content
	lastKey    string
}

func newNoiseCache() *noiseCache {
	return &noiseCache{lastByRole: make(map[string]string)}
}

// isNoise returns true if msg is a duplicate that contributes no new information.
// Messages the noise key recognises dedup on that key (for a scheduled job, the
// job fingerprint when present, else the payload) so that repeated fires of the
// same job — which embed differing timestamps — collapse to one. All other
// messages dedup on identical same-role content. A nil noise recognises nothing.
func isNoise(msg StoredMessage, cache *noiseCache, noise NoiseKeyFunc) bool {
	if cache == nil {
		return false
	}
	if noise != nil {
		if key, ok := noise(msg.Content); ok {
			return cache.lastKey == key
		}
	}
	return cache.lastByRole[msg.Role] == msg.Content && msg.Content != ""
}

// updateNoiseCache records msg in the cache for future noise checks.
func updateNoiseCache(msg StoredMessage, cache *noiseCache, noise NoiseKeyFunc) {
	if cache == nil {
		return
	}
	if noise != nil {
		if key, ok := noise(msg.Content); ok {
			cache.lastKey = key
			return
		}
	}
	cache.lastByRole[msg.Role] = msg.Content
}

// JSONLStore implements Store using append-only JSONL files.
//
// Each session is stored as two files:
//
//	{sanitized_key}.jsonl      — one JSON-encoded StoredMessage per line, append-only
//	{sanitized_key}.meta.json  — session metadata (summary, logical truncation offset)
//
// The all-time message archive is now stored as a SQLite database
// ({sanitized_key}.archive.db) managed by the ContextManager via ArchiveStore.
// JSONLStore does not write to the archive.
//
// Messages are never physically deleted from the JSONL file. Instead,
// TruncateHistory records a "skip" offset in the metadata file and
// GetHistory ignores lines before that offset. This keeps all writes
// append-only, which is both fast and crash-safe.
type JSONLStore struct {
	dir         string
	locks       [numLockShards]sync.Mutex
	noiseMu     sync.Mutex // protects noiseCaches map and noiseKey
	noiseCaches map[string]*noiseCache
	// noiseKey recognises repeated-source messages for the noise classifier.
	// Nil (the default) means only identical same-role content is noise.
	noiseKey NoiseKeyFunc
}

// NewJSONLStore creates a new JSONL-backed store rooted at dir.
func NewJSONLStore(dir string) (*JSONLStore, error) {
	err := os.MkdirAll(dir, 0o755)
	if err != nil {
		return nil, fmt.Errorf("memory: create directory: %w", err)
	}
	return &JSONLStore{
		dir:         dir,
		noiseCaches: make(map[string]*noiseCache),
	}, nil
}

// sessionLock returns a mutex for the given session key.
// Keys are mapped to a fixed pool of shards via FNV hash, so
// memory usage is O(1) regardless of total session count.
func (s *JSONLStore) sessionLock(key string) *sync.Mutex {
	h := fnv.New32a()
	h.Write([]byte(key))
	return &s.locks[h.Sum32()%numLockShards]
}

func (s *JSONLStore) jsonlPath(key string) string {
	return filepath.Join(s.dir, sanitizeKey(key)+".jsonl")
}

func (s *JSONLStore) metaPath(key string) string {
	return filepath.Join(s.dir, sanitizeKey(key)+".meta.json")
}

func (s *JSONLStore) summaryHistoryPath(key string) string {
	return filepath.Join(s.dir, sanitizeKey(key)+".summaries.jsonl")
}

// sanitizeKey converts a session key to a safe filename component.
// Mirrors session.sanitizeFilename so that migration paths match.
// Replaces ':' with '_' (session key separator) and '/' and '\' with '_'
// so composite IDs (e.g. Telegram forum "chatID/threadID", Slack "channel/thread_ts")
// do not create subdirectories or break on Windows.
func sanitizeKey(key string) string {
	s := strings.ReplaceAll(key, ":", "_")
	s = strings.ReplaceAll(s, "/", "_")
	s = strings.ReplaceAll(s, "\\", "_")
	return s
}

// SanitizeSessionKey is the single filename rule for every per-session file
// (.jsonl, .meta.json, .archive.db): ':' '/' '\' become '_'. Every
// package that names a session file must use it rather than carry a copy.
func SanitizeSessionKey(key string) string { return sanitizeKey(key) }

// ArchivePath returns the SQLite archive path for a session under dir
// (normally <workspace>/sessions): <dir>/<sanitized-key>.archive.db.
func ArchivePath(dir, sessionKey string) string {
	return filepath.Join(dir, sanitizeKey(sessionKey)+".archive.db")
}

// SetNoiseKey installs the function that recognises repeated fires of one
// source (the host's scheduled-job wrapper, say) so they count as noise even
// though each carries a different timestamp. Without it only identical
// same-role content is noise.
func (s *JSONLStore) SetNoiseKey(fn NoiseKeyFunc) {
	s.noiseMu.Lock()
	defer s.noiseMu.Unlock()
	s.noiseKey = fn
}

// getNoiseCache returns the noise cache for the given session key, creating
// one if it does not exist, together with the installed noise key. The cache
// contents are only accessed while holding the per-session lock; noiseMu
// protects the map itself.
func (s *JSONLStore) getNoiseCache(key string) (*noiseCache, NoiseKeyFunc) {
	s.noiseMu.Lock()
	defer s.noiseMu.Unlock()
	if c, ok := s.noiseCaches[key]; ok {
		return c, s.noiseKey
	}
	c := newNoiseCache()
	s.noiseCaches[key] = c
	return c, s.noiseKey
}

// ForgetSession drops in-memory per-session state (the noise cache) for key.
// Durable data on disk is untouched: a subsequent access lazily recreates the
// cache. Called when a session's context manager is evicted so the noiseCaches
// map does not grow unbounded over the lifetime of the process.
func (s *JSONLStore) ForgetSession(key string) {
	s.noiseMu.Lock()
	defer s.noiseMu.Unlock()
	delete(s.noiseCaches, key)
}

// readMeta loads the metadata file for a session.
// Returns a zero-value sessionMeta if the file does not exist.
func (s *JSONLStore) readMeta(key string) (sessionMeta, error) {
	data, err := os.ReadFile(s.metaPath(key))
	if os.IsNotExist(err) {
		return sessionMeta{Key: key}, nil
	}
	if err != nil {
		return sessionMeta{}, fmt.Errorf("memory: read meta: %w", err)
	}
	var meta sessionMeta
	err = json.Unmarshal(data, &meta)
	if err != nil {
		return sessionMeta{}, fmt.Errorf("memory: decode meta: %w", err)
	}
	return meta, nil
}

// writeMeta atomically writes the metadata file (temp + fsync + rename).
func (s *JSONLStore) writeMeta(key string, meta sessionMeta) error {
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("memory: encode meta: %w", err)
	}
	return writeFileAtomic(s.metaPath(key), data, 0o644)
}

// readStoredMessages reads valid JSON lines from a .jsonl file, skipping
// the first `skip` lines without unmarshaling them, and returns StoredMessages
// with seq numbers. Legacy lines (seq == 0) are assigned approximate seq
// numbers of (skip + lineNum).
func readStoredMessages(path string, skip int) ([]StoredMessage, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return []StoredMessage{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("memory: open jsonl: %w", err)
	}
	defer f.Close()

	var msgs []StoredMessage
	scanner := bufio.NewScanner(f)
	// Allow large lines for tool results (read_file, web search, etc.).
	scanner.Buffer(make([]byte, 0, 64*1024), maxLineSize)

	lineNum := 0
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		lineNum++
		if lineNum <= skip {
			continue
		}
		var stored StoredMessage
		if err := json.Unmarshal(line, &stored); err != nil {
			// Corrupt line — likely a partial write from a crash.
			logger.WarnCF("memory", "skipping corrupt line",
				map[string]any{"line": lineNum, "path": filepath.Base(path), "error": err.Error()})
			continue
		}
		// Migration: legacy lines written before Phase 2 have seq == 0.
		// Assign an approximate seq based on position.
		if stored.Seq == 0 {
			stored.Seq = int64(skip + lineNum)
		}
		msgs = append(msgs, stored)
	}
	if scanner.Err() != nil {
		return nil, fmt.Errorf("memory: scan jsonl: %w", scanner.Err())
	}

	if msgs == nil {
		msgs = []StoredMessage{}
	}
	return msgs, nil
}

// readMessages reads valid JSON lines from a .jsonl file, skipping
// the first `skip` lines without unmarshaling them. This avoids the
// cost of json.Unmarshal on logically truncated messages.
// Malformed trailing lines (e.g. from a crash) are silently skipped.
func readMessages(path string, skip int) ([]spawnllm.Message, error) {
	stored, err := readStoredMessages(path, skip)
	if err != nil {
		return nil, err
	}
	msgs := make([]spawnllm.Message, len(stored))
	for i, sm := range stored {
		msgs[i] = sm.Message
	}
	return msgs, nil
}

// countLines counts the total number of non-empty lines in a .jsonl file.
// Used by TruncateHistory to reconcile a stale meta.Count without
// the overhead of unmarshaling every message.
func countLines(path string) (int, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return 0, nil
	}
	if err != nil {
		return 0, fmt.Errorf("memory: open jsonl: %w", err)
	}
	defer f.Close()

	n := 0
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), maxLineSize)
	for scanner.Scan() {
		if len(scanner.Bytes()) > 0 {
			n++
		}
	}
	return n, scanner.Err()
}

func (s *JSONLStore) AddMessage(
	_ context.Context, sessionKey, role, content string,
) error {
	_, err := s.addMsg(sessionKey, spawnllm.Message{
		Role:    role,
		Content: content,
	})
	return err
}

func (s *JSONLStore) AddFullMessage(
	_ context.Context, sessionKey string, msg spawnllm.Message,
) (int64, error) {
	return s.addMsg(sessionKey, msg)
}

// addMsg is the shared implementation for AddMessage and AddFullMessage.
// It returns the monotonic sequence number assigned to the written message.
func (s *JSONLStore) addMsg(sessionKey string, msg spawnllm.Message) (int64, error) {
	l := s.sessionLock(sessionKey)
	l.Lock()
	defer l.Unlock()

	// Read meta first so we can assign the next seq number.
	meta, err := s.readMeta(sessionKey)
	if err != nil {
		return 0, err
	}

	// Assign monotonically increasing sequence number.
	seq := meta.NextSeq + 1
	if seq <= 0 {
		seq = 1
	}
	stored := NewStoredMessage(seq, msg)

	// Serialize as StoredMessage (includes seq field).
	line, err := json.Marshal(stored)
	if err != nil {
		return 0, fmt.Errorf("memory: marshal message: %w", err)
	}
	line = append(line, '\n')

	f, err := os.OpenFile(
		s.jsonlPath(sessionKey),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND,
		0o644,
	)
	if err != nil {
		return 0, fmt.Errorf("memory: open jsonl for append: %w", err)
	}
	_, writeErr := f.Write(line)
	if writeErr != nil {
		f.Close()
		return 0, fmt.Errorf("memory: append message: %w", writeErr)
	}
	// Flush to physical storage before closing. This matches the
	// durability guarantee of writeMeta and rewriteStoredJSONL (which use
	// WriteFileAtomic with fsync). Without Sync, a power loss could
	// leave the append in the kernel page cache only — lost on reboot.
	if syncErr := f.Sync(); syncErr != nil {
		f.Close()
		return 0, fmt.Errorf("memory: sync jsonl: %w", syncErr)
	}
	if closeErr := f.Close(); closeErr != nil {
		return 0, fmt.Errorf("memory: close jsonl: %w", closeErr)
	}

	// Determine if this message is noise (no new information).
	cache, noise := s.getNoiseCache(sessionKey)
	isNoisy := isNoise(stored, cache, noise)

	// Update metadata.
	now := time.Now()
	if meta.Count == 0 && meta.CreatedAt.IsZero() {
		meta.CreatedAt = now
	}
	meta.NextSeq = seq
	meta.Count++
	if !isNoisy {
		meta.MeaningfulCount++
	}
	meta.UpdatedAt = now

	// Update noise cache after computing isNoisy.
	updateNoiseCache(stored, cache, noise)

	logger.DebugCF("memory", "message_stored",
		map[string]any{
			"seq":     seq,
			"length":  len(msg.Content),
			"counted": !isNoisy,
			"role":    msg.Role,
			"session": sessionKey,
		})

	if writeMetaErr := s.writeMeta(sessionKey, meta); writeMetaErr != nil {
		return 0, writeMetaErr
	}
	return seq, nil
}

// GetHistoryWithSeqs returns the active history window for sessionKey with seq
// numbers intact. It mirrors GetHistory but returns []StoredMessage instead of
// stripping the seq field.
func (s *JSONLStore) GetHistoryWithSeqs(
	_ context.Context, sessionKey string,
) ([]StoredMessage, error) {
	l := s.sessionLock(sessionKey)
	l.Lock()
	defer l.Unlock()

	meta, err := s.readMeta(sessionKey)
	if err != nil {
		return nil, err
	}

	if meta.NextSeq == 0 {
		n, countErr := countLines(s.jsonlPath(sessionKey))
		if countErr != nil {
			return nil, countErr
		}
		if n > 0 {
			meta.NextSeq = int64(n)
			if saveErr := s.writeMeta(sessionKey, meta); saveErr != nil {
				logger.WarnCF("memory", "could not save migrated NextSeq",
					map[string]any{"session": sessionKey, "error": saveErr.Error()})
			}
		}
	}

	return readStoredMessages(s.jsonlPath(sessionKey), meta.Skip)
}

func (s *JSONLStore) GetHistory(
	_ context.Context, sessionKey string,
) ([]spawnllm.Message, error) {
	l := s.sessionLock(sessionKey)
	l.Lock()
	defer l.Unlock()

	meta, err := s.readMeta(sessionKey)
	if err != nil {
		return nil, err
	}

	// Migrate legacy files: if NextSeq is 0 and the file exists with
	// content, set NextSeq to the line count so future writes have
	// correct sequence numbers.
	if meta.NextSeq == 0 {
		n, countErr := countLines(s.jsonlPath(sessionKey))
		if countErr != nil {
			return nil, countErr
		}
		if n > 0 {
			meta.NextSeq = int64(n)
			if saveErr := s.writeMeta(sessionKey, meta); saveErr != nil {
				// Non-fatal: log and continue; seq will be approximate.
				logger.WarnCF("memory", "could not save migrated NextSeq",
					map[string]any{"session": sessionKey, "error": saveErr.Error()})
			}
		}
	}

	// Pass meta.Skip so readMessages skips those lines without
	// unmarshaling them — avoids wasted CPU on truncated messages.
	msgs, err := readMessages(s.jsonlPath(sessionKey), meta.Skip)
	if err != nil {
		return nil, err
	}

	return msgs, nil
}

func (s *JSONLStore) GetSummary(
	_ context.Context, sessionKey string,
) (string, error) {
	l := s.sessionLock(sessionKey)
	l.Lock()
	defer l.Unlock()

	meta, err := s.readMeta(sessionKey)
	if err != nil {
		return "", err
	}
	return meta.Summary, nil
}

func (s *JSONLStore) SetSummary(
	_ context.Context, sessionKey, summary string,
) error {
	l := s.sessionLock(sessionKey)
	l.Lock()
	defer l.Unlock()

	meta, err := s.readMeta(sessionKey)
	if err != nil {
		return err
	}
	now := time.Now()
	if meta.CreatedAt.IsZero() {
		meta.CreatedAt = now
	}
	meta.Summary = summary
	meta.UpdatedAt = now

	return s.writeMeta(sessionKey, meta)
}

func (s *JSONLStore) TruncateHistory(
	_ context.Context, sessionKey string, keepLast int,
) error {
	l := s.sessionLock(sessionKey)
	l.Lock()
	defer l.Unlock()

	meta, err := s.readMeta(sessionKey)
	if err != nil {
		return err
	}

	// Always reconcile meta.Count with the actual line count on disk.
	// A crash between the JSONL append and the meta update in addMsg
	// leaves meta.Count stale (e.g. file has 101 lines but meta says
	// 100). Counting lines is cheap — no unmarshal, just a scan — and
	// TruncateHistory is not a hot path, so always re-count.
	n, countErr := countLines(s.jsonlPath(sessionKey))
	if countErr != nil {
		return countErr
	}
	meta.Count = n

	if keepLast <= 0 {
		meta.Skip = meta.Count
		// Full reset: clear tracking fields.
		// Archive deletion is handled by ContextManager.Reset() via ArchiveStore.Delete().
		meta.MeaningfulCount = 0
		meta.CompressedAtMeaningfulCount = 0
		meta.CompressionCooling = false
	} else {
		effective := meta.Count - meta.Skip
		if keepLast < effective {
			meta.Skip = meta.Count - keepLast
		}
	}
	meta.UpdatedAt = time.Now()

	return s.writeMeta(sessionKey, meta)
}

func (s *JSONLStore) SetHistory(
	_ context.Context,
	sessionKey string,
	history []spawnllm.Message,
) error {
	l := s.sessionLock(sessionKey)
	l.Lock()
	defer l.Unlock()

	meta, err := s.readMeta(sessionKey)
	if err != nil {
		return err
	}
	now := time.Now()
	if meta.CreatedAt.IsZero() {
		meta.CreatedAt = now
	}
	meta.Skip = 0
	meta.Count = len(history)
	meta.UpdatedAt = now

	// Write meta BEFORE rewriting the JSONL file. If we crash between
	// the two writes, meta has Skip=0 and the old file is still intact,
	// so GetHistory reads from line 1 — returning "too many" messages
	// rather than losing data. The next SetHistory call corrects this.
	err = s.writeMeta(sessionKey, meta)
	if err != nil {
		return err
	}

	// Assign new seq numbers starting from current NextSeq + 1 so that
	// the seq numbers remain monotonically increasing across rewrites.
	startSeq := meta.NextSeq
	stored := make([]StoredMessage, len(history))
	for i, msg := range history {
		startSeq++
		// SetHistory takes []spawnllm.Message — no prior CreatedAt to carry
		// over, so NewStoredMessage stamps time.Now(). This loses the original
		// per-message timestamps, which is a known limitation of the rewrite
		// path; before the constructor existed, the field was simply zero.
		stored[i] = NewStoredMessage(startSeq, msg)
	}

	if err := s.rewriteStoredJSONL(sessionKey, stored); err != nil {
		return err
	}

	// Update NextSeq now that we have written new seq numbers.
	meta.NextSeq = startSeq
	return s.writeMeta(sessionKey, meta)
}

// SetHistoryWithSeqs replaces all messages while preserving stable seq numbers.
// It is intended for compaction: retained tail messages keep the IDs already
// advertised in summaries and written to the archive.
func (s *JSONLStore) SetHistoryWithSeqs(
	_ context.Context,
	sessionKey string,
	history []StoredMessage,
) error {
	l := s.sessionLock(sessionKey)
	l.Lock()
	defer l.Unlock()

	meta, err := s.readMeta(sessionKey)
	if err != nil {
		return err
	}
	now := time.Now()
	if meta.CreatedAt.IsZero() {
		meta.CreatedAt = now
	}
	meta.Skip = 0
	meta.Count = len(history)
	meta.UpdatedAt = now

	maxSeq := meta.NextSeq
	stored := make([]StoredMessage, len(history))
	for i, sm := range history {
		if sm.Seq <= 0 {
			maxSeq++
			sm.Seq = maxSeq
		}
		if sm.Seq > maxSeq {
			maxSeq = sm.Seq
		}
		stored[i] = NewStoredMessageAt(sm.Seq, sm.Message, sm.CreatedAt)
	}
	meta.NextSeq = maxSeq

	err = s.writeMeta(sessionKey, meta)
	if err != nil {
		return err
	}
	return s.rewriteStoredJSONL(sessionKey, stored)
}

// Compact physically rewrites the JSONL file, dropping all logically
// skipped lines. This reclaims disk space that accumulates after
// repeated TruncateHistory calls.
//
// It is safe to call at any time; if there is nothing to compact
// (skip == 0) the method returns immediately.
func (s *JSONLStore) Compact(
	_ context.Context, sessionKey string,
) error {
	l := s.sessionLock(sessionKey)
	l.Lock()
	defer l.Unlock()

	meta, err := s.readMeta(sessionKey)
	if err != nil {
		return err
	}
	if meta.Skip == 0 {
		return nil
	}

	// Read only the active StoredMessages, preserving their seq numbers.
	active, err := readStoredMessages(s.jsonlPath(sessionKey), meta.Skip)
	if err != nil {
		return err
	}

	// Write meta BEFORE rewriting the JSONL file. If the process
	// crashes between the two writes, meta has Skip=0 and the old
	// (uncompacted) file is still intact, so GetHistory reads from
	// line 1 — returning previously-truncated messages rather than
	// losing data. The next Compact or TruncateHistory corrects this.
	meta.Skip = 0
	meta.Count = len(active)
	meta.UpdatedAt = time.Now()

	err = s.writeMeta(sessionKey, meta)
	if err != nil {
		return err
	}

	return s.rewriteStoredJSONL(sessionKey, active)
}

// AppendSummaryCheckpoint appends a generated summary to the per-session
// checkpoint log. It fills ID, PrevHash, SummaryHash, and GeneratedAt.
func (s *JSONLStore) AppendSummaryCheckpoint(
	_ context.Context,
	sessionKey string,
	checkpoint SummaryCheckpoint,
) error {
	l := s.sessionLock(sessionKey)
	l.Lock()
	defer l.Unlock()

	path := s.summaryHistoryPath(sessionKey)
	prevHash := ""
	nextID := 1
	if f, err := os.Open(path); err == nil {
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 0, 64*1024), maxLineSize)
		for scanner.Scan() {
			var prior SummaryCheckpoint
			if json.Unmarshal(scanner.Bytes(), &prior) == nil {
				if prior.ID >= nextID {
					nextID = prior.ID + 1
				}
				if prior.SummaryHash != "" {
					prevHash = prior.SummaryHash
				}
			}
		}
		if scanErr := scanner.Err(); scanErr != nil {
			f.Close()
			return fmt.Errorf("memory: scan summary history: %w", scanErr)
		}
		if closeErr := f.Close(); closeErr != nil {
			return fmt.Errorf("memory: close summary history: %w", closeErr)
		}
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("memory: open summary history: %w", err)
	}

	if checkpoint.GeneratedAt.IsZero() {
		checkpoint.GeneratedAt = time.Now().UTC()
	} else {
		checkpoint.GeneratedAt = checkpoint.GeneratedAt.UTC()
	}
	checkpoint.ID = nextID
	checkpoint.PrevHash = prevHash
	sum := sha256.Sum256([]byte(checkpoint.Summary))
	checkpoint.SummaryHash = hex.EncodeToString(sum[:])

	line, err := json.Marshal(checkpoint)
	if err != nil {
		return fmt.Errorf("memory: marshal summary checkpoint: %w", err)
	}
	line = append(line, '\n')

	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("memory: open summary history for append: %w", err)
	}
	if _, err := f.Write(line); err != nil {
		f.Close()
		return fmt.Errorf("memory: append summary checkpoint: %w", err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return fmt.Errorf("memory: sync summary history: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("memory: close summary history: %w", err)
	}
	return nil
}

// rewriteStoredJSONL atomically replaces the JSONL file with the given
// StoredMessages using the project's standard WriteFileAtomic (temp + fsync + rename).
func (s *JSONLStore) rewriteStoredJSONL(
	sessionKey string, msgs []StoredMessage,
) error {
	var buf bytes.Buffer
	for i, msg := range msgs {
		line, err := json.Marshal(msg)
		if err != nil {
			return fmt.Errorf("memory: marshal message %d: %w", i, err)
		}
		buf.Write(line)
		buf.WriteByte('\n')
	}
	return writeFileAtomic(s.jsonlPath(sessionKey), buf.Bytes(), 0o644)
}

func (s *JSONLStore) SetPendingTurn(ctx context.Context, sessionKey string) error {
	l := s.sessionLock(sessionKey)
	l.Lock()
	defer l.Unlock()
	meta, err := s.readMeta(sessionKey)
	if err != nil {
		return err
	}
	meta.PendingTurn = true
	return s.writeMeta(sessionKey, meta)
}

func (s *JSONLStore) ClearPendingTurn(ctx context.Context, sessionKey string) error {
	l := s.sessionLock(sessionKey)
	l.Lock()
	defer l.Unlock()
	meta, err := s.readMeta(sessionKey)
	if err != nil {
		return err
	}
	meta.PendingTurn = false
	return s.writeMeta(sessionKey, meta)
}

// ListPendingSessions scans the store directory for .meta.json files and
// returns the session key of every session where PendingTurn is true.
// Files that cannot be read or parsed are silently skipped.
// Returns nil, nil if the directory does not exist.
func (s *JSONLStore) ListPendingSessions(_ context.Context) ([]string, error) {
	entries, err := os.ReadDir(s.dir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("memory: list pending sessions: %w", err)
	}

	var keys []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasSuffix(name, ".meta.json") {
			continue
		}
		path := filepath.Join(s.dir, name)
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			continue
		}
		var meta sessionMeta
		if json.Unmarshal(data, &meta) != nil {
			continue
		}
		if meta.PendingTurn && meta.Key != "" {
			keys = append(keys, meta.Key)
		}
	}
	return keys, nil
}

// GetArchiveBounds returns (0, 0, nil). Archive bounds are now owned by
// ContextManager via ArchiveStore.Bounds(); the JSONLStore no longer tracks them.
func (s *JSONLStore) GetArchiveBounds(_ context.Context, _ string) (minSeq, maxSeq int64, err error) {
	return 0, 0, nil
}

// CompactionState holds the durable compression counters for a session.
// Defined here so JSONLStore can return it without importing llmcontext
// (which would create a circular import).
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

// GetCompactionState reads the compaction counters from the session meta file.
func (s *JSONLStore) GetCompactionState(sessionKey string) (CompactionState, error) {
	l := s.sessionLock(sessionKey)
	l.Lock()
	defer l.Unlock()

	meta, err := s.readMeta(sessionKey)
	if err != nil {
		return CompactionState{}, err
	}
	return CompactionState{
		MeaningfulCount:             meta.MeaningfulCount,
		CompressedAtMeaningfulCount: meta.CompressedAtMeaningfulCount,
		Cooling:                     meta.CompressionCooling,
		CoolingSinceCount:           meta.CoolingSinceCount,
		SummaryGeneratedAt:          meta.SummaryGeneratedAt,
		SummaryModel:                meta.SummaryModel,
		ActiveModelIndex:            meta.ActiveModelIndex,
		ExposeReasoning:             meta.ExposeReasoning,
		ShowToolActivity:            meta.ShowToolActivity,
	}, nil
}

// SetCompactionState writes the compaction counters back to the session meta file.
func (s *JSONLStore) SetCompactionState(sessionKey string, state CompactionState) error {
	l := s.sessionLock(sessionKey)
	l.Lock()
	defer l.Unlock()

	meta, err := s.readMeta(sessionKey)
	if err != nil {
		return err
	}
	meta.MeaningfulCount = state.MeaningfulCount
	meta.CompressedAtMeaningfulCount = state.CompressedAtMeaningfulCount
	meta.CompressionCooling = state.Cooling
	meta.CoolingSinceCount = state.CoolingSinceCount
	meta.SummaryGeneratedAt = state.SummaryGeneratedAt
	meta.SummaryModel = state.SummaryModel
	meta.ActiveModelIndex = state.ActiveModelIndex
	meta.ExposeReasoning = state.ExposeReasoning
	meta.ShowToolActivity = state.ShowToolActivity
	meta.UpdatedAt = time.Now()
	return s.writeMeta(sessionKey, meta)
}

func (s *JSONLStore) Close() error {
	return nil
}
