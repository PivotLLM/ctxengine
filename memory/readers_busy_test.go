// ctxengine
// License: MIT

package memory

import (
	"context"
	"database/sql"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/PivotLLM/spawnllm"
)

// TestReadOnlyConnections_WaitOutLock holds an exclusive lock on an archive
// while every read path opens its own connection against it. The readers must
// wait for the lock to clear and then succeed, not fail with "database is
// locked". A WAL reader is never blocked by a WAL writer, so to make the lock
// observable the file is switched to a rollback journal first — the shape a
// reader also meets during WAL recovery, a journal-mode switch, or on a
// database that has not been opened by the writer yet.
func TestReadOnlyConnections_WaitOutLock(t *testing.T) {
	path := filepath.Join(t.TempDir(), "busy.archive.db")
	a, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	now := time.Now()
	if err = a.Append(1, spawnllm.Message{Role: "user", Content: "hello archive"}, now); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if _, err = a.AppendSummary(SummaryRecord{Summary: "s"}); err != nil {
		t.Fatalf("AppendSummary: %v", err)
	}
	if err = a.ReplaceWindow([]StoredMessage{NewStoredMessage(1, spawnllm.Message{Role: "user", Content: "hello archive"})},
		SessionState{Key: "k", NextSeq: 1}); err != nil {
		t.Fatalf("ReplaceWindow: %v", err)
	}
	if err = a.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Take the lock: rollback journal, then an exclusive write transaction.
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("raw open: %v", err)
	}
	defer func() { noErr(t, raw.Close()) }()
	raw.SetMaxOpenConns(1) // BEGIN and COMMIT must run on the same connection
	if _, err = raw.Exec("PRAGMA journal_mode=DELETE"); err != nil {
		t.Fatalf("journal_mode=DELETE: %v", err)
	}
	if _, err = raw.Exec("BEGIN EXCLUSIVE"); err != nil {
		t.Fatalf("BEGIN EXCLUSIVE: %v", err)
	}
	const hold = 500 * time.Millisecond
	release := time.AfterFunc(hold, func() {
		if _, err = raw.Exec("COMMIT"); err != nil {
			t.Errorf("COMMIT: %v", err)
		}
	})
	defer release.Stop()

	ro, err := OpenReadOnly(path)
	if err != nil {
		t.Fatalf("OpenReadOnly: %v", err)
	}
	reads := map[string]func() error{
		"Bounds":        func() error { _, _, err := ro.Bounds(); return err },
		"QueryRange":    func() error { _, err := ro.QueryRange(1, 1); return err },
		"Search":        func() error { _, err := ro.Search(context.Background(), "hello", "", 10); return err },
		"Stats":         func() error { _, _, _, err := ro.Stats(); return err },
		"MinSeqAfter":   func() error { _, err := ro.MinSeqAfter(now.Add(-time.Hour)); return err },
		"ListSummaries": func() error { _, err := ro.ListSummaries(); return err },
		"GetSummary":    func() error { _, _, err := ro.GetSummary(1); return err },
		"Window":        func() error { _, err := ro.Window(); return err },
		"State":         func() error { _, err := ro.State(); return err },
	}

	start := time.Now()
	var wg sync.WaitGroup
	var mu sync.Mutex
	failures := map[string]error{}
	for name, read := range reads {
		wg.Go(func() {
			if err := read(); err != nil {
				mu.Lock()
				failures[name] = err
				mu.Unlock()
			}
		})
	}
	wg.Wait()

	for name, err := range failures {
		t.Errorf("%s failed while the writer held the lock: %v", name, err)
	}
	if waited := time.Since(start); waited < hold/2 {
		t.Errorf("readers returned after %v; the lock was held for %v, so they cannot have waited for it", waited, hold)
	}
}
