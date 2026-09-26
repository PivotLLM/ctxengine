// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine/memory"
	"github.com/PivotLLM/ctxengine/session"
)

// The real store commits a compaction atomically.
var _ CompactionCommitter = (*session.SQLiteStore)(nil)

// TestCompaction_CommitsThroughSQLiteStore drives a compaction on the real
// store and checks that the window, the current summary, the summary
// checkpoint and the compaction counters all describe the same pass — the
// path that is one transaction on this store.
func TestCompaction_CommitsThroughSQLiteStore(t *testing.T) {
	dir := t.TempDir()
	store, err := session.NewSQLiteStore(dir)
	if err != nil {
		t.Fatalf("NewSQLiteStore: %v", err)
	}
	defer func() { noErr(t, store.Close()) }()

	for range 10 {
		for _, role := range []string{"user", "assistant"} {
			if _, err = store.AddFullMessage("sess", spawnllm.Message{Role: role, Content: strings.Repeat(role[:1], 200)}); err != nil {
				t.Fatalf("seed: %v", err)
			}
		}
	}

	llm := &mockLLM{model: "committer", responses: []string{validSummaryJSON("committed goal")}}
	mgr := asManager(t, New("sess", store,
		WithArchiveDir(dir),
		WithModelCaller(llm),
		WithContextWindow(1000),
		WithOverheadTokens(0),
		WithRetainTokenPercent(20),
		WithRetainMinMessages(2),
	))
	defer func() { noErr(t, mgr.Close(context.Background())) }()

	if err = mgr.Compact(context.Background()); err != nil {
		t.Fatalf("Compact: %v", err)
	}

	window := store.GetHistoryWithSeqs("sess")
	if len(window) == 0 || len(window) >= 20 || window[len(window)-1].Seq != 20 {
		t.Fatalf("window after compaction = %d rows ending at seq %d, want a shorter tail ending at #20",
			len(window), window[len(window)-1].Seq)
	}
	raw := store.GetSummary("sess")
	if !strings.Contains(raw, "committed goal") {
		t.Fatalf("current summary = %q, want the committed one", raw)
	}

	a, err := memory.OpenReadOnly(memory.ArchivePath(dir, "sess"))
	if err != nil {
		t.Fatalf("OpenReadOnly: %v", err)
	}
	metas, err := a.ListSummaries()
	if err != nil || len(metas) != 1 {
		t.Fatalf("checkpoints = %+v, %v; want exactly one", metas, err)
	}
	rec, ok, err := a.GetSummary(metas[0].ID)
	if err != nil || !ok {
		t.Fatalf("GetSummary: ok=%v err=%v", ok, err)
	}
	if rec.Summary != raw {
		t.Errorf("checkpoint body differs from the current summary:\n checkpoint: %s\n current:    %s", rec.Summary, raw)
	}
	if rec.Model != "committer" {
		t.Errorf("checkpoint model = %q, want committer", rec.Model)
	}

	st, err := store.GetCompactionState("sess")
	if err != nil {
		t.Fatalf("GetCompactionState: %v", err)
	}
	if st.CompressedAtMeaningfulCount != mgr.compressedAtCount || st.SummaryModel != "committer" {
		t.Errorf("compaction state = %+v, want compressed_at %d from model committer", st, mgr.compressedAtCount)
	}
}

// failingWriteStore is a mockStore whose message and truncate writes fail.
type failingWriteStore struct {
	*mockStore
	err error
}

func (s *failingWriteStore) AddFullMessage(string, spawnllm.Message) (int64, error) { return 0, s.err }
func (s *failingWriteStore) TruncateHistory(string, int) error                      { return s.err }

// TestAddMessages_ReturnStoreError checks that a failed store write comes back
// from every Add* and that the unstored message is not counted.
func TestAddMessages_ReturnStoreError(t *testing.T) {
	boom := errors.New("disk full")
	store := &failingWriteStore{mockStore: newMockStore(), err: boom}
	mgr := asManager(t, New("sess", store, WithContextWindow(10000)))
	ctx := context.Background()
	msg := spawnllm.Message{Role: "user", Content: "hi"}

	adds := map[string]func() (int64, error){
		"AddUserMessage":      func() (int64, error) { return mgr.AddUserMessage(ctx, msg) },
		"AddAssistantMessage": func() (int64, error) { return mgr.AddAssistantMessage(ctx, msg) },
		"AddToolCallMessage":  func() (int64, error) { return mgr.AddToolCallMessage(ctx, msg) },
		"AddToolResult":       func() (int64, error) { return mgr.AddToolResult(ctx, msg) },
	}
	for name, add := range adds {
		seq, err := add()
		if !errors.Is(err, boom) {
			t.Errorf("%s: err = %v, want the store's error", name, err)
		}
		if seq != 0 {
			t.Errorf("%s: seq = %d on a failed write, want 0", name, seq)
		}
	}
	if mgr.msgCount != 0 {
		t.Errorf("msgCount = %d after failed writes, want 0", mgr.msgCount)
	}
}

// TestReset_ReturnsStoreError checks that Reset reports a failed truncate
// instead of claiming the session was cleared.
func TestReset_ReturnsStoreError(t *testing.T) {
	boom := errors.New("disk full")
	store := &failingWriteStore{mockStore: newMockStore(), err: boom}
	mgr := asManager(t, New("sess", store, WithContextWindow(10000)))
	if err := mgr.Reset(context.Background()); !errors.Is(err, boom) {
		t.Errorf("Reset err = %v, want the store's error", err)
	}
}
