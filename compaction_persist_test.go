// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"testing"

	"github.com/PivotLLM/ctxengine/memory"
	"github.com/PivotLLM/spawnllm"
)

// persistentMockStore wraps mockStore and adds CompactionStateStore support,
// simulating the SQLiteStore's durable state persistence.
type persistentMockStore struct {
	*mockStore
	states map[string]memory.CompactionState
}

func newPersistentMockStore() *persistentMockStore {
	return &persistentMockStore{
		mockStore: newMockStore(),
		states:    make(map[string]memory.CompactionState),
	}
}

// AddFullMessage advances the meaningful count the way a real compaction-state
// store does: the store owns the count and the manager reads it back.
func (s *persistentMockStore) AddFullMessage(sessionKey string, msg spawnllm.Message) (int64, error) {
	seq, err := s.mockStore.AddFullMessage(sessionKey, msg)
	if err != nil {
		return 0, err
	}
	st := s.states[sessionKey]
	st.MeaningfulCount++
	s.states[sessionKey] = st
	return seq, nil
}

func (s *persistentMockStore) GetCompactionState(sessionKey string) (memory.CompactionState, error) {
	return s.states[sessionKey], nil
}

func (s *persistentMockStore) SetCompactionState(sessionKey string, state memory.CompactionState) error {
	s.states[sessionKey] = state
	return nil
}

// TestCompactionState_PersistedAndRestoredOnRestart simulates a process restart
// by creating a second Manager on the same store and verifying that the
// compaction state (msgCount, cooldown) is restored from durable storage.
func TestCompactionState_PersistedAndRestoredOnRestart(t *testing.T) {
	const sessionKey = "restart-session"

	store := newPersistentMockStore()

	// Pre-populate history with enough messages to trigger compression.
	history := makeConversation(10, 200)
	store.SetHistory(sessionKey, history)

	// Build a Manager with a successful LLM client so compression succeeds.
	llm := &mockLLM{
		responses: []string{validSummaryJSON("persistent goal")},
	}
	mgr1 := New(sessionKey, store,
		WithContextWindow(2000),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithRetainTokenPercent(20),
		WithRetainMinMessages(2),
		WithModelCaller(llm),
	).(*Manager)
	// The store owns the meaningful count; seed it the way a real store would
	// have after len(history) non-noise messages, and let the manager read it.
	store.states[sessionKey] = memory.CompactionState{MeaningfulCount: len(history)}
	mgr1.msgCount = len(history)

	// Run compression to produce persisted state.
	if err := mgr1.doCompress(context.Background(), false); err != nil {
		t.Fatalf("doCompress returned error: %v", err)
	}

	// Check that state was written to the store.
	state, _ := store.GetCompactionState(sessionKey)
	if state.MeaningfulCount == 0 {
		t.Error("expected MeaningfulCount to be persisted after compression")
	}

	// Simulate restart: create a second Manager on the same store.
	mgr2 := New(sessionKey, store,
		WithContextWindow(2000),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithRetainTokenPercent(20),
		WithRetainMinMessages(2),
	).(*Manager)

	// Verify that the in-memory state was loaded from durable storage.
	if mgr2.msgCount != mgr1.msgCount {
		t.Errorf("msgCount not restored: got %d, want %d", mgr2.msgCount, mgr1.msgCount)
	}
	if mgr2.compressedAtCount != mgr1.compressedAtCount {
		t.Errorf("compressedAtCount not restored: got %d, want %d",
			mgr2.compressedAtCount, mgr1.compressedAtCount)
	}
}

// TestCompactionState_CoolingRestoredOnRestart verifies that the cooling flag
// and coolingSinceCount are correctly restored from durable storage.
func TestCompactionState_CoolingRestoredOnRestart(t *testing.T) {
	const sessionKey = "cooling-session"
	store := newPersistentMockStore()

	// Write a state that has cooling = true.
	wantState := memory.CompactionState{
		MeaningfulCount:             42,
		CompressedAtMeaningfulCount: 40,
		Cooling:                     true,
		CoolingSinceCount:           40,
	}
	if err := store.SetCompactionState(sessionKey, wantState); err != nil {
		t.Fatalf("SetCompactionState: %v", err)
	}

	mgr := New(sessionKey, store).(*Manager)

	if mgr.msgCount != wantState.MeaningfulCount {
		t.Errorf("msgCount: got %d, want %d", mgr.msgCount, wantState.MeaningfulCount)
	}
	if !mgr.cooling {
		t.Error("expected cooling=true to be restored")
	}
	if mgr.coolingSinceCount != wantState.CoolingSinceCount {
		t.Errorf("coolingSinceCount: got %d, want %d", mgr.coolingSinceCount, wantState.CoolingSinceCount)
	}
}

// TestCompactionState_InMemoryStoreWorksWithZeroState verifies that a store
// that does not implement CompactionStateStore (e.g. the in-memory mockStore)
// causes the Manager to start with zero state, not an error.
func TestCompactionState_InMemoryStoreWorksWithZeroState(t *testing.T) {
	// mockStore does NOT implement CompactionStateStore.
	store := newMockStore()
	store.history["zero-session"] = []spawnllm.Message{
		{Role: "user", Content: "hello"},
	}

	mgr := New("zero-session", store).(*Manager)

	if mgr.msgCount != 0 {
		t.Errorf("expected msgCount=0 for in-memory store; got %d", mgr.msgCount)
	}
	if mgr.cooling {
		t.Error("expected cooling=false for in-memory store")
	}
}

// TestStats_ReturnsMsgCount verifies that Stats().MeaningfulMessages reflects
// the in-memory msgCount rather than returning 0.
func TestStats_ReturnsMsgCount(t *testing.T) {
	store := newMockStore()
	store.history["stats-session"] = makeConversation(3, 50)

	mgr := New("stats-session", store).(*Manager)
	mgr.msgCount = 7

	stats := mgr.Stats()
	if stats.MeaningfulMessages != 7 {
		t.Errorf("Stats().MeaningfulMessages: got %d, want 7", stats.MeaningfulMessages)
	}
}

// TestCompactionState_WrittenAfterPersistResult verifies that SetCompactionState
// is called on the store after a successful persistResult.
func TestCompactionState_WrittenAfterPersistResult(t *testing.T) {
	const sessionKey = "persist-state-session"
	store := newPersistentMockStore()

	history := makeConversation(10, 200)
	store.SetHistory(sessionKey, history)

	llm := &mockLLM{responses: []string{validSummaryJSON("write-back goal")}}
	mgr := New(sessionKey, store,
		WithContextWindow(2000),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithRetainTokenPercent(20),
		WithRetainMinMessages(2),
		WithModelCaller(llm),
	).(*Manager)
	mgr.msgCount = len(history)

	if err := mgr.doCompress(context.Background(), false); err != nil {
		t.Fatalf("doCompress: %v", err)
	}

	state, err := store.GetCompactionState(sessionKey)
	if err != nil {
		t.Fatalf("GetCompactionState: %v", err)
	}
	// After doCompress, compressedAtCount is set to msgCount.
	if state.CompressedAtMeaningfulCount != mgr.compressedAtCount {
		t.Errorf("CompressedAtMeaningfulCount: got %d, want %d",
			state.CompressedAtMeaningfulCount, mgr.compressedAtCount)
	}
}
