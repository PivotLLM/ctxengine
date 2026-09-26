// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"errors"
	"testing"
)

// failingLLM returns a model that fails every call, with enough failures to
// trip the breaker and then fail the safety-net pass that follows.
func failingLLM(n int) *mockLLM {
	errs := make([]error, n)
	for i := range errs {
		errs[i] = errors.New("model down")
	}
	return &mockLLM{model: "down", errors: errs}
}

// tripBreaker fails the automatic path until the breaker trips.
func tripBreaker(t *testing.T, mgr *Manager) {
	t.Helper()
	for range defaultMaxConsecutiveCompactFailures {
		_ = mgr.compress(context.Background(), false)
	}
	if !mgr.autoCompactionSuppressed() {
		t.Fatal("breaker did not trip after the failure threshold")
	}
}

// TestBreaker_SafetyNetBypassesBreaker is the regression for the breaker bug:
// with the breaker tripped, compress() used to return nil for a safety-net
// pass, the emergency paths read that as "compacted", and an oversized
// request went out to the provider. The safety net must run regardless and,
// with the model still failing, fall through to the drop-only fallback so the
// assembled request fits.
func TestBreaker_SafetyNetBypassesBreaker(t *testing.T) {
	// contextWindow=1000, safety=80 → line at 800 tokens. 10 pairs × 200
	// chars = 4000 chars ≈ 1000 tokens: the window is at 100%.
	store := &compressTestStore{history: makeConversation(10, 200)}
	llm := failingLLM(60)
	mgr := newCompressManager(store, []*mockLLM{llm}, WithContextWindow(1000))
	mgr.msgCount = len(store.history)

	tripBreaker(t, mgr)
	callsBefore := llm.callCount

	asm, err := mgr.Assemble(context.Background(), AssembleRequest{})
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if !asm.Compacted {
		t.Fatal("safety-net pass did not run while the breaker was tripped")
	}
	if llm.callCount == callsBefore {
		t.Error("safety-net pass did not reach the model: the breaker still gates it")
	}
	if pct := mgr.contextPercent(asm.Messages); pct >= float64(mgr.cfg.safetyPercent) {
		t.Errorf("assembled request still past the safety line: %.0f%% of the window (%d messages)", pct, len(asm.Messages))
	}
	if pct := mgr.contextPercent(store.GetHistory("sess")); pct >= float64(mgr.cfg.safetyPercent) {
		t.Errorf("stored window still past the safety line: %.0f%%", pct)
	}
}

// TestBreaker_NormalPathStillSuppressed pins the other half: the breaker still
// gates the normal-trigger path, which is what it exists for.
func TestBreaker_NormalPathStillSuppressed(t *testing.T) {
	store := &compressTestStore{history: makeConversation(10, 200)}
	llm := failingLLM(60)
	mgr := newCompressManager(store, []*mockLLM{llm})
	mgr.msgCount = len(store.history)

	tripBreaker(t, mgr)
	callsBefore := llm.callCount
	if err := mgr.compress(context.Background(), false); err != nil {
		t.Fatalf("suppressed compress returned %v", err)
	}
	if llm.callCount != callsBefore {
		t.Errorf("normal path called the model %d times while the breaker was tripped", llm.callCount-callsBefore)
	}
}

// TestBreaker_TrippedHookFires verifies the host hook fires once per trip with
// the session key and the failure count, and not before the threshold.
func TestBreaker_TrippedHookFires(t *testing.T) {
	store := &compressTestStore{history: makeConversation(10, 200)}
	llm := failingLLM(60)

	type trip struct {
		key      string
		failures int
	}
	var trips []trip
	mgr := newCompressManager(store, []*mockLLM{llm}, WithBreakerTrippedHook(func(key string, failures int) {
		trips = append(trips, trip{key, failures})
	}))
	mgr.msgCount = len(store.history)

	for range defaultMaxConsecutiveCompactFailures - 1 {
		_ = mgr.compress(context.Background(), false)
	}
	if len(trips) != 0 {
		t.Fatalf("hook fired before the threshold: %+v", trips)
	}
	_ = mgr.compress(context.Background(), false)
	if len(trips) != 1 {
		t.Fatalf("hook fired %d times, want once", len(trips))
	}
	if trips[0].key != "sess" || trips[0].failures != defaultMaxConsecutiveCompactFailures {
		t.Errorf("hook args = %+v, want {sess %d}", trips[0], defaultMaxConsecutiveCompactFailures)
	}

	// Further suppressed attempts do not re-fire the hook.
	_ = mgr.compress(context.Background(), false)
	if len(trips) != 1 {
		t.Errorf("hook re-fired while already tripped: %d calls", len(trips))
	}
}
