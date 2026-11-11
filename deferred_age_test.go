/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/PivotLLM/spawnllm"
)

// TestAgeTrigger_DeferredFromAdds checks that the age trigger alone never
// compacts inside the adds a turn waits on: it marks the pass due, CompactIfDue
// runs it once and clears it, and a second CompactIfDue is a no-op.
func TestAgeTrigger_DeferredFromAdds(t *testing.T) {
	tests := []struct {
		name string
		add  func(*Manager, context.Context, spawnllm.Message) (int64, error)
		role string
	}{
		{"user", (*Manager).AddUserMessage, "user"},
		{"assistant", (*Manager).AddAssistantMessage, "assistant"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			store := newAgingStore()
			mgr := newTestManager(t, store,
				WithContextWindow(1_000_000),
				WithMinPercent(20),
				WithMessageThreshold(0),
				WithTriggerDays(7),
				WithRetainMaxAgeDays(5),
			)
			calls := 0
			mgr.SetTestCompressHook(func(bool) { calls++ })

			store.ages[0] = 30 * 24 * time.Hour
			if _, err := tt.add(mgr, ctx, spawnllm.Message{Role: tt.role, Content: "hi Bob"}); err != nil {
				t.Fatalf("add: %v", err)
			}
			if calls != 0 {
				t.Fatalf("the add compacted %d times; the age trigger alone must be deferred", calls)
			}
			if !mgr.ageCompactionDue {
				t.Fatal("the add did not mark the age compaction due")
			}

			ran, err := mgr.CompactIfDue(ctx)
			if err != nil || !ran {
				t.Fatalf("CompactIfDue = %v, %v; want a pass", ran, err)
			}
			if calls != 1 {
				t.Fatalf("CompactIfDue compacted %d times, want 1", calls)
			}
			if mgr.ageCompactionDue {
				t.Error("CompactIfDue left the age compaction due")
			}

			ran, err = mgr.CompactIfDue(ctx)
			if err != nil || ran {
				t.Fatalf("second CompactIfDue = %v, %v; want a no-op", ran, err)
			}
			if calls != 1 {
				t.Errorf("second CompactIfDue compacted; calls = %d", calls)
			}
		})
	}
}

// TestCompactIfDue_NothingDue checks the no-op on a session nothing marked.
func TestCompactIfDue_NothingDue(t *testing.T) {
	mgr := newTestManager(t, newMockStore(), WithTriggerDays(7))
	calls := 0
	mgr.SetTestCompressHook(func(bool) { calls++ })

	ran, err := mgr.CompactIfDue(context.Background())
	if err != nil || ran || calls != 0 {
		t.Errorf("CompactIfDue = %v, %v with %d passes; want a no-op", ran, err, calls)
	}
}

// TestCompactIfDue_RechecksAge checks that a pass marked due but no longer owed
// (the window moved inside the trigger age in the meantime) does not run.
func TestCompactIfDue_RechecksAge(t *testing.T) {
	store := newAgingStore()
	mgr := newTestManager(t, store,
		WithContextWindow(1_000_000),
		WithMinPercent(20),
		WithMessageThreshold(0),
		WithTriggerDays(7),
	)
	calls := 0
	mgr.SetTestCompressHook(func(bool) { calls++ })

	store.ages[0] = 30 * 24 * time.Hour
	if _, err := mgr.AddUserMessage(context.Background(), msgWithContent("hi")); err != nil {
		t.Fatal(err)
	}
	delete(store.ages, 0) // the old message is now fresh

	ran, err := mgr.CompactIfDue(context.Background())
	if err != nil || ran || calls != 0 {
		t.Errorf("CompactIfDue = %v, %v with %d passes; want a no-op", ran, err, calls)
	}
	if mgr.ageCompactionDue {
		t.Error("a pass no longer owed must clear the due mark")
	}
}

// TestAgeTrigger_WithThresholdCompactsInline checks that when the percentage
// trigger fires too, the add compacts inline as before and nothing is left due.
func TestAgeTrigger_WithThresholdCompactsInline(t *testing.T) {
	store := newAgingStore()
	mgr := newTestManager(t, store,
		WithContextWindow(100),
		WithMinPercent(10),
		WithNormalPercent(50),
		WithSafetyPercent(95),
		WithMessageThreshold(0),
		WithTriggerDays(7),
	)
	calls := 0
	mgr.SetTestCompressHook(func(bool) { calls++ })

	store.ages[0] = 30 * 24 * time.Hour
	big := msgWithContent(strings.Repeat("a", 240)) // 60 tokens: 60% of the window
	if _, err := mgr.AddUserMessage(context.Background(), big); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want the threshold pass inline", calls)
	}
	if mgr.ageCompactionDue {
		t.Error("an inline pass must not leave the age compaction due")
	}
}

// startRecorder collects compaction-start events and the ctx value they saw.
type startRecorder struct {
	events []CompactionStart
	chats  []any
}

type chatKey struct{}

func (r *startRecorder) record(ctx context.Context, info CompactionStart) {
	r.events = append(r.events, info)
	r.chats = append(r.chats, ctx.Value(chatKey{}))
}

// windowFor returns a context window in which the incident history occupies
// about 100/pct percent, measured the way the triggers measure it.
func windowFor(t *testing.T, now time.Time, divisor int) int {
	t.Helper()
	store := newSeqWindowStore(brokenHistory(now))
	probe := newAgeTriggeredManager(t, store, &summaryRecorder{})
	return probe.contextTokens(store.GetHistory("test-session")) * divisor
}

// TestCompactionStart_ReasonsAndBlocking checks the start hook: one event per
// pass, before the model call, with the right reason, Blocking for the paths a
// turn waits on, and the ctx of the call that started the pass.
func TestCompactionStart_ReasonsAndBlocking(t *testing.T) {
	now := time.Now()
	// Thresholds sized so the history is past the safety line (window = 1×
	// the history) or between normal and safety (window = 3×).
	safetyOpts := func(t *testing.T) []Option {
		t.Helper()
		return []Option{WithContextWindow(windowFor(t, now, 1)), WithMinPercent(10), WithNormalPercent(20), WithSafetyPercent(80)}
	}
	thresholdOpts := func(t *testing.T) []Option {
		t.Helper()
		return []Option{WithContextWindow(windowFor(t, now, 3)), WithMinPercent(10), WithNormalPercent(20), WithSafetyPercent(90)}
	}
	ageOpts := func(*testing.T) []Option { return nil }
	addFire := func(ctx context.Context, m *Manager) error {
		_, err := m.AddUserMessage(ctx, brokenFire(now))
		return err
	}

	tests := []struct {
		name   string
		opts   func(*testing.T) []Option
		run    func(context.Context, *Manager) error
		reason CompactionReason
		block  bool
	}{
		{"add past the safety line", safetyOpts, addFire, CompactionReasonSafetyNet, true},
		{"assemble past the safety line", safetyOpts, func(ctx context.Context, m *Manager) error {
			_, err := m.Assemble(ctx, AssembleRequest{})
			return err
		}, CompactionReasonSafetyNet, true},
		{"add past the normal threshold", thresholdOpts, addFire, CompactionReasonThreshold, true},
		{"force compress", safetyOpts, func(ctx context.Context, m *Manager) error {
			return m.ForceCompress(ctx)
		}, CompactionReasonOverflow, true},
		{"manual compact", ageOpts, func(ctx context.Context, m *Manager) error {
			return m.Compact(ctx)
		}, CompactionReasonManual, false},
		{"deferred age", ageOpts, func(ctx context.Context, m *Manager) error {
			if err := addFire(context.WithValue(ctx, chatKey{}, "bob-chat"), m); err != nil {
				return err
			}
			ran, err := m.CompactIfDue(ctx)
			if err == nil && !ran {
				return errors.New("CompactIfDue ran no pass")
			}
			return err
		}, CompactionReasonAge, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := &startRecorder{}
			caller := &summaryRecorder{}
			store := newSeqWindowStore(brokenHistory(now))
			opts := append(tt.opts(t), WithCompactionStartFunc(rec.record))
			mgr := newAgeTriggeredManager(t, store, caller, opts...)

			ctx := context.WithValue(context.Background(), chatKey{}, "alice-chat")
			if err := tt.run(ctx, mgr); err != nil {
				t.Fatalf("run: %v", err)
			}
			if len(caller.requests) == 0 {
				t.Fatal("no model call: the case does not exercise the hook")
			}
			if len(rec.events) != 1 {
				t.Fatalf("start events = %+v, want exactly one", rec.events)
			}
			want := CompactionStart{SessionKey: "test-session", Reason: tt.reason, Blocking: tt.block}
			if rec.events[0] != want {
				t.Errorf("event = %+v, want %+v", rec.events[0], want)
			}
			if rec.chats[0] != "alice-chat" {
				t.Errorf("hook ctx carried %v, want the caller's ctx", rec.chats[0])
			}
		})
	}
}

// TestCompactionStart_NotFiredWhenSuppressed checks that a pass the circuit
// breaker suppresses neither calls the model nor reports a start, and that a
// suppressed deferred age pass stays due for later.
func TestCompactionStart_NotFiredWhenSuppressed(t *testing.T) {
	now := time.Now()
	tests := []struct {
		name string
		opts []Option
		run  func(context.Context, *Manager) (bool, error)
	}{
		{
			"inline threshold",
			[]Option{WithContextWindow(windowFor(t, now, 3)), WithMinPercent(10), WithNormalPercent(20), WithSafetyPercent(90)},
			func(ctx context.Context, m *Manager) (bool, error) {
				_, err := m.AddUserMessage(ctx, brokenFire(now))
				return false, err
			},
		},
		{"deferred age", nil, func(ctx context.Context, m *Manager) (bool, error) {
			if _, err := m.AddUserMessage(ctx, brokenFire(now)); err != nil {
				return false, err
			}
			if !m.ageCompactionDue {
				return false, errors.New("the add did not mark the age compaction due")
			}
			ran, err := m.CompactIfDue(ctx)
			if err == nil && !m.ageCompactionDue {
				return ran, errors.New("a suppressed deferred pass must stay due")
			}
			return ran, err
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := &startRecorder{}
			caller := &summaryRecorder{}
			store := newSeqWindowStore(brokenHistory(now))
			mgr := newAgeTriggeredManager(t, store, caller, append(tt.opts, WithCompactionStartFunc(rec.record))...)
			mgr.consecutiveCompactFailures = defaultMaxConsecutiveCompactFailures
			mgr.breakerTrippedUntilCount = 1000

			ran, err := tt.run(context.Background(), mgr)
			if err != nil || ran {
				t.Fatalf("run = %v, %v; want suppressed", ran, err)
			}
			if len(rec.events) != 0 || len(caller.requests) != 0 {
				t.Errorf("reported %d starts and made %d model calls, want none", len(rec.events), len(caller.requests))
			}
		})
	}
}

// TestCompactIfDue_ConcurrentWithTurn drives adds, Assemble and CompactIfDue
// from several goroutines at once; run under -race it checks they serialise.
func TestCompactIfDue_ConcurrentWithTurn(t *testing.T) {
	now := time.Now()
	store := newSeqWindowStore(brokenHistory(now))
	mgr := newAgeTriggeredManager(t, store, &summaryRecorder{},
		WithCompactionStartFunc(func(context.Context, CompactionStart) {}))
	ctx := context.Background()

	const rounds = 20
	errs := make(chan error, 4*rounds)
	var wg sync.WaitGroup
	wg.Go(func() {
		for range rounds {
			if _, err := mgr.AddUserMessage(ctx, msgWithContent("hi from Alice")); err != nil {
				errs <- err
			}
		}
	})
	wg.Go(func() {
		for range rounds {
			if _, err := mgr.AddAssistantMessage(ctx, spawnllm.Message{Role: "assistant", Content: "hi Alice"}); err != nil {
				errs <- err
			}
		}
	})
	wg.Go(func() {
		for range rounds {
			if _, err := mgr.Assemble(ctx, AssembleRequest{}); err != nil {
				errs <- err
			}
		}
	})
	wg.Go(func() {
		for range rounds {
			if _, err := mgr.CompactIfDue(ctx); err != nil {
				errs <- err
			}
		}
	})
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}
