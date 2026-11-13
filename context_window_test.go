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

	"github.com/PivotLLM/spawnllm"
)

// windowManager is a manager with a window of window tokens whose history
// holds chars characters (about chars/4 tokens), counting each compaction
// pass in *fired.
func windowManager(t *testing.T, window, chars int, fired *int) *Manager {
	t.Helper()
	store := newMockStore()
	noErr(t, store.SetHistory("test-session", []spawnllm.Message{{Role: "user", Content: strings.Repeat("a", chars)}}))
	m := asManager(t, New("test-session", store,
		WithContextWindow(window), WithSafetyPercent(80), WithOverheadTokens(0)))
	m.SetTestCompressHook(func(bool) { *fired++ })
	return m
}

// A smaller window the history no longer fits is enforced by the next
// Assemble: the safety-net pass runs before the request is built.
func TestSetContextWindow_ShrinkCompactsOnNextAssemble(t *testing.T) {
	fired := 0
	// About 2500 tokens: 2.5% of 100000, past 80% of 3000.
	m := windowManager(t, 100_000, 10_000, &fired)
	if asm, err := m.Assemble(context.Background(), AssembleRequest{}); err != nil || asm.Compacted {
		t.Fatalf("before the change: compacted=%v err=%v, want neither", asm.Compacted, err)
	}

	noErr(t, m.SetContextWindow(3_000))
	asm, err := m.Assemble(context.Background(), AssembleRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if !asm.Compacted || fired != 1 {
		t.Fatalf("after shrinking: compacted=%v passes=%d, want a safety-net pass", asm.Compacted, fired)
	}
}

// A larger window takes the history off the safety line: Assemble no longer
// compacts.
func TestSetContextWindow_GrowStopsCompaction(t *testing.T) {
	fired := 0
	// About 10000 tokens: past 80% of 10000, 10% of 100000.
	m := windowManager(t, 10_000, 40_000, &fired)

	noErr(t, m.SetContextWindow(100_000))
	asm, err := m.Assemble(context.Background(), AssembleRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if asm.Compacted || fired != 0 {
		t.Fatalf("after growing: compacted=%v passes=%d, want none", asm.Compacted, fired)
	}
	if pct := m.Stats().ContextWindowPct; pct < 9 || pct > 13 {
		t.Fatalf("ContextWindowPct = %.1f, want about 10-12 (with the estimate margin) of the new window", pct)
	}
}

// A changed window clears the post-compaction cooldown, which was judged
// against the previous window; setting the current window changes nothing.
func TestSetContextWindow_Cooldown(t *testing.T) {
	tests := []struct {
		name        string
		window      int
		wantCooling bool
	}{
		{"unchanged window is a no-op", 10_000, true},
		{"smaller window clears the cooldown", 5_000, false},
		{"larger window clears the cooldown", 20_000, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fired := 0
			m := windowManager(t, 10_000, 400, &fired)
			m.cooling, m.coolingSinceCount = true, 7

			noErr(t, m.SetContextWindow(tt.window))
			if m.cfg.contextWindow != tt.window {
				t.Errorf("window = %d, want %d", m.cfg.contextWindow, tt.window)
			}
			if m.cooling != tt.wantCooling {
				t.Errorf("cooling = %v, want %v", m.cooling, tt.wantCooling)
			}
		})
	}
}

// A window of zero or fewer tokens is refused and changes nothing.
func TestSetContextWindow_InvalidRefused(t *testing.T) {
	for _, window := range []int{0, -1} {
		fired := 0
		m := windowManager(t, 10_000, 400, &fired)
		m.cooling = true
		if err := m.SetContextWindow(window); !errors.Is(err, ErrInvalidContextWindow) {
			t.Errorf("SetContextWindow(%d) = %v, want ErrInvalidContextWindow", window, err)
		}
		if m.cfg.contextWindow != 10_000 || !m.cooling {
			t.Errorf("SetContextWindow(%d) changed the manager: window=%d cooling=%v", window, m.cfg.contextWindow, m.cooling)
		}
	}
}

// SetContextWindow is serialised with the manager's other operations (run
// with -race).
func TestSetContextWindow_Concurrent(t *testing.T) {
	fired := 0
	m := windowManager(t, 100_000, 400, &fired)
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Go(func() {
			if err := m.SetContextWindow(50_000 + i*1_000); err != nil {
				t.Error(err)
			}
		})
		wg.Go(func() {
			_ = m.Stats()
		})
		wg.Go(func() {
			if _, err := m.Assemble(context.Background(), AssembleRequest{}); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
}
