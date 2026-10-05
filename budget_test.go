/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

import (
	"strings"
	"testing"
	"time"

	"github.com/PivotLLM/spawnllm"
)

// TestValidation_RetainClampedBelowFloor is the fix for the oscillation this
// investigation started from: with retain == min, every pass shaved back to
// exactly the floor and the next message crossed it again, which is why the
// production logs showed 29 consecutive compactions all firing at 20.0-22.9%.
func TestValidation_RetainClampedBelowFloor(t *testing.T) {
	for _, tc := range []struct {
		name              string
		retain, min, want int
	}{
		{"equal clamps to half", 20, 20, 10},
		{"above clamps to half", 30, 20, 10},
		{"below is left alone", 8, 20, 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mgr := newTestManager(t, newMockStore(),
				WithContextWindow(100_000),
				WithMinPercent(tc.min),
				WithNormalPercent(50),
				WithRetainTokenPercent(tc.retain),
			)
			if got := mgr.cfg.retainTokenPercent; got != tc.want {
				t.Errorf("retainTokenPercent = %d, want %d", got, tc.want)
			}
		})
	}
}

// TestValidation_DefaultsDoNotTripTheClamp guards the defaults themselves: if
// the built-in retain and min were equal, every Manager would construct with a
// warning and a silently altered budget.
func TestValidation_DefaultsDoNotTripTheClamp(t *testing.T) {
	if defaultRetainTokenPercent >= defaultMinPercent {
		t.Fatalf("defaultRetainTokenPercent (%d) must be below defaultMinPercent (%d)",
			defaultRetainTokenPercent, defaultMinPercent)
	}
	mgr := newTestManager(t, newMockStore(), WithContextWindow(100_000))
	if mgr.cfg.retainTokenPercent != defaultRetainTokenPercent {
		t.Errorf("defaults were clamped: retain = %d, want %d",
			mgr.cfg.retainTokenPercent, defaultRetainTokenPercent)
	}
}

// TestValidation_TriggerDaysClampedAboveRetain covers the hysteresis rule: a
// trigger that fires before anything is old enough to cut summarizes nothing.
func TestValidation_TriggerDaysClampedAboveRetain(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		trigger, retain, want int
	}{
		{"below retain clamps up", 3, 5, 5},
		{"equal is left alone", 5, 5, 5},
		{"above retain is left alone", 7, 5, 7},
		{"disabled is left alone", 0, 5, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mgr := newTestManager(t, newMockStore(),
				WithContextWindow(100_000),
				WithTriggerDays(tc.trigger),
				WithRetainMaxAgeDays(tc.retain),
			)
			if got := mgr.cfg.triggerDays; got != tc.want {
				t.Errorf("triggerDays = %d, want %d", got, tc.want)
			}
		})
	}
}

// TestRetainBudget_AbsoluteCapWins is the guard against percentages scaling with
// the window: 10% of a million-token model is 100k tokens of retained tail,
// inherited wholesale from a figure tuned for 128k.
func TestRetainBudget_AbsoluteCapWins(t *testing.T) {
	mgr := newTestManager(t, newMockStore(),
		WithContextWindow(1_000_000),
		WithMinPercent(20),
		WithRetainTokenPercent(10),
		WithRetainMaxTokens(30_000),
	)
	if got, want := mgr.retainBudgetTokens(), 30_000; got != want {
		t.Errorf("retainBudgetTokens = %d, want %d (absolute cap below the percentage)", got, want)
	}
}

// TestRetainBudget_PercentWinsWhenSmaller verifies the cap is a ceiling, not an
// override — a small window keeps its small percentage budget.
func TestRetainBudget_PercentWinsWhenSmaller(t *testing.T) {
	mgr := newTestManager(t, newMockStore(),
		WithContextWindow(100_000),
		WithMinPercent(20),
		WithRetainTokenPercent(10),
		WithRetainMaxTokens(30_000),
	)
	if got, want := mgr.retainBudgetTokens(), 10_000; got != want {
		t.Errorf("retainBudgetTokens = %d, want %d (percentage below the cap)", got, want)
	}
}

// TestRetainBudget_CapDisabled is the off path: an explicit 0 removes the
// absolute ceiling and leaves the percentage in charge.
func TestRetainBudget_CapDisabled(t *testing.T) {
	mgr := newTestManager(t, newMockStore(),
		WithContextWindow(1_000_000),
		WithMinPercent(20),
		WithRetainTokenPercent(10),
		WithRetainMaxTokens(0),
	)
	if got, want := mgr.retainBudgetTokens(), 100_000; got != want {
		t.Errorf("retainBudgetTokens = %d, want %d (absolute cap explicitly disabled)", got, want)
	}
}

// TestRetainBudget_DefaultCapBindsOnLargeWindows pins the reason the default
// exists: the percentage scales with the window, so without an absolute ceiling
// a million-token model retains 100k of tail purely because it can. It must stay
// a no-op on small windows, where the percentage already binds tighter.
func TestRetainBudget_DefaultCapBindsOnLargeWindows(t *testing.T) {
	large := newTestManager(t, newMockStore(), WithContextWindow(1_000_000), WithMinPercent(20))
	if got, want := large.retainBudgetTokens(), defaultRetainMaxTokens; got != want {
		t.Errorf("1M window: retainBudgetTokens = %d, want %d (absolute cap binds)", got, want)
	}

	small := newTestManager(t, newMockStore(), WithContextWindow(128_000), WithMinPercent(20))
	if got, want := small.retainBudgetTokens(), 128_000*defaultRetainTokenPercent/100; got != want {
		t.Errorf("128k window: retainBudgetTokens = %d, want %d (percentage binds tighter)", got, want)
	}
}

// TestCompressTargetPercent covers both the explicit setting and the derived
// default, which stays coupled to normalPercent for configs that do not set it.
func TestCompressTargetPercent(t *testing.T) {
	derived := newTestManager(t, newMockStore(), WithContextWindow(100_000), WithNormalPercent(50))
	if got, want := derived.compressTargetPercent(), 25.0; got != want {
		t.Errorf("derived target = %v, want %v", got, want)
	}
	explicit := newTestManager(t, newMockStore(),
		WithContextWindow(100_000), WithNormalPercent(50), WithTargetPercent(15))
	if got, want := explicit.compressTargetPercent(), 15.0; got != want {
		t.Errorf("explicit target = %v, want %v", got, want)
	}
}

// TestContextPercent_CountsNonHistory is the fix for the three trigger paths
// disagreeing: the turn-boundary trigger used to measure stored history alone,
// ignoring the reserve, the tool schemas and everything Build() adds.
func TestContextPercent_CountsNonHistory(t *testing.T) {
	mgr := newTestManager(t, newMockStore(),
		WithContextWindow(100_000),
		WithOverheadTokens(4_000),
	)
	mgr.SetToolDefinitionTokens(6_000)
	mgr.builtOverheadTokens = 2_000

	history := []spawnllm.Message{{Role: "user", Content: strings.Repeat("h", 40_000)}} // 10k tokens
	if got, want := mgr.contextTokens(history), 22_000; got != want {
		t.Fatalf("contextTokens = %d, want %d (10k history + 4k reserve + 6k tools + 2k build)", got, want)
	}
	if got, want := mgr.contextPercent(history), 22.0; got != want {
		t.Errorf("contextPercent = %v, want %v", got, want)
	}
}

// TestSetToolDefinitionTokens_IgnoresNegative keeps a bad caller from making the
// window look smaller than it is.
func TestSetToolDefinitionTokens_IgnoresNegative(t *testing.T) {
	mgr := newTestManager(t, newMockStore(), WithContextWindow(100_000))
	mgr.SetToolDefinitionTokens(5_000)
	mgr.SetToolDefinitionTokens(-1)
	if got := mgr.toolDefTokens; got != 5_000 {
		t.Errorf("toolDefTokens = %d, want 5000 (negative input ignored)", got)
	}
}

// TestRetainMaxAge covers the days→duration conversion and its off switch.
func TestRetainMaxAge(t *testing.T) {
	on := newTestManager(t, newMockStore(), WithContextWindow(100_000), WithRetainMaxAgeDays(5))
	if got, want := on.retainMaxAge(), 5*24*time.Hour; got != want {
		t.Errorf("retainMaxAge = %v, want %v", got, want)
	}
	off := newTestManager(t, newMockStore(), WithContextWindow(100_000), WithRetainMaxAgeDays(0))
	if got := off.retainMaxAge(); got != 0 {
		t.Errorf("retainMaxAge = %v, want 0 (disabled)", got)
	}
}

// TestValidation_AllRetainBoundsDisabled documents the one retention
// configuration that cannot work: with no percentage, no absolute cap and no age
// limit, selectTail has nothing to cut against and the window grows forever.
// Each bound alone is legitimate — retaining by age only, or by tokens only —
// so this is a warning rather than a clamp, but it must be reachable and it must
// not fire for the sensible single-bound cases.
func TestValidation_AllRetainBoundsDisabled(t *testing.T) {
	unbounded := newTestManager(t, newMockStore(),
		WithContextWindow(100_000),
		WithRetainTokenPercent(0),
		WithRetainMaxTokens(0),
		WithRetainMaxAgeDays(0),
	)
	if got := unbounded.retainBudgetTokens(); got != 0 {
		t.Errorf("budget = %d, want 0 (unbounded)", got)
	}
	if got := unbounded.retainMaxAge(); got != 0 {
		t.Errorf("maxAge = %v, want 0 (unbounded)", got)
	}

	// Age-only retention is a supported configuration and must stay bounded.
	ageOnly := newTestManager(t, newMockStore(),
		WithContextWindow(100_000),
		WithRetainTokenPercent(0),
		WithRetainMaxTokens(0),
		WithRetainMaxAgeDays(5),
	)
	if ageOnly.retainMaxAge() == 0 {
		t.Error("age-only retention must keep its age bound")
	}
}
