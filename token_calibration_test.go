// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"
)

// TestTokenSafetyMargin_Default pins the default margin the estimate is
// inflated by.
func TestTokenSafetyMargin_Default(t *testing.T) {
	if got := SettingsFromOptions().TokenSafetyMargin; got != 1.15 {
		t.Fatalf("default token safety margin = %v, want 1.15", got)
	}
	m := asManager(t, New("s", newMockStore()))
	if got := m.tokenMargin(); got != 1.15 {
		t.Fatalf("uncalibrated tokenMargin = %v, want the 1.15 default", got)
	}
}

// TestObserveUsage_ConvergesAndClamps: the calibrated ratio replaces the static
// margin only after enough observations, tracks the observed ratio, and is
// clamped between the static margin (floor) and calibrationMaxMargin.
func TestObserveUsage_ConvergesAndClamps(t *testing.T) {
	m := asManager(t, New("s", newMockStore()))

	// Fewer than the minimum observations: still the static margin.
	for range calibrationMinObservations - 1 {
		m.ObserveUsage(1000, 1500)
	}
	if got := m.tokenMargin(); got != 1.15 {
		t.Fatalf("margin after %d observations = %v, want the static 1.15", calibrationMinObservations-1, got)
	}
	m.ObserveUsage(1000, 1500)
	if got := m.tokenMargin(); math.Abs(got-1.5) > 1e-9 {
		t.Fatalf("margin after %d identical 1.5 observations = %v, want 1.5", calibrationMinObservations, got)
	}

	// Converges toward a new steady ratio.
	for range 20 {
		m.ObserveUsage(1000, 1250)
	}
	if got := m.tokenMargin(); math.Abs(got-1.25) > 0.01 {
		t.Errorf("margin after converging on 1.25 = %v", got)
	}

	// Ceiling.
	for range 30 {
		m.ObserveUsage(1000, 6000)
	}
	if got := m.tokenMargin(); got != calibrationMaxMargin {
		t.Errorf("margin with a 6.0 observed ratio = %v, want the %v ceiling", got, calibrationMaxMargin)
	}

	// Floor: a provider that counts fewer tokens than the heuristic never
	// pulls the margin under the configured value.
	for range 30 {
		m.ObserveUsage(1000, 500)
	}
	if got := m.tokenMargin(); got != 1.15 {
		t.Errorf("margin with a 0.5 observed ratio = %v, want the 1.15 floor", got)
	}

	// Non-positive reports are ignored.
	before := m.usageObservations
	m.ObserveUsage(0, 100)
	m.ObserveUsage(100, 0)
	if m.usageObservations != before {
		t.Errorf("non-positive reports were counted")
	}
}

// TestObserveUsage_CalibratedMarginDrivesPreBuildCheck: a window that the
// static margin puts under the safety line is pushed over it once the host
// reports the provider counts 50% more tokens than the heuristic, so the
// pre-build safety net fires on the next Assemble.
func TestObserveUsage_CalibratedMarginDrivesPreBuildCheck(t *testing.T) {
	// contextWindow=1000, safety=80 → line at 800 tokens. 2600 chars is 650
	// raw tokens: 747 under the 1.15 margin (fits), 975 at a 1.5 ratio.
	store := newMockStore()
	noErr(t, store.SetHistory("s", []spawnllm.Message{{Role: "user", Content: strings.Repeat("a", 2600)}}))
	m := asManager(t, New("s", store, WithContextWindow(1000), WithOverheadTokens(0), WithSafetyPercent(80)))
	var fired []bool
	m.SetTestCompressHook(func(safetyNet bool) { fired = append(fired, safetyNet) })

	asm, err := m.Assemble(context.Background(), AssembleRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if len(fired) != 0 {
		t.Fatalf("safety net fired under the static margin: %v", fired)
	}
	if asm.PromptTokenEstimate != 650 {
		t.Fatalf("PromptTokenEstimate = %d, want the raw 650 (no margin)", asm.PromptTokenEstimate)
	}

	for range calibrationMinObservations {
		m.ObserveUsage(asm.PromptTokenEstimate, 975)
	}
	if _, err := m.Assemble(context.Background(), AssembleRequest{}); err != nil {
		t.Fatal(err)
	}
	if len(fired) != 1 || !fired[0] {
		t.Fatalf("safety net did not fire under the calibrated margin: %v", fired)
	}
}
