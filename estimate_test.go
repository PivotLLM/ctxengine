/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

import (
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"
)

// TestEstimateTokensWith_DefaultMatchesLegacy verifies that the default tuning
// (4.0 chars/token, 1.0 margin) reproduces the original runes/4 behaviour, so
// existing thresholds are unchanged when no override is configured.
func TestEstimateTokensWith_DefaultMatchesLegacy(t *testing.T) {
	msgs := []spawnllm.Message{{Role: "user", Content: "0123456789"}} // 10 runes
	if got := estimateTokensWith(msgs, defaultCharsPerToken, defaultTokenSafetyMargin); got != 2 {
		t.Fatalf("default estimate: want 2, got %d", got)
	}
	if got := estimateTokens(msgs); got != 2 {
		t.Fatalf("package estimateTokens: want 2, got %d", got)
	}
}

// TestEstimateTokensWith_DivisorAndMargin verifies the divisor and safety margin
// both push the estimate upward (more conservative).
func TestEstimateTokensWith_DivisorAndMargin(t *testing.T) {
	msgs := []spawnllm.Message{{Role: "user", Content: "01234567890123456789"}} // 20 runes

	// 20 / 4.0 = 5
	if got := estimateTokensWith(msgs, 4.0, 1.0); got != 5 {
		t.Fatalf("4.0/1.0: want 5, got %d", got)
	}
	// 20 / 3.5 ≈ 5.71 → 5 (truncated), still >= the 4.0 case is equal here;
	// use a clearer divisor: 20 / 2.0 = 10.
	if got := estimateTokensWith(msgs, 2.0, 1.0); got != 10 {
		t.Fatalf("2.0/1.0: want 10, got %d", got)
	}
	// margin 1.5: 20 / 4.0 * 1.5 = 7
	if got := estimateTokensWith(msgs, 4.0, 1.5); got != 7 {
		t.Fatalf("4.0/1.5: want 7, got %d", got)
	}
}

// TestEstimateTokensWith_NonPositiveFallback verifies non-positive tuning values
// fall back to the defaults rather than dividing by zero.
func TestEstimateTokensWith_NonPositiveFallback(t *testing.T) {
	msgs := []spawnllm.Message{{Role: "user", Content: "0123456789"}} // 10 runes
	if got := estimateTokensWith(msgs, 0, 0); got != 2 {
		t.Fatalf("zero tuning fallback: want 2, got %d", got)
	}
	if got := estimateTokensWith(msgs, -3, -1); got != 2 {
		t.Fatalf("negative tuning fallback: want 2, got %d", got)
	}
}

// TestManagerEstTokens_UsesConfig verifies the Manager's estTokens applies the
// configured divisor and margin.
func TestManagerEstTokens_UsesConfig(t *testing.T) {
	cm := New("sess", newMockStore(),
		WithContextWindow(1000),
		WithCharsPerToken(2.0),
		WithTokenSafetyMargin(1.5),
	)
	m := asManager(t, cm)
	msgs := []spawnllm.Message{{Role: "user", Content: "01234567890123456789"}} // 20 runes
	// 20 / 2.0 * 1.5 = 15
	if got := m.estTokens(msgs); got != 15 {
		t.Fatalf("manager estTokens: want 15, got %d", got)
	}
}

// TestArchiveTruncateContent_Limit verifies the truncation limit is honoured and
// that a non-positive limit falls back to the default.
func TestArchiveTruncateContent_Limit(t *testing.T) {
	long := make([]byte, 8192)
	for i := range long {
		long[i] = 'x'
	}
	msg := spawnllm.Message{Role: "tool", Content: string(long)}

	got := archiveTruncateContent(msg, 100)
	if len(got.Content) <= 100 {
		t.Fatalf("expected truncated content with note appended, got len %d", len(got.Content))
	}
	if len(got.Content) > 200 {
		t.Fatalf("truncated content unexpectedly long: %d", len(got.Content))
	}

	// Non-positive limit falls back to the default cap, so 8192 bytes are kept
	// whole and content over the cap is truncated.
	if fallback := archiveTruncateContent(msg, 0); len(fallback.Content) != len(long) {
		t.Fatalf("fallback limit truncated %d bytes under the default cap: len %d", len(long), len(fallback.Content))
	}
	over := spawnllm.Message{Role: "tool", Content: strings.Repeat("x", archiveContentMaxBytes*2)}
	if fallback := archiveTruncateContent(over, 0); len(fallback.Content) >= len(over.Content) {
		t.Fatalf("fallback limit did not truncate content over the default cap: len %d", len(fallback.Content))
	}

	// Content within the limit is returned unchanged.
	short := spawnllm.Message{Role: "tool", Content: "small"}
	if archiveTruncateContent(short, 100).Content != "small" {
		t.Fatalf("short content should be unchanged")
	}
}
