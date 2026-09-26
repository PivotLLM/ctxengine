// ctxengine
// License: MIT

package ctxengine

import (
	"encoding/json"
	"strings"
	"testing"
)

// makeLargeSummary produces a Summary with n key_moments and n message_index
// entries, each containing verbose text so the serialized form is large.
func makeLargeSummary(n int) *Summary {
	s := &Summary{
		Version: 2,
		State: SummaryState{
			Goals:       []SummaryItem{{Text: strings.Repeat("a goal that is somewhat verbose ", 5), Refs: []SeqRange{{SeqStart: 1}}}},
			Progress:    []SummaryItem{{Text: strings.Repeat("making good progress on many tasks ", 5), Refs: []SeqRange{{SeqStart: 2}}}},
			Pending:     []SummaryItem{{Text: strings.Repeat("waiting on various things ", 5), Refs: []SeqRange{{SeqStart: 3}}}},
			Constraints: []SummaryItem{{Text: strings.Repeat("important constraint to remember ", 5), Refs: []SeqRange{{SeqStart: 4}}}},
		},
		CoveredSeqStart: 1,
		CoveredSeqEnd:   int64(n),
	}
	for i := range n {
		s.KeyMoments = append(s.KeyMoments, KeyMoment{
			Seq:     int64(i + 1),
			Role:    "user",
			Summary: strings.Repeat("very important moment that happened at step ", 3),
		})
		s.MessageIndex = append(s.MessageIndex, IndexEntry{
			SeqStart: int64(i + 1),
			SeqEnd:   int64(i + 1),
			Role:     "user",
			Label:    strings.Repeat("descriptive label for archival purpose ", 3),
		})
	}
	return s
}

// estimateSummaryTokens returns the runes/4 token estimate for s.
func estimateSummaryTokens(s *Summary) int {
	data, err := json.Marshal(s)
	if err != nil {
		return 0
	}
	return len([]rune(string(data))) / 4
}

// TestTruncateToFit_FitsWithinLimit verifies that TruncateToFit truncates a
// verbose summary to fit within the given token limit.
func TestTruncateToFit_FitsWithinLimit(t *testing.T) {
	s := makeLargeSummary(50) // many entries → large summary
	initialTokens := estimateSummaryTokens(s)
	limit := initialTokens / 2 // target half the original size

	truncated := s.TruncateToFit(limit)
	if !truncated {
		t.Error("TruncateToFit should have returned true (truncation occurred)")
	}

	finalTokens := estimateSummaryTokens(s)
	if finalTokens > limit {
		t.Errorf("summary still over limit after TruncateToFit: %d > %d tokens", finalTokens, limit)
	}
}

// TestTruncateToFit_PreservesNewest verifies that the most recent entries
// (highest index) are preserved and the oldest (index 0) are dropped first.
func TestTruncateToFit_PreservesNewest(t *testing.T) {
	s := &Summary{
		Version:         2,
		CoveredSeqStart: 1,
		CoveredSeqEnd:   10,
		KeyMoments: []KeyMoment{
			{Refs: []SeqRange{{SeqStart: 1}}, Role: "user", Summary: strings.Repeat("old moment ", 30)},
			{Refs: []SeqRange{{SeqStart: 5}}, Role: "user", Summary: strings.Repeat("middle moment ", 30)},
			{Refs: []SeqRange{{SeqStart: 10}}, Role: "user", Summary: strings.Repeat("newest moment ", 30)},
		},
	}

	initialTokens := estimateSummaryTokens(s)
	// Set limit that forces dropping oldest but allows keeping newest.
	limit := initialTokens * 2 / 3

	s.TruncateToFit(limit)

	// The most recent key moment (seq 10) should still be present.
	found := false
	for _, km := range s.KeyMoments {
		if len(km.Refs) > 0 && km.Refs[0].SeqStart == 10 {
			found = true
		}
	}
	if !found {
		t.Error("TruncateToFit should preserve the newest (seq=10) key moment")
	}
}

// TestTruncateToFit_AlreadyFits returns false when summary already fits.
func TestTruncateToFit_AlreadyFits(t *testing.T) {
	s := &Summary{
		Version: 2,
		State:   SummaryState{Goals: []SummaryItem{{Text: "short", Refs: []SeqRange{{SeqStart: 1}}}}},
	}
	if s.TruncateToFit(10000) {
		t.Error("TruncateToFit should return false when summary already fits")
	}
}

// TestTruncateToFit_NilSafe verifies that TruncateToFit on nil summary does not panic.
func TestTruncateToFit_NilSafe(t *testing.T) {
	var s *Summary
	if s.TruncateToFit(100) {
		t.Error("nil summary TruncateToFit should return false")
	}
}

// TestTruncateToFit_ZeroLimitNoop verifies that a zero limit is a no-op.
func TestTruncateToFit_ZeroLimitNoop(t *testing.T) {
	s := makeLargeSummary(5)
	initial := len(s.KeyMoments)
	if s.TruncateToFit(0) {
		t.Error("zero limit TruncateToFit should return false (no-op)")
	}
	if len(s.KeyMoments) != initial {
		t.Error("zero limit TruncateToFit should not modify the summary")
	}
}

// TestWithMaxSummaryTokensOption verifies that WithMaxSummaryTokens is applied
// to the managerConfig correctly.
func TestWithMaxSummaryTokensOption(t *testing.T) {
	cfg := defaultManagerConfig()
	if cfg.maxSummaryTokens != 0 {
		t.Errorf("expected default maxSummaryTokens=0, got %d", cfg.maxSummaryTokens)
	}

	WithMaxSummaryTokens(1500)(&cfg)
	if cfg.maxSummaryTokens != 1500 {
		t.Errorf("expected maxSummaryTokens=1500, got %d", cfg.maxSummaryTokens)
	}
}

// TestStats_SummaryTokensPopulated verifies that Stats() returns a non-zero
// SummaryTokens when a summary is stored.
func TestStats_SummaryTokensPopulated(t *testing.T) {
	store := newMockStore()
	m := newTestManager(t, store, WithContextWindow(128000))

	// newTestManager uses sessionKey "test-session" — inject on the correct key.
	noErr(t, store.SetSummary("test-session", `{"version":2,"state":{"goals":[{"text":"test","refs":[{"seq_start":1}]}]},"covered_seq_start":0,"covered_seq_end":5,"generated_at":"2026-01-01T00:00:00Z"}`))

	stats := m.Stats()
	if stats.SummaryTokens == 0 {
		t.Error("expected non-zero SummaryTokens in Stats() when summary is stored")
	}
}

// TestStats_SummaryTokensZeroWithoutSummary verifies that Stats() returns 0
// SummaryTokens when no summary exists.
func TestStats_SummaryTokensZeroWithoutSummary(t *testing.T) {
	store := newMockStore()
	m := newTestManager(t, store, WithContextWindow(128000))

	stats := m.Stats()
	if stats.SummaryTokens != 0 {
		t.Errorf("expected SummaryTokens=0 without summary, got %d", stats.SummaryTokens)
	}
}
