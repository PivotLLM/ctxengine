// ctxengine
// License: MIT

package ctxengine

import (
	"testing"

	"github.com/PivotLLM/spawnllm"
)

// buildHighSeqSummary returns a Summary whose state items, key moment, and
// message index all cite HIGH memory seqs (memStart..memEnd). These are the
// seqs the summarizer would emit because GetHistoryWithSeqs feeds it memory
// seqs. The fix makes the archive key on these same seqs so the refs survive
// the out-of-range strip.
func buildHighSeqSummary(memStart, memEnd int64) *Summary {
	return &Summary{
		Version: summaryVersion,
		State: SummaryState{
			Goals: []SummaryItem{
				{Text: "ship the int64 seq fix", Refs: []SeqRange{{SeqStart: memStart, SeqEnd: memStart}}},
			},
			Progress: []SummaryItem{
				{Text: "archive now keyed by memory seq", Refs: []SeqRange{{SeqStart: memStart + 1, SeqEnd: memEnd - 1}}},
			},
			Pending: []SummaryItem{
				{Text: "verify retrieval round-trips", Refs: []SeqRange{{SeqStart: memEnd, SeqEnd: memEnd}}},
			},
		},
		KeyMoments: []KeyMoment{
			{Refs: []SeqRange{{SeqStart: memEnd, SeqEnd: memEnd}}, Role: "user", Summary: "user approved option A"},
		},
		MessageIndex: []IndexEntry{
			{SeqStart: memStart, SeqEnd: memEnd, Role: "user", Label: "recent work"},
		},
		CoveredSeqStart: memStart,
		CoveredSeqEnd:   memEnd,
	}
}

// TestArchiveKeyedByMemorySeq_HighSeqRefsSurviveStrip is the regression test for
// the context-compaction seq-divergence bug.
//
// Bug (old behavior): the archive used a SEPARATE private counter (archiveSeq)
// seeded from 0/Bounds and incremented per append, so archive seqs were small
// (1..N) even when the message's MEMORY seq was large (5000+). archiveWindow()
// returned those small archive-space bounds, and StripOutOfRangeSeqRefs then
// stripped every summary ref whose seq exceeded the small archive max —
// deleting ALL recent state from the summary and breaking get_session_messages
// (which retrieves by archive seq while the summary cited memory seqs).
//
// Fix (current behavior): the archive is keyed by the MEMORY seq that the store
// assigns, so there is ONE sequence space. archiveWindow() returns memory-space
// bounds, the high-seq refs the summarizer cited fall inside that window, and
// the strip keeps them. The same seqs are retrievable from the archive.
//
// This test simulates a long-lived agent whose memory seq (5000..5200) is far
// above any private archive counter (which would have been ~1..201). It would
// FAIL on the old code path: archiveWindow() would return ~(1, 201), so every
// 5000+ ref would be stripped and the surviving summary would be empty.
func TestArchiveKeyedByMemorySeq_HighSeqRefsSurviveStrip(t *testing.T) {
	const (
		memStart int64 = 5000
		memEnd   int64 = 5200
	)

	store := newMockStore()
	mgr := asManager(t, New("longlived", store,
		WithContextWindow(100000),
		WithArchiveDir(t.TempDir()),
	))

	// Archive 201 messages keyed by their MEMORY seq (5000..5200), exactly as
	// AddUserMessage/AddAssistantMessage now do with the seq returned by the
	// store. On the old code path these would have landed under archive seqs
	// 1..201 regardless of the memory seq.
	for seq := memStart; seq <= memEnd; seq++ {
		mgr.archiveAppend(seq, spawnllm.Message{
			Role:    "user",
			Content: "recent message",
		})
	}

	// archiveWindow() must report the MEMORY-space bounds, not 1..201.
	minSeq, maxSeq := mgr.archiveWindow()
	if minSeq != memStart || maxSeq != memEnd {
		t.Fatalf("archiveWindow() = (%d, %d), want (%d, %d); "+
			"a small upper bound here is the old bug (archive used a private counter)",
			minSeq, maxSeq, memStart, memEnd)
	}

	// Build a summary citing the HIGH memory seqs and run the strip with the
	// bounds from archiveWindow() — the exact call doCompress makes.
	summary := buildHighSeqSummary(memStart, memEnd)
	summary.StripOutOfRangeSeqRefs(minSeq, maxSeq)

	// All high-seq state must SURVIVE. Under the old behavior every ref exceeded
	// the (tiny) archive max and was stripped, leaving nothing.
	if !summary.HasMaterial() {
		t.Fatal("all summary material was stripped — high memory-seq refs did not survive " +
			"(this is the regression the fix prevents)")
	}
	if len(summary.State.Goals) != 1 {
		t.Errorf("Goals stripped: got %d, want 1", len(summary.State.Goals))
	}
	if len(summary.State.Progress) != 1 {
		t.Errorf("Progress stripped: got %d, want 1", len(summary.State.Progress))
	}
	if len(summary.State.Pending) != 1 {
		t.Errorf("Pending stripped: got %d, want 1", len(summary.State.Pending))
	}
	if len(summary.KeyMoments) != 1 {
		t.Errorf("KeyMoments stripped: got %d, want 1", len(summary.KeyMoments))
	}
	if len(summary.MessageIndex) != 1 {
		t.Errorf("MessageIndex stripped: got %d, want 1", len(summary.MessageIndex))
	}

	// The summary advertises memory seqs; those same seqs must be retrievable
	// from the archive (this is what get_session_messages does). Under the old
	// behavior the archive stored them under seqs 1..201, so a query at 5000+
	// returned nothing.
	archive := mgr.getOrOpenArchive()
	if archive == nil {
		t.Fatal("expected an open archive")
	}
	got, err := archive.QueryRange(memStart, memStart)
	if err != nil {
		t.Fatalf("QueryRange(%d): %v", memStart, err)
	}
	if len(got) != 1 || got[0].Seq != memStart {
		t.Fatalf("expected to retrieve message at memory seq %d; got %d results",
			memStart, len(got))
	}
}
