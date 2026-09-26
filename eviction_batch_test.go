// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"strings"
	"testing"

	"github.com/PivotLLM/ctxengine/memory"
)

// staleReads builds a history with n stale reader results of size bytes each
// (all older than EvictTurns), followed by enough plain turns to age them.
func staleReads(n, bytes int) []memory.StoredMessage {
	specs := make([]turnSpec, 0, n+12)
	for i := range n {
		specs = append(specs, turnSpec{
			tool: "file_read_bytes", id: "r" + string(rune('a'+i)),
			args:    map[string]any{"path": "f" + string(rune('a'+i)) + ".md"},
			content: strings.Repeat("x", bytes),
		})
	}
	for range 12 {
		specs = append(specs, turnSpec{text: "t"})
	}
	return buildHistory(specs...)
}

// TestSweep_BatchedBelowThreshold: through Assemble a sweep that would free
// less than MinSweepPercent of the window applies nothing — the window and so
// the cached prefix stay put — while one that clears the threshold applies
// every pending eviction at once.
func TestSweep_BatchedBelowThreshold(t *testing.T) {
	ctx := context.Background()
	p := basePolicy()
	p.MinSweepPercent = 5 // 5% of a 100k-token window = 20 000 bytes

	// One 5000-byte stale read: below the threshold, deferred.
	small := newSeqStore(staleReads(1, 5000))
	mgr := asManager(t, New("sess", small, WithContextWindow(100_000), WithEvictionPolicy(p)))
	asm, err := mgr.Assemble(ctx, AssembleRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if len(asm.Evictions) != 0 || asm.Changed() {
		t.Fatalf("a 5000-byte eviction was applied under a 20000-byte threshold: %+v", asm.Evictions)
	}
	if isEvicted(findToolResult(small, "ra")) {
		t.Fatal("stored content was rewritten although the sweep was deferred")
	}

	// Five of them: 25 000 bytes clears the threshold and all five go.
	big := newSeqStore(staleReads(5, 5000))
	mgr = asManager(t, New("sess", big, WithContextWindow(100_000), WithEvictionPolicy(p)))
	asm, err = mgr.Assemble(ctx, AssembleRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if len(asm.Evictions) != 5 {
		t.Fatalf("evictions = %d, want all 5 once the batch clears the threshold", len(asm.Evictions))
	}
	for _, id := range []string{"ra", "rb", "rc", "rd", "re"} {
		if !isEvicted(findToolResult(big, id)) {
			t.Errorf("read %s not evicted in the batch", id)
		}
	}
}

// TestSweep_ExplicitSweepIgnoresThreshold: SweepEvictions is a request to
// reclaim the window now and applies what it finds regardless of the batch
// threshold.
func TestSweep_ExplicitSweepIgnoresThreshold(t *testing.T) {
	p := basePolicy()
	p.MinSweepPercent = 5
	store := newSeqStore(staleReads(1, 5000))
	mgr := asManager(t, New("sess", store, WithContextWindow(100_000), WithEvictionPolicy(p)))
	if events := mgr.SweepEvictions(context.Background()); len(events) != 1 {
		t.Fatalf("explicit sweep applied %d evictions, want 1", len(events))
	}
}

// TestSweep_AppliedAtCompaction: a compaction pass applies the pending
// evictions the batched sweep was holding back, since it rewrites the window
// anyway.
func TestSweep_AppliedAtCompaction(t *testing.T) {
	p := basePolicy()
	p.MinSweepPercent = 5
	store := newSeqStore(staleReads(1, 5000))
	// No model: the pass reports "nothing", but the sweep at its start runs.
	mgr := asManager(t, New("sess", store, WithContextWindow(100_000), WithEvictionPolicy(p)))
	if err := mgr.Compact(context.Background()); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if !isEvicted(findToolResult(store, "ra")) {
		t.Fatal("compaction did not apply the pending eviction")
	}
}

// TestSweep_ThresholdDisabled: MinSweepPercent 0 (and a policy literal that
// leaves it unset) applies every sweep at once, the previous behaviour.
func TestSweep_ThresholdDisabled(t *testing.T) {
	store := newSeqStore(staleReads(1, 50))
	mgr := asManager(t, New("sess", store, WithContextWindow(100_000), WithEvictionPolicy(basePolicy())))
	asm, err := mgr.Assemble(context.Background(), AssembleRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if len(asm.Evictions) != 1 {
		t.Fatalf("evictions = %d, want 1 with no threshold", len(asm.Evictions))
	}
	if DefaultEvictionPolicy().MinSweepPercent != defaultEvictionMinSweepPercent {
		t.Errorf("default MinSweepPercent = %d, want %d", DefaultEvictionPolicy().MinSweepPercent, defaultEvictionMinSweepPercent)
	}
}
