// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"testing"
)

// TestForceCompress_MeasuresFullRequest: history alone is under the safety
// line but the request — history plus the tool schemas — is not. ForceCompress
// must measure the request the way Assemble does and drop until it fits;
// measuring history alone would return without doing anything and the retry
// would hit the same rejection.
func TestForceCompress_MeasuresFullRequest(t *testing.T) {
	// contextWindow=1000, safety=80 → line at 800 tokens. 5 pairs × 200 chars
	// ≈ 500 tokens of history (50%); 400 tokens of tool schemas make 90%.
	store := &compressTestStore{history: makeConversation(5, 200)}
	mgr := newCompressManager(t, store, nil, WithContextWindow(1000))
	mgr.SetToolDefinitionTokens(400)

	if err := mgr.ForceCompress(context.Background()); err != nil {
		t.Fatalf("ForceCompress: %v", err)
	}
	after := store.GetHistory("sess")
	if len(after) >= 10 {
		t.Fatalf("ForceCompress dropped nothing: %d messages remain", len(after))
	}
	if pct := mgr.contextPercent(after); pct >= float64(mgr.cfg.safetyPercent) {
		t.Errorf("request still past the safety line: %.0f%%", pct)
	}

	// Without the schemas the same history fits and nothing is touched.
	untouched := &compressTestStore{history: makeConversation(5, 200)}
	quiet := newCompressManager(t, untouched, nil, WithContextWindow(1000))
	if err := quiet.ForceCompress(context.Background()); err != nil {
		t.Fatalf("ForceCompress on a fitting request: %v", err)
	}
	if len(untouched.GetHistory("sess")) != 10 {
		t.Errorf("ForceCompress dropped from a request that already fit")
	}
}

// TestForceCompress_TriesSummaryBeforeDropping: with a working model the
// 413-recovery path summarizes the oldest history rather than discarding it,
// and the window ends up under the line.
func TestForceCompress_TriesSummaryBeforeDropping(t *testing.T) {
	store := &compressTestStore{history: makeConversation(10, 200)} // ≈1000 tokens
	llm := &mockLLM{model: "m", responses: []string{validSummaryJSON("forced goal")}}
	mgr := newCompressManager(t, store, []*mockLLM{llm}, WithContextWindow(1000))
	mgr.msgCount = len(store.history)

	if err := mgr.ForceCompress(context.Background()); err != nil {
		t.Fatalf("ForceCompress: %v", err)
	}
	if llm.callCount == 0 {
		t.Fatal("ForceCompress did not try the model")
	}
	if store.summary == "" {
		t.Fatal("ForceCompress produced no summary although the model succeeded")
	}
	if pct := mgr.contextPercent(store.GetHistory("sess")); pct >= float64(mgr.cfg.safetyPercent) {
		t.Errorf("window still past the safety line after summary: %.0f%%", pct)
	}
	if len(store.history) >= 20 {
		t.Errorf("window did not shrink: %d messages", len(store.history))
	}
}

// TestForceCompress_FailingModelFallsThroughToDrops: when every model fails
// the recovery still completes by dropping the oldest groups, leaves the
// (absent) summary alone and reports success because the request now fits.
func TestForceCompress_FailingModelFallsThroughToDrops(t *testing.T) {
	store := &compressTestStore{history: makeConversation(10, 200)}
	llm := failingLLM(60)
	mgr := newCompressManager(t, store, []*mockLLM{llm}, WithContextWindow(1000))
	mgr.msgCount = len(store.history)

	if err := mgr.ForceCompress(context.Background()); err != nil {
		t.Fatalf("ForceCompress: %v", err)
	}
	if llm.callCount == 0 {
		t.Fatal("ForceCompress did not try the model")
	}
	if store.summary != "" {
		t.Errorf("a failed summary was stored: %q", store.summary)
	}
	if pct := mgr.contextPercent(store.GetHistory("sess")); pct >= float64(mgr.cfg.safetyPercent) {
		t.Errorf("window still past the safety line after drops: %.0f%%", pct)
	}
}
