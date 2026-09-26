// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine/memory"
)

// agedCompressStore is compressTestStore with per-message ages. Ages are keyed
// by content, not by position: compaction rewrites the history in place, so an
// index-keyed age would silently re-label the surviving tail after the very
// operation under test.
type agedCompressStore struct {
	*compressTestStore
	ageDays map[string]int // content → days old; missing means "now"
}

func (s *agedCompressStore) GetHistoryWithSeqs(_ string) []memory.StoredMessage {
	out := make([]memory.StoredMessage, len(s.history))
	now := time.Now()
	for i, msg := range s.history {
		out[i] = memory.StoredMessage{
			Seq:       int64(i + 1),
			CreatedAt: now.AddDate(0, 0, -s.ageDays[msg.Content]),
			Message:   msg,
		}
	}
	return out
}

// ageAll marks every message in msgs as `days` old.
func ageAll(msgs []spawnllm.Message, days int) map[string]int {
	out := make(map[string]int, len(msgs))
	for _, m := range msgs {
		out[m.Content] = days
	}
	return out
}

// distinctConversation builds alternating user/assistant messages with unique
// content, so noise collapse never shrinks the tail and message counts mean what
// they say.
func distinctConversation(pairs, charsPerMessage int) []spawnllm.Message {
	msgs := make([]spawnllm.Message, 0, pairs*2)
	for i := range pairs {
		pad := func(tag string) string {
			s := fmt.Sprintf("%s-%d-", tag, i)
			for len(s) < charsPerMessage {
				s += "x"
			}
			return s
		}
		msgs = append(msgs,
			spawnllm.Message{Role: "user", Content: pad("u")},
			spawnllm.Message{Role: "assistant", Content: pad("a")},
		)
	}
	return msgs
}

// TestCompress_AgeCapRemovesOldMessages is the end-to-end proof of the behaviour
// this whole change is about: a session comfortably inside its token budget, but
// weeks old, must actually get smaller. Before the age cap, retention was purely
// token-budgeted, so a pass like this one retained everything and the oldest
// messages stayed in the window indefinitely.
func TestCompress_AgeCapRemovesOldMessages(t *testing.T) {
	history := distinctConversation(10, 100) // 20 messages, ~500 tokens total
	ages := map[string]int{}
	for i, m := range history {
		if i < 12 {
			ages[m.Content] = 30 // the first 12 messages are a month old
		} else {
			ages[m.Content] = 1
		}
	}
	store := &agedCompressStore{compressTestStore: &compressTestStore{history: history}, ageDays: ages}

	mgr := newCompressManager(store.compressTestStore, []*mockLLM{
		{responses: []string{validSummaryJSON("age test")}},
	},
		WithContextWindow(1_000_000), // budget is enormous: only age can cut here
		WithMinPercent(20),
		WithRetainTokenPercent(10),
		WithRetainMaxAgeDays(5),
		WithRetainMinMessages(2),
	)
	// Re-point the manager at the aging store so timestamps are visible.
	mgr.store = store
	mgr.msgCount = len(history)

	if err := mgr.doCompress(context.Background(), false); err != nil {
		t.Fatalf("doCompress: %v", err)
	}

	remaining := store.GetHistoryWithSeqs("sess")
	if len(remaining) >= len(history) {
		t.Fatalf("history did not shrink: %d messages before, %d after", len(history), len(remaining))
	}
	for _, sm := range remaining {
		if age := time.Since(sm.CreatedAt); age > 6*24*time.Hour {
			t.Errorf("a %v-old message survived a 5-day retain cap: %.20q", age.Round(time.Hour), sm.Content)
		}
	}
	if store.summary == "" {
		t.Error("expected the summarized prefix to produce a summary")
	}
}

// TestCompress_NoAgeCapRetainsEverything is the off path and the regression this
// investigation found: with age retention disabled and a budget this large,
// compaction correctly has nothing to do — which is exactly why old messages
// used to accumulate forever.
func TestCompress_NoAgeCapRetainsEverything(t *testing.T) {
	history := distinctConversation(10, 100)
	store := &agedCompressStore{
		compressTestStore: &compressTestStore{history: history},
		ageDays:           ageAll(history, 30),
	}

	mgr := newCompressManager(store.compressTestStore, []*mockLLM{
		{responses: []string{validSummaryJSON("no age cap")}},
	},
		WithContextWindow(1_000_000),
		WithMinPercent(20),
		WithRetainTokenPercent(10),
		WithRetainMaxAgeDays(0), // disabled
	)
	mgr.store = store
	mgr.msgCount = len(history)

	err := mgr.doCompress(context.Background(), false)
	if err == nil {
		if got := len(store.GetHistoryWithSeqs("sess")); got != len(history) {
			t.Errorf("without an age cap the whole history should be retained; got %d of %d", got, len(history))
		}
		return
	}
	if !errors.Is(err, ErrNothingToCompress) {
		t.Fatalf("expected ErrNothingToCompress with no age cap, got %v", err)
	}
}

// TestCompress_AgeCapKeepsLatestUserMessage guards the clamp that outranks every
// retention bound: archiving past the most recent user message leaves a payload
// strict providers reject outright.
func TestCompress_AgeCapKeepsLatestUserMessage(t *testing.T) {
	history := distinctConversation(6, 100)
	// Everything, including the last user turn, is ancient.
	store := &agedCompressStore{
		compressTestStore: &compressTestStore{history: history},
		ageDays:           ageAll(history, 90),
	}

	mgr := newCompressManager(store.compressTestStore, []*mockLLM{
		{responses: []string{validSummaryJSON("clamp")}},
	},
		WithContextWindow(1_000_000),
		WithMinPercent(20),
		WithRetainMaxAgeDays(5),
		WithRetainMinMessages(1),
	)
	mgr.store = store
	mgr.msgCount = len(history)

	if err := mgr.doCompress(context.Background(), false); err != nil && !errors.Is(err, ErrNothingToCompress) {
		t.Fatalf("doCompress: %v", err)
	}

	remaining := store.GetHistory("sess")
	if len(remaining) == 0 {
		t.Fatal("the live window must never be emptied")
	}
	sawUser := false
	for _, m := range remaining {
		if m.Role == "user" {
			sawUser = true
		}
	}
	if !sawUser {
		t.Error("the most recent user message must survive any age cap")
	}
}
