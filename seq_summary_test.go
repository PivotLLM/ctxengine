// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine/memory"
)

// seqTrackingLLM captures the request sent to the model for inspection.
type seqTrackingLLM struct {
	captured ModelRequest
	called   bool
	response string
	err      error
}

func (l *seqTrackingLLM) Complete(_ context.Context, req ModelRequest) (ModelReply, error) {
	l.captured = req
	l.called = true
	if l.err != nil {
		return ModelReply{}, l.err
	}
	return ModelReply{Content: l.response}, nil
}

// seqStore is a SessionStore that returns StoredMessages with explicit seq numbers.
type seqStore struct {
	stored  []memory.StoredMessage
	summary string
}

func newSeqStore(msgs []memory.StoredMessage) *seqStore {
	return &seqStore{stored: msgs}
}

func (s *seqStore) GetHistory(_ string) []spawnllm.Message {
	msgs := make([]spawnllm.Message, len(s.stored))
	for i, sm := range s.stored {
		msgs[i] = sm.Message
	}
	return msgs
}

func (s *seqStore) GetHistoryWithSeqs(_ string) []memory.StoredMessage {
	cp := make([]memory.StoredMessage, len(s.stored))
	copy(cp, s.stored)
	return cp
}

func (s *seqStore) GetSummary(_ string) string { return s.summary }
func (s *seqStore) SetSummary(_, v string) error {
	s.summary = v
	return nil
}

func (s *seqStore) SetHistory(_ string, h []spawnllm.Message) error {
	stored := make([]memory.StoredMessage, len(h))
	for i, msg := range h {
		stored[i] = memory.StoredMessage{Seq: int64(i + 1), Message: msg}
	}
	s.stored = stored
	return nil
}

func (s *seqStore) SetHistoryWithSeqs(_ string, h []memory.StoredMessage) error {
	cp := make([]memory.StoredMessage, len(h))
	copy(cp, h)
	s.stored = cp
	return nil
}
func (s *seqStore) Save(_ string) error                                        { return nil }
func (s *seqStore) AddMessage(_, _, _ string) error                            { return nil }
func (s *seqStore) AddFullMessage(_ string, _ spawnllm.Message) (int64, error) { return 0, nil }
func (s *seqStore) TruncateHistory(_ string, _ int) error                      { return nil }
func (s *seqStore) SetPendingTurn(_ string) error                              { return nil }
func (s *seqStore) ClearPendingTurn(_ string) error                            { return nil }
func (s *seqStore) GetArchiveBounds(_ string) (int64, int64)                   { return 0, 0 }
func (s *seqStore) ListPendingSessions() ([]string, error)                     { return nil, nil }
func (s *seqStore) Close() error                                               { return nil }

// buildSeqSummaryJSON builds a valid summary JSON with the given coverage fields.
func buildSeqSummaryJSON(goals string, coveredStart, coveredEnd int) string {
	return fmt.Sprintf(
		`{"version":2,"state":{"goals":[{"text":%q,"refs":[{"seq_start":%d}]}]},"covered_seq_start":%d,"covered_seq_end":%d}`,
		goals, coveredStart, coveredStart, coveredEnd,
	)
}

// TestSeqAware_CoveredRangeSetFromActualSeqs verifies that after compression,
// CoveredSeqStart and CoveredSeqEnd in the stored summary reflect the actual
// seq numbers of the messages passed to the LLM, not any values the LLM emitted.
func TestSeqAware_CoveredRangeSetFromActualSeqs(t *testing.T) {
	// Create 10 stored messages with seq numbers 11–20 (simulating a session
	// that has been truncated; the first 10 were skipped).
	const startSeq = 11
	stored := make([]memory.StoredMessage, 0, 11)
	for i := range 10 {
		stored = append(stored, memory.StoredMessage{
			Seq:     int64(startSeq + i),
			Message: spawnllm.Message{Role: "user", Content: strings.Repeat("a", 200)},
		})
	}
	if len(stored)%2 != 0 {
		stored = append(stored, memory.StoredMessage{
			Seq:     int64(startSeq + len(stored)),
			Message: spawnllm.Message{Role: "assistant", Content: strings.Repeat("b", 200)},
		})
	}

	store := newSeqStore(stored)

	// The LLM emits a summary that claims to cover seq 1–5 (wrong — the actual
	// range is 11–18 for the summarized portion). callModel must override these
	// with the actual min/max seq from the passed slice.
	fakeSummary := buildSeqSummaryJSON("fake goal", 1, 5)
	llm := &seqTrackingLLM{response: fakeSummary}

	mgr := asManager(t, New("sess", store,
		WithContextWindow(2000),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithRetainTokenPercent(20),
		WithRetainMinMessages(2),
		WithModelCaller(llm),
	))
	mgr.msgCount = len(stored)

	err := mgr.doCompress(context.Background(), false)
	if err != nil {
		t.Fatalf("doCompress returned error: %v", err)
	}

	if store.summary == "" {
		t.Fatal("expected summary to be stored")
	}

	var s Summary
	if err := json.Unmarshal([]byte(store.summary), &s); err != nil {
		t.Fatalf("stored summary is not valid JSON: %v", err)
	}

	// CoveredSeqStart must match the actual min seq of the summarized slice.
	if s.CoveredSeqStart < startSeq {
		t.Errorf("CoveredSeqStart %d is less than the minimum actual seq %d",
			s.CoveredSeqStart, startSeq)
	}
	// CoveredSeqEnd must match the actual max seq of the summarized portion
	// (at most maxSeq of the slice, not the LLM-emitted 5).
	if s.CoveredSeqEnd < s.CoveredSeqStart {
		t.Errorf("CoveredSeqEnd %d < CoveredSeqStart %d", s.CoveredSeqEnd, s.CoveredSeqStart)
	}
	// CoveredSeqEnd must not be the LLM-fabricated value of 5.
	if s.CoveredSeqEnd == 5 {
		t.Error("CoveredSeqEnd appears to be the LLM-emitted value (5), not the actual max seq")
	}
}

// TestSeqAware_PromptContainsSeqPrefixes verifies that messages sent to the
// summarizer LLM include [#N] prefixes with the actual stored sequence numbers.
func TestSeqAware_PromptContainsSeqPrefixes(t *testing.T) {
	stored := []memory.StoredMessage{
		{Seq: 42, Message: spawnllm.Message{Role: "user", Content: strings.Repeat("x", 500)}},
		{Seq: 43, Message: spawnllm.Message{Role: "assistant", Content: strings.Repeat("y", 500)}},
		{Seq: 44, Message: spawnllm.Message{Role: "user", Content: strings.Repeat("z", 500)}},
		{Seq: 45, Message: spawnllm.Message{Role: "assistant", Content: strings.Repeat("w", 500)}},
	}

	store := newSeqStore(stored)
	llm := &seqTrackingLLM{
		response: buildSeqSummaryJSON("prompt test", 42, 44),
	}

	mgr := asManager(t, New("sess", store,
		WithContextWindow(1000),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithRetainTokenPercent(20),
		WithRetainMinMessages(2),
		WithModelCaller(llm),
	))
	mgr.msgCount = len(stored)

	if err := mgr.doCompress(context.Background(), false); err != nil {
		t.Fatalf("doCompress returned error: %v", err)
	}

	if !llm.called {
		t.Fatal("no request captured by LLM — compression may not have fired")
	}

	// The user message sent to the LLM contains the formatted conversation and
	// must carry the [#N] prefixes.
	user := llm.captured.User
	if !strings.Contains(user, "[#42]") && !strings.Contains(user, "[#43]") && !strings.Contains(user, "[#44]") {
		t.Errorf("expected LLM prompt to contain [#N] seq prefixes; prompt: %q",
			user[:min(200, len(user))])
	}
}

func TestSeqAware_PromptContainsToolMetadata(t *testing.T) {
	stored := []memory.StoredMessage{
		{
			Seq: 50,
			Message: spawnllm.Message{
				Role:    "assistant",
				Content: "I will inspect the file.",
				ToolCalls: []spawnllm.ToolCall{
					{
						ID: "call_123",
						Function: &spawnllm.FunctionCall{
							Name:      "read_file",
							Arguments: `{"path":"llmcontext/compress.go"}`,
						},
					},
				},
			},
		},
		{
			Seq: 51,
			Message: spawnllm.Message{
				Role:       "tool",
				ToolCallID: "call_123",
				Content:    "file contents",
			},
		},
	}
	llm := &seqTrackingLLM{
		response: buildSeqSummaryJSON("tool metadata", 50, 51),
	}

	mgr := &Manager{sessionKey: "test", caller: llm}
	if _, ok := mgr.callModel(context.Background(), nil, stored, 1, 100, false, "", 0, &compactionRecorder{sessionKey: "test"}); !ok {
		t.Fatal("expected callModel to accept fake summary")
	}
	if !llm.called {
		t.Fatal("model was not called")
	}
	userPrompt := llm.captured.User
	for _, want := range []string{
		"[#50] [assistant]",
		"tool_calls:",
		"id: call_123",
		"name: read_file",
		`arguments: {"path":"llmcontext/compress.go"}`,
		"[#51] [tool]",
		"tool_call_id: call_123",
	} {
		if !strings.Contains(userPrompt, want) {
			t.Fatalf("summarization prompt missing %q; prompt:\n%s", want, userPrompt)
		}
	}
}

// TestSeqAware_OutOfRangeRetrievableHistoryStripped verifies that
// retrievable_history entries whose seq references fall outside the valid range
// are stripped from the summary before it is accepted.
func TestSeqAware_OutOfRangeRetrievableHistoryStripped(t *testing.T) {
	// Summary emitted by the LLM with three message_index entries:
	//   - seq 10–12: within range (covered is 10–20, archive max is 20)
	//   - seq 5–7: below coveredStart (10) → out of range
	//   - seq 25–27: above archiveMax (20) → out of range
	summaryWithBadEntries := `{
			"version": 2,
			"state": {"goals": [{"text": "test", "refs": [{"seq_start": 10}]}]},
			"key_moments": [
				{"refs": [{"seq_start": 10}], "role": "user", "summary": "valid moment"},
				{"refs": [{"seq_start": 3}],  "role": "user", "summary": "too old"},
				{"refs": [{"seq_start": 25}], "role": "user", "summary": "future seq"}
			],
		"message_index": [
			{"seq_start": 10, "seq_end": 12, "role": "user", "label": "valid entry"},
			{"seq_start": 5,  "seq_end": 7,  "role": "user", "label": "before covered start"},
			{"seq_start": 25, "seq_end": 27, "role": "user", "label": "beyond archive max"}
		],
		"covered_seq_start": 10,
		"covered_seq_end": 20
	}`

	s, err := validateAndUnmarshalLLMResponse(summaryWithBadEntries)
	if err != nil {
		t.Fatalf("validateAndUnmarshalLLMResponse failed: %v", err)
	}

	// Apply the validation with archiveMax = 20.
	s.StripOutOfRangeSeqRefs(10, 20)

	// Only the entry with seq 10–12 should survive in MessageIndex.
	if len(s.MessageIndex) != 1 {
		t.Errorf("expected 1 message_index entry after stripping; got %d", len(s.MessageIndex))
	} else if s.MessageIndex[0].SeqStart != 10 {
		t.Errorf("expected surviving entry SeqStart=10; got %d", s.MessageIndex[0].SeqStart)
	}

	// Only the key_moment with seq 10 should survive.
	if len(s.KeyMoments) != 1 {
		t.Errorf("expected 1 key_moment after stripping; got %d", len(s.KeyMoments))
	} else if s.KeyMoments[0].Seq != 10 {
		t.Errorf("expected surviving key_moment Seq=10; got %d", s.KeyMoments[0].Seq)
	}
}

// TestSeqAware_AllKeyMomentsInRangeKept verifies that valid key_moments (within
// range) are not stripped.
func TestSeqAware_AllKeyMomentsInRangeKept(t *testing.T) {
	s := &Summary{
		Version: summaryVersion,
		KeyMoments: []KeyMoment{
			{Refs: []SeqRange{{SeqStart: 5}}, Role: "user", Summary: "moment at 5"},
			{Refs: []SeqRange{{SeqStart: 10}}, Role: "assistant", Summary: "moment at 10"},
			{Refs: []SeqRange{{SeqStart: 15}}, Role: "user", Summary: "moment at 15"},
		},
		MessageIndex: []IndexEntry{
			{SeqStart: 5, SeqEnd: 8, Role: "user", Label: "block A"},
		},
		CoveredSeqStart: 5,
		CoveredSeqEnd:   15,
	}

	s.StripOutOfRangeSeqRefs(5, 20)

	if len(s.KeyMoments) != 3 {
		t.Errorf("expected all 3 key_moments to be kept; got %d", len(s.KeyMoments))
	}
	if len(s.MessageIndex) != 1 {
		t.Errorf("expected 1 message_index entry to be kept; got %d", len(s.MessageIndex))
	}
}

func TestSeqAware_ExistingSummaryCoverageAndRefsSurviveNextCompaction(t *testing.T) {
	stored := make([]memory.StoredMessage, 10)
	for i := range stored {
		stored[i] = memory.StoredMessage{
			Seq:     int64(11 + i),
			Message: spawnllm.Message{Role: "user", Content: strings.Repeat("x", 500)},
		}
	}
	store := newSeqStore(stored)
	existing := &Summary{
		Version:             summaryVersion,
		State:               SummaryState{Goals: []SummaryItem{{Text: "old goal", Refs: []SeqRange{{SeqStart: 5}}}}},
		KeyMoments:          []KeyMoment{{Refs: []SeqRange{{SeqStart: 5}}, Role: "user", Summary: "old decision"}},
		CoveredSeqStart:     1,
		CoveredSeqEnd:       10,
		CoveredRanges:       []SeqRange{{SeqStart: 1, SeqEnd: 10}},
		LastSummarizedSeq:   10,
		LastSummarizedRange: SeqRange{SeqStart: 1, SeqEnd: 10},
	}
	data, err := json.Marshal(existing)
	if err != nil {
		t.Fatalf("marshal existing: %v", err)
	}
	store.summary = string(data)

	response := `{
		"version": 2,
		"state": {"goals": [
			{"text": "old goal", "refs": [{"seq_start": 5}]},
			{"text": "new goal", "refs": [{"seq_start": 11}]}
		]},
		"key_moments": [
			{"refs": [{"seq_start": 5}], "role": "user", "summary": "old decision"},
			{"refs": [{"seq_start": 11}], "role": "user", "summary": "new decision"}
		],
		"message_index": [
			{"seq_start": 11, "seq_end": 12, "role": "user", "label": "new work"}
		]
	}`
	llm := &seqTrackingLLM{response: response}
	mgr := asManager(t, New("sess", store,
		WithContextWindow(1000),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithRetainTokenPercent(20),
		WithRetainMinMessages(2),
		WithModelCaller(llm),
	))
	mgr.msgCount = len(stored)

	if err := mgr.doCompress(context.Background(), false); err != nil {
		t.Fatalf("doCompress returned error: %v", err)
	}

	var got Summary
	if err := json.Unmarshal([]byte(store.summary), &got); err != nil {
		t.Fatalf("stored summary is not valid JSON: %v", err)
	}
	if got.CoveredSeqStart != 1 {
		t.Fatalf("CoveredSeqStart = %d, want cumulative start 1", got.CoveredSeqStart)
	}
	foundOld := false
	for _, km := range got.KeyMoments {
		if len(km.Refs) > 0 && km.Refs[0].SeqStart == 5 {
			foundOld = true
		}
	}
	if !foundOld {
		t.Fatalf("old key moment ref #5 was not preserved: %+v", got.KeyMoments)
	}
}
