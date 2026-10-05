/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

import (
	"context"
	"errors"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/PivotLLM/spawnllm"

	cronmsg "github.com/PivotLLM/ctxengine/internal/testcron"
	"github.com/PivotLLM/ctxengine/memory"
)

// brokenJob is the fingerprint of the scheduled job in the incident.
const brokenJob = "828e07b7"

// brokenFire builds a fire of the incident's job, wrapped exactly as the host's
// scheduler wraps it; cronmsg.CollapseKey keys it by the bracketed fingerprint.
func brokenFire(at time.Time) spawnllm.Message {
	return msg("user", cronmsg.Build(brokenJob, at, "Run the morning report."))
}

// brokenHistory is the stored conversation of the incident, seqs 536-545, with
// the CreatedAt ages that made the age trigger cut between 537 and 538. The
// message the turn adds, 546, is not included.
func brokenHistory(now time.Time) []memory.StoredMessage {
	day := 24 * time.Hour
	type row struct {
		m   spawnllm.Message
		age time.Duration
	}
	rows := []row{
		{msg("assistant", "Report: everything nominal (536)."), 10 * day},
		{brokenFire(now.Add(-10 * day)), 10 * day}, // 537
		{msg("assistant", "Report for 538."), 4 * day},
		{brokenFire(now.Add(-4 * day)), 4 * day}, // 539
		{brokenFire(now.Add(-3 * day)), 3 * day}, // 540: adjacent to 539
		{msg("assistant", "Report for 541."), 3 * day},
		{brokenFire(now.Add(-2 * day)), 2 * day}, // 542
		{msg("assistant", "Report for 543."), 2 * day},
		{brokenFire(now.Add(-1 * day)), 1 * day}, // 544
		{msg("assistant", "Report for 545."), 1 * day},
	}
	out := make([]memory.StoredMessage, len(rows))
	for i, r := range rows {
		out[i] = memory.StoredMessage{Seq: int64(536 + i), CreatedAt: now.Add(-r.age), Message: r.m}
	}
	return out
}

// seqWindowStore is a session store that keeps real seqs and CreatedAt stamps
// and persists a compacted window with them, as the SQLite store does.
type seqWindowStore struct {
	*mockStore
	window  []memory.StoredMessage
	nextSeq int64
}

func newSeqWindowStore(history []memory.StoredMessage) *seqWindowStore {
	s := &seqWindowStore{mockStore: newMockStore(), window: slices.Clone(history)}
	if n := len(history); n > 0 {
		s.nextSeq = history[n-1].Seq
	}
	return s
}

func (s *seqWindowStore) AddFullMessage(_ string, m spawnllm.Message) (int64, error) {
	s.nextSeq++
	s.window = append(s.window, memory.StoredMessage{Seq: s.nextSeq, CreatedAt: time.Now(), Message: m})
	return s.nextSeq, nil
}

func (s *seqWindowStore) GetHistoryWithSeqs(_ string) []memory.StoredMessage {
	return slices.Clone(s.window)
}

func (s *seqWindowStore) GetHistory(_ string) []spawnllm.Message {
	return storedToPlain(s.window)
}

func (s *seqWindowStore) SetHistoryWithSeqs(_ string, h []memory.StoredMessage) error {
	s.window = slices.Clone(h)
	return nil
}

func (s *seqWindowStore) SetHistory(_ string, _ []spawnllm.Message) error {
	return errors.New("seqWindowStore: SetHistory would lose seqs")
}

// summaryRecorder is a summarization model that records each request and
// returns a valid summary: reply, or validSummaryJSON when reply is empty.
type summaryRecorder struct {
	requests []ModelRequest
	reply    string
}

func (c *summaryRecorder) Complete(_ context.Context, req ModelRequest) (ModelReply, error) {
	c.requests = append(c.requests, req)
	if c.reply != "" {
		return ModelReply{Content: c.reply}, nil
	}
	return ModelReply{Content: validSummaryJSON("reports")}, nil
}

var summarySeqRe = regexp.MustCompile(`(?m)^\[#(\d+)\] \[`)

// summarizedSeqs returns the seqs the summarizer was shown, from the framed
// transcript headers.
func (c *summaryRecorder) summarizedSeqs(t *testing.T) []int64 {
	t.Helper()
	var out []int64
	for _, req := range c.requests {
		for _, m := range summarySeqRe.FindAllStringSubmatch(req.User, -1) {
			n, err := strconv.ParseInt(m[1], 10, 64)
			if err != nil {
				t.Fatalf("parse seq %q: %v", m[1], err)
			}
			out = append(out, n)
		}
	}
	return out
}

// newAgeTriggeredManager returns a manager whose age trigger fires on store
// (anything 7 days old) and retains 5 days, with the host's cron noise key.
func newAgeTriggeredManager(t *testing.T, store *seqWindowStore, caller ModelCaller, opts ...Option) *Manager {
	t.Helper()
	return newTestManager(t, store, append([]Option{
		WithContextWindow(1_000_000), // far below every percentage trigger
		WithMinPercent(20),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithMessageThreshold(0),
		WithTriggerDays(7),
		WithRetainMaxAgeDays(5),
		WithRetainMinMessages(2),
		WithNoiseKey(cronmsg.CollapseKey),
		WithModelCaller(caller),
	}, opts...)...)
}

// assertAccounted fails unless the summarizer input, the window and the
// dropped adjacent repeats partition want: every seq in exactly one of them.
func assertAccounted(t *testing.T, want, summarized, window, dropped []int64) {
	t.Helper()
	where := map[int64]string{}
	for name, seqs := range map[string][]int64{"summarized": summarized, "window": window, "dropped": dropped} {
		for _, s := range seqs {
			if prev, ok := where[s]; ok {
				t.Errorf("seq %d is both %s and %s", s, prev, name)
			}
			where[s] = name
		}
	}
	for _, s := range want {
		if _, ok := where[s]; !ok {
			t.Errorf("seq %d is not summarized, kept or a dropped repeat: lost", s)
		}
	}
}

// assertCoveredRange fails unless the stored summary covers exactly first..last.
func assertCoveredRange(t *testing.T, store *seqWindowStore, first, last int64) {
	t.Helper()
	sum, err := unmarshalSummary(store.GetSummary("test-session"))
	if err != nil || sum == nil {
		t.Fatalf("stored summary: %v", err)
	}
	if sum.CoveredSeqStart != first || sum.CoveredSeqEnd != last {
		t.Errorf("summary covers %d..%d, want %d..%d", sum.CoveredSeqStart, sum.CoveredSeqEnd, first, last)
	}
}

// assertLastUserAssembled fails unless the request Assemble builds ends its
// user messages on want.
func assertLastUserAssembled(t *testing.T, mgr *Manager, want string) {
	t.Helper()
	asm, err := mgr.Assemble(context.Background(), AssembleRequest{})
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	lastUser := ""
	for _, m := range asm.Messages {
		if m.Role == "user" {
			lastUser = m.Content
		}
	}
	if lastUser != want {
		t.Errorf("last user message assembled = %q, want %q", lastUser, want)
	}
}

// TestAddUserMessage_AgeCompactionKeepsTheNewCronFire reproduces the incident:
// a fire of a scheduled job added while the age trigger is due must survive
// the compaction that the add runs, and be the request the model is sent.
// Before the fix, every later fire of the job was collapsed as a duplicate of
// an earlier one with replies in between, the new fire with them, and the
// keep-newest-user clamp missed it because it compared indexes into the
// uncollapsed history.
func TestAddUserMessage_AgeCompactionKeepsTheNewCronFire(t *testing.T) {
	ctx := context.Background()
	now := time.Now()
	store := newSeqWindowStore(brokenHistory(now))
	caller := &summaryRecorder{}
	mgr := newAgeTriggeredManager(t, store, caller)

	fire := brokenFire(now)
	seq, err := mgr.AddUserMessage(ctx, fire)
	if err != nil {
		t.Fatalf("AddUserMessage: %v", err)
	}
	if seq != 546 {
		t.Fatalf("seq = %d, want 546", seq)
	}
	if len(caller.requests) == 0 {
		t.Fatal("the age trigger did not run a compaction")
	}

	window := seqsOf(store.window)
	// 536 and 537 are past the retain age; 540 repeats 539 with nothing
	// between them. Everything else is kept, the new fire included.
	want := []int64{538, 539, 541, 542, 543, 544, 545, 546}
	if !slices.Equal(window, want) {
		t.Errorf("window = %v, want %v", window, want)
	}
	if lu := lastUserStoredIndex(store.window); lu < 0 || store.window[lu].Seq != 546 {
		t.Errorf("the newest user message in the window is not 546: %v", window)
	}

	summarized := caller.summarizedSeqs(t)
	if !slices.Equal(summarized, []int64{536, 537}) {
		t.Errorf("summarized = %v, want [536 537]", summarized)
	}
	assertCoveredRange(t, store, 536, 537)
	all := make([]int64, 0, 11)
	for s := int64(536); s <= 546; s++ {
		all = append(all, s)
	}
	assertAccounted(t, all, summarized, window, []int64{540})

	assertLastUserAssembled(t, mgr, fire.Content)
}

// TestAddUserMessage_AnsweredRunFoldsAndKeepsTheNewFire covers the persisted
// window's run folding: three fires of one job each answered by the same
// short reply fold into one anchor, and the fire just added, which has no
// reply yet, stays after it as the request the model is sent.
func TestAddUserMessage_AnsweredRunFoldsAndKeepsTheNewFire(t *testing.T) {
	now := time.Now()
	day := 24 * time.Hour
	at := func(seq int64, age time.Duration, m spawnllm.Message) memory.StoredMessage {
		return memory.StoredMessage{Seq: seq, CreatedAt: now.Add(-age), Message: m}
	}
	history := []memory.StoredMessage{
		at(1, 10*day, msg("user", "old question")),
		at(2, 10*day, msg("assistant", "old answer")),
		at(3, 3*day, brokenFire(now.Add(-3*day))),
		at(4, 3*day, msg("assistant", "ok")),
		at(5, 2*day, brokenFire(now.Add(-2*day))),
		at(6, 2*day, msg("assistant", "ok")),
		at(7, 1*day, brokenFire(now.Add(-1*day))),
		at(8, 1*day, msg("assistant", "ok")),
	}
	store := newSeqWindowStore(history)
	mgr := newAgeTriggeredManager(t, store, &summaryRecorder{})

	fire := brokenFire(now)
	if _, err := mgr.AddUserMessage(context.Background(), fire); err != nil {
		t.Fatalf("AddUserMessage: %v", err)
	}

	if len(store.window) != 2 {
		t.Fatalf("window = %+v, want the run's anchor and the new fire", storedToPlain(store.window))
	}
	if !strings.HasPrefix(store.window[0].Content, "[scheduled job "+brokenJob+" fired ×3") {
		t.Errorf("window[0] = %q, want the folded run's anchor", store.window[0].Content)
	}
	if got := store.window[1]; got.Seq != 9 || got.Content != fire.Content {
		t.Errorf("window[1] = #%d %q, want the new fire #9", got.Seq, got.Content)
	}
	assertLastUserAssembled(t, mgr, fire.Content)
}

// TestCompress_CollapsedRepeatsAreNotSummarized checks that an adjacent
// repeat the tail collapses is neither summarized nor kept in the window, and
// is still in the archive; the summary covers the summarized prefix only.
func TestCompress_CollapsedRepeatsAreNotSummarized(t *testing.T) {
	now := time.Now()
	day := 24 * time.Hour
	history := []memory.StoredMessage{
		{Seq: 10, CreatedAt: now.Add(-10 * day), Message: msg("user", "old question")},
		{Seq: 11, CreatedAt: now.Add(-10 * day), Message: msg("assistant", "old answer")},
		{Seq: 12, CreatedAt: now.Add(-day), Message: brokenFire(now.Add(-day))},
		{Seq: 13, CreatedAt: now.Add(-day), Message: brokenFire(now.Add(-day + time.Minute))}, // repeats 12
		{Seq: 14, CreatedAt: now.Add(-day), Message: msg("assistant", "Report for 14.")},
		{Seq: 15, CreatedAt: now, Message: msg("user", "thanks")},
	}
	store := newSeqWindowStore(history)
	// The summary's refs must lie inside the archive's seqs to be accepted.
	caller := &summaryRecorder{reply: `{"version":2,"state":{"goals":[{"text":"reports","refs":[{"seq_start":10}]}]}}`}
	mgr := newAgeTriggeredManager(t, store, caller, WithArchiveDir(t.TempDir()))
	defer func() { noErr(t, mgr.Close(context.Background())) }()
	for _, sm := range history {
		mgr.archiveAppend(sm.Seq, sm.Message) // as AddUserMessage and friends do
	}
	mgr.msgCount = len(history)

	if err := mgr.doCompress(context.Background(), false); err != nil {
		t.Fatalf("doCompress: %v", err)
	}

	window := seqsOf(store.window)
	if want := []int64{12, 14, 15}; !slices.Equal(window, want) {
		t.Errorf("window = %v, want %v", window, want)
	}
	summarized := caller.summarizedSeqs(t)
	if want := []int64{10, 11}; !slices.Equal(summarized, want) {
		t.Errorf("summarized = %v, want %v", summarized, want)
	}
	assertCoveredRange(t, store, 10, 11)
	assertAccounted(t, seqsOf(history), summarized, window, []int64{13})

	archived, err := mgr.getOrOpenArchive().QueryRange(13, 13)
	if err != nil {
		t.Fatalf("archive: %v", err)
	}
	if len(archived) != 1 || archived[0].Content != history[3].Content {
		t.Errorf("archive seq 13 = %+v, want the dropped repeat", archived)
	}
}

// TestCollapseStoredNoise covers which repeats the tail collapses.
func TestCollapseStoredNoise(t *testing.T) {
	now := time.Now()
	stored := func(msgs ...spawnllm.Message) []memory.StoredMessage {
		out := make([]memory.StoredMessage, len(msgs))
		for i, m := range msgs {
			out[i] = memory.StoredMessage{Seq: int64(i + 1), Message: m}
		}
		return out
	}
	fire := func(n int) spawnllm.Message { return brokenFire(now.Add(time.Duration(n) * time.Minute)) }

	tests := []struct {
		name  string
		input []memory.StoredMessage
		want  []int64
	}{
		{
			name:  "incident sequence: only the adjacent repeat collapses",
			input: append(brokenHistory(now), memory.StoredMessage{Seq: 546, Message: fire(0)}),
			want:  []int64{536, 537, 538, 539, 541, 542, 543, 544, 545, 546},
		},
		{
			name:  "fires with a reply between are all kept",
			input: stored(fire(1), msg("assistant", "a"), fire(2), msg("assistant", "b"), fire(3), msg("assistant", "c")),
			want:  []int64{1, 2, 3, 4, 5, 6},
		},
		{
			name:  "consecutive fires collapse to the first",
			input: stored(fire(1), fire(2), fire(3), msg("assistant", "done"), msg("user", "thanks")),
			want:  []int64{1, 4, 5},
		},
		{
			name:  "consecutive fires ending on the newest user message collapse to it",
			input: stored(fire(1), fire(2), fire(3), msg("assistant", "done")),
			want:  []int64{3, 4},
		},
		{
			name:  "consecutive identical text collapses",
			input: stored(msg("user", "same"), msg("user", "same"), msg("assistant", "ok"), msg("user", "next")),
			want:  []int64{1, 3, 4},
		},
		{
			name:  "consecutive identical text ending on the newest user message collapses to it",
			input: stored(msg("user", "same"), msg("user", "same"), msg("assistant", "ok")),
			want:  []int64{2, 3},
		},
		{
			name:  "identical text with a reply between is kept",
			input: stored(msg("user", "hello"), msg("assistant", "hi"), msg("user", "hello"), msg("assistant", "hi")),
			want:  []int64{1, 2, 3, 4},
		},
		{
			name:  "a different job ends the run",
			input: stored(fire(1), msg("user", cronmsg.Build("a1b2c3d4", now, "other")), fire(2), msg("assistant", "ok")),
			want:  []int64{1, 2, 3, 4},
		},
		{
			name:  "newest message is never collapsed",
			input: stored(msg("assistant", "x"), msg("assistant", "x")),
			want:  []int64{2},
		},
		{
			name:  "newest user message is never collapsed",
			input: stored(msg("assistant", "r"), fire(1), fire(2), toolCallMsg("", "t1"), toolResultMsg("out", "t1")),
			want:  []int64{1, 3, 4, 5},
		},
		{
			name:  "newest fire repeating the one before it is kept",
			input: stored(msg("assistant", "r"), fire(1), fire(2)),
			want:  []int64{1, 3},
		},
		{
			name:  "tool plumbing never collapses",
			input: stored(msg("user", "go"), toolCallMsg("", "a"), toolResultMsg("same", "a"), toolCallMsg("", "b"), toolResultMsg("same", "b")),
			want:  []int64{1, 2, 3, 4, 5},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := seqsOf(collapseStoredNoise(tt.input, cronmsg.CollapseKey))
			if !slices.Equal(got, tt.want) {
				t.Errorf("kept %v, want %v", got, tt.want)
			}
		})
	}
}

// TestSelectTail_StartIgnoresCollapse checks that start marks where the
// retained tail begins, not len(stored)-len(tail): collapse must not push
// retained messages into the summarized prefix.
func TestSelectTail_StartIgnoresCollapse(t *testing.T) {
	history := []spawnllm.Message{
		msg("user", "same"),
		msg("user", "same"),
		msg("assistant", "ok"),
	}
	tail, start := selectTail(storedAt(history), 0, 0, 0, testNow, estimateTokens, nil)
	if start != 0 {
		t.Errorf("start = %d, want 0", start)
	}
	if got := seqsOf(tail); !slices.Equal(got, []int64{2, 3}) {
		t.Errorf("tail = %v, want [2 3] (the newest user message replaces its repeat)", got)
	}
}

// TestDroppedRepeatSeqs covers the repeats a pass reports as dropped.
func TestDroppedRepeatSeqs(t *testing.T) {
	window := storedAt([]spawnllm.Message{
		msg("user", "c"), msg("user", "c"), msg("assistant", "d"),
	})
	if got := droppedRepeatSeqs(window, window); got != nil {
		t.Errorf("nothing collapsed: got %v, want none", got)
	}
	if got := droppedRepeatSeqs(window, []memory.StoredMessage{window[1], window[2]}); !slices.Equal(got, []int64{1}) {
		t.Errorf("got %v, want [1]", got)
	}
}
