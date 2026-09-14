// ctxengine
// License: MIT

package ctxengine

// End-to-end tests: every test here drives a Manager over the REAL
// session.SQLiteStore and the REAL archive (both on the same per-session
// database under one temp dir), through the public Add*/Assemble/Compact/
// Reset/Close surface, and asserts against the rows that land on disk.

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	cronmsg "github.com/PivotLLM/ctxengine/internal/testcron"
	"github.com/PivotLLM/ctxengine/memory"
	"github.com/PivotLLM/ctxengine/session"
	"github.com/PivotLLM/spawnllm"
)

// e2eKey carries a ':' so the on-disk sanitisation is exercised too.
const e2eKey = "agent:alice:main"

const (
	e2eLayerBefore = "You are Alice, a careful assistant."
	e2eLayerAfter  = "Answer briefly and cite seqs when you refer to history."
	e2eInjection   = "INJECTED-MEMORY-7f3a: the user prefers tea over coffee"
	e2eChannel     = "telegram-alice"
	e2eChatID      = "chat-42"
)

// e2eSummarizer is a ModelCaller that reads the [#N] seq prefixes out of the
// summarization prompt and answers with a valid summary citing them, so every
// reference in the stored summary points at a message the engine actually
// sent it. It records every request so tests can inspect Exclude lists and
// the prompt text. Setting fail makes it answer with non-JSON.
type e2eSummarizer struct {
	model    string
	fail     bool
	requests []ModelRequest
}

var promptSeqRE = regexp.MustCompile(`(?m)^\[#(\d+)\] \[`)

// promptSeqs returns the seqs named by the [#N] prefixes of a summarization
// prompt, in prompt order.
func promptSeqs(user string) []int64 {
	var out []int64
	for _, m := range promptSeqRE.FindAllStringSubmatch(user, -1) {
		n, _ := strconv.ParseInt(m[1], 10, 64)
		out = append(out, n)
	}
	return out
}

func (s *e2eSummarizer) Complete(_ context.Context, req ModelRequest) (ModelReply, error) {
	s.requests = append(s.requests, req)
	if s.fail {
		return ModelReply{Content: "I have nothing structured to say.", Model: s.model}, nil
	}
	seqs := promptSeqs(req.User)
	if len(seqs) == 0 {
		return ModelReply{Content: "{}", Model: s.model}, nil
	}
	first, last, mid := seqs[0], seqs[len(seqs)-1], seqs[len(seqs)/2]
	body := fmt.Sprintf(`{"version":2,`+
		`"state":{"goals":[{"text":"e2e goal","refs":[{"seq_start":%d,"seq_end":%d}]}],`+
		`"progress":[{"text":"e2e progress","refs":[{"seq_start":%d}]}]},`+
		`"key_moments":[{"refs":[{"seq_start":%d}],"role":"user","summary":"e2e moment"}],`+
		`"message_index":[{"seq_start":%d,"seq_end":%d,"role":"user","label":"e2e range"}]}`,
		first, last, mid, last, first, last)
	return ModelReply{Content: body, Model: s.model}, nil
}

// e2eStore opens the real SQLite session store rooted at dir.
func e2eStore(t *testing.T, dir string) *session.SQLiteStore {
	t.Helper()
	store, err := session.NewSQLiteStore(dir)
	if err != nil {
		t.Fatalf("NewSQLiteStore: %v", err)
	}
	return store
}

// e2eOpts is the small-window configuration the conversation tests run under:
// a 3000-token window so ten-odd turns of 240-character messages cross the
// normal threshold, a 5% retain budget so one pass lands well under target,
// and the archive on the same directory as the store.
func e2eOpts(dir string, caller ModelCaller, extra ...Option) []Option {
	opts := []Option{
		WithArchiveDir(dir),
		WithModelCaller(caller),
		WithContextWindow(3000),
		WithOverheadTokens(0),
		WithMinPercent(20),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithRetainTokenPercent(5),
		WithRetainMinMessages(2),
		WithMessageThreshold(0),
	}
	return append(opts, extra...)
}

func e2eRequest() AssembleRequest {
	return AssembleRequest{
		Layers: []Layer{
			{Name: "static", Text: e2eLayerBefore},
			{Name: "dynamic", Text: e2eLayerAfter, AfterSummary: true},
		},
		Injections: []Injection{{Placement: PlaceCurrentUser, Text: e2eInjection}},
		Channel:    e2eChannel,
		ChatID:     e2eChatID,
	}
}

// pad right-pads s with a marker so a message has a predictable size.
func pad(s string, n int) string {
	if len(s) >= n {
		return s
	}
	return s + strings.Repeat(".", n-len(s))
}

func e2eCall(id, name, args string) spawnllm.ToolCall {
	return spawnllm.ToolCall{ID: id, Function: &spawnllm.FunctionCall{Name: name, Arguments: args}}
}

// e2eDB opens a read-only SQL connection to the session's database so tests
// can assert against the tables directly.
func e2eDB(t *testing.T, dir string) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", "file:"+memory.ArchivePath(dir, e2eKey)+"?mode=ro")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

func queryInt(t *testing.T, db *sql.DB, q string, args ...any) int64 {
	t.Helper()
	var n int64
	if err := db.QueryRow(q, args...).Scan(&n); err != nil {
		t.Fatalf("query %q: %v", q, err)
	}
	return n
}

func queryString(t *testing.T, db *sql.DB, q string, args ...any) string {
	t.Helper()
	var s string
	if err := db.QueryRow(q, args...).Scan(&s); err != nil {
		t.Fatalf("query %q: %v", q, err)
	}
	return s
}

func seqsOf(stored []memory.StoredMessage) []int64 {
	out := make([]int64, len(stored))
	for i, sm := range stored {
		out[i] = sm.Seq
	}
	return out
}

// summaryRefs returns every SeqRange a stored summary cites: state items, key
// moments, message index entries and the covered ranges.
func summaryRefs(t *testing.T, raw string) []SeqRange {
	t.Helper()
	s, err := unmarshalSummary(raw)
	if err != nil || s == nil {
		t.Fatalf("stored summary is not a valid Summary: %v (%q)", err, raw)
	}
	var refs []SeqRange
	for _, items := range [][]SummaryItem{s.State.Goals, s.State.Progress, s.State.Pending, s.State.Constraints, s.CarryForward} {
		for _, it := range items {
			refs = append(refs, it.Refs...)
		}
	}
	for _, km := range s.KeyMoments {
		refs = append(refs, km.Refs...)
	}
	for _, e := range s.MessageIndex {
		refs = append(refs, SeqRange{SeqStart: e.SeqStart, SeqEnd: e.SeqEnd})
	}
	refs = append(refs, s.CoveredRanges...)
	if len(refs) == 0 {
		t.Fatal("stored summary cites nothing")
	}
	return refs
}

// assertRefsResolve checks that every cited range resolves, row for row,
// through the archive query API.
func assertRefsResolve(t *testing.T, a *memory.ArchiveStore, refs []SeqRange) {
	t.Helper()
	for _, r := range refs {
		rows, err := a.QueryRange(r.SeqStart, r.SeqEnd)
		if err != nil {
			t.Fatalf("QueryRange(%d,%d): %v", r.SeqStart, r.SeqEnd, err)
		}
		if want := int(r.SeqEnd - r.SeqStart + 1); len(rows) != want {
			t.Errorf("ref #%d-#%d resolves to %d rows, want %d", r.SeqStart, r.SeqEnd, len(rows), want)
			continue
		}
		for i, row := range rows {
			if row.Seq != r.SeqStart+int64(i) {
				t.Errorf("ref #%d-#%d row %d has seq %d", r.SeqStart, r.SeqEnd, i, row.Seq)
			}
		}
	}
}

// assertWellFormed is the provider-shape invariant: exactly one system
// message and it leads; every tool result answers a call declared by the
// nearest preceding assistant tool-call turn; every assistant tool-call turn
// is immediately followed by results for all of its ids.
func assertWellFormed(t *testing.T, msgs []spawnllm.Message) {
	t.Helper()
	if len(msgs) == 0 || msgs[0].Role != "system" {
		t.Fatalf("messages must start with a system message; got %d messages, first role %q", len(msgs), firstRole(msgs))
	}
	for i := 1; i < len(msgs); i++ {
		if msgs[i].Role == "system" {
			t.Errorf("message %d is a second system message", i)
		}
	}
	var declared map[string]bool // ids of the nearest preceding tool-call turn
	for i := 1; i < len(msgs); i++ {
		m := msgs[i]
		switch {
		case m.Role == "assistant" && len(m.ToolCalls) > 0:
			declared = make(map[string]bool, len(m.ToolCalls))
			for _, tc := range m.ToolCalls {
				declared[tc.ID] = true
			}
			// Results for every id must follow immediately.
			answered := make(map[string]bool)
			for j := i + 1; j < len(msgs) && msgs[j].Role == "tool"; j++ {
				answered[msgs[j].ToolCallID] = true
			}
			for id := range declared {
				if !answered[id] {
					t.Errorf("message %d declares tool call %q with no result following it", i, id)
				}
			}
		case m.Role == "tool":
			if !declared[m.ToolCallID] {
				t.Errorf("message %d: tool result %q is not declared by the nearest preceding tool-call turn", i, m.ToolCallID)
			}
		}
	}
}

func firstRole(msgs []spawnllm.Message) string {
	if len(msgs) == 0 {
		return ""
	}
	return msgs[0].Role
}

// e2eSession is one driven conversation over the real store and archive.
type e2eSession struct {
	t     *testing.T
	dir   string
	store *session.SQLiteStore
	mgr   *Manager
	sum   *e2eSummarizer
	// seqs records every seq the Add* calls returned, in order.
	seqs []int64
	// compactedAfterSeq is the seq whose Add* call triggered the compaction,
	// 0 while none has run.
	compactedAfterSeq int64
	assemblies        []Assembly
}

func (s *e2eSession) add(seq int64, err error) int64 {
	t := s.t
	t.Helper()
	if err != nil {
		t.Fatalf("Add*: %v", err)
	}
	if seq <= 0 {
		t.Fatalf("Add* minted seq %d", seq)
	}
	if len(s.seqs) > 0 && seq != s.seqs[len(s.seqs)-1]+1 {
		t.Fatalf("seq %d does not follow %d", seq, s.seqs[len(s.seqs)-1])
	}
	s.seqs = append(s.seqs, seq)
	if s.compactedAfterSeq == 0 && s.mgr.LastCompactionReport() != nil {
		s.compactedAfterSeq = seq
	}
	return seq
}

func (s *e2eSession) assemble(t *testing.T) Assembly {
	t.Helper()
	asm, err := s.mgr.Assemble(context.Background(), e2eRequest())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	s.assemblies = append(s.assemblies, asm)
	return asm
}

// runE2EConversation drives fifteen turns through a fresh Manager on dir:
// plain user/assistant turns, a parallel two-call tool turn at turn 3 and a
// single-call tool turn at turn 7, with Assemble before every dispatch. The
// sizes are chosen so the normal compaction trigger fires exactly once, on an
// Add* call somewhere in the middle, and never again.
func runE2EConversation(t *testing.T, dir string, sum *e2eSummarizer, extra ...Option) *e2eSession {
	t.Helper()
	ctx := context.Background()
	store := e2eStore(t, dir)
	s := &e2eSession{t: t, dir: dir, store: store, sum: sum}
	s.mgr = New(e2eKey, store, e2eOpts(dir, sum, extra...)...).(*Manager)

	for turn := 1; turn <= 15; turn++ {
		s.add(s.mgr.AddUserMessage(ctx, spawnllm.Message{
			Role: "user", Content: pad(fmt.Sprintf("turn %d: user asks about topic %d ", turn, turn), 240),
		}))
		s.assemble(t)
		switch turn {
		case 3:
			s.add(s.mgr.AddToolCallMessage(ctx, spawnllm.Message{Role: "assistant", ToolCalls: []spawnllm.ToolCall{
				e2eCall("t3-a", "weather_lookup", `{"city":"Ottawa"}`),
				e2eCall("t3-b", "weather_lookup", `{"city":"Toronto"}`),
			}}))
			s.add(s.mgr.AddToolResult(ctx, spawnllm.Message{
				Role: "tool", ToolCallID: "t3-a", Content: pad("turn 3 result a: Ottawa forecast sunny with a light breeze ", 400),
			}))
			s.add(s.mgr.AddToolResult(ctx, spawnllm.Message{
				Role: "tool", ToolCallID: "t3-b", Content: pad("turn 3 result b: Toronto forecast overcast, chance of rain ", 400),
			}))
			s.assemble(t)
		case 7:
			s.add(s.mgr.AddToolCallMessage(ctx, spawnllm.Message{Role: "assistant", ToolCalls: []spawnllm.ToolCall{
				e2eCall("t7-a", "weather_lookup", `{"city":"Montreal"}`),
			}}))
			s.add(s.mgr.AddToolResult(ctx, spawnllm.Message{
				Role: "tool", ToolCallID: "t7-a", Content: pad("turn 7 result: Montreal forecast snow flurries ", 400),
			}))
			s.assemble(t)
		}
		s.add(s.mgr.AddAssistantMessage(ctx, spawnllm.Message{
			Role: "assistant", Content: pad(fmt.Sprintf("turn %d: assistant answers about topic %d ", turn, turn), 240),
		}))
	}
	return s
}

// TestE2E_FullLoopCompactsOnRealStore is the whole turn shape against the real
// store: fifteen turns with tool groups, one compaction crossing the normal
// threshold, then the assembled request, the archive, the summaries table and
// the window are all checked against the seqs the Add* calls returned.
func TestE2E_FullLoopCompactsOnRealStore(t *testing.T) {
	dir := t.TempDir()
	sum := &e2eSummarizer{model: "e2e-model"}
	s := runE2EConversation(t, dir, sum)
	defer s.store.Close()
	defer s.mgr.Close(context.Background())

	// Exactly one compaction, mid-conversation, and every Assemble was clean.
	if len(sum.requests) != 1 {
		t.Fatalf("summarizer called %d times, want exactly 1", len(sum.requests))
	}
	if s.compactedAfterSeq == 0 || s.compactedAfterSeq >= s.seqs[len(s.seqs)-1]-6 {
		t.Fatalf("compaction fired after seq %d of %d; want it well before the end", s.compactedAfterSeq, s.seqs[len(s.seqs)-1])
	}
	rep := s.mgr.LastCompactionReport()
	if rep.Outcome != "success" || len(rep.Attempts) != 1 || rep.Attempts[0].Model != "e2e-model" || rep.Attempts[0].Status != "ok" {
		t.Fatalf("report = %+v", rep)
	}
	for i, asm := range s.assemblies {
		if asm.Changed() {
			t.Errorf("assembly %d reported changes (evictions=%d compacted=%v); the normal trigger runs on Add*, not Assemble", i, len(asm.Evictions), asm.Compacted)
		}
	}

	// The post-compaction system message: layer, summary block, layer — once.
	last := s.assemblies[len(s.assemblies)-1].Messages
	if last[0].Role != "system" {
		t.Fatalf("messages[0].Role = %q, want system", last[0].Role)
	}
	for i := 1; i < len(last); i++ {
		if last[i].Role == "system" {
			t.Errorf("messages[%d] is a second system message", i)
		}
	}
	rendered := s.mgr.RenderedSummary()
	if rendered == "" || !strings.Contains(rendered, "e2e goal") {
		t.Fatalf("RenderedSummary() = %q", rendered)
	}
	sys := last[0].Content
	prefix, suffix := e2eLayerBefore+systemSeparator, systemSeparator+e2eLayerAfter
	if !strings.HasPrefix(sys, prefix) || !strings.HasSuffix(sys, suffix) {
		t.Fatalf("system message is not layer/summary/layer:\n%s", sys)
	}
	middle := strings.TrimSuffix(strings.TrimPrefix(sys, prefix), suffix)
	if !strings.HasPrefix(middle, "CONTEXT_SUMMARY:") || !strings.HasSuffix(middle, rendered) {
		t.Fatalf("the block between the layers is not the rendered summary:\n%s", middle)
	}
	if strings.Count(sys, "CONTEXT_SUMMARY:") != 1 {
		t.Errorf("summary block appears %d times", strings.Count(sys, "CONTEXT_SUMMARY:"))
	}
	assertWellFormed(t, last)

	// The injection rode on the last user message of the built slice only.
	lastUser := -1
	for i := len(last) - 1; i >= 0; i-- {
		if last[i].Role == "user" {
			lastUser = i
			break
		}
	}
	if lastUser < 0 || !strings.HasSuffix(last[lastUser].Content, systemSeparator+e2eInjection) {
		t.Fatalf("injection not folded into the last user message: %q", last[lastUser].Content)
	}
	db := e2eDB(t, dir)
	for _, table := range []string{"messages", "window"} {
		if n := queryInt(t, db, "SELECT count(*) FROM "+table+" WHERE payload LIKE ?", "%INJECTED-MEMORY%"); n != 0 {
			t.Errorf("injection persisted in %s (%d rows)", table, n)
		}
	}
	if n := queryInt(t, db, "SELECT count(*) FROM summaries WHERE summary LIKE ?", "%INJECTED-MEMORY%"); n != 0 {
		t.Errorf("injection persisted in summaries (%d rows)", n)
	}
	if strings.Contains(queryString(t, db, "SELECT summary FROM session_state WHERE id = 1"), "INJECTED-MEMORY") {
		t.Error("injection persisted in session_state.summary")
	}
	if strings.Contains(sum.requests[0].User, "INJECTED-MEMORY") {
		t.Error("injection reached the summarization prompt")
	}

	// The archive holds every seq ever minted, once.
	archive := s.mgr.getOrOpenArchive()
	rows, err := archive.QueryRange(1, s.seqs[len(s.seqs)-1])
	if err != nil {
		t.Fatalf("QueryRange: %v", err)
	}
	if got := seqsOf(rows); !slices.Equal(got, s.seqs) {
		t.Errorf("archive seqs = %v, want every minted seq %v", got, s.seqs)
	}
	if n := queryInt(t, db, "SELECT count(*) FROM messages"); n != int64(len(s.seqs)) {
		t.Errorf("messages rows = %d, want %d", n, len(s.seqs))
	}

	// Exactly one summaries row, covering seq 1 up to the row before the tail.
	window := s.store.GetHistoryWithSeqs(e2eKey)
	if len(window) == 0 {
		t.Fatal("empty window")
	}
	if n := queryInt(t, db, "SELECT count(*) FROM summaries"); n != 1 {
		t.Fatalf("summaries rows = %d, want 1", n)
	}
	cited := promptSeqs(sum.requests[0].User)
	wantEnd := cited[len(cited)-1]
	var srcStart, srcEnd, covStart, covEnd int64
	if err := db.QueryRow("SELECT source_seq_start, source_seq_end, covered_seq_start, covered_seq_end FROM summaries").Scan(&srcStart, &srcEnd, &covStart, &covEnd); err != nil {
		t.Fatalf("summaries row: %v", err)
	}
	if srcStart != 1 || srcEnd != wantEnd || covStart != 1 || covEnd != wantEnd {
		t.Errorf("summaries row range = source #%d-#%d covered #%d-#%d, want #1-#%d", srcStart, srcEnd, covStart, covEnd, wantEnd)
	}
	if wantEnd >= s.compactedAfterSeq {
		t.Errorf("summary covers up to #%d but compaction fired on #%d", wantEnd, s.compactedAfterSeq)
	}

	// The window is the retained tail plus what came after: a suffix of the
	// minted seqs starting right after the summarized range.
	winSeqs := seqsOf(window)
	k := slices.Index(s.seqs, wantEnd+1)
	if k < 0 || !slices.Equal(winSeqs, s.seqs[k:]) {
		t.Errorf("window seqs = %v, want minted suffix from #%d: %v", winSeqs, wantEnd+1, s.seqs[max(k, 0):])
	}
	if n := queryInt(t, db, "SELECT count(*) FROM window"); n != int64(len(window)) {
		t.Errorf("window rows = %d, want %d", n, len(window))
	}
	if n := queryInt(t, db, "SELECT count(*) FROM window WHERE seq <= ?", wantEnd); n != 0 {
		t.Errorf("%d summarized rows still in the window", n)
	}

	// Every seq the summary cites resolves through the archive.
	assertRefsResolve(t, archive, summaryRefs(t, s.store.GetSummary(e2eKey)))
}

// TestE2E_CloseAndReopenPreservesState closes the manager and the store after
// a compacted conversation, reopens both on the same directory, and expects
// the reopened manager to be indistinguishable: stats, rendered summary,
// window rows, the next seq and the assembled request.
func TestE2E_CloseAndReopenPreservesState(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	sum := &e2eSummarizer{model: "e2e-model"}
	s := runE2EConversation(t, dir, sum)
	if len(sum.requests) != 1 {
		t.Fatalf("summarizer called %d times, want 1", len(sum.requests))
	}

	before := s.mgr.Stats()
	renderedBefore := s.mgr.RenderedSummary()
	windowBefore := s.store.GetHistoryWithSeqs(e2eKey)
	asmBefore := s.assemble(t)
	db := e2eDB(t, dir)
	nextSeqBefore := queryInt(t, db, "SELECT next_seq FROM session_state WHERE id = 1")
	if nextSeqBefore != s.seqs[len(s.seqs)-1] {
		t.Fatalf("next_seq = %d, want last minted %d", nextSeqBefore, s.seqs[len(s.seqs)-1])
	}

	if err := s.mgr.Close(ctx); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := s.store.Close(); err != nil {
		t.Fatalf("store.Close: %v", err)
	}

	store2 := e2eStore(t, dir)
	defer store2.Close()
	mgr2 := New(e2eKey, store2, e2eOpts(dir, sum)...).(*Manager)
	defer mgr2.Close(ctx)

	after := mgr2.Stats()
	if after.TotalMessages != before.TotalMessages || after.MeaningfulMessages != before.MeaningfulMessages ||
		after.EstimatedTokens != before.EstimatedTokens || after.ContextWindowPct != before.ContextWindowPct ||
		after.SummaryTokens != before.SummaryTokens || after.CompressionCooling != before.CompressionCooling ||
		after.CoolingSinceCount != before.CoolingSinceCount || !after.LastCompressedAt.Equal(before.LastCompressedAt) {
		t.Errorf("Stats after reopen = %+v, want %+v", after, before)
	}
	if before.MeaningfulMessages != len(s.seqs) || before.LastCompressedAt.IsZero() || before.SummaryTokens == 0 {
		t.Errorf("pre-close stats are not what a compacted session should report: %+v", before)
	}
	// LastCompressionGain is in-memory only: CompactionState has no gain
	// column, so a reopened manager reports 0 by design.

	if got := mgr2.RenderedSummary(); got != renderedBefore {
		t.Errorf("RenderedSummary after reopen:\n%s\nwant:\n%s", got, renderedBefore)
	}
	if got := store2.GetHistoryWithSeqs(e2eKey); !reflect.DeepEqual(got, windowBefore) {
		t.Errorf("window rows differ after reopen:\n%+v\nwant:\n%+v", got, windowBefore)
	}
	if n := queryInt(t, db, "SELECT next_seq FROM session_state WHERE id = 1"); n != nextSeqBefore {
		t.Errorf("next_seq after reopen = %d, want %d", n, nextSeqBefore)
	}

	asmAfter, err := mgr2.Assemble(ctx, e2eRequest())
	if err != nil {
		t.Fatalf("Assemble after reopen: %v", err)
	}
	if asmAfter.Changed() {
		t.Errorf("reopened Assemble rewrote history: %+v", asmAfter)
	}
	if asmAfter.Messages[0].Content != asmBefore.Messages[0].Content {
		t.Errorf("system message differs after reopen:\n%s\nwant:\n%s", asmAfter.Messages[0].Content, asmBefore.Messages[0].Content)
	}
	if !reflect.DeepEqual(asmAfter.Messages[1:], asmBefore.Messages[1:]) {
		t.Errorf("history slice differs after reopen")
	}

	seq, err := mgr2.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: "after reopen"})
	if err != nil || seq != nextSeqBefore+1 {
		t.Errorf("first seq after reopen = %d (%v), want %d", seq, err, nextSeqBefore+1)
	}
}

// seedParallelGroupHistory writes the nine-message history the well-formedness
// tests share through a manager whose window is too large to trigger anything,
// then closes it. Sizes (tokens ≈ chars/4): six small messages of 10 tokens, a
// two-call tool turn of ~45 tokens and two 500-token results — 1105 in all.
//
//	#1 user  #2 assistant  #3 user
//	#4 assistant(tc-a, tc-b)  #5 tool tc-a (big)  #6 tool tc-b (big)
//	#7 assistant  #8 user  #9 assistant
func seedParallelGroupHistory(t *testing.T, dir string) {
	t.Helper()
	ctx := context.Background()
	store := e2eStore(t, dir)
	mgr := New(e2eKey, store, WithArchiveDir(dir), WithContextWindow(1_000_000), WithMessageThreshold(0)).(*Manager)
	small := func(role, text string) spawnllm.Message { return spawnllm.Message{Role: role, Content: pad(text, 40)} }
	big := strings.Repeat("z", 2000)
	must := func(seq int64, err error) {
		t.Helper()
		if err != nil || seq <= 0 {
			t.Fatalf("seed Add*: seq=%d err=%v", seq, err)
		}
	}
	must(mgr.AddUserMessage(ctx, small("user", "u1 first question")))
	must(mgr.AddAssistantMessage(ctx, small("assistant", "a1 first answer")))
	must(mgr.AddUserMessage(ctx, small("user", "u2 look both up")))
	must(mgr.AddToolCallMessage(ctx, spawnllm.Message{Role: "assistant", ToolCalls: []spawnllm.ToolCall{
		e2eCall("tc-a", "weather_lookup", `{"city":"Ottawa"}`),
		e2eCall("tc-b", "weather_lookup", `{"city":"Toronto"}`),
	}}))
	must(mgr.AddToolResult(ctx, spawnllm.Message{Role: "tool", ToolCallID: "tc-a", Content: "result a " + big}))
	must(mgr.AddToolResult(ctx, spawnllm.Message{Role: "tool", ToolCallID: "tc-b", Content: "result b " + big}))
	must(mgr.AddAssistantMessage(ctx, small("assistant", "a2 both looked up")))
	must(mgr.AddUserMessage(ctx, small("user", "u3 thanks")))
	must(mgr.AddAssistantMessage(ctx, small("assistant", "a3 welcome")))
	if len(store.GetHistory(e2eKey)) != 9 {
		t.Fatalf("seed produced %d messages, want 9", len(store.GetHistory(e2eKey)))
	}
	if err := mgr.Close(ctx); err != nil {
		t.Fatalf("seed Close: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("seed store.Close: %v", err)
	}
}

// tightOpts is the small-window configuration the well-formedness tests
// reopen the seeded session under: 1300 tokens, safety at 80% (1040), so the
// seeded 1105 tokens are past the safety line, and a 130-token tail budget
// that fits #7-#9 (30 tokens) but not the 1045-token tool group before them —
// the budget boundary lands inside the parallel group.
func tightOpts(dir string, caller ModelCaller) []Option {
	return []Option{
		WithArchiveDir(dir),
		WithModelCaller(caller),
		WithContextWindow(1300),
		WithOverheadTokens(0),
		WithMinPercent(20),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithRetainTokenPercent(10),
		WithRetainMinMessages(2),
		WithMessageThreshold(0),
	}
}

// TestE2E_WellFormedAfterSafetyNetCompaction reopens the seeded session under
// the tight window and lets Assemble's pre-build safety net compact it with a
// working summarizer. The tool group must be kept or summarized whole.
func TestE2E_WellFormedAfterSafetyNetCompaction(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	seedParallelGroupHistory(t, dir)

	sum := &e2eSummarizer{model: "e2e-model"}
	store := e2eStore(t, dir)
	defer store.Close()
	mgr := New(e2eKey, store, tightOpts(dir, sum)...).(*Manager)
	defer mgr.Close(ctx)

	asm, err := mgr.Assemble(ctx, e2eRequest())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if !asm.Compacted || !asm.Changed() {
		t.Fatalf("safety net did not compact: %+v", asm)
	}
	if len(sum.requests) != 1 {
		t.Fatalf("summarizer called %d times, want 1", len(sum.requests))
	}
	assertWellFormed(t, asm.Messages)

	// The whole group went to the summary: #1-#6 summarized, #7-#9 retained.
	if got := promptSeqs(sum.requests[0].User); !slices.Equal(got, []int64{1, 2, 3, 4, 5, 6}) {
		t.Errorf("summarized seqs = %v, want #1-#6 (the group whole)", got)
	}
	window := store.GetHistoryWithSeqs(e2eKey)
	if got := seqsOf(window); !slices.Equal(got, []int64{7, 8, 9}) {
		t.Errorf("window seqs = %v, want [7 8 9]", got)
	}
	assertWellFormed(t, append([]spawnllm.Message{{Role: "system", Content: "x"}}, storedToPlain(window)...))
	db := e2eDB(t, dir)
	var srcStart, srcEnd int64
	if err := db.QueryRow("SELECT source_seq_start, source_seq_end FROM summaries").Scan(&srcStart, &srcEnd); err != nil {
		t.Fatalf("summaries row: %v", err)
	}
	if srcStart != 1 || srcEnd != 6 {
		t.Errorf("summary source range #%d-#%d, want #1-#6", srcStart, srcEnd)
	}
	if got := len(asm.Messages); got != 4 {
		t.Errorf("built %d messages, want system + #7-#9", got)
	}
}

// TestE2E_WellFormedAfterSafetyNetDrop is the safety net with every model
// failing: the drop path removes the oldest turn groups whole, so the
// parallel group #4-#6 goes together and the stored window is left
// well-formed, not merely the built request. Regression: the drop once
// resolved groups backwards from the head and stripped the tool-call turn
// while leaving its results behind.
func TestE2E_WellFormedAfterSafetyNetDrop(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	seedParallelGroupHistory(t, dir)

	failures := make([]error, 20)
	for i := range failures {
		failures[i] = errors.New("model down")
	}
	failing := &mockLLM{model: "down", errors: failures}
	store := e2eStore(t, dir)
	defer store.Close()
	mgr := New(e2eKey, store, tightOpts(dir, chainOf([]*mockLLM{failing}))...).(*Manager)
	defer mgr.Close(ctx)

	asm, err := mgr.Assemble(ctx, e2eRequest())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if !asm.Compacted {
		t.Fatalf("safety net did not run: %+v", asm)
	}
	if failing.callCount == 0 {
		t.Fatal("model never called")
	}
	assertWellFormed(t, asm.Messages)

	window := store.GetHistoryWithSeqs(e2eKey)
	if len(window) >= 9 || len(window) == 0 {
		t.Fatalf("drop path left %d window rows, want fewer than 9", len(window))
	}
	if got := seqsOf(window); len(got) != 3 || got[0] != 7 || got[2] != 9 {
		t.Errorf("window seqs = %v, want #7-#9 (the parallel group dropped whole)", got)
	}
	for _, m := range window {
		if m.ToolCallID != "" || len(m.ToolCalls) > 0 {
			t.Errorf("tool plumbing left in the stored window: seq %d role %s", m.Seq, m.Role)
		}
	}
	if got := mgr.LastCompactionReport().Outcome; got != "success" {
		t.Errorf("report outcome = %q, want success (drops got under the safety line)", got)
	}
	if n := queryInt(t, e2eDB(t, dir), "SELECT count(*) FROM summaries"); n != 0 {
		t.Errorf("summaries rows = %d, want 0 (no model succeeded)", n)
	}
	// The built slice carries only the clean tail.
	if got := len(asm.Messages); got != 4 {
		t.Errorf("built %d messages, want system + #7-#9", got)
	}
}

// TestE2E_WellFormedAfterForceCompress runs the 413-recovery path on the
// seeded session under the tight window and checks the group is dropped
// whole rather than split.
func TestE2E_WellFormedAfterForceCompress(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	seedParallelGroupHistory(t, dir)

	store := e2eStore(t, dir)
	defer store.Close()
	mgr := New(e2eKey, store, tightOpts(dir, &e2eSummarizer{model: "e2e-model"})...).(*Manager)
	defer mgr.Close(ctx)

	if err := mgr.ForceCompress(ctx); err != nil {
		t.Fatalf("ForceCompress: %v", err)
	}
	window := store.GetHistory(e2eKey)
	if len(window) != 3 || window[0].Role != "assistant" || window[1].Role != "user" || window[2].Role != "assistant" {
		t.Fatalf("window after ForceCompress = %d messages (%v), want #7-#9", len(window), rolesOf(window))
	}
	for _, m := range window {
		if len(m.ToolCalls) > 0 || m.ToolCallID != "" {
			t.Errorf("tool plumbing survived a group drop: %+v", m)
		}
	}
	asm, err := mgr.Assemble(ctx, e2eRequest())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if asm.Changed() {
		t.Errorf("Assemble after ForceCompress rewrote history again: %+v", asm)
	}
	assertWellFormed(t, asm.Messages)
	if got := len(asm.Messages); got != 4 {
		t.Errorf("built %d messages, want system + 3", got)
	}
}

func rolesOf(msgs []spawnllm.Message) []string {
	out := make([]string, len(msgs))
	for i, m := range msgs {
		out[i] = m.Role
	}
	return out
}

// TestE2E_ForceCompressKeepsWindowSeqsInArchive checks that the tail
// ForceCompress retains keeps the seqs the archive knows it by.
//
// Regression: ForceCompress once persisted through SessionStore.SetHistory,
// which minted fresh seqs after next_seq for every retained message while the
// archive rows kept the originals, so a summary citing the retained tail was
// stripped as out of range and session_messages could not fetch what the
// window showed. It now persists the way compaction does.
func TestE2E_ForceCompressKeepsWindowSeqsInArchive(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	seedParallelGroupHistory(t, dir)

	store := e2eStore(t, dir)
	defer store.Close()
	mgr := New(e2eKey, store, tightOpts(dir, &e2eSummarizer{model: "e2e-model"})...).(*Manager)
	defer mgr.Close(ctx)

	if err := mgr.ForceCompress(ctx); err != nil {
		t.Fatalf("ForceCompress: %v", err)
	}
	winSeqs := seqsOf(store.GetHistoryWithSeqs(e2eKey))
	if !slices.Equal(winSeqs, []int64{7, 8, 9}) {
		t.Errorf("window seqs after ForceCompress = %v, want the original [7 8 9]", winSeqs)
	}
	db := e2eDB(t, dir)
	for _, seq := range winSeqs {
		if n := queryInt(t, db, "SELECT count(*) FROM messages WHERE seq = ?", seq); n != 1 {
			t.Errorf("window seq %d has no archive row", seq)
		}
	}
	if n := queryInt(t, db, "SELECT next_seq FROM session_state WHERE id = 1"); n != 9 {
		t.Errorf("next_seq = %d after ForceCompress, want 9 (no reseq)", n)
	}
}

// TestE2E_SeqStabilityAcrossCompaction records every seq the Add* calls mint,
// compacts on the real store, and checks the surviving window rows carry
// their original seqs (SetHistoryWithSeqs, not a reseq), that next_seq did
// not move, and that the archive has each seq exactly once.
func TestE2E_SeqStabilityAcrossCompaction(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	sum := &e2eSummarizer{model: "e2e-model"}
	s := runE2EConversation(t, dir, sum)
	defer s.store.Close()
	defer s.mgr.Close(ctx)
	if s.compactedAfterSeq == 0 {
		t.Fatal("no compaction ran")
	}

	window := s.store.GetHistoryWithSeqs(e2eKey)
	winSeqs := seqsOf(window)
	for _, seq := range winSeqs {
		if !slices.Contains(s.seqs, seq) {
			t.Errorf("window seq %d was never minted by Add*", seq)
		}
	}
	k := slices.Index(s.seqs, winSeqs[0])
	if k < 0 || !slices.Equal(winSeqs, s.seqs[k:]) {
		t.Errorf("window seqs %v are not the minted suffix %v", winSeqs, s.seqs[max(k, 0):])
	}
	// The retained tail was the tail at the moment of compaction: seqs below
	// the trigger seq survive with their original numbers.
	if winSeqs[0] > s.compactedAfterSeq {
		t.Errorf("window starts at %d, after the compaction trigger %d: the tail was renumbered", winSeqs[0], s.compactedAfterSeq)
	}
	for i, sm := range window {
		if sm.CreatedAt.IsZero() {
			t.Errorf("window row %d (seq %d) lost its timestamp", i, sm.Seq)
		}
	}

	db := e2eDB(t, dir)
	last := s.seqs[len(s.seqs)-1]
	if n := queryInt(t, db, "SELECT next_seq FROM session_state WHERE id = 1"); n != last {
		t.Errorf("next_seq = %d, want %d (a reseq would have moved it)", n, last)
	}
	if n := queryInt(t, db, "SELECT count(*) FROM messages"); n != int64(len(s.seqs)) {
		t.Errorf("messages rows = %d, want %d", n, len(s.seqs))
	}
	if n := queryInt(t, db, "SELECT count(DISTINCT seq) FROM messages"); n != int64(len(s.seqs)) {
		t.Errorf("distinct archive seqs = %d, want %d", n, len(s.seqs))
	}
	if n := queryInt(t, db, "SELECT max(seq) FROM messages"); n != last {
		t.Errorf("archive max seq = %d, want %d", n, last)
	}
	seq, err := s.mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: "one more"})
	if err != nil || seq != last+1 {
		t.Errorf("next minted seq = %d (%v), want %d", seq, err, last+1)
	}
}

// TestE2E_EvictionObservedThroughAssemble builds a stale file read through
// Add*, ages it past EvictTurns with plain turns, and watches Assemble evict
// it: the event, the placeholder in the built slice, the rewritten window row
// under the same seq, and the untouched archive row.
func TestE2E_EvictionObservedThroughAssemble(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	store := e2eStore(t, dir)
	defer store.Close()
	mgr := New(e2eKey, store,
		WithArchiveDir(dir),
		WithContextWindow(100_000),
		WithMessageThreshold(0),
		WithModelCaller(&e2eSummarizer{model: "e2e-model"}),
	).(*Manager)
	defer mgr.Close(ctx)

	must := func(seq int64, err error) int64 {
		t.Helper()
		if err != nil || seq <= 0 {
			t.Fatalf("Add*: seq=%d err=%v", seq, err)
		}
		return seq
	}
	const path = "notes/ch1.md"
	content := "chapter one begins here " + strings.Repeat("x", 4976) // 5000 bytes
	must(mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: "read chapter one"}))
	must(mgr.AddToolCallMessage(ctx, spawnllm.Message{Role: "assistant", ToolCalls: []spawnllm.ToolCall{
		e2eCall("read-1", "file_read_bytes", `{"path":"`+path+`"}`),
	}}))
	readSeq := must(mgr.AddToolResult(ctx, spawnllm.Message{Role: "tool", ToolCallID: "read-1", Content: content}))
	must(mgr.AddAssistantMessage(ctx, spawnllm.Message{Role: "assistant", Content: "chapter one is about a journey"}))
	for i := 1; i <= 6; i++ { // twelve more turn groups: the read is now 14 groups old
		must(mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: fmt.Sprintf("follow-up %d", i)}))
		must(mgr.AddAssistantMessage(ctx, spawnllm.Message{Role: "assistant", Content: fmt.Sprintf("answer %d", i)}))
	}
	before := store.GetHistoryWithSeqs(e2eKey)

	asm, err := mgr.Assemble(ctx, e2eRequest())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if len(asm.Evictions) != 1 || !asm.Changed() || asm.Compacted {
		t.Fatalf("Assembly = evictions %+v compacted %v; want exactly one eviction", asm.Evictions, asm.Compacted)
	}
	ev := asm.Evictions[0]
	if ev.Seq != readSeq || ev.Tool != "file_read_bytes" || ev.Resource != path || ev.Bytes != len(content) || ev.Reason != "stale" || ev.AgeTurns <= 10 {
		t.Errorf("eviction event = %+v", ev)
	}
	placeholder := evictionPlaceholder("file_read_bytes", path, len(content))

	// The built slice carries the placeholder in the tool result's slot.
	found := false
	for _, m := range asm.Messages {
		if m.Role == "tool" && m.ToolCallID == "read-1" {
			found = true
			if m.Content != placeholder {
				t.Errorf("built tool result = %q, want placeholder %q", m.Content, placeholder)
			}
		}
	}
	if !found {
		t.Error("tool result missing from the built slice")
	}
	assertWellFormed(t, asm.Messages)

	// The window row was rewritten in place: same seq, same neighbours.
	after := store.GetHistoryWithSeqs(e2eKey)
	if !slices.Equal(seqsOf(after), seqsOf(before)) {
		t.Fatalf("window seqs changed: %v -> %v", seqsOf(before), seqsOf(after))
	}
	for i, sm := range after {
		switch {
		case sm.Seq == readSeq:
			if sm.Content != placeholder {
				t.Errorf("window row %d content = %q, want placeholder", sm.Seq, sm.Content)
			}
			if !sm.CreatedAt.Equal(before[i].CreatedAt) {
				t.Errorf("window row %d timestamp changed", sm.Seq)
			}
		case sm.Content != before[i].Content:
			t.Errorf("window row %d changed although it was not evicted", sm.Seq)
		}
	}
	db := e2eDB(t, dir)
	if got := queryString(t, db, "SELECT payload FROM window WHERE seq = ?", readSeq); !strings.Contains(got, "[evicted:") {
		t.Errorf("window table row %d not rewritten: %s", readSeq, got)
	}
	// The archive keeps the original content (truncated to the tool cap).
	if got := queryString(t, db, "SELECT payload FROM messages WHERE seq = ?", readSeq); !strings.Contains(got, "chapter one begins here") || strings.Contains(got, "[evicted:") {
		t.Errorf("archive row %d changed by eviction: %.80s", readSeq, got)
	}

	// Idempotent: a second sweep finds nothing.
	again, err := mgr.Assemble(ctx, e2eRequest())
	if err != nil {
		t.Fatalf("second Assemble: %v", err)
	}
	if again.Changed() {
		t.Errorf("second Assemble evicted again: %+v", again.Evictions)
	}
}

// TestE2E_ResetContinueCompactReopen resets a compacted session and checks
// the window and live summary go while the archive rows stay, that new
// messages continue the seq space, that a later compaction cites only the
// new range, and that all of it survives a reopen.
func TestE2E_ResetContinueCompactReopen(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	sum := &e2eSummarizer{model: "e2e-model"}
	s := runE2EConversation(t, dir, sum)
	if len(sum.requests) != 1 {
		t.Fatalf("summarizer called %d times, want 1", len(sum.requests))
	}
	preMax := s.seqs[len(s.seqs)-1]
	db := e2eDB(t, dir)

	if err := s.mgr.Reset(ctx); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	if n := len(s.store.GetHistoryWithSeqs(e2eKey)); n != 0 {
		t.Errorf("window has %d rows after Reset", n)
	}
	if n := queryInt(t, db, "SELECT count(*) FROM window"); n != 0 {
		t.Errorf("window table has %d rows after Reset", n)
	}
	if got := queryString(t, db, "SELECT summary FROM session_state WHERE id = 1"); got != "" {
		t.Errorf("session_state.summary = %q after Reset, want empty", got)
	}
	if got := s.mgr.RenderedSummary(); got != "" {
		t.Errorf("RenderedSummary after Reset = %q", got)
	}
	if n := queryInt(t, db, "SELECT count(*) FROM messages"); n != int64(len(s.seqs)) {
		t.Errorf("messages rows = %d after Reset, want %d preserved", n, len(s.seqs))
	}
	if n := queryInt(t, db, "SELECT count(*) FROM summaries"); n != 1 {
		t.Errorf("summaries rows = %d after Reset, want 1 preserved", n)
	}
	if got := s.mgr.Stats(); got.TotalMessages != 0 || got.MeaningfulMessages != 0 || got.SummaryTokens != 0 {
		t.Errorf("Stats after Reset = %+v", got)
	}

	// Continue: seqs carry on above the pre-Reset maximum.
	var newSeqs []int64
	for i := 1; i <= 6; i++ {
		seq, err := s.mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: pad(fmt.Sprintf("after reset %d: user ", i), 240)})
		if err != nil {
			t.Fatalf("AddUserMessage: %v", err)
		}
		newSeqs = append(newSeqs, seq)
		seq, err = s.mgr.AddAssistantMessage(ctx, spawnllm.Message{Role: "assistant", Content: pad(fmt.Sprintf("after reset %d: assistant ", i), 240)})
		if err != nil {
			t.Fatalf("AddAssistantMessage: %v", err)
		}
		newSeqs = append(newSeqs, seq)
	}
	if newSeqs[0] != preMax+1 {
		t.Fatalf("first seq after Reset = %d, want %d", newSeqs[0], preMax+1)
	}
	if len(sum.requests) != 1 {
		t.Fatalf("summarizer called %d times before the manual compaction, want still 1", len(sum.requests))
	}

	if err := s.mgr.Compact(ctx); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if len(sum.requests) != 2 {
		t.Fatalf("summarizer called %d times, want 2", len(sum.requests))
	}
	cited := promptSeqs(sum.requests[1].User)
	if cited[0] != preMax+1 {
		t.Errorf("post-Reset compaction summarized from #%d, want #%d", cited[0], preMax+1)
	}
	raw := s.store.GetSummary(e2eKey)
	for _, r := range summaryRefs(t, raw) {
		if r.SeqStart <= preMax {
			t.Errorf("post-Reset summary cites #%d-#%d, inside the pre-Reset range (max #%d)", r.SeqStart, r.SeqEnd, preMax)
		}
	}
	if n := queryInt(t, db, "SELECT count(*) FROM summaries"); n != 2 {
		t.Fatalf("summaries rows = %d, want 2", n)
	}
	var srcStart, covStart int64
	if err := db.QueryRow("SELECT source_seq_start, covered_seq_start FROM summaries WHERE id = 2").Scan(&srcStart, &covStart); err != nil {
		t.Fatalf("summaries row 2: %v", err)
	}
	if srcStart != preMax+1 || covStart != preMax+1 {
		t.Errorf("summary 2 starts at source #%d covered #%d, want #%d", srcStart, covStart, preMax+1)
	}
	window := s.store.GetHistoryWithSeqs(e2eKey)
	if len(window) == 0 || window[0].Seq <= preMax || window[0].Seq != cited[len(cited)-1]+1 {
		t.Errorf("window after compaction = %v, want the tail after #%d", seqsOf(window), cited[len(cited)-1])
	}
	assertRefsResolve(t, s.mgr.getOrOpenArchive(), summaryRefs(t, raw))

	// Reopen preserves all of it.
	renderedBefore := s.mgr.RenderedSummary()
	statsBefore := s.mgr.Stats()
	if err := s.mgr.Close(ctx); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := s.store.Close(); err != nil {
		t.Fatalf("store.Close: %v", err)
	}
	store2 := e2eStore(t, dir)
	defer store2.Close()
	mgr2 := New(e2eKey, store2, e2eOpts(dir, sum)...).(*Manager)
	defer mgr2.Close(ctx)
	if got := mgr2.RenderedSummary(); got != renderedBefore || got == "" {
		t.Errorf("RenderedSummary after reopen = %q, want %q", got, renderedBefore)
	}
	if got := store2.GetHistoryWithSeqs(e2eKey); !reflect.DeepEqual(got, window) {
		t.Errorf("window differs after reopen: %v vs %v", seqsOf(got), seqsOf(window))
	}
	if got := mgr2.Stats(); got.TotalMessages != statsBefore.TotalMessages || got.MeaningfulMessages != statsBefore.MeaningfulMessages || got.SummaryTokens != statsBefore.SummaryTokens {
		t.Errorf("Stats after reopen = %+v, want %+v", got, statsBefore)
	}
	if n := queryInt(t, db, "SELECT count(*) FROM messages"); n != int64(len(s.seqs)+len(newSeqs)) {
		t.Errorf("messages rows = %d after reopen, want %d", n, len(s.seqs)+len(newSeqs))
	}
	if n := queryInt(t, db, "SELECT count(*) FROM summaries"); n != 2 {
		t.Errorf("summaries rows = %d after reopen, want 2", n)
	}
	seq, err := mgr2.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: "after reopen"})
	if err != nil || seq != newSeqs[len(newSeqs)-1]+1 {
		t.Errorf("seq after reopen = %d (%v), want %d", seq, err, newSeqs[len(newSeqs)-1]+1)
	}
}

// TestE2E_ManagerConcurrencySmoke drives one Manager from several goroutines
// at once — writers calling Add*, readers calling Assemble, one Compact —
// under the race detector, and checks nothing panics and the archive holds
// one row per successful Add*.
//
// The manager serialises its public operations (ContextManager states the
// contract), so this runs under the race detector as the proof.
func TestE2E_ManagerConcurrencySmoke(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	store := e2eStore(t, dir)
	defer store.Close()
	sum := &e2eSummarizer{model: "e2e-model"}
	mgr := New(e2eKey, store,
		WithArchiveDir(dir),
		WithModelCaller(sum),
		WithContextWindow(200_000),
		WithOverheadTokens(0),
		WithRetainTokenPercent(1),
		WithRetainMaxTokens(200),
		WithRetainMinMessages(2),
		WithMessageThreshold(0),
	).(*Manager)
	defer mgr.Close(ctx)

	// Seed so the compaction has something to summarize.
	var added atomic.Int64
	for i := 0; i < 10; i++ {
		if seq, _ := mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: pad(fmt.Sprintf("seed user %d ", i), 200)}); seq > 0 {
			added.Add(1)
		}
		if seq, _ := mgr.AddAssistantMessage(ctx, spawnllm.Message{Role: "assistant", Content: pad(fmt.Sprintf("seed assistant %d ", i), 200)}); seq > 0 {
			added.Add(1)
		}
	}

	var wg sync.WaitGroup
	var panics atomic.Int64
	guard := func(name string, fn func()) {
		defer wg.Done()
		defer func() {
			if r := recover(); r != nil {
				panics.Add(1)
				t.Errorf("%s panicked: %v", name, r)
			}
		}()
		fn()
	}
	const writers, perWriter, readers, perReader = 4, 20, 3, 15
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go guard(fmt.Sprintf("writer %d", w), func() {
			for i := 0; i < perWriter; i++ {
				var seq int64
				if i%2 == 0 {
					seq, _ = mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: fmt.Sprintf("writer %d message %d", w, i)})
				} else {
					seq, _ = mgr.AddAssistantMessage(ctx, spawnllm.Message{Role: "assistant", Content: fmt.Sprintf("writer %d reply %d", w, i)})
				}
				if seq > 0 {
					added.Add(1)
				}
			}
		})
	}
	for r := 0; r < readers; r++ {
		wg.Add(1)
		go guard(fmt.Sprintf("reader %d", r), func() {
			for i := 0; i < perReader; i++ {
				if _, err := mgr.Assemble(ctx, e2eRequest()); err != nil {
					t.Errorf("Assemble: %v", err)
				}
			}
		})
	}
	wg.Add(1)
	go guard("compactor", func() {
		if err := mgr.Compact(ctx); err != nil {
			t.Errorf("Compact: %v", err)
		}
	})
	wg.Wait()

	if panics.Load() != 0 {
		t.Fatalf("%d goroutines panicked", panics.Load())
	}
	if got := len(sum.requests); got == 0 {
		t.Error("Compact never reached the model")
	}
	db := e2eDB(t, dir)
	if n := queryInt(t, db, "SELECT count(*) FROM messages"); n != added.Load() {
		t.Errorf("archive rows = %d, want %d (one per successful Add*)", n, added.Load())
	}
	if n := queryInt(t, db, "SELECT count(DISTINCT seq) FROM messages"); n != added.Load() {
		t.Errorf("distinct archive seqs = %d, want %d", n, added.Load())
	}
	if n := queryInt(t, db, "SELECT next_seq FROM session_state WHERE id = 1"); n != added.Load() {
		t.Errorf("next_seq = %d, want %d", n, added.Load())
	}
}

// cronTime is the fire time of the i-th scheduled fire, one hour apart.
func cronTime(i int) time.Time {
	return time.Date(2026, 9, 14, 9+i, 0, 0, 0, time.UTC)
}

// cronFires adds n scheduled-job fires of one job (different fire times,
// identical replies) through the manager and returns the seqs minted.
func cronFires(t *testing.T, mgr *Manager, n int) []int64 {
	t.Helper()
	ctx := context.Background()
	var seqs []int64
	for i := 0; i < n; i++ {
		fire := cronmsg.Build("3f9a1c0d", cronTime(i), "self-check: anything new?")
		seq, err := mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: fire})
		if err != nil || seq <= 0 {
			t.Fatalf("fire %d: seq=%d err=%v", i, seq, err)
		}
		seqs = append(seqs, seq)
		seq, err = mgr.AddAssistantMessage(ctx, spawnllm.Message{Role: "assistant", Content: "No changes."})
		if err != nil || seq <= 0 {
			t.Fatalf("reply %d: seq=%d err=%v", i, seq, err)
		}
		seqs = append(seqs, seq)
	}
	return seqs
}

// TestE2E_NoiseKeyCollapsesCronFires wires the same cron key into the store
// (SetNoiseKey, which drives session_state.meaningful_count) and the manager
// (WithNoiseKey, which drives the summarizer input), fires the job four times
// and checks both: the real row advances by one for the four fires, and the
// summarizer sees a single counted anchor instead of four fires.
func TestE2E_NoiseKeyCollapsesCronFires(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	store := e2eStore(t, dir)
	defer store.Close()
	store.SetNoiseKey(cronmsg.CollapseKey)
	sum := &e2eSummarizer{model: "e2e-model"}
	mgr := New(e2eKey, store,
		WithArchiveDir(dir),
		WithModelCaller(sum),
		WithContextWindow(3000),
		WithOverheadTokens(0),
		WithRetainTokenPercent(5),
		WithRetainMaxTokens(1), // the tail is the floor alone
		WithRetainMinMessages(1),
		WithMessageThreshold(0),
		WithNoiseKey(cronmsg.CollapseKey),
	).(*Manager)
	defer mgr.Close(ctx)

	seqs := cronFires(t, mgr, 4)
	state, err := store.GetCompactionState(e2eKey)
	if err != nil {
		t.Fatalf("GetCompactionState: %v", err)
	}
	// First fire and first reply count; the three repeats of each are noise.
	if state.MeaningfulCount != 2 {
		t.Errorf("meaningful_count = %d after 4 fires + 4 identical replies, want 2", state.MeaningfulCount)
	}
	if got := mgr.Stats().TotalMessages; got != 8 {
		t.Errorf("TotalMessages = %d, want 8 (noise is stored, only not counted)", got)
	}

	// A real question, then a manual compaction: the summarizer gets one anchor.
	if _, err := mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: "real question"}); err != nil {
		t.Fatal(err)
	}
	if err := mgr.Compact(ctx); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if len(sum.requests) != 1 {
		t.Fatalf("summarizer called %d times, want 1", len(sum.requests))
	}
	prompt := sum.requests[0].User
	wantAnchor := fmt.Sprintf(`[scheduled job 3f9a1c0d fired ×4 (#%d-#%d); routine, replies identical: "No changes."]`, seqs[0], seqs[7])
	if !strings.Contains(prompt, wantAnchor) {
		t.Errorf("summarizer prompt lacks the collapsed anchor %q:\n%s", wantAnchor, prompt)
	}
	if n := strings.Count(prompt, "self-check: anything new?"); n != 0 {
		t.Errorf("summarizer prompt still carries %d raw fires", n)
	}
	if got := promptSeqs(prompt); !slices.Equal(got, []int64{seqs[0]}) {
		t.Errorf("prompt seqs = %v, want just the anchor's %d", got, seqs[0])
	}
}

// TestE2E_NoNoiseKeyCountsEveryFire is the off path: without a key each fire
// differs by its timestamp and counts; only the identical replies are noise.
func TestE2E_NoNoiseKeyCountsEveryFire(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	store := e2eStore(t, dir)
	defer store.Close()
	sum := &e2eSummarizer{model: "e2e-model"}
	mgr := New(e2eKey, store,
		WithArchiveDir(dir),
		WithModelCaller(sum),
		WithContextWindow(3000),
		WithOverheadTokens(0),
		WithRetainTokenPercent(5),
		WithRetainMaxTokens(1),
		WithRetainMinMessages(1),
		WithMessageThreshold(0),
	).(*Manager)
	defer mgr.Close(ctx)

	cronFires(t, mgr, 4)
	state, err := store.GetCompactionState(e2eKey)
	if err != nil {
		t.Fatalf("GetCompactionState: %v", err)
	}
	if state.MeaningfulCount != 5 {
		t.Errorf("meaningful_count = %d without a noise key, want 5 (4 fires + 1 reply)", state.MeaningfulCount)
	}
	if _, err := mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: "real question"}); err != nil {
		t.Fatal(err)
	}
	if err := mgr.Compact(ctx); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	prompt := sum.requests[0].User
	if strings.Contains(prompt, "scheduled job") {
		t.Errorf("fires collapsed without a noise key:\n%s", prompt)
	}
	if n := strings.Count(prompt, "self-check: anything new?"); n != 4 {
		t.Errorf("summarizer prompt carries %d raw fires, want 4", n)
	}
}

// TestE2E_NoiseAwareCountSurvivesClose checks the noise-aware count the store
// keeps in session_state.meaningful_count is what a reopened session sees.
//
// Regression: the manager once counted every Add* itself and wrote that
// number over the row on compaction and Close, discarding the store's noise
// classification. The store owns the count now and the manager reads it back.
func TestE2E_NoiseAwareCountSurvivesClose(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	store := e2eStore(t, dir)
	defer store.Close()
	store.SetNoiseKey(cronmsg.CollapseKey)
	mgr := New(e2eKey, store, WithArchiveDir(dir), WithContextWindow(100_000), WithMessageThreshold(0), WithNoiseKey(cronmsg.CollapseKey)).(*Manager)

	cronFires(t, mgr, 4)
	if got := mgr.Stats().MeaningfulMessages; got != 2 {
		t.Errorf("Stats().MeaningfulMessages = %d, want the store's noise-aware 2", got)
	}
	if err := mgr.Close(ctx); err != nil {
		t.Fatalf("Close: %v", err)
	}
	state, err := store.GetCompactionState(e2eKey)
	if err != nil {
		t.Fatalf("GetCompactionState: %v", err)
	}
	if state.MeaningfulCount != 2 {
		t.Errorf("meaningful_count = %d after Close, want the store's noise-aware 2", state.MeaningfulCount)
	}
}

// TestE2E_CompactionReporterDeliversToLastAssembleChannel runs the automatic
// path and checks the report lands once, for the channel and chat the last
// Assemble named, with the same text LastCompactionReport renders.
func TestE2E_CompactionReporterDeliversToLastAssembleChannel(t *testing.T) {
	dir := t.TempDir()
	type delivery struct{ channel, chatID, text string }
	var deliveries []delivery
	sum := &e2eSummarizer{model: "e2e-model"}
	s := runE2EConversation(t, dir, sum, WithCompactionReporter(func(channel, chatID, text string) {
		deliveries = append(deliveries, delivery{channel, chatID, text})
	}))
	defer s.store.Close()
	defer s.mgr.Close(context.Background())

	if len(deliveries) != 1 {
		t.Fatalf("reporter called %d times, want 1: %+v", len(deliveries), deliveries)
	}
	d := deliveries[0]
	if d.channel != e2eChannel || d.chatID != e2eChatID {
		t.Errorf("report delivered to (%q, %q), want (%q, %q)", d.channel, d.chatID, e2eChannel, e2eChatID)
	}
	if want := s.mgr.LastCompactionReport().String(); d.text != want {
		t.Errorf("report text:\n%s\nwant:\n%s", d.text, want)
	}
	if !strings.Contains(d.text, "Model e2e-model: ok") || !strings.Contains(d.text, "Compacted to") {
		t.Errorf("report text does not describe the pass:\n%s", d.text)
	}
}

// excludeRecorder wraps a ModelCaller and records the Exclude list of every
// request it forwards.
type excludeRecorder struct {
	inner    ModelCaller
	excludes [][]string
}

func (r *excludeRecorder) Complete(ctx context.Context, req ModelRequest) (ModelReply, error) {
	r.excludes = append(r.excludes, append([]string(nil), req.Exclude...))
	return r.inner.Complete(ctx, req)
}

// refusalFixture builds a session with enough history to compact twice
// against a two-model chain: the first model always answers with a marker the
// built-in classifier does not know, the second answers valid summaries.
func refusalFixture(t *testing.T, classifier RefusalClassifier) (*Manager, *session.SQLiteStore, *excludeRecorder, *mockLLM) {
	t.Helper()
	dir := t.TempDir()
	store := e2eStore(t, dir)
	t.Cleanup(func() { store.Close() })
	refuser := &mockLLM{model: "m-refuser", responses: slices.Repeat([]string{"NOPE-CUSTOM: this session is off limits"}, 12)}
	worker := &mockLLM{model: "m-worker", responses: []string{validSummaryJSON("first pass"), validSummaryJSON("second pass")}}
	rec := &excludeRecorder{inner: chainOf([]*mockLLM{refuser, worker})}
	opts := []Option{
		WithArchiveDir(dir),
		WithModelCaller(rec),
		WithContextWindow(3000),
		WithOverheadTokens(0),
		WithRetainTokenPercent(5),
		WithRetainMinMessages(2),
		WithMessageThreshold(0),
	}
	if classifier != nil {
		opts = append(opts, WithRefusalClassifier(classifier))
	}
	mgr := New(e2eKey, store, opts...).(*Manager)
	t.Cleanup(func() { mgr.Close(context.Background()) })
	return mgr, store, rec, refuser
}

func addTurns(t *testing.T, mgr *Manager, label string, n int) {
	t.Helper()
	ctx := context.Background()
	for i := 1; i <= n; i++ {
		if _, err := mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: pad(fmt.Sprintf("%s %d: user ", label, i), 240)}); err != nil {
			t.Fatal(err)
		}
		if _, err := mgr.AddAssistantMessage(ctx, spawnllm.Message{Role: "assistant", Content: pad(fmt.Sprintf("%s %d: assistant ", label, i), 240)}); err != nil {
			t.Fatal(err)
		}
	}
}

// TestE2E_CustomRefusalClassifierExcludesModel: a custom classifier turns the
// unknown marker into a refusal, so the refusing model is excluded on the
// retry within the pass and, session-wide, from the first request of the
// next pass.
func TestE2E_CustomRefusalClassifierExcludesModel(t *testing.T) {
	ctx := context.Background()
	mgr, store, rec, refuser := refusalFixture(t, func(_, content string) (bool, string) {
		if strings.Contains(content, "NOPE-CUSTOM") {
			return true, "custom refusal marker"
		}
		return false, ""
	})

	addTurns(t, mgr, "first", 5)
	if err := mgr.Compact(ctx); err != nil {
		t.Fatalf("first Compact: %v", err)
	}
	if len(rec.excludes) != 2 {
		t.Fatalf("first pass made %d requests, want 2 (refused, then retried): %v", len(rec.excludes), rec.excludes)
	}
	if len(rec.excludes[0]) != 0 || !slices.Equal(rec.excludes[1], []string{"m-refuser"}) {
		t.Errorf("first pass Exclude lists = %v, want [] then [m-refuser]", rec.excludes)
	}
	rep := mgr.LastCompactionReport()
	if len(rep.Attempts) != 2 || rep.Attempts[0].Model != "m-refuser" || rep.Attempts[0].Status != "refused" ||
		rep.Attempts[0].Detail != "custom refusal marker" || rep.Attempts[1].Model != "m-worker" || rep.Attempts[1].Status != "ok" {
		t.Errorf("report attempts = %+v", rep.Attempts)
	}
	if !strings.Contains(rep.String(), "refused (custom refusal marker)") {
		t.Errorf("report text lacks the custom detail:\n%s", rep.String())
	}
	if store.GetSummary(e2eKey) == "" {
		t.Fatal("no summary stored after the worker succeeded")
	}

	// Second pass: the refuser is excluded before any call.
	addTurns(t, mgr, "second", 5)
	if err := mgr.Compact(ctx); err != nil {
		t.Fatalf("second Compact: %v", err)
	}
	if len(rec.excludes) != 3 || !slices.Equal(rec.excludes[2], []string{"m-refuser"}) {
		t.Errorf("second pass Exclude lists = %v, want one request excluding m-refuser", rec.excludes[2:])
	}
	if refuser.callCount != 1 {
		t.Errorf("refuser called %d times across two passes, want 1", refuser.callCount)
	}
}

// TestE2E_DefaultRefusalClassifierTreatsMarkerAsError is the off path: with
// the built-in classifier the same marker is an invalid reply, excluded only
// for the rest of that pass, and the model is asked again next time.
func TestE2E_DefaultRefusalClassifierTreatsMarkerAsError(t *testing.T) {
	ctx := context.Background()
	mgr, _, rec, refuser := refusalFixture(t, nil)

	addTurns(t, mgr, "first", 5)
	if err := mgr.Compact(ctx); err != nil {
		t.Fatalf("first Compact: %v", err)
	}
	if len(rec.excludes) != 2 || !slices.Equal(rec.excludes[1], []string{"m-refuser"}) {
		t.Fatalf("first pass Exclude lists = %v, want [] then [m-refuser]", rec.excludes)
	}
	if got := mgr.LastCompactionReport().Attempts[0]; got.Status != "error" || !strings.HasPrefix(got.Detail, "invalid JSON response") {
		t.Errorf("attempt[0] = %+v, want an invalid-JSON error", got)
	}

	addTurns(t, mgr, "second", 5)
	if err := mgr.Compact(ctx); err != nil {
		t.Fatalf("second Compact: %v", err)
	}
	if len(rec.excludes) != 4 || len(rec.excludes[2]) != 0 {
		t.Errorf("second pass Exclude lists = %v, want [] first (nothing remembered)", rec.excludes[2:])
	}
	if refuser.callCount != 2 {
		t.Errorf("refuser called %d times across two passes, want 2", refuser.callCount)
	}
}

// TestE2E_FailureDumpOnlyForFailingPass gives the manager a dump sink that
// writes files, runs one succeeding and one failing compaction, and checks
// files land only for the failing one, carrying the request and reply.
func TestE2E_FailureDumpOnlyForFailingPass(t *testing.T) {
	dir := t.TempDir()
	dumpDir := filepath.Join(dir, "dumps")
	if err := os.MkdirAll(dumpDir, 0o755); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	store := e2eStore(t, dir)
	defer store.Close()
	sum := &e2eSummarizer{model: "e2e-model"}
	dumps := 0
	mgr := New(e2eKey, store,
		WithArchiveDir(dir),
		WithModelCaller(sum),
		WithContextWindow(3000),
		WithOverheadTokens(0),
		WithRetainTokenPercent(5),
		WithRetainMinMessages(2),
		WithMessageThreshold(0),
		WithFailureDump(func(kind string, meta map[string]any, input, output string) error {
			dumps++
			body, err := json.Marshal(map[string]any{"meta": meta, "input": input, "output": output})
			if err != nil {
				return err
			}
			return os.WriteFile(filepath.Join(dumpDir, fmt.Sprintf("%s-%02d.json", kind, dumps)), body, 0o600)
		}),
	).(*Manager)
	defer mgr.Close(ctx)

	addTurns(t, mgr, "good", 5)
	if err := mgr.Compact(ctx); err != nil {
		t.Fatalf("succeeding Compact: %v", err)
	}
	if files, _ := os.ReadDir(dumpDir); len(files) != 0 {
		t.Fatalf("%d dump files after a succeeding pass, want 0", len(files))
	}

	sum.fail = true
	addTurns(t, mgr, "bad", 5)
	err := mgr.Compact(ctx)
	if !errors.Is(err, ErrCompressionFailed) {
		t.Fatalf("failing Compact returned %v, want ErrCompressionFailed", err)
	}
	files, _ := os.ReadDir(dumpDir)
	if len(files) == 0 {
		t.Fatal("no dump files after a failing pass")
	}
	if got := mgr.LastCompactionReport().Outcome; got != "failed" {
		t.Errorf("report outcome = %q, want failed", got)
	}
	for _, f := range files {
		if !strings.HasPrefix(f.Name(), "compress_fail-") {
			t.Errorf("dump file %q is not a compress_fail dump", f.Name())
		}
	}
	raw, err := os.ReadFile(filepath.Join(dumpDir, files[0].Name()))
	if err != nil {
		t.Fatal(err)
	}
	var dump struct {
		Meta   map[string]any `json:"meta"`
		Input  string         `json:"input"`
		Output string         `json:"output"`
	}
	if err := json.Unmarshal(raw, &dump); err != nil {
		t.Fatalf("dump is not JSON: %v", err)
	}
	if dump.Meta["model"] != "e2e-model" || dump.Meta["status"] != "error" || dump.Meta["session"] != e2eKey {
		t.Errorf("dump meta = %v", dump.Meta)
	}
	var req []spawnllm.Message
	if err := json.Unmarshal([]byte(dump.Input), &req); err != nil || len(req) != 2 || req[0].Role != "system" || req[1].Role != "user" {
		t.Errorf("dump input is not the system+user request: %v (%d messages)", err, len(req))
	}
	if !strings.Contains(req[1].Content, "bad 1: user") {
		t.Errorf("dump request does not carry the failing pass's history")
	}
	var reply string
	if err := json.Unmarshal([]byte(dump.Output), &reply); err != nil || reply != "I have nothing structured to say." {
		t.Errorf("dump output = %q (%v), want the raw reply", dump.Output, err)
	}
	// The window is untouched by a failed normal pass.
	if n := len(store.GetHistory(e2eKey)); n < 10 {
		t.Errorf("window has %d rows after a failed pass; the bad turns should still be live", n)
	}
}

// TestE2E_HostSettingsSurviveCompactionCloseAndReset pins the ownership rule
// for the shared session_state record: the manager rewrites only the
// compaction fields it owns, so the host's per-session settings that live in
// the same record (active model index, reasoning and tool-activity toggles)
// survive a compaction, a Close and a Reset, and the store's noise-aware
// meaningful count is what a compaction and a Close leave behind.
func TestE2E_HostSettingsSurviveCompactionCloseAndReset(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	sum := &e2eSummarizer{model: "e2e-model"}
	s := runE2EConversation(t, dir, sum)
	defer s.store.Close()

	st, err := s.store.GetCompactionState(e2eKey)
	if err != nil {
		t.Fatalf("GetCompactionState: %v", err)
	}
	st.ActiveModelIndex = 2
	st.ExposeReasoning = true
	st.ShowToolActivity = true
	if err := s.store.SetCompactionState(e2eKey, st); err != nil {
		t.Fatalf("SetCompactionState: %v", err)
	}
	countBefore := st.MeaningfulCount

	check := func(stage string, wantCount int) {
		t.Helper()
		got, err := s.store.GetCompactionState(e2eKey)
		if err != nil {
			t.Fatalf("%s: GetCompactionState: %v", stage, err)
		}
		if got.ActiveModelIndex != 2 || !got.ExposeReasoning || !got.ShowToolActivity {
			t.Errorf("%s: host settings clobbered: model=%d reasoning=%v tools=%v",
				stage, got.ActiveModelIndex, got.ExposeReasoning, got.ShowToolActivity)
		}
		if got.MeaningfulCount != wantCount {
			t.Errorf("%s: meaningful_count = %d, want %d", stage, got.MeaningfulCount, wantCount)
		}
	}

	if err := s.mgr.Compact(ctx); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	check("after compaction", countBefore)

	if err := s.mgr.Close(ctx); err != nil {
		t.Fatalf("Close: %v", err)
	}
	check("after Close", countBefore)

	reopened := New(e2eKey, s.store, e2eOpts(dir, sum)...).(*Manager)
	if err := reopened.Reset(ctx); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	check("after Reset", 0)
	if got := reopened.Stats().MeaningfulMessages; got != 0 {
		t.Errorf("Stats().MeaningfulMessages after Reset = %d, want 0", got)
	}
}

// TestE2E_StaleSummaryIsNotCheckpointedAgain runs a safety-net pass in which
// every model fails after an earlier compaction has produced a summary. The
// pass falls back to the stale summary and drops groups instead; the stored
// summary, its provenance and the checkpoint log must all be exactly what
// the successful pass left. Regression: the fallback used to append the stale
// summary to the checkpoint log again on every such pass.
func TestE2E_StaleSummaryIsNotCheckpointedAgain(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	seedParallelGroupHistory(t, dir)

	sum := &e2eSummarizer{model: "e2e-model"}
	store := e2eStore(t, dir)
	defer store.Close()
	mgr := New(e2eKey, store, tightOpts(dir, sum)...).(*Manager)
	defer mgr.Close(ctx)

	if _, err := mgr.Assemble(ctx, e2eRequest()); err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	db := e2eDB(t, dir)
	if n := queryInt(t, db, "SELECT count(*) FROM summaries"); n != 1 {
		t.Fatalf("summaries rows after the successful pass = %d, want 1", n)
	}
	summaryBefore := store.GetSummary(e2eKey)
	stateBefore, err := store.GetCompactionState(e2eKey)
	if err != nil {
		t.Fatalf("GetCompactionState: %v", err)
	}
	if summaryBefore == "" || stateBefore.SummaryModel != "e2e-model" {
		t.Fatalf("no summary recorded by the successful pass: summary=%q model=%q", summaryBefore, stateBefore.SummaryModel)
	}

	// Every model fails from here on. Push the window past the safety line
	// with two 600-token messages and a small one after them, so the drop
	// path can get back under the line by dropping the oldest of the three
	// (retainMinMessages keeps the last two).
	sum.fail = true
	requestsBefore := len(sum.requests)
	if _, err := mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: strings.Repeat("q", 2400)}); err != nil {
		t.Fatalf("AddUserMessage: %v", err)
	}
	if _, err := mgr.AddAssistantMessage(ctx, spawnllm.Message{Role: "assistant", Content: strings.Repeat("a", 2400)}); err != nil {
		t.Fatalf("AddAssistantMessage: %v", err)
	}
	if _, err := mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: pad("and now?", 40)}); err != nil {
		t.Fatalf("AddUserMessage: %v", err)
	}
	// The adds themselves trigger passes (the last one runs the safety net
	// and gets under the line by dropping); Assemble then finds nothing left
	// to do. Either way every pass asked the models and every model failed.
	if _, err := mgr.Assemble(ctx, e2eRequest()); err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if len(sum.requests) == requestsBefore {
		t.Fatal("no compaction pass asked the models")
	}
	window := store.GetHistory(e2eKey)
	if len(window) >= 6 || mgr.contextPercent(window) >= 80 {
		t.Fatalf("drop path did not run: %d messages at %.0f%%", len(window), mgr.contextPercent(window))
	}
	if n := queryInt(t, db, "SELECT count(*) FROM summaries"); n != 1 {
		t.Errorf("summaries rows after the failed passes = %d, want still 1", n)
	}
	if got := store.GetSummary(e2eKey); got != summaryBefore {
		t.Errorf("stored summary changed by the failed pass")
	}
	stateAfter, err := store.GetCompactionState(e2eKey)
	if err != nil {
		t.Fatalf("GetCompactionState: %v", err)
	}
	if stateAfter.SummaryModel != stateBefore.SummaryModel || !stateAfter.SummaryGeneratedAt.Equal(stateBefore.SummaryGeneratedAt) {
		t.Errorf("summary provenance changed by the failed pass: %q %v -> %q %v",
			stateBefore.SummaryModel, stateBefore.SummaryGeneratedAt, stateAfter.SummaryModel, stateAfter.SummaryGeneratedAt)
	}
}
