// ctxengine
// License: MIT

package tools

// End-to-end: the session tools read the archive the engine itself wrote.
// A Manager over the real session.SQLiteStore drives a conversation with a
// tool result, compacts it, and the archive-backed tools are then pointed at
// the same directory and asked for what the compaction cited.

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine"
	"github.com/PivotLLM/ctxengine/session"
)

const (
	e2eToolsKey    = "agent:bob:main"
	e2eToolsPhrase = "the quick zebra manifest lists seven crates"
)

// citingSummarizer answers every summarization request with a summary whose
// goal cites the first and last seq in the prompt and whose key moment cites
// the seq the test names (the tool result), so the checkpoint the engine
// writes points at archived rows the tools can be asked for.
type citingSummarizer struct {
	cite     int64
	requests []ctxengine.ModelRequest
}

var toolsPromptSeqRE = regexp.MustCompile(`(?m)^\[#(\d+)\] \[`)

func (s *citingSummarizer) Complete(_ context.Context, req ctxengine.ModelRequest) (ctxengine.ModelReply, error) {
	s.requests = append(s.requests, req)
	var seqs []int64
	for _, m := range toolsPromptSeqRE.FindAllStringSubmatch(req.User, -1) {
		n, err := strconv.ParseInt(m[1], 10, 64)
		if err != nil {
			return ctxengine.ModelReply{}, fmt.Errorf("prompt seq %q: %w", m[1], err)
		}
		seqs = append(seqs, n)
	}
	if len(seqs) == 0 {
		return ctxengine.ModelReply{Content: "{}", Model: "e2e-model"}, nil
	}
	first, last := seqs[0], seqs[len(seqs)-1]
	body := fmt.Sprintf(`{"version":2,`+
		`"state":{"goals":[{"text":"track the zebra manifest","refs":[{"seq_start":%d,"seq_end":%d}]}]},`+
		`"key_moments":[{"refs":[{"seq_start":%d}],"role":"tool","summary":"manifest fetched"}]}`,
		first, last, s.cite)
	return ctxengine.ModelReply{Content: body, Model: "e2e-model"}, nil
}

// TestE2E_ToolsReadTheEngineArchive compacts a real session and then answers
// session_messages, session_search, session_summary_list and
// session_summary_get from the archive the engine wrote.
func TestE2E_ToolsReadTheEngineArchive(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	store, err := session.NewSQLiteStore(dir)
	if err != nil {
		t.Fatalf("NewSQLiteStore: %v", err)
	}
	defer func() { noErr(t, store.Close()) }()
	sum := &citingSummarizer{}
	mgr := ctxengine.New(e2eToolsKey, store,
		ctxengine.WithArchiveDir(dir),
		ctxengine.WithModelCaller(sum),
		ctxengine.WithContextWindow(3000),
		ctxengine.WithOverheadTokens(0),
		ctxengine.WithRetainTokenPercent(5),
		ctxengine.WithRetainMinMessages(2),
		ctxengine.WithMessageThreshold(0),
	)
	defer func() { noErr(t, mgr.Close(ctx)) }()

	must := func(seq int64, err error) int64 {
		t.Helper()
		if err != nil || seq <= 0 {
			t.Fatalf("Add*: seq=%d err=%v", seq, err)
		}
		return seq
	}
	pad := func(s string) string { return s + strings.Repeat(".", 240-len(s)) }
	firstSeq := must(mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: pad("turn 1: what is in the manifest? ")}))
	must(mgr.AddToolCallMessage(ctx, spawnllm.Message{Role: "assistant", ToolCalls: []spawnllm.ToolCall{{
		ID: "call-manifest", Function: &spawnllm.FunctionCall{Name: "manifest_lookup", Arguments: `{"name":"zebra"}`},
	}}}))
	toolSeq := must(mgr.AddToolResult(ctx, spawnllm.Message{Role: "tool", ToolCallID: "call-manifest", Content: e2eToolsPhrase}))
	sum.cite = toolSeq
	must(mgr.AddAssistantMessage(ctx, spawnllm.Message{Role: "assistant", Content: pad("turn 1: seven crates are listed ")}))
	for i := 2; i <= 6; i++ {
		must(mgr.AddUserMessage(ctx, spawnllm.Message{Role: "user", Content: pad(fmt.Sprintf("turn %d: user ", i))}))
		must(mgr.AddAssistantMessage(ctx, spawnllm.Message{Role: "assistant", Content: pad(fmt.Sprintf("turn %d: assistant ", i))}))
	}

	if err := mgr.Compact(ctx); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if len(sum.requests) != 1 {
		t.Fatalf("summarizer called %d times, want 1", len(sum.requests))
	}
	if !strings.Contains(sum.requests[0].User, fmt.Sprintf("[#%d] [tool]", toolSeq)) {
		t.Fatalf("the tool result #%d was not summarized:\n%s", toolSeq, sum.requests[0].User)
	}
	rendered := mgr.RenderedSummary()
	if !strings.Contains(rendered, fmt.Sprintf("[#%d] tool: manifest fetched", toolSeq)) {
		t.Fatalf("live summary does not cite the tool result:\n%s", rendered)
	}

	h := Host{SessionsDir: dir}

	// session_messages for the seq the summary cites returns the real row.
	res := run(t, h, "messages", e2eToolsKey, map[string]any{"seq": toolSeq})
	if res.IsError {
		t.Fatalf("messages: %s", res.ForLLM)
	}
	var entries []struct {
		Seq     int64  `json:"seq"`
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal([]byte(res.ForLLM), &entries); err != nil {
		t.Fatalf("messages result is not JSON: %v: %s", err, res.ForLLM)
	}
	if len(entries) != 1 || entries[0].Seq != toolSeq || entries[0].Role != "tool" || entries[0].Content != e2eToolsPhrase {
		t.Errorf("messages(seq=%d) = %+v", toolSeq, entries)
	}

	// The cited range resolves too, and the tool-call turn carries its output.
	res = run(t, h, "messages", e2eToolsKey, map[string]any{"seq_start": firstSeq, "seq_end": toolSeq})
	if res.IsError {
		t.Fatalf("messages range: %s", res.ForLLM)
	}
	var ranged []struct {
		Seq       int64 `json:"seq"`
		ToolCalls []struct {
			Name   string `json:"name"`
			Output string `json:"output"`
			Status string `json:"status"`
			Seq    int64  `json:"result_seq"`
		} `json:"tool_calls"`
	}
	if err := json.Unmarshal([]byte(res.ForLLM), &ranged); err != nil {
		t.Fatalf("messages range result is not JSON: %v", err)
	}
	if len(ranged) != 3 || ranged[0].Seq != firstSeq || ranged[2].Seq != toolSeq {
		t.Fatalf("messages(#%d-#%d) returned %d rows: %+v", firstSeq, toolSeq, len(ranged), ranged)
	}
	if tc := ranged[1].ToolCalls; len(tc) != 1 || tc[0].Output != e2eToolsPhrase || tc[0].Status != "success" || tc[0].Seq != toolSeq {
		t.Errorf("tool-call turn = %+v, want the result joined in", ranged[1].ToolCalls)
	}

	// session_search finds the archived tool result by phrase.
	res = run(t, h, "search", e2eToolsKey, map[string]any{"query": `"zebra manifest"`, "role": "tool"})
	if res.IsError {
		t.Fatalf("search: %s", res.ForLLM)
	}
	var hits []struct {
		Seq     int64  `json:"seq"`
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal([]byte(res.ForLLM), &hits); err != nil {
		t.Fatalf("search result is not JSON: %v: %s", err, res.ForLLM)
	}
	if len(hits) != 1 || hits[0].Seq != toolSeq || hits[0].Content != e2eToolsPhrase {
		t.Errorf("search hits = %+v, want the tool result at #%d", hits, toolSeq)
	}

	// session_summary_list shows the checkpoint the compaction wrote.
	res = run(t, h, "summary_list", e2eToolsKey, nil)
	if res.IsError {
		t.Fatalf("summary_list: %s", res.ForLLM)
	}
	cited := toolsPromptSeqRE.FindAllStringSubmatch(sum.requests[0].User, -1)
	lastCited := cited[len(cited)-1][1]
	wantList := fmt.Sprintf("1 context summary checkpoint(s) (newest first):\n- id 1: covers #%d-#%s, generated ", firstSeq, lastCited)
	if !strings.HasPrefix(res.ForLLM, wantList) || !strings.HasSuffix(res.ForLLM, ", model e2e-model") {
		t.Errorf("summary_list = %q, want prefix %q and the model suffix", res.ForLLM, wantList)
	}

	// session_summary_get renders the checkpoint body with its citations.
	res = run(t, h, "summary_get", e2eToolsKey, map[string]any{"id": 1})
	if res.IsError {
		t.Fatalf("summary_get: %s", res.ForLLM)
	}
	wantHeader := fmt.Sprintf("Context summary checkpoint id 1 (covers #%d-#%s, generated ", firstSeq, lastCited)
	if !strings.HasPrefix(res.ForLLM, wantHeader) {
		t.Errorf("summary_get header = %.80q, want prefix %q", res.ForLLM, wantHeader)
	}
	for _, want := range []string{
		fmt.Sprintf("- [#%d-#%s] track the zebra manifest", firstSeq, lastCited),
		fmt.Sprintf("- [#%d] tool: manifest fetched", toolSeq),
		"## Current State",
		"## Key Moments",
	} {
		if !strings.Contains(res.ForLLM, want) {
			t.Errorf("summary_get lacks %q:\n%s", want, res.ForLLM)
		}
	}
	// The checkpoint body is the live summary the manager renders.
	if !strings.HasSuffix(res.ForLLM, rendered) {
		t.Errorf("summary_get body differs from RenderedSummary():\n%s\nwant suffix:\n%s", res.ForLLM, rendered)
	}

	// A seq past the archive is reported as unavailable, not as an error.
	res = run(t, h, "messages", e2eToolsKey, map[string]any{"seq": 999})
	if res.IsError || !strings.Contains(res.ForLLM, "not available") {
		t.Errorf("messages(seq=999) = %q (isErr=%v)", res.ForLLM, res.IsError)
	}
}
