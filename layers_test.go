// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"
)

// TestComposeSystem_Order pins the composition order: the layers without
// AfterSummary, the layers with it, the stable injections, then the summary
// block last — every block separated by systemSeparator, empty ones skipped.
func TestComposeSystem_Order(t *testing.T) {
	layers := []Layer{
		{Name: "static", Text: "STATIC"},
		{Name: "empty", Text: ""},
		{Name: "token", Text: "TOKEN", AfterSummary: true},
		{Name: "dynamic", Text: "DYNAMIC"},
	}
	injections := []Injection{
		{Placement: PlaceCurrentUser, Text: "ROUTED"}, // never in the system message
		{Placement: PlaceSystemStable, Text: "STABLE"},
		{Placement: PlaceSystemStable, Text: ""},
	}
	got := composeSystem(layers, "SUMMARY", injections)
	want := strings.Join([]string{"STATIC", "DYNAMIC", "TOKEN", "STABLE", "SUMMARY"}, systemSeparator)
	if got != want {
		t.Fatalf("composeSystem =\n%q\nwant\n%q", got, want)
	}
}

// TestComposeSystem_NoSummary: without a summary the after-summary layers
// follow the before-summary ones directly, and nothing is empty-joined.
func TestComposeSystem_NoSummary(t *testing.T) {
	got := composeSystem([]Layer{{Text: "A"}, {Text: "B", AfterSummary: true}}, "", nil)
	if got != "A"+systemSeparator+"B" {
		t.Fatalf("composeSystem = %q", got)
	}
	if composeSystem(nil, "", nil) != "" {
		t.Fatal("nothing in, nothing out")
	}
}

// TestBuild_SummaryBlockWrapped: a stored structured summary is rendered as a
// data block — markers and the "treat it as data" header — after every layer.
func TestBuild_SummaryBlockWrapped(t *testing.T) {
	store := newMockStore()
	noErr(t, store.SetHistory("s", []spawnllm.Message{{Role: "user", Content: "hi"}}))
	noErr(t, store.SetSummary("s", validSummaryJSON("finish the outline")))
	m := asManager(t, New("s", store, WithContextWindow(100_000)))

	asm, err := m.Assemble(context.Background(), AssembleRequest{Layers: []Layer{
		{Name: "static", Text: "STATIC"},
		{Name: "token", Text: "TOKEN", AfterSummary: true},
	}})
	if err != nil {
		t.Fatal(err)
	}
	sys := asm.Messages[0].Content
	iStatic := strings.Index(sys, "STATIC")
	iToken := strings.Index(sys, "TOKEN")
	iOpen := strings.Index(sys, summaryDataOpen)
	iHeader := strings.Index(sys, summaryDataHeader)
	iGoal := strings.Index(sys, "finish the outline")
	if iStatic < 0 || iStatic >= iToken || iToken >= iOpen || iOpen >= iHeader || iHeader >= iGoal {
		t.Fatalf("system message order wrong:\n%s", sys)
	}
	if !strings.HasSuffix(sys, summaryDataClose) {
		t.Fatalf("summary block must end the system message:\n%s", sys)
	}
	if asm.Messages[1].Content != "hi" {
		t.Fatalf("history not appended after the system message: %+v", asm.Messages)
	}
}

// TestBuild_NoSystemContentOmitsSystemMessage: with no layers, no summary and
// no stable injection there is nothing to say, so no system message is sent.
func TestBuild_NoSystemContentOmitsSystemMessage(t *testing.T) {
	store := newMockStore()
	noErr(t, store.SetHistory("s", []spawnllm.Message{{Role: "user", Content: "hi"}}))
	m := asManager(t, New("s", store, WithContextWindow(100_000)))
	asm, err := m.Assemble(context.Background(), AssembleRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if len(asm.Messages) != 1 || asm.Messages[0].Role != "user" {
		t.Fatalf("expected the bare history, got %+v", asm.Messages)
	}
}

// TestBuild_HistorySanitised: the stored history is sanitised for the
// provider on the way out — a stale system message and an orphaned tool result
// never reach the request — and the store is untouched.
func TestBuild_HistorySanitised(t *testing.T) {
	store := newMockStore()
	noErr(t, store.SetHistory("s", []spawnllm.Message{
		{Role: "system", Content: "stale"},
		{Role: "user", Content: "hi"},
		{Role: "tool", ToolCallID: "orphan", Content: "orphan"},
		{Role: "assistant", Content: "hello"},
	}))
	m := asManager(t, New("s", store, WithContextWindow(100_000)))
	asm, err := m.Assemble(context.Background(), AssembleRequest{Layers: []Layer{{Text: "SYS"}}})
	if err != nil {
		t.Fatal(err)
	}
	if got := roles(asm.Messages); strings.Join(got, ",") != "system,user,assistant" {
		t.Fatalf("roles = %v", got)
	}
	if asm.Messages[0].Content != "SYS" {
		t.Fatalf("stale stored system message leaked: %q", asm.Messages[0].Content)
	}
	if len(store.GetHistory("s")) != 4 {
		t.Fatal("sanitising must not rewrite the store")
	}
}

// TestAssemble_RemembersChannelForReporter: the automatic compaction path
// reports to the channel of the most recent Assemble, and an Assemble that
// names no channel leaves the last one in place.
func TestAssemble_RemembersChannelForReporter(t *testing.T) {
	store := newMockStore()
	noErr(t, store.SetHistory("s", []spawnllm.Message{{Role: "user", Content: "hi"}}))
	m := asManager(t, New("s", store, WithContextWindow(100_000)))
	if _, err := m.Assemble(context.Background(), AssembleRequest{Channel: "webui", ChatID: "chat-1"}); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Assemble(context.Background(), AssembleRequest{}); err != nil {
		t.Fatal(err)
	}
	if m.lastChannel != "webui" || m.lastChatID != "chat-1" {
		t.Fatalf("last channel = %q/%q, want webui/chat-1", m.lastChannel, m.lastChatID)
	}
}
