// ctxengine
// License: MIT

package tools

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/PivotLLM/ctxengine"
)

func TestSessionClearTool(t *testing.T) {
	var gotKey, gotMsg string
	h := Host{Clear: func(_ context.Context, sessionKey, message string) error {
		gotKey, gotMsg = sessionKey, message
		return nil
	}}

	res := run(t, h, "clear", "sess-1", map[string]any{"message": "do task B"})
	if res.IsError {
		t.Fatalf("unexpected error: %s", res.ForLLM)
	}
	if gotKey != "sess-1" || gotMsg != "do task B" {
		t.Errorf("clear called with (%q, %q), want (sess-1, do task B)", gotKey, gotMsg)
	}
	if !strings.Contains(res.ForLLM, "End your turn") {
		t.Errorf("result should tell the agent to end its turn: %q", res.ForLLM)
	}

	// Missing session key.
	if r := run(t, h, "clear", "", nil); !r.IsError {
		t.Error("expected error without a session key")
	}

	// Clear callback error is surfaced (e.g. rate-limited).
	failing := Host{Clear: func(context.Context, string, string) error {
		return errors.New("rate-limited")
	}}
	r := run(t, failing, "clear", "s", nil)
	if !r.IsError || !strings.Contains(r.ForLLM, "rate-limited") {
		t.Errorf("expected the clear error surfaced, got %q (isErr=%v)", r.ForLLM, r.IsError)
	}
}

func TestSessionCompactTool(t *testing.T) {
	var gotKey string
	h := Host{Compact: func(_ context.Context, sessionKey string) (string, string, error) {
		gotKey = sessionKey
		return "compacted 10 messages", "the summary", nil
	}}

	res := run(t, h, "compact", "sess-1", map[string]any{"message": "next: task C"})
	if res.IsError {
		t.Fatalf("unexpected error: %s", res.ForLLM)
	}
	if gotKey != "sess-1" {
		t.Errorf("compact called with %q, want sess-1", gotKey)
	}
	for _, want := range []string{"compacted 10 messages", "the summary", "next: task C"} {
		if !strings.Contains(res.ForLLM, want) {
			t.Errorf("result missing %q: %s", want, res.ForLLM)
		}
	}

	// Nothing to compress is a plain outcome, not an error.
	empty := Host{Compact: func(context.Context, string) (string, string, error) {
		return "", "", ctxengine.ErrNothingToCompress
	}}
	if r := run(t, empty, "compact", "s", nil); r.IsError || !strings.Contains(r.ForLLM, "already compact") {
		t.Errorf("expected the already-compact message, got %q (isErr=%v)", r.ForLLM, r.IsError)
	}

	// Other failures are surfaced.
	failing := Host{Compact: func(context.Context, string) (string, string, error) {
		return "", "", errors.New("model down")
	}}
	if r := run(t, failing, "compact", "s", nil); !r.IsError || !strings.Contains(r.ForLLM, "model down") {
		t.Errorf("expected the compact error surfaced, got %q (isErr=%v)", r.ForLLM, r.IsError)
	}
}

func TestSessionInfoTool(t *testing.T) {
	h := Host{SessionInfo: func(_ context.Context, sessionKey string) (*SessionInfo, error) {
		return &SessionInfo{SessionKey: sessionKey, Channel: "telegram", TotalArchived: 42}, nil
	}}

	res := run(t, h, "info", "sess-1", nil)
	if res.IsError {
		t.Fatalf("unexpected error: %s", res.ForLLM)
	}
	for _, want := range []string{`"session_key":"sess-1"`, `"channel":"telegram"`, `"total_archived":42`} {
		if !strings.Contains(res.ForLLM, want) {
			t.Errorf("result missing %s: %s", want, res.ForLLM)
		}
	}

	failing := Host{SessionInfo: func(context.Context, string) (*SessionInfo, error) {
		return nil, errors.New("no such session")
	}}
	if r := run(t, failing, "info", "s", nil); !r.IsError || !strings.Contains(r.ForLLM, "no such session") {
		t.Errorf("expected the info error surfaced, got %q (isErr=%v)", r.ForLLM, r.IsError)
	}
}

// TestDefinitions_Unavailable verifies that a zero Host (no sessions directory,
// no closures — what a deps-free catalogue enumeration produces) yields clear
// error results from every tool rather than a panic or a disk access.
func TestDefinitions_Unavailable(t *testing.T) {
	var h Host
	args := map[string]map[string]any{
		"messages":     {"seq": 1},
		"search":       {"query": "anything"},
		"summary_list": {},
		"summary_get":  {"id": 1},
		"compact":      {},
		"info":         {},
		"clear":        {},
	}
	for _, def := range Definitions(h) {
		a, ok := args[def.Name]
		if !ok {
			t.Fatalf("unexpected tool %q", def.Name)
		}
		r := run(t, h, def.Name, "sess", a)
		if !r.IsError || strings.TrimSpace(r.ForLLM) == "" {
			t.Errorf("%s on a zero Host: want a non-empty error result, got %q (isErr=%v)", def.Name, r.ForLLM, r.IsError)
		}
	}
}

// TestDefinitions_Metadata pins the published tool set: names, session
// scoping, and the default-allow values the MCP integration test relies on.
func TestDefinitions_Metadata(t *testing.T) {
	wantAllow := map[string]bool{
		"messages":     true,
		"search":       true,
		"summary_list": true,
		"summary_get":  true,
		"compact":      true,
		"info":         true,
		"clear":        false, // denied by default — opt-in only
	}
	defs := Definitions(Host{})
	if len(defs) != len(wantAllow) {
		t.Fatalf("got %d definitions, want %d", len(defs), len(wantAllow))
	}
	for _, def := range defs {
		allow, ok := wantAllow[def.Name]
		if !ok {
			t.Errorf("unexpected tool %q", def.Name)
			continue
		}
		if !def.SessionScoped {
			t.Errorf("%s: SessionScoped = false, want true", def.Name)
		}
		if def.Description == "" || def.RawSchema == nil || def.Handler == nil {
			t.Errorf("%s: description, schema and handler must all be set", def.Name)
		}
		if got := def.DefaultAllow != nil && *def.DefaultAllow; got != allow {
			t.Errorf("%s: DefaultAllow = %v, want %v", def.Name, got, allow)
		}
	}
}
