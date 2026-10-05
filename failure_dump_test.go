/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine/logger"
)

// captureBackend collects the engine's log events for assertions.
type captureBackend struct {
	mu     sync.Mutex
	events []string
}

func (c *captureBackend) Log(level, component, message string, fields map[string]any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, fmt.Sprintf("%s %s %s %v", level, component, message, fields))
}

func (c *captureBackend) String() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return strings.Join(c.events, "\n")
}

// TestCompactionRecorder_LogsOutcome verifies every attempt logs a per-model
// success/failure line (independent of debug capture), naming the model,
// through the injectable logging seam.
func TestCompactionRecorder_LogsOutcome(t *testing.T) {
	capture := &captureBackend{}
	logger.SetBackend(capture)
	defer logger.SetBackend(nil)

	rec := &compactionRecorder{sessionKey: "sess"}
	rec.record("openai/gpt-5.4", "error", "invalid JSON response", time.Second, nil, "")
	rec.record("deepseek/deepseek-v4-pro", "ok", "", time.Second, nil, "{}")

	out := capture.String()
	if !strings.Contains(out, "compression model failed") || !strings.Contains(out, "openai/gpt-5.4") {
		t.Errorf("missing failure outcome log for gpt-5.4:\n%s", out)
	}
	if !strings.Contains(out, "compression model succeeded") || !strings.Contains(out, "deepseek/deepseek-v4-pro") {
		t.Errorf("missing success outcome log for deepseek:\n%s", out)
	}
}

// dumpSink is a FailureDumpFunc that records what it was handed.
type dumpSink struct {
	kinds   []string
	metas   []map[string]any
	inputs  []string
	outputs []string
}

func (d *dumpSink) write(kind string, meta map[string]any, input, output string) error {
	d.kinds = append(d.kinds, kind)
	d.metas = append(d.metas, meta)
	d.inputs = append(d.inputs, input)
	d.outputs = append(d.outputs, output)
	return nil
}

// TestCompactionRecorder_DumpsFailuresOnly verifies that with a failure-dump
// sink set, a failed attempt hands over a compress_fail dump (request + raw
// response as JSON) while a successful attempt hands over nothing.
func TestCompactionRecorder_DumpsFailuresOnly(t *testing.T) {
	sink := &dumpSink{}
	rec := &compactionRecorder{sessionKey: "sess-1", failureDump: sink.write}

	req := []spawnllm.Message{{Role: "user", Content: "summarize the conversation"}}
	rec.record("gpt-5.4", "error", "invalid JSON response", time.Second, req, "Sorry, I can't comply.")

	if len(sink.kinds) != 1 || sink.kinds[0] != "compress_fail" {
		t.Fatalf("expected 1 compress_fail dump, got %v", sink.kinds)
	}
	meta := sink.metas[0]
	if meta["model"] != "gpt-5.4" || meta["status"] != "error" || meta["detail"] != "invalid JSON response" || meta["session"] != "sess-1" {
		t.Errorf("dump meta = %v", meta)
	}
	var gotReq []spawnllm.Message
	if err := json.Unmarshal([]byte(sink.inputs[0]), &gotReq); err != nil || len(gotReq) != 1 || gotReq[0].Content != req[0].Content {
		t.Errorf("dump input is not the JSON request: %q (%v)", sink.inputs[0], err)
	}
	var gotResp string
	if err := json.Unmarshal([]byte(sink.outputs[0]), &gotResp); err != nil || gotResp != "Sorry, I can't comply." {
		t.Errorf("dump output is not the JSON-encoded raw response: %q (%v)", sink.outputs[0], err)
	}

	// A successful attempt must not add a dump.
	rec.record("gpt-5.4", "ok", "", time.Second, req, `{"version":2}`)
	if got := len(sink.kinds); got != 1 {
		t.Errorf("ok attempt should not dump; dump count = %d, want 1", got)
	}
}

// TestCompactionRecorder_NoDumpWhenDisabled verifies that with no sink set,
// failures are recorded without any dump attempt.
func TestCompactionRecorder_NoDumpWhenDisabled(t *testing.T) {
	rec := &compactionRecorder{sessionKey: "sess-2"} // failureDump nil
	rec.record("m", "error", "boom", time.Second, nil, "x")
	if len(rec.attempts) != 1 {
		t.Fatalf("attempt not recorded: %+v", rec.attempts)
	}
}
