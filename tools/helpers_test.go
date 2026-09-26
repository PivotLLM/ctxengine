// ctxengine
// License: MIT

package tools

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/PivotLLM/spawnllm"
	"github.com/PivotLLM/toolspec"

	"github.com/PivotLLM/ctxengine/memory"
)

// handler returns the bound handler for the named tool from Definitions(h).
func handler(t *testing.T, h Host, name string) toolspec.ToolHandler {
	t.Helper()
	for _, def := range Definitions(h) {
		if def.Name == name {
			return def.Handler
		}
	}
	t.Fatalf("no tool named %q", name)
	return nil
}

// run invokes the named tool with the given session key and args.
func run(t *testing.T, h Host, name, session string, args map[string]any) *toolspec.Result {
	t.Helper()
	res, err := handler(t, h, name)(&toolspec.ToolCall{
		Ctx:     context.Background(),
		Args:    args,
		Session: session,
	})
	if err != nil {
		t.Fatalf("%s: unexpected handler error: %v", name, err)
	}
	if res == nil {
		t.Fatalf("%s: nil result", name)
	}
	return res
}

// archiveHost is a Host whose archive-backed tools read from dir.
func archiveHost(dir string) Host { return Host{SessionsDir: dir} }

// writeArchive creates a .archive.db file in dir and populates it with msgs.
func writeArchive(t *testing.T, dir, sessionKey string, msgs []memory.StoredMessage) string {
	t.Helper()
	filename := memory.SanitizeSessionKey(sessionKey) + ".archive.db"
	path := filepath.Join(dir, filename)
	a, err := memory.Open(path)
	if err != nil {
		t.Fatalf("writeArchive Open: %v", err)
	}
	defer func() { noErr(t, a.Close()) }()
	now := time.Now()
	for _, m := range msgs {
		if err := a.Append(m.Seq, m.Message, now); err != nil {
			t.Fatalf("writeArchive Append seq=%d: %v", m.Seq, err)
		}
	}
	return path
}

func archiveMsg(seq int64, role, content string) memory.StoredMessage {
	return memory.StoredMessage{
		Seq: seq,
		Message: spawnllm.Message{
			Role:    role,
			Content: content,
		},
	}
}
