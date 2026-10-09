/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/PivotLLM/ctxengine/memory"
)

func permOf(t *testing.T, path string) os.FileMode {
	t.Helper()
	info, err := os.Lstat(path)
	if err != nil {
		t.Fatalf("lstat %s: %v", path, err)
	}
	return info.Mode().Perm()
}

// TestManager_ArchivePermissions verifies that the permission options reach
// the archive database and the directory created for it.
func TestManager_ArchivePermissions(t *testing.T) {
	tests := []struct {
		name     string
		opts     []Option
		wantDir  os.FileMode
		wantFile os.FileMode
	}{
		{name: "defaults", wantDir: DefaultFolderPermissions, wantFile: DefaultFilePermissions},
		{
			name:     "custom",
			opts:     []Option{WithFolderPermissions(0o750), WithFilePermissions(0o640)},
			wantDir:  0o750,
			wantFile: 0o640,
		},
		{
			name:     "umask overridden",
			opts:     []Option{WithFolderPermissions(0o777), WithFilePermissions(0o666)},
			wantDir:  0o777,
			wantFile: 0o666,
		},
		{
			name:     "zero keeps defaults",
			opts:     []Option{WithFolderPermissions(0), WithFilePermissions(0)},
			wantDir:  DefaultFolderPermissions,
			wantFile: DefaultFilePermissions,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const key = "alice"
			dir := filepath.Join(t.TempDir(), "archive")
			mgr := newResetManager(t, newResetStore(key), key, append([]Option{WithArchiveDir(dir)}, tt.opts...)...)
			t.Cleanup(func() { noErr(t, mgr.Close(context.Background())) })
			if mgr.getOrOpenArchive() == nil {
				t.Fatal("archive did not open")
			}
			if got := permOf(t, dir); got != tt.wantDir {
				t.Errorf("archive directory mode = %#o; want %#o", got, tt.wantDir)
			}
			if got := permOf(t, memory.ArchivePath(dir, key)); got != tt.wantFile {
				t.Errorf("archive database mode = %#o; want %#o", got, tt.wantFile)
			}
		})
	}
}

// TestCompactionDebugCapture_Permissions verifies the mode of compact.jsonl
// when it is created and that an existing file is changed to it and appended
// to.
func TestCompactionDebugCapture_Permissions(t *testing.T) {
	tests := []struct {
		name     string
		opts     []Option
		existing os.FileMode // 0: no compact.jsonl before the pass
		want     os.FileMode
	}{
		{name: "created at default", want: DefaultFilePermissions},
		{name: "created at custom", opts: []Option{WithFilePermissions(0o640)}, want: 0o640},
		// A normal umask strips group/other write; the chmod must restore it.
		{name: "umask overridden", opts: []Option{WithFilePermissions(0o666)}, want: 0o666},
		{name: "existing tightened to default", existing: 0o644, want: DefaultFilePermissions},
		{name: "existing tightened to custom", existing: 0o666, opts: []Option{WithFilePermissions(0o640)}, want: 0o640},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "compact.jsonl")
			const prior = `{"prior":true}` + "\n"
			if tt.existing != 0 {
				if err := os.WriteFile(path, []byte(prior), 0o600); err != nil {
					t.Fatal(err)
				}
				if err := os.Chmod(path, tt.existing); err != nil {
					t.Fatal(err)
				}
			}
			store := &compressTestStore{history: makeConversation(10, 200)}
			llm := &mockLLM{model: "claude-haiku-4-5", responses: []string{validSummaryJSON("goal")}}
			opts := append([]Option{WithCompressionProfileDir(dir), WithCompactDebug(true)}, tt.opts...)
			mgr := newCompressManager(t, store, []*mockLLM{llm}, opts...)
			mgr.msgCount = len(store.history)

			if err := mgr.doCompress(context.Background(), false); err != nil {
				t.Fatalf("doCompress: %v", err)
			}
			if got := permOf(t, path); got != tt.want {
				t.Errorf("compact.jsonl mode = %#o; want %#o", got, tt.want)
			}
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if tt.existing != 0 && !strings.HasPrefix(string(data), prior) {
				t.Error("existing compact.jsonl content was not kept")
			}
			if len(data) <= len(prior) {
				t.Error("no debug record was written")
			}
		})
	}
}
