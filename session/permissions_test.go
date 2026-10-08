/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package session

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/PivotLLM/spawnllm"

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

// TestNewSQLiteStore_Permissions verifies the modes of the sessions directory
// and of the archive databases the store creates.
func TestNewSQLiteStore_Permissions(t *testing.T) {
	tests := []struct {
		name        string
		opts        []StoreOption
		existingDir os.FileMode // 0: the directory does not exist
		wantDir     os.FileMode
		wantFile    os.FileMode
	}{
		{
			name:     "defaults",
			wantDir:  memory.DefaultFolderPermissions,
			wantFile: memory.DefaultFilePermissions,
		},
		{
			name:     "custom",
			opts:     []StoreOption{WithFolderPermissions(0o750), WithFilePermissions(0o640)},
			wantDir:  0o750,
			wantFile: 0o640,
		},
		{
			name:     "zero keeps defaults",
			opts:     []StoreOption{WithFolderPermissions(0), WithFilePermissions(0)},
			wantDir:  memory.DefaultFolderPermissions,
			wantFile: memory.DefaultFilePermissions,
		},
		{
			// A normal umask strips group/other write; the chmod must restore it.
			name:     "umask overridden",
			opts:     []StoreOption{WithFolderPermissions(0o777), WithFilePermissions(0o666)},
			wantDir:  0o777,
			wantFile: 0o666,
		},
		{
			name:        "existing directory untouched",
			opts:        []StoreOption{WithFolderPermissions(0o700)},
			existingDir: 0o755,
			wantDir:     0o755,
			wantFile:    memory.DefaultFilePermissions,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "sessions")
			if tt.existingDir != 0 {
				if err := os.Mkdir(dir, tt.existingDir); err != nil {
					t.Fatal(err)
				}
				if err := os.Chmod(dir, tt.existingDir); err != nil {
					t.Fatal(err)
				}
			}
			s, err := NewSQLiteStore(dir, tt.opts...)
			if err != nil {
				t.Fatalf("NewSQLiteStore: %v", err)
			}
			t.Cleanup(func() { noErr(t, s.Close()) })
			if got := permOf(t, dir); got != tt.wantDir {
				t.Errorf("directory mode = %#o; want %#o", got, tt.wantDir)
			}

			mustAdd(t, s, "alice", spawnllm.Message{Role: "user", Content: "hi"})
			path := memory.ArchivePath(dir, "alice")
			for _, p := range []string{path, path + "-wal", path + "-shm"} {
				if got := permOf(t, p); got != tt.wantFile {
					t.Errorf("%s mode = %#o; want %#o", filepath.Base(p), got, tt.wantFile)
				}
			}
		})
	}
}

// TestNewSQLiteStore_NotADirectory verifies that a dir that exists but is not
// a directory is an error.
func TestNewSQLiteStore_NotADirectory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sessions")
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	s, err := NewSQLiteStore(path)
	if err == nil {
		noErr(t, s.Close())
		t.Fatal("NewSQLiteStore succeeded on a regular file; want an error")
	}
	if want := "session: " + path + " is not a directory"; err.Error() != want {
		t.Errorf("error = %q; want %q", err, want)
	}
}
