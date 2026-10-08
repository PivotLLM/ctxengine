/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package memory

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/PivotLLM/spawnllm"
)

// fileMode returns the permission bits of path, failing the test if it cannot
// be read.
func fileMode(t *testing.T, path string) os.FileMode {
	t.Helper()
	info, err := os.Lstat(path)
	if err != nil {
		t.Fatalf("lstat %s: %v", path, err)
	}
	return info.Mode().Perm()
}

// openAndWrite opens path with opts, appends one message so the -wal and -shm
// files exist, and closes the store on cleanup.
func openAndWrite(t *testing.T, path string, opts ...OpenOption) {
	t.Helper()
	a, err := Open(path, opts...)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { noErr(t, a.Close()) })
	if err := a.Append(1, spawnllm.Message{Role: "user", Content: "hello"}, time.Now()); err != nil {
		t.Fatalf("Append: %v", err)
	}
}

// TestOpen_FileModes verifies the modes of a new database and its side files,
// and that existing files are changed to the configured mode.
func TestOpen_FileModes(t *testing.T) {
	tests := []struct {
		name     string
		opts     []OpenOption
		existing os.FileMode // 0: no files before Open
		want     os.FileMode
	}{
		{name: "default", want: DefaultFilePermissions},
		{name: "custom", opts: []OpenOption{WithFilePermissions(0o640)}, want: 0o640},
		{name: "zero keeps default", opts: []OpenOption{WithFilePermissions(0)}, want: DefaultFilePermissions},
		// A normal umask strips group/other write; the chmod must restore it.
		{name: "umask overridden", opts: []OpenOption{WithFilePermissions(0o666)}, want: 0o666},
		{name: "existing tightened to default", existing: 0o644, want: DefaultFilePermissions},
		{name: "existing tightened to custom", existing: 0o666, opts: []OpenOption{WithFilePermissions(0o640)}, want: 0o640},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "s.archive.db")
			if tt.existing != 0 {
				// A real database with its side files, left at a loose mode.
				a, err := Open(path)
				if err != nil {
					t.Fatalf("seed Open: %v", err)
				}
				if err := a.Append(1, spawnllm.Message{Role: "user", Content: "seed"}, time.Now()); err != nil {
					t.Fatalf("seed Append: %v", err)
				}
				for _, p := range []string{path, path + "-wal", path + "-shm"} {
					if err := os.Chmod(p, tt.existing); err != nil {
						t.Fatalf("chmod %s: %v", p, err)
					}
				}
				// Open again while the seed handle still holds the side files,
				// so they exist when the second Open checks them.
				openAndWrite(t, path, tt.opts...)
				noErr(t, a.Close())
			} else {
				openAndWrite(t, path, tt.opts...)
			}
			for _, p := range []string{path, path + "-wal", path + "-shm"} {
				if got := fileMode(t, p); got != tt.want {
					t.Errorf("%s mode = %#o; want %#o", filepath.Base(p), got, tt.want)
				}
			}
		})
	}
}

// TestOpen_ParentDirectory verifies that a missing parent directory (and any
// missing ancestors) is created at the folder mode, and that an existing one
// keeps its mode.
func TestOpen_ParentDirectory(t *testing.T) {
	tests := []struct {
		name     string
		opts     []OpenOption
		existing os.FileMode // 0: the parent does not exist
		want     os.FileMode
	}{
		{name: "created at default", want: DefaultFolderPermissions},
		{name: "created at custom", opts: []OpenOption{WithFolderPermissions(0o750)}, want: 0o750},
		{name: "zero keeps default", opts: []OpenOption{WithFolderPermissions(0)}, want: DefaultFolderPermissions},
		{name: "umask overridden", opts: []OpenOption{WithFolderPermissions(0o777)}, want: 0o777},
		{name: "existing untouched", existing: 0o755, want: 0o755},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := t.TempDir()
			top := filepath.Join(root, "a")
			dir := filepath.Join(top, "b")
			if tt.existing != 0 {
				if err := os.MkdirAll(dir, tt.existing); err != nil {
					t.Fatal(err)
				}
				for _, d := range []string{top, dir} {
					if err := os.Chmod(d, tt.existing); err != nil {
						t.Fatal(err)
					}
				}
			}
			openAndWrite(t, filepath.Join(dir, "s.archive.db"), tt.opts...)
			for _, d := range []string{top, dir} {
				if got := fileMode(t, d); got != tt.want {
					t.Errorf("%s mode = %#o; want %#o", d, got, tt.want)
				}
			}
		})
	}
}

// TestOpen_SymlinkNotChanged verifies that Open does not change the mode of
// the target of a database path that is a symbolic link.
func TestOpen_SymlinkNotChanged(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target.db")
	if err := os.WriteFile(target, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(target, 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "s.archive.db")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	openAndWrite(t, link)
	if got := fileMode(t, target); got != 0o644 {
		t.Errorf("symlink target mode = %#o; want 0644 (unchanged)", got)
	}
}

// TestOpen_CreateFailureUnavailable verifies that a parent that cannot be
// created makes the archive unavailable.
func TestOpen_CreateFailureUnavailable(t *testing.T) {
	dir := t.TempDir()
	blocker := filepath.Join(dir, "file")
	if err := os.WriteFile(blocker, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	a, err := Open(filepath.Join(blocker, "sub", "s.archive.db"))
	if !errors.Is(err, ErrArchiveUnavailable) {
		t.Fatalf("Open error = %v; want ErrArchiveUnavailable", err)
	}
	noErr(t, a.Close())
}
