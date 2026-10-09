/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

// Package iox holds small I/O helpers shared by ctxengine's packages.
package iox

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/PivotLLM/ctxengine/logger"
)

// CloseQuietly closes c and logs at debug level, under component, if Close
// reports an error. It is for handles whose close error carries nothing the
// caller can act on: read-only database connections, row iterators, files
// opened for reading, and handles being discarded on an error path. A handle
// that was written to must have its Close error checked by the caller instead,
// because that is where a flush failure surfaces.
func CloseQuietly(component string, c io.Closer) {
	if err := c.Close(); err != nil {
		logger.DebugCF(component, "close failed", map[string]any{"error": err.Error()})
	}
}

// EnsureDir creates dir and any missing parents and sets each directory it
// created to perm with chmod, so the umask does not alter it. A directory that
// already exists is left as it is: it may be the host's. An existing dir that
// is not a directory is an error.
func EnsureDir(dir string, perm os.FileMode) error {
	var missing []string
	for d := dir; ; d = filepath.Dir(d) {
		info, err := os.Stat(d)
		if err == nil {
			if d == dir && !info.IsDir() {
				return fmt.Errorf("%s is not a directory", dir)
			}
			break
		}
		if !errors.Is(err, fs.ErrNotExist) {
			return fmt.Errorf("stat directory: %w", err)
		}
		missing = append(missing, d)
		if parent := filepath.Dir(d); parent == d {
			break
		}
	}
	if len(missing) == 0 {
		return nil
	}
	if err := os.MkdirAll(dir, perm); err != nil {
		return fmt.Errorf("create directory: %w", err)
	}
	for _, d := range missing {
		if err := os.Chmod(d, perm); err != nil {
			return fmt.Errorf("set directory mode: %w", err)
		}
	}
	return nil
}

// SetFileMode sets the existing regular file at path to perm when its
// permission bits differ, whether that tightens or loosens them. A missing
// file is ignored. A symbolic link is not followed, since its target is not
// ctxengine's to change; it is left alone and logged at debug level. Other
// failures are logged as warnings under component and otherwise ignored.
func SetFileMode(component, path string, perm os.FileMode) {
	info, err := os.Lstat(path)
	if errors.Is(err, fs.ErrNotExist) {
		return
	}
	if err != nil {
		logger.WarnCF(component, "file mode check failed",
			map[string]any{"path": path, "error": err.Error()})
		return
	}
	if info.Mode()&os.ModeSymlink != 0 {
		logger.DebugCF(component, "file is a symbolic link; mode not changed",
			map[string]any{"path": path})
		return
	}
	if !info.Mode().IsRegular() || info.Mode().Perm() == perm {
		return
	}
	if err := os.Chmod(path, perm); err != nil {
		logger.WarnCF(component, "file mode change failed",
			map[string]any{"path": path, "mode": fmt.Sprintf("%#o", perm), "error": err.Error()})
	}
}
