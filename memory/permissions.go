/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package memory

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/PivotLLM/ctxengine/internal/iox"
)

// Default modes for the directories and files ctxengine creates: owner-only.
const (
	DefaultFolderPermissions os.FileMode = 0o700
	DefaultFilePermissions   os.FileMode = 0o600
)

// OpenOption configures Open.
type OpenOption func(*openConfig)

type openConfig struct {
	folderPerm os.FileMode
	filePerm   os.FileMode
}

// WithFolderPermissions sets the mode of the parent directories Open creates
// (default DefaultFolderPermissions). A directory that already exists is left
// as it is. Only the permission bits are used; 0 keeps the default.
func WithFolderPermissions(perm os.FileMode) OpenOption {
	return func(c *openConfig) {
		if perm &= os.ModePerm; perm != 0 {
			c.folderPerm = perm
		}
	}
}

// WithFilePermissions sets the mode of the database file and its -wal and
// -shm side files (default DefaultFilePermissions). Existing files whose mode
// differs are changed to it. Only the permission bits are used; 0 keeps the
// default.
func WithFilePermissions(perm os.FileMode) OpenOption {
	return func(c *openConfig) {
		if perm &= os.ModePerm; perm != 0 {
			c.filePerm = perm
		}
	}
}

// prepareArchiveFiles creates the database's parent directory and an empty
// database file at the configured modes, and tightens the modes of an
// existing database and its side files. The modes are set with chmod so the
// umask does not alter them. A failure to create returns an error; a failure
// to change the mode of an existing file is logged and ignored.
func prepareArchiveFiles(path string, cfg openConfig) error {
	if err := iox.EnsureDir(filepath.Dir(path), cfg.folderPerm); err != nil {
		return err
	}
	if err := ensureFile(path, cfg.filePerm); err != nil {
		return err
	}
	iox.SetFileMode("memory", path+"-wal", cfg.filePerm)
	iox.SetFileMode("memory", path+"-shm", cfg.filePerm)
	return nil
}

// ensureFile creates path empty at perm if it does not exist (SQLite treats an
// empty file as a new database), or tightens its mode if it does.
func ensureFile(path string, perm os.FileMode) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, perm) //nolint:gosec // G304: the archive path is built by ArchivePath from the host's directory
	if errors.Is(err, fs.ErrExist) {
		iox.SetFileMode("memory", path, perm)
		return nil
	}
	if err != nil {
		return fmt.Errorf("create database file: %w", err)
	}
	// Set the mode on the descriptor of the file just created, so the umask
	// does not alter it.
	chmodErr := f.Chmod(perm)
	closeErr := f.Close()
	if chmodErr != nil {
		return fmt.Errorf("set database file mode: %w", chmodErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close database file: %w", closeErr)
	}
	return nil
}
