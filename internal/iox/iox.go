// ctxengine
// License: MIT

// Package iox holds small I/O helpers shared by ctxengine's packages.
package iox

import (
	"io"

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
