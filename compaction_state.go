// ctxengine
// License: MIT

package ctxengine

import (
	"github.com/PivotLLM/ctxengine/memory"
	"github.com/PivotLLM/ctxengine/session"
)

// CompactionCommitter is an optional interface a session store implements
// when it can write a whole compaction result — retained window, new summary,
// its checkpoint and the compaction counters — in one transaction.
// session.SQLiteStore does; the Manager uses it when present and otherwise
// writes the parts separately, which leaves a crash window between them.
type CompactionCommitter interface {
	CommitCompaction(sessionKey string, c session.CompactionCommit) error
}

// CompactionStateStore is an optional interface that session store backends
// may implement to persist compression state across process restarts.
// Manager checks for this interface via type assertion; backends that do not
// implement it (an in-memory test store) use zero-state initialization.
//
// CompactionState is defined in memory so that implementing backends in
// session can satisfy this interface without importing the engine package
// (which would create a circular import). Go's structural typing means
// session.SQLiteStore satisfies CompactionStateStore without an explicit
// declaration.
type CompactionStateStore interface {
	GetCompactionState(sessionKey string) (memory.CompactionState, error)
	SetCompactionState(sessionKey string, state memory.CompactionState) error
}
