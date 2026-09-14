// ctxengine
// License: MIT

package ctxengine

import "github.com/PivotLLM/ctxengine/memory"

// CompactionStateStore is an optional interface that session store backends
// may implement to persist compression state across process restarts.
// Manager checks for this interface via type assertion; backends that do not
// implement it (e.g. in-memory SessionManager) use zero-state initialization.
//
// CompactionState is defined in memory so that implementing backends in
// session can satisfy this interface without importing llmcontext
// (which would create a circular import). Go's structural typing means
// JSONLBackend satisfies CompactionStateStore without an explicit declaration.
type CompactionStateStore interface {
	GetCompactionState(sessionKey string) (memory.CompactionState, error)
	SetCompactionState(sessionKey string, state memory.CompactionState) error
}
