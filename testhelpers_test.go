// ctxengine
// License: MIT

package ctxengine

import "testing"

// asManager returns the *Manager behind a ContextManager from New, failing the
// test if New returned some other implementation.
func asManager(t *testing.T, cm ContextManager) *Manager {
	t.Helper()
	m, ok := cm.(*Manager)
	if !ok {
		t.Fatalf("New returned %T, want *Manager", cm)
	}
	return m
}

// noErr fails the test when err is not nil. It must be called from the test's
// own goroutine.
func noErr(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
