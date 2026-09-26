// ctxengine
// License: MIT

package tools

import "testing"

// noErr fails the test when err is not nil. It must be called from the test's
// own goroutine.
func noErr(t testing.TB, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
