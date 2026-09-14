// ctxengine
// License: MIT

package memory

import (
	"context"
	"testing"
)

// TestForgetSession_DropsNoiseCache verifies that ForgetSession removes the
// in-memory noise cache for a session while leaving durable data intact.
func TestForgetSession_DropsNoiseCache(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	const key = "telegram_42"

	if err := store.AddMessage(ctx, key, "user", "hello"); err != nil {
		t.Fatalf("AddMessage: %v", err)
	}

	// A noise cache entry should exist after a write.
	store.noiseMu.Lock()
	_, present := store.noiseCaches[key]
	store.noiseMu.Unlock()
	if !present {
		t.Fatalf("expected noise cache entry for %q after write", key)
	}

	store.ForgetSession(key)

	store.noiseMu.Lock()
	_, present = store.noiseCaches[key]
	store.noiseMu.Unlock()
	if present {
		t.Fatalf("noise cache entry for %q should be gone after ForgetSession", key)
	}

	// Durable history is untouched.
	hist, err := store.GetHistory(ctx, key)
	if err != nil {
		t.Fatalf("GetHistory: %v", err)
	}
	if len(hist) != 1 || hist[0].Content != "hello" {
		t.Fatalf("history altered by ForgetSession: %+v", hist)
	}
}

// TestForgetSession_UnknownKeyNoPanic verifies forgetting an unknown session is
// a harmless no-op.
func TestForgetSession_UnknownKeyNoPanic(t *testing.T) {
	store := newTestStore(t)
	store.ForgetSession("does-not-exist")
}
