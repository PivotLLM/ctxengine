package memory

import (
	"context"
	"testing"
)

func TestListPendingSessions_Empty(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	keys, err := store.ListPendingSessions(ctx)
	if err != nil {
		t.Fatalf("ListPendingSessions: %v", err)
	}
	if len(keys) != 0 {
		t.Errorf("expected empty, got %v", keys)
	}
}

func TestListPendingSessions_DirectoryNotExist(t *testing.T) {
	// Point directly at a non-existent sub-directory; do NOT create it via NewJSONLStore.
	dir := t.TempDir() + "/does-not-exist"
	store := &JSONLStore{
		dir:         dir,
		noiseCaches: make(map[string]*noiseCache),
	}
	ctx := context.Background()

	keys, err := store.ListPendingSessions(ctx)
	if err != nil {
		t.Fatalf("expected nil error for missing dir, got %v", err)
	}
	if len(keys) != 0 {
		t.Errorf("expected nil/empty, got %v", keys)
	}
}

func TestListPendingSessions_ReturnsOnlyPending(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	// Session A: PendingTurn = true (set and left set)
	if err := store.AddMessage(ctx, "session-a", "user", "hello"); err != nil {
		t.Fatalf("AddMessage session-a: %v", err)
	}
	if err := store.SetPendingTurn(ctx, "session-a"); err != nil {
		t.Fatalf("SetPendingTurn session-a: %v", err)
	}

	// Session B: PendingTurn was set then cleared
	if err := store.AddMessage(ctx, "session-b", "user", "hi"); err != nil {
		t.Fatalf("AddMessage session-b: %v", err)
	}
	if err := store.SetPendingTurn(ctx, "session-b"); err != nil {
		t.Fatalf("SetPendingTurn session-b: %v", err)
	}
	if err := store.ClearPendingTurn(ctx, "session-b"); err != nil {
		t.Fatalf("ClearPendingTurn session-b: %v", err)
	}

	// Session C: only has a message, no meta file manipulation
	if err := store.AddMessage(ctx, "session-c", "user", "hey"); err != nil {
		t.Fatalf("AddMessage session-c: %v", err)
	}

	keys, err := store.ListPendingSessions(ctx)
	if err != nil {
		t.Fatalf("ListPendingSessions: %v", err)
	}
	if len(keys) != 1 {
		t.Fatalf("expected 1 pending session, got %d: %v", len(keys), keys)
	}
	if keys[0] != "session-a" {
		t.Errorf("expected session-a, got %q", keys[0])
	}
}

func TestListPendingSessions_MultiplePending(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	for _, key := range []string{"s1", "s2", "s3"} {
		if err := store.AddMessage(ctx, key, "user", "msg"); err != nil {
			t.Fatalf("AddMessage %s: %v", key, err)
		}
		if err := store.SetPendingTurn(ctx, key); err != nil {
			t.Fatalf("SetPendingTurn %s: %v", key, err)
		}
	}

	keys, err := store.ListPendingSessions(ctx)
	if err != nil {
		t.Fatalf("ListPendingSessions: %v", err)
	}
	if len(keys) != 3 {
		t.Fatalf("expected 3 pending sessions, got %d: %v", len(keys), keys)
	}
}
