/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package tools

import (
	"strings"
	"testing"

	"github.com/PivotLLM/ctxengine/memory"
)

// TestSessionHistory_SingleSeq retrieves one message by exact seq.
func TestSessionHistory_SingleSeq(t *testing.T) {
	dir := t.TempDir()
	msgs := []memory.StoredMessage{
		archiveMsg(1, "user", "hello"),
		archiveMsg(2, "assistant", "world"),
		archiveMsg(3, "user", "bye"),
	}
	writeArchive(t, dir, "testsession", msgs)

	result := run(t, archiveHost(dir), "messages", "testsession", map[string]any{"seq": 2})
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.ForLLM)
	}
	if result.ForLLM == "" {
		t.Fatal("expected non-empty result")
	}
	if !strings.Contains(result.ForLLM, "world") {
		t.Errorf("expected 'world' in result, got: %s", result.ForLLM)
	}
}

// TestSessionHistory_Range retrieves multiple messages.
func TestSessionHistory_Range(t *testing.T) {
	dir := t.TempDir()
	msgs := []memory.StoredMessage{
		archiveMsg(1, "user", "first"),
		archiveMsg(2, "assistant", "second"),
		archiveMsg(3, "user", "third"),
		archiveMsg(4, "assistant", "fourth"),
	}
	writeArchive(t, dir, "sess1", msgs)

	result := run(t, archiveHost(dir), "messages", "sess1", map[string]any{"seq_start": 2, "seq_end": 3})
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.ForLLM)
	}
	if !strings.Contains(result.ForLLM, "second") || !strings.Contains(result.ForLLM, "third") {
		t.Errorf("expected seq 2 and 3, got: %s", result.ForLLM)
	}
	if strings.Contains(result.ForLLM, "first") || strings.Contains(result.ForLLM, "fourth") {
		t.Errorf("unexpected out-of-range messages in result: %s", result.ForLLM)
	}
}

// TestSessionHistory_BelowMin returns "not available" for seq below archive range.
func TestSessionHistory_BelowMin(t *testing.T) {
	dir := t.TempDir()
	writeArchive(t, dir, "mysession", []memory.StoredMessage{archiveMsg(10, "user", "recent")})

	result := run(t, archiveHost(dir), "messages", "mysession", map[string]any{"seq": 2})
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.ForLLM)
	}
	if !strings.Contains(result.ForLLM, "not available") {
		t.Errorf("expected 'not available', got: %s", result.ForLLM)
	}
}

// TestSessionHistory_AboveMax returns "not available" for seq above archive range.
func TestSessionHistory_AboveMax(t *testing.T) {
	dir := t.TempDir()
	writeArchive(t, dir, "mysession", []memory.StoredMessage{archiveMsg(1, "user", "old")})

	result := run(t, archiveHost(dir), "messages", "mysession", map[string]any{"seq": 999})
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.ForLLM)
	}
	if !strings.Contains(result.ForLLM, "not available") {
		t.Errorf("expected 'not available', got: %s", result.ForLLM)
	}
}

// TestSessionHistory_NoArchive returns "unavailable" when archive file is missing.
func TestSessionHistory_NoArchive(t *testing.T) {
	dir := t.TempDir()
	// Don't create the archive file.
	result := run(t, archiveHost(dir), "messages", "nosession", map[string]any{"seq": 1})
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.ForLLM)
	}
	// Missing archive returns "archive unavailable" message (file doesn't exist).
	if !strings.Contains(result.ForLLM, "unavailable") {
		t.Errorf("expected 'unavailable', got: %s", result.ForLLM)
	}
}

// TestSessionHistory_MissingSessionKey errors when the call carries no session key.
func TestSessionHistory_MissingSessionKey(t *testing.T) {
	result := run(t, archiveHost(t.TempDir()), "messages", "", map[string]any{"seq": 1})
	if !result.IsError {
		t.Errorf("expected error for missing session key, got: %s", result.ForLLM)
	}
}

// TestSessionHistory_InvalidArgs errors on missing required parameters.
func TestSessionHistory_InvalidArgs(t *testing.T) {
	result := run(t, archiveHost(t.TempDir()), "messages", "s", map[string]any{"seq_start": 1}) // seq_end missing
	if !result.IsError {
		t.Errorf("expected error for missing seq_end, got: %s", result.ForLLM)
	}
}

// TestSessionHistory_SessionKeyInFileName verifies that ":" in session keys is
// replaced with "_" to form the archive filename.
func TestSessionHistory_SessionKeyInFileName(t *testing.T) {
	dir := t.TempDir()
	sessionKey := "agent:main"
	writeArchive(t, dir, sessionKey, []memory.StoredMessage{archiveMsg(5, "user", "content")})

	result := run(t, archiveHost(dir), "messages", sessionKey, map[string]any{"seq": 5})
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.ForLLM)
	}
	if !strings.Contains(result.ForLLM, "content") {
		t.Errorf("expected message content, got: %s", result.ForLLM)
	}
}
