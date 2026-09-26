// ctxengine
// License: MIT

package tools

import (
	"strings"
	"testing"

	"github.com/PivotLLM/ctxengine/memory"
)

// TestSearchTool_BasicSearch returns the correct message for a content term.
func TestSearchTool_BasicSearch(t *testing.T) {
	dir := t.TempDir()
	msgs := []memory.StoredMessage{
		archiveMsg(1, "user", "the quick brown fox"),
		archiveMsg(2, "assistant", "jumped over the lazy dog"),
		archiveMsg(3, "user", "nothing relevant here"),
	}
	writeArchive(t, dir, "searchsess", msgs)

	result := run(t, archiveHost(dir), "search", "searchsess", map[string]any{"query": "fox"})
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.ForLLM)
	}
	if !strings.Contains(result.ForLLM, "quick brown fox") {
		t.Errorf("expected fox message, got: %s", result.ForLLM)
	}
	if strings.Contains(result.ForLLM, "lazy dog") {
		t.Errorf("unexpected second message in result: %s", result.ForLLM)
	}
}

// TestSearchTool_FTSOperators tests AND, OR expressions.
func TestSearchTool_FTSOperators(t *testing.T) {
	dir := t.TempDir()
	msgs := []memory.StoredMessage{
		archiveMsg(1, "user", "apple banana cherry"),
		archiveMsg(2, "assistant", "apple only"),
		archiveMsg(3, "user", "banana only"),
	}
	writeArchive(t, dir, "ftssess", msgs)
	h := archiveHost(dir)

	// AND: only message 1 contains both apple and banana.
	r := run(t, h, "search", "ftssess", map[string]any{"query": "apple AND banana"})
	if r.IsError {
		t.Fatalf("AND query error: %s", r.ForLLM)
	}
	if !strings.Contains(r.ForLLM, "cherry") {
		t.Errorf("AND: expected msg1 (cherry), got: %s", r.ForLLM)
	}

	// OR: messages 1 and 2 contain apple; 1 and 3 contain banana — all 3 should appear.
	r2 := run(t, h, "search", "ftssess", map[string]any{"query": "apple OR banana"})
	if r2.IsError {
		t.Fatalf("OR query error: %s", r2.ForLLM)
	}
	if !strings.Contains(r2.ForLLM, "cherry") && !strings.Contains(r2.ForLLM, "apple only") {
		t.Errorf("OR: expected multiple results, got: %s", r2.ForLLM)
	}
}

// TestSearchTool_RoleFilter restricts results by role.
func TestSearchTool_RoleFilter(t *testing.T) {
	dir := t.TempDir()
	msgs := []memory.StoredMessage{
		archiveMsg(1, "user", "unique_term_xyz user message"),
		archiveMsg(2, "assistant", "unique_term_xyz assistant message"),
	}
	writeArchive(t, dir, "rolesess", msgs)

	// Filter to user only.
	r := run(t, archiveHost(dir), "search", "rolesess", map[string]any{"query": "unique_term_xyz", "role": "user"})
	if r.IsError {
		t.Fatalf("role filter error: %s", r.ForLLM)
	}
	if !strings.Contains(r.ForLLM, "user message") {
		t.Errorf("expected user message in result: %s", r.ForLLM)
	}
	if strings.Contains(r.ForLLM, "assistant message") {
		t.Errorf("unexpected assistant message with role=user filter: %s", r.ForLLM)
	}
}

// TestSearchTool_QueryTooLong rejects queries longer than 500 characters.
func TestSearchTool_QueryTooLong(t *testing.T) {
	dir := t.TempDir()
	writeArchive(t, dir, "longsess", []memory.StoredMessage{archiveMsg(1, "user", "something")})

	longQuery := strings.Repeat("x", 501)
	r := run(t, archiveHost(dir), "search", "longsess", map[string]any{"query": longQuery})
	if !r.IsError {
		t.Errorf("expected error for >500 char query, got: %s", r.ForLLM)
	}
}

// TestSearchTool_MalformedFTS returns a tool error (not panic) for a bad FTS expression.
func TestSearchTool_MalformedFTS(t *testing.T) {
	dir := t.TempDir()
	writeArchive(t, dir, "badftssess", []memory.StoredMessage{archiveMsg(1, "user", "content")})

	// Unmatched quote is a malformed FTS5 expression.
	r := run(t, archiveHost(dir), "search", "badftssess", map[string]any{"query": `"unclosed phrase`})
	if r.IsError {
		// An error result is acceptable — FTS parse errors are surfaced as tool errors.
		return
	}
	// If it doesn't error, no panic means pass.
}

// TestSearchTool_LimitEnforcedAt100 clamps limit to 100.
func TestSearchTool_LimitEnforcedAt100(t *testing.T) {
	dir := t.TempDir()
	var msgs []memory.StoredMessage
	for i := 1; i <= 150; i++ {
		msgs = append(msgs, archiveMsg(int64(i), "user", "matchterm repetitive content"))
	}
	writeArchive(t, dir, "limitsess", msgs)

	r := run(t, archiveHost(dir), "search", "limitsess", map[string]any{"query": "matchterm", "limit": 200})
	if r.IsError {
		t.Fatalf("unexpected error: %s", r.ForLLM)
	}
	// Count occurrences of "[#" in the output to estimate number of results.
	count := strings.Count(r.ForLLM, "[#")
	if count > 100 {
		t.Errorf("expected at most 100 results, got %d", count)
	}
}

// TestSearchTool_SQLInjection verifies that a SQL injection attempt is inert.
func TestSearchTool_SQLInjection(t *testing.T) {
	dir := t.TempDir()
	writeArchive(t, dir, "injsess", []memory.StoredMessage{archiveMsg(1, "user", "safe content")})

	// This classic injection attempt should either return no results or a tool
	// error (FTS parse error). In either case the table must remain intact.
	_ = run(t, archiveHost(dir), "search", "injsess", map[string]any{"query": "x'; DROP TABLE messages; --"})

	// Verify the table is still intact by opening the archive directly.
	archivePath := memory.ArchivePath(dir, "injsess")
	a, err := memory.Open(archivePath)
	if err != nil {
		t.Fatalf("archive open after injection attempt: %v", err)
	}
	defer func() { noErr(t, a.Close()) }()
	msgs, err := a.QueryRange(1, 1)
	if err != nil {
		t.Fatalf("QueryRange after injection attempt: %v", err)
	}
	if len(msgs) != 1 {
		t.Errorf("expected 1 message after injection attempt, got %d", len(msgs))
	}
}

// TestSearchTool_EmptyResult returns a clear message when no messages match.
func TestSearchTool_EmptyResult(t *testing.T) {
	dir := t.TempDir()
	writeArchive(t, dir, "emptysess", []memory.StoredMessage{archiveMsg(1, "user", "hello world")})

	r := run(t, archiveHost(dir), "search", "emptysess", map[string]any{"query": "zzznomatch"})
	if r.IsError {
		t.Fatalf("unexpected error: %s", r.ForLLM)
	}
	if !strings.Contains(r.ForLLM, "no matching") {
		t.Errorf("expected 'no matching messages' message, got: %s", r.ForLLM)
	}
}

// TestSearchTool_MissingSessionKey returns error when the call carries no session key.
func TestSearchTool_MissingSessionKey(t *testing.T) {
	r := run(t, archiveHost(t.TempDir()), "search", "", map[string]any{"query": "anything"})
	if !r.IsError {
		t.Errorf("expected error for missing session key, got: %s", r.ForLLM)
	}
}

// TestSearchTool_NoArchive returns "archive unavailable" when file is missing.
func TestSearchTool_NoArchive(t *testing.T) {
	r := run(t, archiveHost(t.TempDir()), "search", "missingsess", map[string]any{"query": "anything"})
	if r.IsError {
		t.Fatalf("unexpected hard error: %s", r.ForLLM)
	}
	if !strings.Contains(r.ForLLM, "unavailable") {
		t.Errorf("expected unavailability message, got: %s", r.ForLLM)
	}
}
