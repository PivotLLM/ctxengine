// ctxengine
// License: MIT

package ctxengine

import (
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"
)

// TestArchiveTruncate_ConversationGetsLargerCap covers the split: user and
// assistant content is not re-retrievable and feeds cognitive-memory
// consolidation, so a clipped instruction yields a memory built on a fragment.
// Measured production p99 for these roles is ~1.2KB and ~2.7KB, which the old
// shared 4KB cap sat just inside.
func TestArchiveTruncate_ConversationGetsLargerCap(t *testing.T) {
	body := strings.Repeat("u", 8_000) // over the tool cap, under the conversation cap
	for _, role := range []string{"user", "assistant"} {
		got := archiveTruncateContent(spawnllm.Message{Role: role, Content: body}, archiveContentMaxBytes)
		if len(got.Content) != len(body) {
			t.Errorf("%s content truncated at %d bytes; conversation should survive to %d",
				role, len(got.Content), archiveConversationMaxBytes)
		}
	}
}

// TestArchiveTruncate_ToolKeepsTightCap verifies tool results still get the
// small cap: they carry the real bulk (a single production row measured 5.7MB)
// and the file they came from is still on disk.
func TestArchiveTruncate_ToolKeepsTightCap(t *testing.T) {
	body := strings.Repeat("t", 8_000)
	got := archiveTruncateContent(spawnllm.Message{Role: "tool", Content: body}, archiveContentMaxBytes)
	if len(got.Content) >= len(body) {
		t.Fatalf("tool content not truncated: %d bytes", len(got.Content))
	}
	if !strings.Contains(got.Content, "[content truncated:") {
		t.Error("truncation marker missing")
	}
}

// TestArchiveTruncate_ConversationStillCappedEventually confirms the larger cap
// is a cap, not an exemption.
func TestArchiveTruncate_ConversationStillCappedEventually(t *testing.T) {
	body := strings.Repeat("u", archiveConversationMaxBytes*2)
	got := archiveTruncateContent(spawnllm.Message{Role: "user", Content: body}, archiveContentMaxBytes)
	if len(got.Content) >= len(body) {
		t.Errorf("oversized conversation content must still be truncated, got %d bytes", len(got.Content))
	}
}

// TestArchiveTruncate_ExplicitLimitAboveConversationCapWins verifies an operator
// raising the limit raises it for every role — the split must never silently
// lower a configured value.
func TestArchiveTruncate_ExplicitLimitAboveConversationCapWins(t *testing.T) {
	limit := archiveConversationMaxBytes * 4
	body := strings.Repeat("t", archiveConversationMaxBytes*2)
	got := archiveTruncateContent(spawnllm.Message{Role: "tool", Content: body}, limit)
	if len(got.Content) != len(body) {
		t.Errorf("an explicit limit of %d should keep %d bytes, kept %d", limit, len(body), len(got.Content))
	}
}

// TestArchiveTruncate_ShortContentUntouched is the no-op path.
func TestArchiveTruncate_ShortContentUntouched(t *testing.T) {
	msg := spawnllm.Message{Role: "tool", Content: "small"}
	if got := archiveTruncateContent(msg, archiveContentMaxBytes); got.Content != "small" {
		t.Errorf("short content altered: %q", got.Content)
	}
}
