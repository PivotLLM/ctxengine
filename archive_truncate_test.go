// ctxengine
// License: MIT

package ctxengine

import (
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"
)

// TestArchiveTruncate_DefaultCapIs256K pins the default tool cap: the archive
// is the durable record an evicted result points back to, so a large but
// ordinary result (100 KB) is stored whole and only a pathological one is
// clipped.
func TestArchiveTruncate_DefaultCapIs256K(t *testing.T) {
	if archiveContentMaxBytes != 256*1024 {
		t.Fatalf("archiveContentMaxBytes = %d, want 256 KB", archiveContentMaxBytes)
	}
	whole := strings.Repeat("t", 100*1024)
	if got := archiveTruncateContent(spawnllm.Message{Role: "tool", Content: whole}, archiveContentMaxBytes); got.Content != whole {
		t.Errorf("a 100 KB tool result was truncated to %d bytes", len(got.Content))
	}
	huge := strings.Repeat("t", archiveContentMaxBytes*2)
	got := archiveTruncateContent(spawnllm.Message{Role: "tool", Content: huge}, archiveContentMaxBytes)
	if len(got.Content) >= len(huge) {
		t.Fatalf("tool content over the cap not truncated: %d bytes", len(got.Content))
	}
	if !strings.HasPrefix(got.Content, huge[:archiveContentMaxBytes]) || !strings.Contains(got.Content, "[content truncated:") {
		t.Error("truncation must keep the first cap bytes and append the marker")
	}
}

// TestArchiveTruncate_ConversationFloor covers the split: user and assistant
// content is not re-retrievable and feeds cognitive-memory consolidation, so a
// clipped instruction yields a memory built on a fragment. An explicit cap
// below the conversation floor is raised to the floor for those roles and
// applied as is to tool results.
func TestArchiveTruncate_ConversationFloor(t *testing.T) {
	body := strings.Repeat("u", 8_000) // over a 4 KB cap, under the floor
	for _, role := range []string{"user", "assistant"} {
		got := archiveTruncateContent(spawnllm.Message{Role: role, Content: body}, 4096)
		if len(got.Content) != len(body) {
			t.Errorf("%s content truncated at %d bytes; conversation should survive to %d",
				role, len(got.Content), archiveConversationMaxBytes)
		}
	}
	if got := archiveTruncateContent(spawnllm.Message{Role: "tool", Content: body}, 4096); len(got.Content) >= len(body) {
		t.Errorf("an explicit 4 KB cap did not apply to a tool result: %d bytes", len(got.Content))
	}
}

// TestArchiveTruncate_ConversationStillCappedEventually confirms the floor is
// a floor, not an exemption: conversation over the effective cap is clipped.
func TestArchiveTruncate_ConversationStillCappedEventually(t *testing.T) {
	body := strings.Repeat("u", archiveConversationMaxBytes*2)
	got := archiveTruncateContent(spawnllm.Message{Role: "user", Content: body}, 4096)
	if len(got.Content) >= len(body) {
		t.Errorf("conversation content over the floor must still be truncated, got %d bytes", len(got.Content))
	}
	if !strings.HasPrefix(got.Content, body[:archiveConversationMaxBytes]) {
		t.Error("conversation must be clipped at the floor, not the explicit cap")
	}
}

// TestArchiveTruncate_ExplicitLimitAboveConversationFloorWins verifies an
// operator raising the limit raises it for every role — the split must never
// silently lower a configured value.
func TestArchiveTruncate_ExplicitLimitAboveConversationFloorWins(t *testing.T) {
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
