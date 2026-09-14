package session

import (
	"regexp"
	"strings"
)

// systemReminderRe matches <system-reminder>...</system-reminder> blocks,
// including multiline content. The (?s) flag makes '.' match newlines.
var systemReminderRe = regexp.MustCompile(`(?s)<system-reminder>.*?</system-reminder>`)

// SanitizeContent removes injected XML blocks from message content before it
// is stored in session history. Currently strips <system-reminder> blocks
// produced by the claude CLI auto-memory system.
//
// Additional sanitization rules can be added here in the future by appending
// further regexp replacements or string transformations before the final Trim.
func SanitizeContent(content string) string {
	content = systemReminderRe.ReplaceAllString(content, "")
	return strings.TrimSpace(content)
}
