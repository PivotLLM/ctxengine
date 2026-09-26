// ctxengine
// License: MIT

// Package testcron is a test-only copy of the host's cron-message wrapper
// format (ClawEh's cronmsg package). A host scheduler wraps a scheduled message
// in a header announcing the fire time; the engine's storage and compaction
// layers recognise that wrapper to detect repeated fires of the same job. The
// tests import this package as cronmsg to produce wrapped messages exactly as
// the scheduler does.
//
// The wrapped form is:
//
//	[<fp>] The following message is from a cron job that fired at <ts>:\n\n<message>
//
// where "[<fp>] " is an optional leading lowercase-hex fingerprint marker. When
// the fingerprint is empty the un-marked legacy form (no "[<fp>] " prefix) is
// produced and remains parseable.
package testcron

import (
	"fmt"
	"strings"
	"time"
)

// prefix is the cron-wrapper header that precedes the fire timestamp. It is the
// single definition of this literal; all producers and parsers reference it.
const prefix = "The following message is from a cron job that fired at "

// timeFormat is the layout used for the embedded fire timestamp. It must match
// the format historically produced by the scheduler so that already-persisted
// messages continue to round-trip.
const timeFormat = "2006-01-02 15:04 MST"

// Build returns message wrapped with the cron metadata header indicating the
// fire time so the LLM knows the message originated from a scheduled job. When
// fingerprint is non-empty it is emitted FIRST as a leading "[<fp>] " marker so
// the context-compaction layer can group repeated fires of the same job
// regardless of the embedded (always-changing) timestamp. When fingerprint is
// empty the legacy un-marked form is produced.
func Build(fingerprint string, fireTime time.Time, message string) string {
	ts := fireTime.Format(timeFormat)
	body := fmt.Sprintf("%s%s:\n\n%s", prefix, ts, message)
	if fingerprint == "" {
		return body
	}
	return fmt.Sprintf("[%s] %s", fingerprint, body)
}

// Parse reports whether content is a cron-wrapper message. It tolerates an
// optional leading "[<hex>] " fingerprint marker (new format) and also the
// legacy un-marked form. It returns the fingerprint ("" if absent), the payload
// (the substring after the first '\n' that follows the prefix line; "" when the
// prefix line has no trailing newline), and ok.
//
// The semantics replicate the historical llmcontext marker parser exactly: a
// leading '[' that does not resolve to a valid "[<lowerhex>] " marker (bounds
// 1 < end <= 12, lowercase-hex token, followed by "] ") is left in place and
// the whole content is then required to start with the prefix.
func Parse(content string) (fingerprint, payload string, ok bool) {
	work := content
	if strings.HasPrefix(work, "[") {
		if end := strings.IndexByte(work, ']'); end > 1 && end <= 12 {
			token := work[1:end]
			if isLowerHex(token) && strings.HasPrefix(work[end:], "] ") {
				fingerprint = token
				work = work[end+2:]
			}
		}
	}

	if !strings.HasPrefix(work, prefix) {
		return "", "", false
	}
	if idx := strings.IndexByte(work[len(prefix):], '\n'); idx >= 0 {
		payload = work[len(prefix)+idx+1:]
	}
	return fingerprint, payload, true
}

// eventPrefix is the header of a message delivered by a continuous monitor (a
// listen job). It is deliberately a different literal from prefix, and Parse
// does not recognise it: repeated fires of one scheduled job are deduplicated
// and collapsed as noise, but each event a monitor delivers is distinct data
// and must be stored and kept like any other message.
const eventPrefix = "The following event was received by a continuous monitor at "

// eventTimeFormat carries seconds, unlike the cron fire time. The store drops
// a user message identical to the previous one as noise; a listener may
// deliver the same event twice in a row on purpose, and a listener never
// delivers more than once every two seconds, so second granularity keeps two
// identical events distinct.
const eventTimeFormat = "2006-01-02 15:04:05 MST"

// BuildEvent wraps an event delivered by a listen job: the operator's note,
// then the source tool's full result introduced by name. With an empty result
// (an operational notice from the monitor itself) only the note is included.
// The output is not a cron-wrapper message; see eventPrefix.
func BuildEvent(at time.Time, message, source, result string) string {
	body := fmt.Sprintf("%s%s:\n\n%s", eventPrefix, at.Format(eventTimeFormat), message)
	if result == "" {
		return body
	}
	return fmt.Sprintf("%s\n\n%s returned the following:\n%s", body, source, result)
}

// CollapseKey returns the dedup/collapse key for a cron-wrapper message: the
// fingerprint when present, otherwise the payload (legacy fallback). The bool is
// false for non-cron content. Two messages with the same key represent repeated
// fires of the same job and may be collapsed.
func CollapseKey(content string) (key string, ok bool) {
	fp, payload, parsed := Parse(content)
	if !parsed {
		return "", false
	}
	if fp != "" {
		return fp, true
	}
	return payload, true
}

// isLowerHex reports whether s is non-empty and consists solely of lowercase
// hexadecimal digits.
func isLowerHex(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') {
			continue
		}
		return false
	}
	return true
}
