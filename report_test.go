// ctxengine
// License: MIT

package ctxengine

import (
	"strings"
	"testing"
)

// TestCompactionReport_NoBeforeStatsHeader confirms an early-exit report (no
// conversation read) renders a bare "Compaction:" header instead of the
// misleading "0 messages (0 B)".
func TestCompactionReport_NoBeforeStatsHeader(t *testing.T) {
	r := &CompactionReport{
		SessionKey: "s",
		Attempts: []CompactionAttempt{
			{Model: "abliterated-model", Status: "skipped", Detail: "in cooldown (9m remaining)"},
		},
		Outcome: "failed",
	}
	s := r.String()
	if strings.Contains(s, "0 messages") || strings.Contains(s, "0 B") {
		t.Fatalf("header should omit bogus 0/0 count:\n%s", s)
	}
	if !strings.HasPrefix(s, "Compaction:") {
		t.Fatalf("expected bare 'Compaction:' header, got:\n%s", s)
	}
	if !strings.Contains(s, "cooldown") {
		t.Fatalf("expected cooldown detail in report:\n%s", s)
	}
}

// TestCompactionReport_WithStatsHeader confirms a normal report still shows the
// message count and size.
func TestCompactionReport_WithStatsHeader(t *testing.T) {
	r := &CompactionReport{SessionKey: "s", BeforeMsgs: 12, BeforeBytes: 2048, Outcome: "success", AfterMsgs: 3, AfterBytes: 512}
	s := r.String()
	if !strings.Contains(s, "12 messages") {
		t.Fatalf("expected message count in header:\n%s", s)
	}
}
