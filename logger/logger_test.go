/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package logger

import "testing"

type captureBackend struct {
	events []string
}

func (c *captureBackend) Log(level, component, message string, _ map[string]any) {
	c.events = append(c.events, level+"/"+component+"/"+message)
}

// TestSilentByDefault: with no backend installed every helper is a no-op.
func TestSilentByDefault(t *testing.T) {
	SetBackend(nil)
	DebugCF("c", "m", nil)
	InfoCF("c", "m", nil)
	WarnCF("c", "m", nil)
	ErrorCF("c", "m", nil)
	InfoC("c", "m")
}

// TestBackendReceivesEvents: an installed backend sees every level with its
// component and message, and uninstalling silences it again.
func TestBackendReceivesEvents(t *testing.T) {
	c := &captureBackend{}
	SetBackend(c)
	defer SetBackend(nil)

	DebugCF("llmcontext", "d", nil)
	InfoCF("llmcontext", "i", nil)
	WarnCF("llmcontext", "w", nil)
	ErrorCF("llmcontext", "e", nil)
	InfoC("llmcontext", "ic")

	want := []string{"debug/llmcontext/d", "info/llmcontext/i", "warn/llmcontext/w", "error/llmcontext/e", "info/llmcontext/ic"}
	if len(c.events) != len(want) {
		t.Fatalf("events = %v, want %v", c.events, want)
	}
	for i := range want {
		if c.events[i] != want[i] {
			t.Errorf("event[%d] = %q, want %q", i, c.events[i], want[i])
		}
	}

	SetBackend(nil)
	InfoC("llmcontext", "after")
	if len(c.events) != len(want) {
		t.Fatalf("backend still receiving after SetBackend(nil): %v", c.events)
	}
}
