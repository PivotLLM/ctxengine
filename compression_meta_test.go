// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// --- Item 3: profile fingerprint ---

func TestProfileFingerprint(t *testing.T) {
	if got := profileFingerprint(""); got != "" {
		t.Errorf("empty profile = %q, want \"\"", got)
	}
	if got := profileFingerprint("   \n  "); got != "" {
		t.Errorf("whitespace-only profile = %q, want \"\"", got)
	}
	a := profileFingerprint("preserve in-flight state")
	if len(a) != 8 {
		t.Fatalf("fingerprint len = %d (%q), want 8", len(a), a)
	}
	for _, c := range a {
		if !strings.ContainsRune("0123456789abcdef", c) {
			t.Fatalf("fingerprint %q is not lowercase hex", a)
		}
	}
	// Stable for identical content (ignoring surrounding whitespace).
	if b := profileFingerprint("  preserve in-flight state\n"); b != a {
		t.Errorf("fingerprint not stable across whitespace: %q vs %q", a, b)
	}
	// Sensitive to content changes.
	if c := profileFingerprint("preserve something else"); c == a {
		t.Errorf("different profiles produced the same fingerprint %q", a)
	}
}

// --- Item 3: render stamps version + profile ---

func TestRender_StampsVersionAndProfile(t *testing.T) {
	gen := time.Date(2026, 6, 1, 9, 30, 0, 0, time.UTC)
	base := func() *Summary {
		return &Summary{
			Version: summaryVersion,
			State: SummaryState{
				Goals: []SummaryItem{{Text: "ship it", Refs: []SeqRange{{SeqStart: 5, SeqEnd: 5}}}},
			},
			CoveredSeqStart: 5,
			CoveredSeqEnd:   9,
			GeneratedAt:     gen,
		}
	}

	withProfile := base()
	withProfile.Model = "claude"
	withProfile.Profile = "ab12cd34"
	out := withProfile.Render(1, 100)
	if !strings.Contains(out, "by claude (v2, profile ab12cd34)") {
		t.Errorf("render missing model+version+profile stamp:\n%s", out)
	}

	noProfile := base()
	noProfile.Model = "claude"
	out = noProfile.Render(1, 100)
	if !strings.Contains(out, "by claude (v2)") {
		t.Errorf("render missing v2 stamp without profile:\n%s", out)
	}
	if strings.Contains(out, "profile") {
		t.Errorf("render should not mention profile when none set:\n%s", out)
	}

	noModel := base()
	out = noModel.Render(1, 100)
	if !strings.Contains(out, "Generated:") || !strings.Contains(out, "(v2)") {
		t.Errorf("render missing version stamp without model:\n%s", out)
	}
}

// --- Items 1 & 2: prompt guidance present in both prompts ---

func TestSummarizationPrompt_ConstraintGuidance(t *testing.T) {
	for _, aggressive := range []bool{false, true} {
		p := buildSummarizationPrompt(nil, 1, 100, aggressive, "")
		if !strings.Contains(p, "tightest range") {
			t.Errorf("aggressive=%v: prompt missing narrow-ref guidance", aggressive)
		}
		if !strings.Contains(p, "combine multiple facts") {
			t.Errorf("aggressive=%v: prompt missing one-fact-per-constraint guidance", aggressive)
		}
	}
}

// --- Item 4: a compression profile is appended to the prompt when present ---

func TestSummarizationPrompt_AppendsProfile(t *testing.T) {
	const marker = "ROLE: preserve active PR numbers verbatim"
	p := buildSummarizationPrompt(nil, 1, 100, false, marker)
	if !strings.Contains(p, marker) {
		t.Errorf("prompt did not append the compression profile content")
	}
	if !strings.Contains(p, "Agent compression profile") {
		t.Errorf("prompt missing the profile section header")
	}
	// Absent profile must not add the header.
	if strings.Contains(buildSummarizationPrompt(nil, 1, 100, false, ""), "Agent compression profile") {
		t.Errorf("empty profile should not add a profile section")
	}
}

// --- Item 4: loadCompressionProfile reads compression.md from the workspace ---

func TestLoadCompressionProfile(t *testing.T) {
	if got := loadCompressionProfile(""); got != "" {
		t.Errorf("empty dir = %q, want \"\"", got)
	}
	dir := t.TempDir()
	if got := loadCompressionProfile(dir); got != "" {
		t.Errorf("absent compression.md = %q, want \"\"", got)
	}
	if err := os.WriteFile(filepath.Join(dir, "compression.md"), []byte("  preserve PR numbers\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := loadCompressionProfile(dir); got != "preserve PR numbers" {
		t.Errorf("loaded profile = %q, want trimmed %q", got, "preserve PR numbers")
	}
}

// TestLoadCompressionProfile_PrefersUppercase verifies COMPRESSION.md is used in
// preference to the legacy lowercase compression.md when both exist.
func TestLoadCompressionProfile_PrefersUppercase(t *testing.T) {
	dir := t.TempDir()
	// Only uppercase present.
	if err := os.WriteFile(filepath.Join(dir, "COMPRESSION.md"), []byte("upper rules"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := loadCompressionProfile(dir); got != "upper rules" {
		t.Errorf("uppercase only = %q, want %q", got, "upper rules")
	}
	// Both present: uppercase wins.
	if err := os.WriteFile(filepath.Join(dir, "compression.md"), []byte("lower rules"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := loadCompressionProfile(dir); got != "upper rules" {
		t.Errorf("both present = %q, want uppercase %q", got, "upper rules")
	}
}

// --- On-demand compaction (the /compact path) stamps model + profile end-to-end ---

func TestCompact_OnDemandStampsProfile(t *testing.T) {
	dir := t.TempDir()
	const profile = "preserve active branch names verbatim"
	if err := os.WriteFile(filepath.Join(dir, "compression.md"), []byte(profile), 0o644); err != nil {
		t.Fatal(err)
	}

	store := &compressTestStore{history: makeConversation(10, 200)}
	llm := &mockLLM{responses: []string{validSummaryJSON("active goal")}}
	mgr := newCompressManager(t, store, []*mockLLM{llm},
		WithCompressionProfileDir(dir),
		WithCompressModel(ModelChain{Primary: "test-model"}),
	)
	mgr.msgCount = len(store.history)

	if err := mgr.Compact(context.Background()); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	s, err := unmarshalSummary(store.summary)
	if err != nil || s == nil {
		t.Fatalf("no valid summary persisted: %v", err)
	}
	if want := profileFingerprint(profile); s.Profile != want {
		t.Errorf("summary profile = %q, want %q", s.Profile, want)
	}
	if s.Model != "test-model" {
		t.Errorf("summary model = %q, want test-model", s.Model)
	}
	// The rendered block carries the full stamp.
	if rendered := s.Render(0, 0); !strings.Contains(rendered, "by test-model (v2, profile "+s.Profile+")") {
		t.Errorf("rendered stamp missing model/version/profile:\n%s", rendered)
	}
}

// TestCompact_NoProfile leaves the profile fingerprint empty when no
// compression.md is configured.
func TestCompact_NoProfile(t *testing.T) {
	store := &compressTestStore{history: makeConversation(10, 200)}
	llm := &mockLLM{responses: []string{validSummaryJSON("active goal")}}
	mgr := newCompressManager(t, store, []*mockLLM{llm})
	mgr.msgCount = len(store.history)

	if err := mgr.Compact(context.Background()); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	s, err := unmarshalSummary(store.summary)
	if err != nil || s == nil {
		t.Fatalf("no valid summary persisted: %v", err)
	}
	if s.Profile != "" {
		t.Errorf("profile = %q, want empty when no compression.md", s.Profile)
	}
}
