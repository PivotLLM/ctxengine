// ctxengine
// License: MIT

package ctxengine

import "testing"

func TestDefaultRefusalClassifier(t *testing.T) {
	cases := []struct {
		finishReason, content string
		want                  bool
		detail                string
	}{
		{"refusal", "", true, "content policy (finish_reason=refusal)"},
		{"CONTENT_FILTER", "", true, "content policy (finish_reason=content_filter)"},
		{"stop", `{"version":2}`, false, ""},
		{"", "partial output\n\nI'm sorry, but I cannot assist with that request.", true, "content policy"},
		{"", "the user discussed a refusal earlier", false, ""},
		{"length", "", false, ""},
	}
	for _, tc := range cases {
		got, detail := defaultRefusalClassifier(tc.finishReason, tc.content)
		if got != tc.want || detail != tc.detail {
			t.Errorf("classify(%q, %q) = (%v, %q), want (%v, %q)", tc.finishReason, tc.content, got, detail, tc.want, tc.detail)
		}
	}
}
