/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

import (
	"slices"
	"strings"
)

// refusalFinishReasons are provider stop reasons that unambiguously signal the
// model declined to produce output on content-policy grounds.
var refusalFinishReasons = []string{
	"refusal",
	"content_filter",
	"content-filter",
	"safety",
}

// refusalMarkers are lowercase substrings that, when present in a response that
// otherwise FAILED to produce the expected structured output, indicate the model
// refused rather than merely returning malformed content. They are deliberately
// specific phrasings of a decline so ordinary output that happens to discuss a
// refusal is not flagged — they are only consulted on an already-failed parse.
var refusalMarkers = []string{
	"i'm sorry, but i cannot",
	"i'm sorry, but i can't",
	"i am sorry, but i cannot",
	"i am sorry, but i can't",
	"i cannot assist with that",
	"i can't assist with that",
	"i cannot help with that",
	"i can't help with that",
	"i'm unable to assist",
	"i am unable to assist",
	"i won't be able to help",
	"i will not be able to help",
	"i must decline",
	"i cannot comply",
	"i can't comply",
	"against my guidelines",
	"i cannot continue with this",
	"i can't continue with this",
}

// defaultRefusalClassifier is the built-in RefusalClassifier. A refusal is
// recognised either from the provider's finishReason (the reliable path when
// the provider sets one) or, as a backstop for providers that do not (many
// OpenAI-compatible gateways, Bedrock), from a refusal marker in content —
// which catches the common case of a model emitting partial output and then
// appending "I'm sorry, but I cannot…", truncating the result.
func defaultRefusalClassifier(finishReason, content string) (bool, string) {
	if fr := strings.ToLower(strings.TrimSpace(finishReason)); fr != "" {
		if slices.Contains(refusalFinishReasons, fr) {
			return true, "content policy (finish_reason=" + fr + ")"
		}
	}
	lc := strings.ToLower(content)
	for _, marker := range refusalMarkers {
		if strings.Contains(lc, marker) {
			return true, "content policy"
		}
	}
	return false, ""
}
