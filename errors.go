// ctxengine
// License: MIT

package ctxengine

import "errors"

// Sentinel errors returned by Compress(), ForceCompress(), and Compact().
var (
	// ErrCompressionNotNeeded is returned when the session is below all
	// compression thresholds and no action was taken.
	ErrCompressionNotNeeded = errors.New("llmcontext: compression not needed")

	// ErrCompressionFailed is returned when the LLM was invoked for compression
	// but every model in the chain failed or produced an unacceptable summary,
	// when Save() fails, or when ForceCompress cannot fit the current turn group
	// within the context window.
	ErrCompressionFailed = errors.New("llmcontext: compression failed")

	// ErrNothingToCompress is returned when compression ran but there was nothing
	// to summarize — the retained tail already covers the entire conversation, so
	// no LLM call was made. This is a benign no-op, not a failure.
	ErrNothingToCompress = errors.New("llmcontext: nothing to compress")

	// ErrCompressionPartial is returned when compression ran but the result
	// is still at or above the safety threshold.
	ErrCompressionPartial = errors.New("llmcontext: compression partial: still over safety threshold")
)
