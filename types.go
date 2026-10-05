/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

import (
	"context"
	"time"
)

// ModelRequest is one summarization call the engine asks the host to make.
// The host owns model selection, fallback and cooldowns; the engine only says
// what to send and which models it will not accept a reply from.
type ModelRequest struct {
	// System and User are the two messages of the call.
	System, User string
	// JSONObject asks for a JSON-object response format where the provider
	// supports it.
	JSONObject bool
	// Exclude lists model names (as reported in ModelReply.Model) that must
	// not serve this request: models that refused this session's content, or
	// returned an unacceptable summary earlier in the same pass.
	Exclude []string
}

// ModelReply is the host's answer to one ModelRequest. Model names the model
// that produced it — the engine keys its per-session refusal memory and the
// compaction report on it. FinishReason carries the provider's stop reason
// (e.g. "stop", "length", "refusal", "content_filter") when available, so the
// summarizer can distinguish a content refusal from a transient error. On an
// error the host may still set Model to the last model it tried so the report
// can name it.
type ModelReply struct {
	Content      string
	FinishReason string
	Model        string
}

// ModelCaller is the host's summarization model. One implementation walks the
// host's whole chain (fallbacks, cooldowns, exclusions); the engine calls it
// again with a longer Exclude list when a reply is unacceptable.
type ModelCaller interface {
	Complete(ctx context.Context, req ModelRequest) (ModelReply, error)
}

// ModelChain records which LLM chain is configured for compression (for stats and logging).
type ModelChain struct {
	Primary   string   `json:"primary,omitempty"`
	Fallbacks []string `json:"fallbacks,omitempty"`
}

// ContextStats holds observable state for a session's context.
type ContextStats struct {
	TotalMessages       int
	MeaningfulMessages  int
	EstimatedTokens     int
	ContextWindowPct    float64
	LastCompressedAt    time.Time
	LastCompressionGain float64
	CompressionCooling  bool
	CoolingSinceCount   int
	// SummaryTokens is the estimated token count of the stored summary (runes/4).
	// Zero when no summary has been generated.
	SummaryTokens int
}
