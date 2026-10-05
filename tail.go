/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

import (
	"time"

	"github.com/PivotLLM/spawnllm"

	"github.com/PivotLLM/ctxengine/memory"
)

// selectTail returns the suffix of history to retain in the context window,
// together with start, the index in stored where that suffix begins. The
// returned tail is stored[start:] less the adjacent repeats step 5 removes, and
// it always holds stored[start:]'s newest message and newest user message.
//
// The caller summarizes stored[:start], exactly. The repeats step 5 removes go
// to neither the summary nor the window: each is an adjacent copy of a message
// the tail keeps, and the archive still holds it. start is not
// len(stored)-len(tail); deriving it from the collapsed length shifted it back
// by the number of repeats, which handed retained messages to the summary and
// hid the newest user message from the caller's keep-newest-user clamp.
//
// The tail is not necessarily what the window ends up holding: when the pass
// persists it, collapseRetainedCronRuns folds an answered run of
// repetitiveRunThreshold or more fires of one job (each answered by the same
// short reply) into one counted anchor. A fire with no reply after it, such as
// the request the next dispatch answers, is never part of such a run.
//
// Algorithm:
//  1. Walk history newest-to-oldest in turn groups (see resolveGroup).
//  2. Accumulate groups while their token cost fits within budget AND they are
//     not older than maxAge.
//  3. Minimum floor: if totalMeaningful < minMessages, keep adding groups
//     regardless of budget or age.
//  4. Advance start past a leading partial tool group so the tail begins on a
//     clean boundary, handing those messages to the summary.
//  5. Collapse consecutive noise repeats in the retained tail to one (see
//     collapseStoredNoise).
//
// A budget <= 0 disables the budget check and a maxAge <= 0 disables the age
// check; the floor always applies. estimate converts a message slice into an
// estimated token count; pass the Manager's estTokens so the configured divisor
// and safety margin apply. noise identifies repeated fires of one source (nil
// means none are recognised).
func selectTail(
	stored []memory.StoredMessage,
	budget, minMessages int,
	maxAge time.Duration,
	now time.Time,
	estimate func([]spawnllm.Message) int,
	noise NoiseKeyFunc,
) ([]memory.StoredMessage, int) {
	if len(stored) == 0 {
		return nil, 0
	}
	if estimate == nil {
		estimate = estimateTokens
	}
	if noise == nil {
		noise = noNoiseKey
	}

	plain := storedToPlain(stored)

	start := len(stored)
	totalTokens := 0
	totalMeaningful := 0

	i := len(stored) - 1
	for i >= 0 {
		g := resolveGroup(plain, i)
		cost := estimate(plain[g.start : g.end+1])
		meaningful := countMeaningfulMessages(plain[g.start:g.end+1], noise)

		fits := budget <= 0 || totalTokens+cost <= budget
		// A group is judged by its OLDEST message: turn groups span seconds, so
		// which end is measured is immaterial in practice, and taking the oldest
		// keeps the cap honest about the age of what is retained.
		fresh := maxAge <= 0 || !isOlderThan(stored[g.start].CreatedAt, maxAge, now)
		belowFloor := minMessages > 0 && totalMeaningful < minMessages

		if (fits && fresh) || belowFloor {
			start = g.start
			totalTokens += cost
			totalMeaningful += meaningful
			i = g.start - 1
			continue
		}
		break
	}

	if start >= len(stored) {
		return nil, len(stored)
	}

	start = advancePastPartialToolGroup(stored, start)
	if start >= len(stored) {
		return nil, len(stored)
	}
	return collapseStoredNoise(stored[start:], noise), start
}

// isOlderThan reports whether ts is more than maxAge before now. A zero
// timestamp is never considered old: messages written before CreatedAt existed
// must not be evicted by an age rule that cannot see them.
func isOlderThan(ts time.Time, maxAge time.Duration, now time.Time) bool {
	if ts.IsZero() {
		return false
	}
	return now.Sub(ts) > maxAge
}

// advancePastPartialToolGroup moves start forward past any leading tool result
// or assistant tool-call message so the retained tail begins on a clean
// boundary — a user message, or an assistant message without tool calls.
// resolveGroup binds a tool result back to its assistant, so a budget or age cut
// can leave the tail starting mid-group; without this the provider sanitizer
// would drop that leading group on every single dispatch (and the context would
// be neither kept nor summarized). Advancing the index rather than trimming the
// returned slice hands those messages to the summary, where they belong.
func advancePastPartialToolGroup(stored []memory.StoredMessage, start int) int {
	for start < len(stored) {
		r := stored[start].Role
		if r == "tool" || (r == "assistant" && len(stored[start].ToolCalls) > 0) {
			start++
			continue
		}
		break
	}
	return start
}

type groupBounds struct{ start, end int }

// resolveGroup returns the index bounds of the atomic turn group ending at end.
// If history[end] has a ToolCallID, the group extends back to the assistant
// message whose ToolCalls slice contains a matching ID.
// If no match is found, the group is just {end, end}.
func resolveGroup(history []spawnllm.Message, end int) groupBounds {
	id := history[end].ToolCallID
	if id == "" {
		return groupBounds{end, end}
	}
	for j := end - 1; j >= 0; j-- {
		for _, tc := range history[j].ToolCalls {
			if tc.ID == id {
				return groupBounds{j, end}
			}
		}
	}
	return groupBounds{end, end}
}

// countMeaningfulMessages counts non-noise messages in a slice: a message that
// repeats the one immediately before it (see isTailNoise) is not counted.
func countMeaningfulMessages(msgs []spawnllm.Message, noise NoiseKeyFunc) int {
	n := 0
	var prev *spawnllm.Message
	for i := range msgs {
		if isTailNoise(msgs[i], prev, noise) {
			continue
		}
		n++
		prev = &msgs[i]
	}
	return n
}

// collapseStoredNoise removes consecutive noise repeats, keeping one message
// of each run of repeats (see isTailNoise): the first, unless the run holds
// the newest message or the newest user message, which then takes its place.
// A repeat is judged against the previous kept message only, so anything in
// between (an assistant reply to a scheduled fire, a different job, a tool
// exchange) ends the run: two fires of one job with a reply between them are
// both kept.
//
// The newest message and the newest user message are never removed, whatever
// they repeat. The newest user message is the request the next dispatch
// answers; collapsing it as a duplicate of an earlier fire leaves the model
// with nothing to respond to.
func collapseStoredNoise(msgs []memory.StoredMessage, noise NoiseKeyFunc) []memory.StoredMessage {
	if len(msgs) == 0 {
		return msgs
	}
	newest := len(msgs) - 1
	newestUser := lastUserStoredIndex(msgs)
	out := make([]memory.StoredMessage, 0, len(msgs))
	var prev *spawnllm.Message
	for i := range msgs {
		if isTailNoise(msgs[i].Message, prev, noise) {
			if i != newest && i != newestUser {
				continue
			}
			// The newest (or newest user) message repeats the kept one before
			// it: keep it in that one's place, so the run still collapses to
			// one message and that message is the current request.
			out = out[:len(out)-1]
		}
		out = append(out, msgs[i])
		prev = &msgs[i].Message
	}
	return out
}

// isTailNoise reports whether m repeats prev, the message kept immediately
// before it (nil when there is none). A repeat has the same role as prev and
// either the same noise key (repeated fires of one source, whose text differs
// by timestamp) or, when neither is keyed, identical content. Only an adjacent
// repeat counts: noise collapse folds a burst of identical messages, it does
// not deduplicate a conversation, where a scheduled job firing again after a
// reply is a new request.
//
// Tool plumbing is never noise, whatever its text, and never anchors a repeat.
// An assistant message that makes a tool call carries empty Content, so a run
// of them looks like a run of identical messages to the content comparison
// below — collapsing one drops the tool_calls it declared and orphans the tool
// results that follow, which strict providers reject outright ("Messages with
// role 'tool' must be a response to a preceding message with 'tool_calls'").
// Tool results are excluded for the mirror reason: two calls to one tool can
// legitimately return the same text, and dropping the second breaks the
// assistant message that expects it.
//
// Noise collapse exists for repeated conversational text — scheduled-job
// wrappers, a user sending the same thing twice — not for structural messages.
func isTailNoise(m spawnllm.Message, prev *spawnllm.Message, noise NoiseKeyFunc) bool {
	if isToolPlumbing(m) || prev == nil || isToolPlumbing(*prev) || m.Role != prev.Role {
		return false
	}
	key, keyed := noise(m.Content)
	prevKey, prevKeyed := noise(prev.Content)
	if keyed || prevKeyed {
		return keyed && prevKeyed && key != "" && key == prevKey
	}
	return m.Content == prev.Content
}

// isToolPlumbing reports whether m is part of a tool exchange: an assistant
// message declaring tool calls, or a tool result.
func isToolPlumbing(m spawnllm.Message) bool {
	return len(m.ToolCalls) > 0 || m.ToolCallID != ""
}
