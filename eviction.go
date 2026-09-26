// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/PivotLLM/ctxengine/logger"
	"github.com/PivotLLM/ctxengine/memory"
	"github.com/PivotLLM/spawnllm"
)

// EvictionPolicy controls the per-turn, LLM-free eviction sweep that collapses
// re-retrievable tool results (file reads, web fetches) in the live window to a
// short placeholder so the window rarely fills enough to trigger summarization
// compaction. Eviction is reversible: the agent re-reads the resource from disk
// or re-fetches it when the placeholder tells it to, so the only cost of an
// over-eager eviction is a re-read, never lost work.
//
// Turn age is 1-based: the newest turn group is age 1.
//
//   - Superseded & age > ProtectTurns                     → evicted (stale duplicate).
//   - Superseded & age <= ProtectTurns                    → kept, but a budget
//     (a later read of the SAME slice, or a later            candidate (reclaimed
//     SUCCESSFUL write to the file)                          under memory pressure).
//   - Not superseded & age <= ProtectTurns                → never evicted.
//   - Not superseded & age > EvictTurns                   → evicted (any size).
//   - Budget valve: if reader-result bytes exceed         → evict superseded-first,
//     BudgetBytes, evict candidates                          then largest-first.
//
// Supersession respects the protect window: a read the model touched in the last
// ProtectTurns is part of the active working set and is kept even when superseded
// (so it can read a file once and make several edits), but it becomes a budget
// candidate so a suddenly-bloated context can still reclaim it. Two refinements
// keep supersession honest: it is range-aware (reading a different page of a large
// file does not evict the earlier page — see readerSliceKey), and only a
// *successful* write supersedes (a failed edit leaves the file unchanged, so it
// must not evict the read the model needs). Beyond the window, superseded reads
// and reads older than EvictTurns are dropped, and the budget valve sheds the
// rest — so nothing becomes immortal.
type EvictionPolicy struct {
	Enabled      bool
	ProtectTurns int
	EvictTurns   int
	BudgetBytes  int  // 0 => derived from contextWindow (~40%)
	NotifyUser   bool // emit a one-line in-conversation notice per eviction

	// ArgBytes is the size at which a tool-call ARGUMENT becomes evictable once
	// it is older than EvictTurns. Arguments live in the assistant message's
	// ToolCalls, not in a tool result, so the reader sweep never touched them —
	// yet a file_write payload is counted in full by the token estimator and can
	// dominate the window. 0 disables argument eviction.
	ArgBytes int

	// MinSweepPercent batches evictions for the prompt cache. Every eviction
	// rewrites an older message in place, which invalidates the provider's
	// cached prefix from that message on, so the per-dispatch sweep applies
	// its evictions only when together they free at least this percentage of
	// the context window (in bytes, via chars-per-token); otherwise they wait
	// and accumulate. A compaction pass and an explicit SweepEvictions apply
	// them regardless — compaction rewrites the window anyway. 0 applies every
	// sweep at once.
	MinSweepPercent int
}

// DefaultEvictionPolicy returns the built-in defaults: enabled, protect the last
// 3 turns, evict reads older than 10 turns, with the budget valve shedding the
// largest reads under memory pressure, and evictions batched until they free
// 5% of the window.
func DefaultEvictionPolicy() EvictionPolicy {
	return EvictionPolicy{
		Enabled:         true,
		ProtectTurns:    3,
		EvictTurns:      10,
		BudgetBytes:     0,
		NotifyUser:      false,
		ArgBytes:        defaultEvictionArgBytes,
		MinSweepPercent: defaultEvictionMinSweepPercent,
	}
}

// defaultEvictionMinSweepPercent is the share of the context window a
// per-dispatch sweep must free before its evictions are applied.
const defaultEvictionMinSweepPercent = 5

// defaultEvictionArgBytes is the threshold above which an aged-out tool-call
// argument is replaced by a placeholder. Set well above ordinary arguments
// (paths, queries, flags) so only genuine payloads — the body of a file_write,
// the replacement text of an edit — are ever touched.
const defaultEvictionArgBytes = 1024

// EvictionEvent records a single tool-result eviction performed by the sweep.
// It is returned to the agent loop for optional in-conversation notification and
// is always written to the DEBUG log by the sweep itself.
type EvictionEvent struct {
	Seq      int64
	Tool     string // originating tool, e.g. "file_read_bytes"
	Resource string // path or URL the tool read
	Bytes    int    // original content size freed
	AgeTurns int    // 1-based turn-group age (1 = newest)
	Reason   string // "superseded" | "stale" | "large" | "budget"
}

// evictionResourceCap caps the resource string in the user-facing notice so a
// very long URL does not flood the conversation.
const evictionResourceCap = 96

// String renders the one-line in-conversation notice:
//
//	[Evicted 18432 bytes at 6 turns (large): file_read_bytes files/novels/ch17.md]
func (e EvictionEvent) String() string {
	reason := ""
	if e.Reason != "" {
		reason = " (" + e.Reason + ")"
	}
	return fmt.Sprintf("[Evicted %d bytes at %d turns%s: %s %s]",
		e.Bytes, e.AgeTurns, reason, e.Tool, capResource(e.Resource, evictionResourceCap))
}

func capResource(s string, max int) string {
	if len(s) <= max || max < 2 {
		return s
	}
	return s[:max-1] + "…"
}

// evictionMarker prefixes the placeholder left in place of evicted content so the
// sweep is idempotent (already-evicted results are skipped) and the LLM is told
// how to recover the content.
const evictionMarker = "[evicted:"

// sessionMessagesTool is the name the host publishes the session history tool
// under (the tools package exports it bare as "messages"; ClawEh mounts it as
// session_messages). Named in placeholders so the model knows how to get the
// exact content back.
const sessionMessagesTool = "session_messages"

// evictionPlaceholder is the text left in place of an evicted tool result. It
// names the tool and resource so the model can re-read, and the archive seq
// so the exact content can be pulled back with the session messages tool.
func evictionPlaceholder(tool, resource string, bytes int, seq int64) string {
	return fmt.Sprintf("%s %s %s (%d bytes, seq #%d) — content evicted to save context; re-read it, or fetch seq #%d with %s]",
		evictionMarker, tool, resource, bytes, seq, seq, sessionMessagesTool)
}

func isEvicted(content string) bool {
	return strings.HasPrefix(content, evictionMarker)
}

// evictionArgPlaceholder replaces the value of an oversized tool-call argument.
// It names the tool, the argument, and the resource the call acted on so the
// model can recover the content the same way it recovers an evicted read: by
// going back to the file. Kept short — the point is to free the window.
func evictionArgPlaceholder(tool, arg, resource string, bytes int) string {
	where := resource
	if where == "" {
		where = "(unknown)"
	}
	return fmt.Sprintf("%s %s.%s for %s (%d bytes) — argument evicted to save context; re-read the resource if you need it again]",
		evictionMarker, tool, arg, capResource(where, evictionResourceCap), bytes)
}

// argEviction records one oversized tool-call argument replaced by the sweep.
type argEviction struct {
	callIdx  int // index into the message's ToolCalls
	arg      string
	tool     string
	resource string
	bytes    int
}

// evictLargeArgs rewrites every argument in tc larger than minBytes to a
// placeholder, returning the evictions performed. It is deliberately keyed on
// SIZE rather than on a per-tool map of payload argument names: the tools that
// bloat the window most are not all native (an MCP document tool's body argument
// costs exactly as much as file_write's content), and a size rule covers every
// writer, present and future, without a registry to keep in sync. Small
// identifying arguments — path, url, query, flags — fall far below the threshold
// and are never touched, so the placeholder can still say what the call acted on.
//
// The rewrite goes through a parsed map and is re-marshalled, so the stored
// arguments remain valid JSON; a provider replaying the message sees a
// well-formed (if abbreviated) tool call rather than a truncated string.
func evictLargeArgs(tc *spawnllm.ToolCall, minBytes int, roles EvictionRoles) []argEviction {
	if minBytes <= 0 || tc.Function == nil {
		return nil
	}
	name, args := normToolCall(*tc)
	if len(args) == 0 {
		return nil
	}
	resource, _ := roles.writerResource(name, args)
	if resource == "" {
		resource, _ = roles.readerResource(name, args)
	}

	var out []argEviction
	for _, key := range sortedArgKeys(args) {
		v, ok := args[key].(string)
		if !ok || len(v) <= minBytes || isEvicted(v) {
			continue
		}
		out = append(out, argEviction{arg: key, tool: name, resource: resource, bytes: len(v)})
		args[key] = evictionArgPlaceholder(name, key, resource, len(v))
	}
	if len(out) == 0 {
		return nil
	}

	data, err := json.Marshal(args)
	if err != nil {
		return nil // leave the call untouched rather than persist something broken
	}
	tc.Function.Arguments = string(data)
	// The runtime Arguments map is json:"-" and so is not persisted, but it is
	// read by normToolCall in preference to the serialized form — keep the two in
	// step or the next sweep would see the original size and re-evict forever.
	if tc.Arguments != nil {
		tc.Arguments = args
	}
	return out
}

// sortedArgKeys returns the argument names in a stable order so a sweep produces
// deterministic events regardless of Go's map iteration order.
func sortedArgKeys(args map[string]any) []string {
	keys := make([]string, 0, len(args))
	for k := range args {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// ReaderRole describes a tool whose result can be fetched again — a file read,
// a web fetch, a search — so the sweep may replace an old result with a
// placeholder and the model can re-run the tool if it needs the content.
type ReaderRole struct {
	// ResourceArg names the argument that says what was read (a path, a URL).
	// Required: a result whose call has no resource is never evicted.
	ResourceArg string
	// ResourceDefault is the resource implied when ResourceArg is absent or
	// empty — "." for a search tool that defaults to the workspace root.
	// Without it a repository-wide search, typically the largest result,
	// would be the one result never evicted.
	ResourceDefault string
	// KeyArg names a string argument that distinguishes two reads of one
	// resource — a search query. Two searches of one tree with different
	// queries return different content, so keying on the bare path would make
	// the second silently supersede the first.
	KeyArg string
	// RangeArg names a numeric argument giving the start of the slice read —
	// start_line, offset. Reads of one file at different starts are distinct
	// pages, not duplicates, so read-vs-read supersession keys on
	// resource+start; otherwise reading page 2 would evict page 1.
	RangeArg string
	// RangeDefault is the slice start implied when RangeArg is absent.
	RangeDefault int64
}

// WriterRole describes a tool that mutates a resource. A successful call
// supersedes every earlier read of that resource, whatever the page.
type WriterRole struct {
	// ResourceArg names the argument that says what was written. For a move
	// it is the source: reads of the moved-away file are now stale.
	ResourceArg string
}

// EvictionRoles maps tool names to their eviction roles. Names are matched
// after any "mcp__server__" prefix is stripped, so a host lists the bare
// names it publishes. DefaultEvictionRoles covers the ClawEh file and web
// tools; a host with other tools supplies its own with WithEvictionRoles,
// usually starting from the defaults.
type EvictionRoles struct {
	Readers map[string]ReaderRole
	Writers map[string]WriterRole
}

// DefaultEvictionRoles returns the built-in roles: the file_* readers and
// writers and web_fetch.
func DefaultEvictionRoles() EvictionRoles {
	return EvictionRoles{
		Readers: map[string]ReaderRole{
			"file_read_bytes":   {ResourceArg: "path", RangeArg: "offset", RangeDefault: 0},
			"file_read_lines":   {ResourceArg: "path", RangeArg: "start_line", RangeDefault: 1},
			"file_list":         {ResourceArg: "path"},
			"file_search_lines": {ResourceArg: "path", ResourceDefault: ".", KeyArg: "query"},
			"file_search_bytes": {ResourceArg: "path", ResourceDefault: ".", KeyArg: "query"},
			"web_fetch":         {ResourceArg: "url"},
		},
		Writers: map[string]WriterRole{
			"file_write":        {ResourceArg: "path"},
			"file_edit":         {ResourceArg: "path"},
			"file_append":       {ResourceArg: "path"},
			"file_copy":         {ResourceArg: "destination_path"},
			"file_edit_lines":   {ResourceArg: "path"},
			"file_edit_bytes":   {ResourceArg: "path"},
			"file_insert_lines": {ResourceArg: "path"},
			"file_insert_bytes": {ResourceArg: "path"},
			"file_delete_lines": {ResourceArg: "path"},
			"file_delete_bytes": {ResourceArg: "path"},
			"file_delete":       {ResourceArg: "path"},
			"file_move":         {ResourceArg: "source_path"},
		},
	}
}

type toolMeta struct {
	tool string
	args map[string]any
}

// normToolCall extracts the (normalized) tool name and argument map from a stored
// ToolCall. Stored history only persists Function.Name/Function.Arguments (the
// runtime Name/Arguments fields are json:"-"), so fall back to those.
func normToolCall(tc spawnllm.ToolCall) (string, map[string]any) {
	name := tc.Name
	if name == "" && tc.Function != nil {
		name = tc.Function.Name
	}
	name = normToolName(name)

	args := tc.Arguments
	if args == nil && tc.Function != nil && strings.TrimSpace(tc.Function.Arguments) != "" {
		_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
	}
	return name, args
}

// normToolName strips any "mcp__server__" namespace prefix so native and
// MCP-published tool names compare equal.
func normToolName(name string) string {
	if i := strings.LastIndex(name, "__"); i >= 0 {
		return name[i+2:]
	}
	return name
}

// readerResource returns the resource a reader tool call read, and whether
// tool is a reader with a resource at all.
func (r EvictionRoles) readerResource(tool string, args map[string]any) (string, bool) {
	role, ok := r.Readers[tool]
	if !ok || args == nil || role.ResourceArg == "" {
		return "", false
	}
	v, _ := args[role.ResourceArg].(string)
	v = strings.TrimSpace(v)
	if v == "" {
		v = role.ResourceDefault
	}
	return v, v != ""
}

// writerResource returns the resource a writer tool call mutated, and whether
// tool is a writer with a resource at all.
func (r EvictionRoles) writerResource(tool string, args map[string]any) (string, bool) {
	role, ok := r.Writers[tool]
	if !ok || args == nil || role.ResourceArg == "" {
		return "", false
	}
	v, _ := args[role.ResourceArg].(string)
	v = strings.TrimSpace(v)
	return v, v != ""
}

// readerSliceKey returns the read-vs-read supersession key: the resource plus
// the KeyArg value or, for paginated readers, the slice start — so distinct
// pages of one file coexist and only a re-read of the SAME page supersedes.
// Readers with neither key on the bare resource. Write-invalidation still keys
// on the bare resource (any edit invalidates every cached page of that file).
func (r EvictionRoles) readerSliceKey(tool string, args map[string]any) (string, bool) {
	res, ok := r.readerResource(tool, args)
	if !ok {
		return "", false
	}
	role := r.Readers[tool]
	if role.KeyArg != "" {
		discriminator, _ := args[role.KeyArg].(string)
		return res + "#" + discriminator, true
	}
	if role.RangeArg == "" {
		return res, true
	}
	start := role.RangeDefault
	if v, present := args[role.RangeArg]; present {
		start = toInt64(v)
	}
	return fmt.Sprintf("%s#%d", res, start), true
}

// toInt64 coerces a JSON-decoded numeric arg (float64 by default) to int64.
func toInt64(v any) int64 {
	switch n := v.(type) {
	case float64:
		return int64(n)
	case int64:
		return n
	case int:
		return int64(n)
	case json.Number:
		i, _ := n.Int64()
		return i
	case string:
		i, _ := strconv.ParseInt(strings.TrimSpace(n), 10, 64)
		return i
	}
	return 0
}

// evictionBudgetBytes resolves the reader-bytes budget: the configured value, or
// ~40% of the context window (converted tokens→bytes) when unset.
func (m *Manager) evictionBudgetBytes() int {
	if m.cfg.eviction.BudgetBytes > 0 {
		return m.cfg.eviction.BudgetBytes
	}
	if m.cfg.contextWindow <= 0 {
		return 0
	}
	cpt := m.cfg.charsPerToken
	if cpt <= 0 {
		cpt = defaultCharsPerToken
	}
	return int(float64(m.cfg.contextWindow) * cpt * 0.40)
}

// evictionMinSweepBytes resolves the batch threshold: MinSweepPercent of the
// context window in bytes, or 0 (apply every sweep) when either is unset.
func (m *Manager) evictionMinSweepBytes() int {
	p := m.cfg.eviction
	if p.MinSweepPercent <= 0 || m.cfg.contextWindow <= 0 {
		return 0
	}
	cpt := m.cfg.charsPerToken
	if cpt <= 0 {
		cpt = defaultCharsPerToken
	}
	return int(float64(m.cfg.contextWindow) * cpt * float64(p.MinSweepPercent) / 100)
}

// sweepEvictions runs the eviction pass over the live window. It is LLM-free,
// idempotent, and rewrites the stored history in place (preserving seqs) so
// the saving persists across turns. It returns the evictions performed (order
// follows history position) for optional in-conversation notification; every
// eviction is also DEBUG-logged.
//
// Unless force is set, the evictions are applied only when together they free
// at least MinSweepPercent of the window; below that nothing is written and
// nil is returned, so the cached prefix survives until a batch is worth the
// invalidation. Compaction and an explicit SweepEvictions pass force.
//
// Best-effort: on an empty window or a disabled policy it returns nil and makes
// no changes.
func (m *Manager) sweepEvictions(_ context.Context, force bool) []EvictionEvent {
	p := m.cfg.eviction
	if !p.Enabled {
		return nil
	}
	roles := m.cfg.evictionRoles
	stored := m.store.GetHistoryWithSeqs(m.sessionKey)
	if len(stored) == 0 {
		return nil
	}

	msgs := make([]spawnllm.Message, len(stored))
	toolErr := make([]bool, len(stored))
	for i := range stored {
		msgs[i] = stored[i].Message
		// A failed tool result is marked with Message.Type (set by the agent loop);
		// used below so a failed write does not supersede the read it needs.
		toolErr[i] = stored[i].Type == MessageTypeToolError
	}

	// 1-based turn-group age per message index (newest group = 1).
	ages := make([]int, len(msgs))
	age := 1
	for i := len(msgs) - 1; i >= 0; {
		g := resolveGroup(msgs, i)
		for j := g.start; j <= g.end; j++ {
			ages[j] = age
		}
		age++
		i = g.start - 1
	}

	// Map tool-call ID → (tool, args); track the latest read and write index per
	// resource for supersession. Assistant tool calls precede their results, so
	// callByID is populated before a result references it.
	callByID := make(map[string]toolMeta, len(msgs))
	maxReadIdx := map[string]int{}  // keyed by slice (path#start) for read-vs-read
	maxWriteIdx := map[string]int{} // keyed by path for write-invalidation
	errByCallID := make(map[string]bool)
	type pendingWrite struct {
		res    string
		idx    int
		callID string
	}
	var writes []pendingWrite
	for idx, mm := range msgs {
		for _, tc := range mm.ToolCalls {
			name, args := normToolCall(tc)
			callByID[tc.ID] = toolMeta{tool: name, args: args}
			if res, ok := roles.writerResource(name, args); ok {
				// Defer recording the write until its result's error status is known.
				writes = append(writes, pendingWrite{res: res, idx: idx, callID: tc.ID})
			}
		}
		if mm.Role == "tool" && mm.ToolCallID != "" {
			errByCallID[mm.ToolCallID] = toolErr[idx]
			meta := callByID[mm.ToolCallID]
			if key, ok := roles.readerSliceKey(meta.tool, meta.args); ok {
				if cur, k := maxReadIdx[key]; !k || idx > cur {
					maxReadIdx[key] = idx
				}
			}
		}
	}
	// A write supersedes earlier reads of the file only if it actually modified it.
	// A failed edit (e.g. "old_text not found") leaves the file unchanged, so it
	// must not evict the read the model needs to build a correct edit — that is
	// exactly the loop this guards against.
	for _, wr := range writes {
		if errByCallID[wr.callID] {
			continue // failed write: no supersession
		}
		if cur, k := maxWriteIdx[wr.res]; !k || wr.idx > cur {
			maxWriteIdx[wr.res] = wr.idx
		}
	}

	// Classify each reader tool-result message.
	type cand struct {
		idx, size  int
		superseded bool
	}
	marks := map[int]string{}
	var budgetCands []cand
	remainingReaderBytes := 0

	for idx, mm := range msgs {
		if mm.Role != "tool" || mm.ToolCallID == "" || isEvicted(mm.Content) {
			continue
		}
		meta := callByID[mm.ToolCallID]
		res, ok := roles.readerResource(meta.tool, meta.args)
		if !ok {
			continue // not a re-retrievable reader → never evicted
		}
		sliceKey, _ := roles.readerSliceKey(meta.tool, meta.args)
		size := len(mm.Content)
		if size == 0 {
			continue
		}

		// Superseded = a strictly-later read of the SAME slice (a genuine re-read of
		// that page), or a later SUCCESSFUL write to the file. Range-aware: reading
		// a different page of the same file is not a duplicate, and a failed write
		// does not count (see maxWriteIdx above).
		superseded := false
		if r, k := maxReadIdx[sliceKey]; k && r > idx {
			superseded = true
		}
		if w, k := maxWriteIdx[res]; k && w > idx {
			superseded = true
		}

		a := ages[idx]
		if superseded {
			// Past the working-set window a superseded read is a pure stale
			// duplicate — drop it now. Within the window keep it (the model may
			// still be editing against it), but demote it to a budget candidate so a
			// suddenly-bloated context can still reclaim it under memory pressure.
			if a > p.ProtectTurns {
				marks[idx] = "superseded"
				continue
			}
			budgetCands = append(budgetCands, cand{idx: idx, size: size, superseded: true})
			remainingReaderBytes += size
			continue
		}

		// Non-superseded reads respect the protect window (the recent working set).
		if a <= p.ProtectTurns {
			remainingReaderBytes += size // protected: stays in window
			continue
		}
		if a > p.EvictTurns {
			marks[idx] = "stale"
			continue
		}
		budgetCands = append(budgetCands, cand{idx: idx, size: size})
		remainingReaderBytes += size // candidate stays unless budget evicts it
	}

	// Budget valve: when reader bytes exceed budget, evict superseded duplicates
	// first (least valuable), then largest-first, until back under budget.
	if budget := m.evictionBudgetBytes(); budget > 0 && remainingReaderBytes > budget {
		sort.Slice(budgetCands, func(i, j int) bool {
			if budgetCands[i].superseded != budgetCands[j].superseded {
				return budgetCands[i].superseded
			}
			return budgetCands[i].size > budgetCands[j].size
		})
		for _, c := range budgetCands {
			if remainingReaderBytes <= budget {
				break
			}
			if c.superseded {
				marks[c.idx] = "superseded"
			} else {
				marks[c.idx] = "budget"
			}
			remainingReaderBytes -= c.size
		}
	}

	// Tool-call ARGUMENT eviction. Independent of the reader classification
	// above: arguments live on the assistant message, are counted in full by the
	// token estimator, and have no result message to collapse. Age is the only
	// gate — a write's content is recoverable from disk the moment it succeeds,
	// so anything past EvictTurns is safe to drop. Deliberately outside the
	// budget valve, which accounts for reader bytes only.
	argMarks := make(map[int][]argEviction)
	if p.ArgBytes > 0 {
		for idx := range msgs {
			if ages[idx] <= p.EvictTurns || len(stored[idx].ToolCalls) == 0 {
				continue
			}
			for c := range stored[idx].ToolCalls {
				evicted := evictLargeArgs(&stored[idx].ToolCalls[c], p.ArgBytes, roles)
				for _, e := range evicted {
					e.callIdx = c
					argMarks[idx] = append(argMarks[idx], e)
				}
			}
		}
	}

	if len(marks) == 0 && len(argMarks) == 0 {
		return nil
	}

	// Batch: below the threshold the sweep leaves the window (and so the
	// cached prefix) untouched; the candidates are still there next time.
	// stored is a copy, so the argument rewrites above are simply discarded.
	if !force {
		freed := 0
		for idx := range marks {
			freed += len(msgs[idx].Content)
		}
		for _, aes := range argMarks {
			for _, ae := range aes {
				freed += ae.bytes
			}
		}
		if minBytes := m.evictionMinSweepBytes(); minBytes > 0 && freed < minBytes {
			logger.DebugCF("llmcontext", "eviction deferred: below the batch threshold", map[string]any{
				"session_key": m.sessionKey,
				"candidates":  len(marks) + len(argMarks),
				"bytes":       freed,
				"threshold":   minBytes,
			})
			return nil
		}
	}

	// Apply in history order so events and the rewritten content are deterministic.
	events := make([]EvictionEvent, 0, len(marks)+len(argMarks))
	for idx := range msgs {
		for _, ae := range argMarks[idx] {
			ev := EvictionEvent{
				Seq:      stored[idx].Seq,
				Tool:     ae.tool,
				Resource: ae.resource,
				Bytes:    ae.bytes,
				AgeTurns: ages[idx],
				Reason:   "args",
			}
			events = append(events, ev)
			logger.DebugCF("llmcontext", "evicted tool call argument", map[string]any{
				"session_key": m.sessionKey,
				"seq":         ev.Seq,
				"tool":        ev.Tool,
				"argument":    ae.arg,
				"resource":    ev.Resource,
				"bytes":       ev.Bytes,
				"age_turns":   ev.AgeTurns,
			})
		}
		reason, ok := marks[idx]
		if !ok {
			continue
		}
		mm := stored[idx].Message
		meta := callByID[mm.ToolCallID]
		res, _ := roles.readerResource(meta.tool, meta.args)
		ev := EvictionEvent{
			Seq:      stored[idx].Seq,
			Tool:     meta.tool,
			Resource: res,
			Bytes:    len(mm.Content),
			AgeTurns: ages[idx],
			Reason:   reason,
		}
		stored[idx].Content = evictionPlaceholder(meta.tool, res, ev.Bytes, ev.Seq)
		events = append(events, ev)
		logger.DebugCF("llmcontext", "evicted tool result", map[string]any{
			"session_key": m.sessionKey,
			"seq":         ev.Seq,
			"tool":        ev.Tool,
			"resource":    ev.Resource,
			"bytes":       ev.Bytes,
			"age_turns":   ev.AgeTurns,
			"reason":      ev.Reason,
		})
	}

	// Persist, preserving seqs when the store supports it. A failed write
	// means nothing was evicted, so nothing is reported.
	var err error
	if sh, ok := m.store.(interface {
		SetHistoryWithSeqs(string, []memory.StoredMessage) error
	}); ok {
		err = sh.SetHistoryWithSeqs(m.sessionKey, stored)
	} else {
		plain := make([]spawnllm.Message, len(stored))
		for i := range stored {
			plain[i] = stored[i].Message
		}
		err = m.store.SetHistory(m.sessionKey, plain)
	}
	if err != nil {
		logger.WarnCF("llmcontext", "eviction persist failed; nothing evicted", map[string]any{
			"session_key": m.sessionKey,
			"error":       err.Error(),
		})
		return nil
	}
	if err := m.store.Save(m.sessionKey); err != nil {
		logger.WarnCF("llmcontext", "eviction save failed", map[string]any{
			"session_key": m.sessionKey,
			"error":       err.Error(),
		})
	}
	return events
}

// SweepEvictions runs the eviction pass on its own and applies whatever it
// finds, ignoring the batch threshold: an explicit call is a request to
// reclaim the window now. Assemble runs the batched form of the same pass.
func (m *Manager) SweepEvictions(ctx context.Context) []EvictionEvent {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.sweepEvictions(ctx, true)
}
