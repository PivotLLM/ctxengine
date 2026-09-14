# ctxengine — context engine for LLM agents

The part of an agent that decides what the model sees. Given a stream of
messages and a token budget, it keeps the transcript, assembles each request,
evicts stale tool results, and summarises older history into a rolling summary
when the window fills — with the whole conversation kept on disk and
searchable.

Extracted from [ClawEh](https://github.com/PivotLLM/ClawEh), where it was
designed, hardened against production logs and first shipped.

## What it owns

- **The transcript.** Every message is ingested once and given a monotonic
  sequence number. That number is what everything else cites: summaries,
  the session tools, and any memory system that derives evidence from the
  conversation.
- **The live window and the archive.** A per-session SQLite archive with
  full-text search holds everything; a live window holds what the model is
  currently shown.
- **Assembly.** One `Assemble` call per model dispatch: the host's prompt
  layers, the rendered summary, the recent history, and any injections the
  host gathered (a memory system's blocks, say), each placed where the host
  says — stable content in the system prefix, per-turn content on the latest
  user message — so prompt caching keeps working.
- **Eviction.** An LLM-free per-turn sweep that collapses superseded or
  oversized re-retrievable tool results (file reads, web fetches) to short
  placeholders.
- **Compaction.** Threshold-, count- and age-triggered summarisation with
  evidence-cited, validated summaries, a safety net when the built request is
  still too large, per-model refusal memory, a failure circuit breaker, and
  durable summary checkpoints.
- **Tools.** `session_messages`, `session_search`, `session_summary_list`,
  `session_summary_get`, `session_info`, `session_compact`, `session_clear`,
  as [toolspec](https://github.com/PivotLLM/toolspec) definitions over a `Host`
  struct.

## What it does not own

- **Models.** It never calls a provider. Compaction goes through a
  `ModelCaller` the host supplies:

  ```go
  type ModelRequest struct {
      System, User string
      JSONObject   bool     // ask for a JSON-object response where supported
      Exclude      []string // models the engine has learned to avoid
  }
  type ModelReply struct{ Content, FinishReason, Model string }
  type ModelCaller interface {
      Complete(ctx context.Context, req ModelRequest) (ModelReply, error)
  }
  ```

  The host owns which model answers, credentials, transport, retries on
  transport errors and cooldowns, and honours `Exclude`. The engine owns its
  output: it classifies refusals, remembers refusing models per session, and
  calls again with a longer `Exclude`. The same shape serves
  [cogmem](https://github.com/PivotLLM/cogmem), so one host client and one
  test fake serve both.
- **The system prompt.** The host passes it as ordered layers; the engine
  only inserts its summary between the layers marked before and after it.
- **Memory.** Nothing here knows what a memory system is. The host asks its
  memory for blocks and hands them to `Assemble` as injections with a
  placement.
- **Logging, failure dumps, refusal rules, message envelopes.** All injected:
  `logger.SetBackend`, `WithFailureDump`, `WithRefusalClassifier`,
  `WithNoiseKey` (how repeated scheduled messages are recognised and
  collapsed without the engine knowing what a scheduler is).

## Host obligations

- Install a logging backend with `logger.SetBackend` (silent otherwise).
- Provide a `ModelCaller` for compaction.
- Call `AddUserMessage`/`AddAssistantMessage`/`AddToolCallMessage`/
  `AddToolResult` as the conversation happens, and `Assemble` before every
  model call.
- Give the tools a `Host` (sessions directory and the compact, clear and
  info closures) if the model should be able to reach the archive.

## Copyright and license

Copyright (c) 2026 Tenebris Technologies Inc.

This software is licensed under the MIT License. Please see LICENSE for details.
