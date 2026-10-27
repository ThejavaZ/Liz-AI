# Liz-AI Roadmap — Consolidated reliability plan

> Source: Phase B audit (2026-10-25) + external review (2026-10-26).
> Every item below was verified against the code before being listed;
> links point at the exact files. This issue is the tracking artifact —
> tick items as they land and reference the closing commit.

## Status legend

- [x] done
- [ ] pending
- 🐛 known bug (verified reproduction)

---

## Phase B — Agent reliability (in progress)

### B1. 🐛 Claude has no tool calling at all

`src/liz/infrastructure/ai/claude.py`

- `_build_kwargs` never sends `tools` — the parameter is silently dropped.
- `chat` only extracts text blocks; `tool_use` blocks are discarded.
- `_convert_messages` has no `Role.TOOL` branch — tool results never reach
  the model, and assistant `tool_use` blocks are not replayed.
- Streaming ignores tool events entirely.

Impact: with `AI_PROVIDER=claude`, Liz *looks* functional but no tool ever
executes. Silent failure — the worst kind.

- [ ] send tool schemas in `_build_kwargs`
- [ ] parse `tool_use` blocks into `ToolCall`
- [ ] map `Role.TOOL` → `tool_result` blocks (id preserved)
- [ ] replay assistant `tool_use` blocks on the next round
- [ ] streaming tool support
- [ ] tests (mocked anthropic client, no network)

### B2. 🐛 Gemini tool loop breaks after round one

`src/liz/infrastructure/ai/gemini.py`

- After a tool round the last history message is a `function_response`
  part whose `.text` is `None` (verified with the installed SDK), and
  `_open_chat` passes that `None` as the live message.
- `name="tool_result"` is hard-coded; the API requires the real function
  name to match the `function_call`.
- History replay only sends `from_text` parts — the assistant's
  `function_call` parts disappear from the conversation.
- Tool ids are fabricated as `call_{name}` — two parallel calls to the
  same tool collide on one id.

- [ ] send `function_response` with the real function name
- [ ] keep assistant `function_call` parts in the replayed history
- [ ] never pass `None` as the live message
- [ ] unique tool-call ids (counter or response id)
- [ ] tests (mocked client, no network)

### B3. ✅ Tool crashes and malformed arguments (done)

Commit `393249b` — "fix: never orphan tool calls when a tool crashes or
arguments are malformed".

- `ToolRegistry.execute` turns a raising tool into `ToolResult.error`.
- `Agent._execute_tool` guards non-dict arguments + last-resort try/except.
- `_short_args` tolerates non-dict arguments.
- 14 tests in `tests/test_tool_resilience.py`; invariant: every
  `tool_call` gets exactly one `TOOL` result.

### B4. Minor hardening (from the same audit)

- [ ] OpenAI adapter: invalid JSON arguments become `{}` silently — feed
  an error back to the model instead.
- [ ] Ollama: models without native tool calling degrade silently — warn
  at startup or on first tool round.
- [ ] `MAX_TOOL_ROUNDS` exhaustion discards the last round's calls with
  no feedback — tell the model (or the user) the budget ran out.

---

## Sprint 3 — Permission coherence (partially done)

Done in `b4d2b88`: interactive confirmations for ASK rules + `FS_ROOTS`
path sandbox for filesystem/terminal tools.

- [ ] 🐛 `/notes del <id>` bypasses the permission layer
  (`src/liz/cli/commands.py:200-207` calls `notes_store.delete()`
  directly, skipping the `("notes", "delete") → ASK` rule that the tool
  path enforces in `app.py`). A typo deletes a note with no confirmation.
- [ ] Richer confirmation preview: include cwd, the rule that authorized
  the request and the expected effect (`app.py:67-76` already shows the
  kwargs preview).
- [ ] Sensitive-file guard for read tools (`.env`, `*.db`, keys) — being
  inside a root does not mean every file should be readable.
- [ ] Authorization audit trail (append-only log of granted/denied
  requests) — stdlib, in SQLite.

---

## Sprint 1 — Terminal experience (new)

- [ ] Multiline input: write/paste paragraphs, edit, explicit send.
  Keep `/commands`, `/talk` and streaming behavior working
  (`src/liz/cli/app.py:217` is a bare `input("You: ")` today).
- [ ] Acceptance: pasting a long multi-paragraph message arrives as one
  user message.

---

## Sprint 4 — Contextual memory (new)

Current: facts, notes, sessions, search, summarization all exist.
Missing: deciding *what* to persist.

- [ ] Categories: profile (stable preferences) / session context
  (temporary) / explicit notes.
- [ ] Auto-persist only stable, useful data; casual chat must not become
  facts.
- [ ] Confirmation policy for modifying or deleting important memories.

---

## Sprint 5 — Behavioral evaluation (new)

Opt-in suite (env flag), skipped by default to keep `pytest` offline and
fast. Scenarios from the review, each runnable against any provider:

| Scenario | Expected behavior |
| --- | --- |
| Greeting another AI | Answer the greeting, no unnecessary tools |
| Paste a long message | Treated as one message |
| Execute `pwd` | Ask permission when required, real output |
| Execute `code .` | Explain the blocking rule, invent no causes |
| Tool denied | Acknowledge the action did not complete |
| Ambiguous request | Ask before consequential actions |
| Remember a preference | Store the right fact, invent nothing |

- [ ] harness that runs scenarios with mocked providers by default and
  real providers behind an opt-in flag
- [ ] comparable per-provider reports (e.g. `llama3.1` vs `qwen2.5`)

---

## Later (parked)

- Project profile (detect repo, load its instructions).
- Linux integration (battery, RAM, processes — read-only first).
- Programming-assistance mode (repo analysis, `git diff` review,
  proposed changes shown before applying).
- Voice: interruption, silence handling, canceling a running action.
- Explicitly **not** planned yet: MCP, REST API, web dashboard,
  multi-agent.
