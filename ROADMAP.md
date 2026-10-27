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

### B1. 🐛 Claude has no tool calling at all — ✅ fixed (`a2c0a12`)

`src/liz/infrastructure/ai/claude.py`

- `_build_kwargs` never sent `tools` — the parameter was silently dropped.
- `chat` only extracted text blocks; `tool_use` blocks were discarded.
- `_convert_messages` had no `Role.TOOL` branch — tool results never
  reached the model, and assistant `tool_use` blocks were not replayed.
- Streaming ignored tool events entirely.

Impact: with `AI_PROVIDER=claude`, Liz *looked* functional but no tool ever
executed. Silent failure — the worst kind.

- [x] send tool schemas in `_build_kwargs` (converted to `input_schema`)
- [x] parse `tool_use` blocks into `ToolCall`
- [x] map `Role.TOOL` → `tool_result` blocks (id preserved, roles merged
  alternately as the API requires)
- [x] replay assistant `tool_use` blocks on the next round
- [x] streaming tool support (`content_block_start` + `input_json_delta`)
- [x] tests: 17 in `tests/test_claude_tools.py` (mocked client, no network)

### B2. 🐛 Gemini tool loop breaks after round one — ✅ fixed (`aa662e8`)

`src/liz/infrastructure/ai/gemini.py`

- After a tool round the live message was `contents[-1].parts[0].text`,
  which is `None` once the last entry is a `function_response`.
- `name="tool_result"` was hard-coded; the API requires the real name.
- History replay only sent `from_text` parts — assistant `function_call`
  parts disappeared from the conversation.
- Tool ids were fabricated as `call_{name}` — parallel calls collided.

- [x] send `function_response` with the real function name
- [x] keep assistant `function_call` parts in the replayed history
- [x] never pass `None` as the live message (parts are forwarded)
- [x] unique tool-call ids (`call_<n>_<name>`)
- [x] tests: 13 in `tests/test_gemini_tools.py` (mocked client, no network)

### B3. ✅ Tool crashes and malformed arguments (done)

Commit `393249b` — "fix: never orphan tool calls when a tool crashes or
arguments are malformed".

- `ToolRegistry.execute` turns a raising tool into `ToolResult.error`.
- `Agent._execute_tool` guards non-dict arguments + last-resort try/except.
- `_short_args` tolerates non-dict arguments.
- 14 tests in `tests/test_tool_resilience.py`; invariant: every
  `tool_call` gets exactly one `TOOL` result.

### B4. Minor hardening (from the same audit) — ✅ done

- [x] OpenAI adapter: invalid JSON arguments are no longer silently `{}`.
  `ToolCall` gained an `error` field; `_parse_arguments` reports
  malformed blobs, and the agent turns the error into an error result
  instead of running the tool with fabricated arguments.
- [x] `MAX_TOOL_ROUNDS` exhaustion no longer discards the last round's
  calls: `_final_answer()` asks once without tool schemas, and any
  residual tool call gets an explicit "round limit reached" error result
  (bounded: at most one retry, never unbounded).
- [ ] Ollama: models without native tool calling degrade silently — warn
  at startup or on first tool round. *(deferred: needs a model-support
  table, product decision)*

---

## Sprint 3 — Permission coherence (partially done)

Done in `b4d2b88`: interactive confirmations for ASK rules + `FS_ROOTS`
path sandbox for filesystem/terminal tools.

- [x] 🐛 `/notes del <id>` and `/forget <key>` bypassed the permission
  layer (`commands.py` called the stores directly, skipping the
  `notes:delete → ASK` and `memory:forget → ASK` rules). Both now route
  through `ToolRegistry.execute`, so a destructive CLI action honours the
  same policy as the model-driven tool, including the confirmation
  prompt.
- [ ] Richer confirmation preview: include cwd, the rule that authorized
  the request and the expected effect (`app.py:67-76` already shows the
  kwargs preview).
- [ ] Sensitive-file guard for read tools (`.env`, `*.db`, keys) — being
  inside a root does not mean every file should be readable.
- [ ] Authorization audit trail (append-only log of granted/denied
  requests) — stdlib, in SQLite.

---

## Sprint 1 — Terminal experience (new) — ✅ done

- [x] Multiline input: write/paste paragraphs, edit, explicit send.
  Keep `/commands`, `/talk` and streaming behavior working
  (`src/liz/cli/app.py:217` was a bare `input("You: ")`).
  Implemented in `cli/prompt.py::read_message` with three mechanisms:
  readline + bracketed paste (the paste lands as one editable buffer,
  sent on Enter), `select()` burst coalescing for terminals without
  bracketed paste, and explicit `\`-continuation lines. Ctrl+C cancels
  a draft without leaving the CLI; only single-line messages are
  routed to the command dispatcher (`is_command_message`), so a pasted
  paragraph starting with `/` goes to the model.
- [x] Acceptance: pasting a long multi-paragraph message arrives as one
  user message. (Unit tests in `tests/test_cli_prompt.py` + verified
  end-to-end on a pty.)

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
