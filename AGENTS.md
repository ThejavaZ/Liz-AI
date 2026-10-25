# Instructions for AI Agents Working on Liz-AI

> Read this file before touching anything in this repository. It is the
> contract for any AI agent (or human) working on the project.

## Overview

Liz-AI is a modular personal AI assistant for Linux written in Python. It runs
locally on resource-constrained machines, persists everything in SQLite, and
talks to multiple LLM providers (Gemini, OpenAI, Claude, DeepSeek, and local
Ollama models).

## Quick Reference

```bash
source .venv/bin/activate
python main.py                     # run the CLI
python -m pytest tests/ -q         # run the test suite (536 tests, must all pass)
python -m pytest tests/ -v -k name # run one test file or case
```

- Python `>=3.11`, installed deps in `pyproject.toml`.
- Optional extras: `pip install -e ".[web]"` (second search backend),
  `pip install -e ".[dev]"` (pytest).
- Configuration comes from `.env` / environment (`src/liz/config/settings.py`).
  **Never commit or edit the user's `.env`** unless explicitly asked.

## Architecture (Hexagonal / Clean)

Dependency direction is always: `cli` and `infrastructure` → `core`.
The `core` layer never imports from `infrastructure` or `cli`, except through
ports (abstract base classes) it defines itself.

```
src/liz/
├── core/                  # Domain: no I/O framework, no provider SDKs
│   ├── agent.py           # Chat loop, tool rounds, streaming, session switch
│   ├── context.py         # In-memory conversation window + trimming
│   ├── memory.py          # MemoryStore PORT + ConversationMemory policy
│   ├── summarizer.py      # ContextSummarizer (collapses old turns)
│   ├── models.py          # Message, AIResponse, StreamChunk, ToolCall, Role
│   ├── notes.py           # Notes domain port
│   └── voice.py           # STT/TTS ports + VAD and wake word policies
├── config/
│   └── settings.py        # Pydantic settings (env + .env)
├── infrastructure/
│   ├── ai/                # Provider adapters implementing AIProvider port
│   │   ├── base.py        # AIProvider ABC (chat + stream_chat)
│   │   ├── factory.py     # create_provider() from settings
│   │   ├── fallback.py    # FallbackProvider (retries only on retryable errors)
│   │   ├── models_registry.py  # known models, defaults, resolve_models()
│   │   └── gemini|openai|claude|deepseek|ollama.py
│   ├── persistence/       # SQLite adapters (stdlib sqlite3 ONLY)
│   │   ├── memory_store.py    # messages, facts, meta tables
│   │   └── notes_store.py
│   └── voice/             # arecord capture, whisper.cpp STT, TTS engines
├── tools/                 # Tool implementations
│   ├── base.py / models.py  # Tool ABC, ToolResult, ToolStatus
│   ├── registry.py        # ToolRegistry (single source of truth for /tools)
│   ├── permissions.py     # PermissionLayer: ALLOW / DENY / ASK
│   └── filesystem|git|terminal|calculator|web|notes|memory.py
└── cli/
    ├── app.py             # Wiring + main loop + streaming output
    ├── markdown.py        # ANSI Markdown renderer for streamed replies
    ├── voice.py           # VoiceLoop: one turn + background listener thread
    └── commands.py        # Internal `/` commands via CommandResult

prompts/liz.txt            # System prompt loaded at startup
tests/                     # pytest suite
```

## Non-Negotiable Rules

1. **No heavy ML libraries.** Never add `transformers`, `torch`, `nltk`,
   `tensorflow`, `elasticsearch`, `langchain`, etc. Liz targets laptops with
   little VRAM/RAM. Prefer the standard library.
2. **Persistence is stdlib `sqlite3`** in `src/liz/infrastructure/persistence/`.
   Do not introduce an ORM or a database server.
3. **No cloud dependencies** unless the user explicitly requests a new provider
   integration. Respect local privacy: telemetry, tracking or phone-home code
   is forbidden.
4. **New dependencies require justification** (size, maintenance, why stdlib is
   not enough) and user approval before adding them to `pyproject.toml`.
5. **Internal commands must never call the AI provider.** `/help`, `/tools`,
   `/clear`, etc. work offline and are handled in `cli/commands.py`.
6. **Commands must not couple to the Agent's internals** beyond what
   `CommandHandler` already receives; `/tools` must read from `ToolRegistry`
   (never hardcode a tool list) and `/clear` must go through `Context`.
7. **Tests must not hit the network.** Mock providers (`MagicMock(spec=AIProvider)`)
   or use the shared doubles.

## Conventions

- **Hexagonal pattern:** new capability = a *port* (ABC) in `core/` +
  an *adapter* in `infrastructure/`.
- **Typing:** full type hints; `from __future__ import annotations` where
  union syntax needs it. Dataclasses for domain values, Pydantic for config.
- **Strings:** user-facing CLI output may be Spanish or English, match the
  surrounding file.
- **Tool results:** return `ToolResult.success(...)` / `ToolResult.error(...)`;
  never raise through the agent loop.
- **Streaming:** providers yield `StreamChunk(text=..., tool_calls=...,
  tool_events=..., done=..., response=...)`. Never buffer a whole reply.
  `stream_chat()` may fall back to a single chunk, but **never switch provider
  mid-stream after text has already been emitted** (it would duplicate output).
- **Fallback policy:** `FallbackProvider` only retries retryable failures
  (429/503/404/quota/timeouts), never a 401/400.

## Gotchas (read before refactoring)

- **Tool pairing:** an assistant message with `tool_calls` and its follow-up
  `Role.TOOL` message (matched by `tool_call_id`) must stay adjacent in
  `Context.messages`. Any trimming/summarizing code must not orphan a tool
  result — `Context._trim` and `ContextSummarizer._safe_split` enforce this;
  keep that invariant in any new code.
- **Session identity:** the active session id is persisted in the `meta` table
  (`current_session`). `ConversationMemory.switch()` writes it; recreate the
  memory object to verify it is restored.
- **OpenAI-compatible providers** (OpenAI, DeepSeek, Ollama) share one adapter;
  tool schemas must be sent in the request or the model invents tool output.
  Ollama reuses the OpenAI SDK with `base_url` — do not add a new dependency.
- **Unknown models:** startup warns when `AI_MODEL` is not in
  `models_registry.list_models()` because every call pays a failed request.
- **Voice engines are external binaries.** STT is `whisper.cpp` invoked by
  subprocess (`WHISPER_BIN` / `WHISPER_MODEL` or auto-detected under
  `whisper.cpp/`), TTS is `spd-say` → `espeak-ng` → `espeak`, capture is
  `arecord`. Never add a Python speech/ML dependency. Voice tests synthesize
  PCM and mock subprocesses — they must never touch a real microphone.
- **Web search backends:** the `[web]` extra installs `ddgs` (formerly
  `duckduckgo_search`). `_load_ddgs()` in `tools/web.py` accepts both
  names, and DuckDuckGo's HTML endpoint (POST) is tried first because the
  library returns empty or junk for some queries.

## Testing

- Every change: `python -m pytest tests/ -q` must end in **536 passed**.
- Put new tests in a descriptive `tests/test_<topic>.py`.
- Reuse `tests/memory_helpers.py::InMemoryStore` for store doubles; do not
  write a fourth copy of an in-memory store.
- SQLite tests use `tmp_path` (see `tests/test_phase6.py`), never the real DB.
- Cover: happy path, error path, and the invariant you are protecting.

## Git Workflow

- Conventional commits in English: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`.
- Do not commit without the user's confirmation; never commit `.env`,
  `*.db`, `__pycache__`, or `.venv`.
- Push to `github.com:ThejavaZ/Liz-AI.git` `main` only when asked.

## Roadmap Status

Phases 1–7 are complete (see `README.md` → Roadmap):

1. Foundation ✅ 2. Tools ✅ 3. CLI ✅ 4. Memory ✅
5. Practical Tools ✅ 6. Conversation Quality ✅ 7. Voice ✅

Next: **Phase 8 — Proactivity** (event monitors, scheduled tasks).
After any change, update the corresponding README checkboxes/counts if a
roadmap item was completed.
