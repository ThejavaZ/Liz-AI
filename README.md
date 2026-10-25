# Liz-AI

> Personal AI assistant for Linux, built with Python.

Liz-AI is a personal assistant designed to help with programming, system tasks, organization, and everyday computer interactions.

Built with **modularity, security, and practical AI tooling** in mind.

## Features

- Multi-provider AI support (Gemini, OpenAI, Claude, DeepSeek, Ollama)
- Model fallback chain — a bad model name degrades instead of failing
- Local model support via Ollama (no API key, no quota)
- Tool calling with permission control
- Persistent conversation memory (SQLite)
- Fact storage and session restore
- Persistent notes with search
- Web search without an API key
- Safe calculator (AST-based, never executes code)
- Streaming responses (token by token, all providers)
- Markdown rendering in the terminal (headings, bold, lists, quotes, links, aligned tables, code fences) with ANSI and zero dependencies
- Remember / forget facts (`/remember`, `/forget`, plus a `memory` tool Liz can use herself)
- Multiple named sessions (`/session`) persisted across runs
- Memory search over past messages and facts (`/recall`, also a `memory` tool action)
- Export / import conversation history as JSON (`/export`, `/import`)
- Context summarization when the conversation grows too long
- Health diagnostics (`/doctor`) and config inspection (`/config`)
- Internal CLI commands (`/help`, `/tools`, `/models`, `/memory`, `/notes`, `/config`, `/doctor`, `/clear`, `/status`)
- Docker support
- 456+ automated tests

## Quick Start

### Installation

```bash
git clone git@github.com:ThejavaZ/Liz-AI.git
cd Liz-AI
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e ".[web]"   # optional second web-search backend
pip install -e ".[dev]"   # test tooling
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
AI_PROVIDER=gemini
AI_MODEL=gemini-2.0-flash
GEMINI_API_KEY=your-key-here
```

### Run

```bash
python main.py
```

### Docker

```bash
docker compose up --build
```

## CLI Commands

Internal commands are available without calling the AI provider:

| Command   | Description                                  |
|-----------|----------------------------------------------|
| `/help`   | Show available commands                      |
| `/tools`  | List registered tools                        |
| `/models` | List models for the current provider         |
| `/memory` | Show stored facts and session info           |
| `/remember`| Store a fact: `/remember editor = neovim`    |
| `/forget` | Remove a stored fact                         |
| `/session`| List sessions, or `/session <name>` to switch|
| `/recall` | Search past messages and facts               |
| `/export` | Save history to JSON: `/export [file.json]`  |
| `/import` | Load history from JSON: `/import <file.json>`|
| `/notes`  | Manage notes: `list` / `add` / `del` / `find`|
| `/config` | Show effective configuration (keys masked)   |
| `/doctor` | Run health diagnostics                       |
| `/clear`  | Clear conversation history                   |
| `/status` | Show provider, model, and state              |
| `/exit`   | Exit Liz AI                                  |
| `/quit`   | Exit Liz AI                                  |

Any input without `/` is sent to the AI provider.

## Supported Providers

| Provider | Variable               | Default Model             | API Key           |
|----------|------------------------|---------------------------|-------------------|
| Gemini   | `AI_PROVIDER=gemini`   | `gemini-2.0-flash`        | `GEMINI_API_KEY`  |
| OpenAI   | `AI_PROVIDER=openai`   | `gpt-4o-mini`             | `OPENAI_API_KEY`  |
| Claude   | `AI_PROVIDER=claude`   | `claude-sonnet-4-20250514`| `ANTHROPIC_API_KEY`|
| DeepSeek | `AI_PROVIDER=deepseek` | `deepseek-chat`           | `DEEPSEEK_API_KEY`|
| Ollama   | `AI_PROVIDER=ollama`   | `llama3.1`                | none (local)      |

Set the provider and model in `.env`:

```env
AI_PROVIDER=openai
AI_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

`AI_MODEL` is optional. If omitted, Liz falls back to the provider default
and then through a fallback chain, so a stale or mistyped model name
degrades instead of failing.

### Local models (Ollama)

Run models locally with no API key and no quota:

```bash
ollama pull llama3.1
```

```env
AI_PROVIDER=ollama
AI_MODEL=llama3.1
```

## Tools

Liz has access to the following tools:

| Tool            | Description                                    |
|-----------------|------------------------------------------------|
| `list_directory`| List files in a directory                      |
| `read_file`     | Read file contents                             |
| `file_exists`   | Check if a file exists                         |
| `git_status`    | Show git status                                |
| `git_log`       | Show git log                                   |
| `git_branch`    | List git branches                              |
| `run_command`   | Execute a shell command                        |
| `calculator`    | Evaluate math safely (AST-based, no `eval`)    |
| `web_search`    | Search the web (DuckDuckGo, no API key)        |
| `notes`         | Create, list, search and delete persistent notes|
| `memory`        | Remember, forget or search facts and past conversation |

Tools are controlled by a permission layer (ALLOW/DENY/ASK).

`web_search` uses the `duckduckgo-search` package when installed and
falls back to DuckDuckGo's HTML endpoint otherwise, so it works either
way.

## Architecture

```text
CLI → CommandHandler (internal commands)
       ↓
     Agent → AIProvider (fallback chain) → Tool Calling → ToolRegistry → PermissionLayer → Tool
       ↓
   ConversationMemory → MemoryStore (SQLite)
```

### Project Structure

```text
src/liz/
├── cli/
│   ├── app.py          # CLI entry point
│   └── commands.py     # Internal commands handler
├── config/
│   └── settings.py     # Pydantic settings
├── core/
│   ├── agent.py        # Agent with tool calling loop
│   ├── context.py      # Conversation context
│   ├── memory.py       # Memory port + ConversationMemory
│   ├── notes.py        # Notes port + Note
│   └── models.py       # Domain models
├── infrastructure/
│   ├── ai/
│   │   ├── base.py           # AIProvider interface
│   │   ├── factory.py        # Provider factory
│   │   ├── fallback.py       # Model fallback chain
│   │   ├── models_registry.py# Provider → models map
│   │   ├── gemini.py         # Google Gemini
│   │   ├── openai.py         # OpenAI
│   │   ├── claude.py         # Anthropic Claude
│   │   ├── deepseek.py       # DeepSeek
│   │   └── ollama.py         # Local models via Ollama
│   └── persistence/
│       ├── memory_store.py   # SQLiteMemoryStore
│       └── notes_store.py    # SQLiteNotesStore
└── tools/
    ├── base.py         # Tool interface
    ├── models.py       # ToolResult, ToolStatus
    ├── registry.py     # ToolRegistry
    ├── permissions.py  # PermissionLayer
    ├── filesystem.py   # File operations
    ├── git.py          # Git operations
    ├── terminal.py     # Shell commands
    ├── calculator.py   # Safe math evaluation
    ├── web.py          # Web search
    └── notes.py        # Notes tool
```

## Development

### Run Tests

```bash
python -m pytest tests/ -v
```

### Test Coverage

```
tests/
├── memory_helpers.py           # Shared in-memory MemoryStore double
├── test_context.py            # Context and conversation history
├── test_models.py             # Domain models
├── test_agent.py              # Agent chat loop and tool rounds
├── test_factory.py            # Provider selection
├── test_providers.py          # Provider integrations
├── test_model_flexibility.py  # Model registry and fallback chain
├── test_memory.py             # Memory persistence
├── test_memory_tool.py        # Memory tool (remember/forget/list/recall)
├── test_phase6.py             # Sessions, recall, export/import, summarizer
├── test_markdown.py           # Terminal Markdown renderer
├── test_notes.py              # Notes persistence and tool
├── test_calculator.py         # Safe math evaluation
├── test_web.py                # Web search tool
├── test_streaming.py          # Streaming responses and fallback
├── test_openai_tools.py       # Tool calling for OpenAI-compatible providers
├── test_filesystem.py         # File operations
├── test_git.py                # Git operations
├── test_terminal.py           # Terminal commands
├── test_permissions.py        # Permission layer
├── test_registry.py           # Tool registry
├── test_tool_models.py        # Tool result models
├── test_tool_calling.py       # Agent-tool integration
└── test_commands.py           # CLI commands
```

## Roadmap

### Phase 1 — Foundation ✅

- [x] Project structure
- [x] Configuration layer
- [x] Agent structure
- [x] Prompt system
- [x] Basic CLI conversation
- [x] AI provider integration

### Phase 2 — Tools ✅

- [x] Filesystem tools
- [x] Git tools
- [x] Terminal tools
- [x] Permission system
- [x] Tool error handling
- [x] Multi-provider support
- [x] Tool calling integration

### Phase 3 — CLI ✅

- [x] Internal commands
- [x] Docker support
- [x] Model registry and fallback chain
- [x] Local model support (Ollama)
- [x] `/models` command

### Phase 4 — Memory ✅

- [x] Persistent conversation history (SQLite)
- [x] Fact storage
- [x] Session restore on startup
- [x] `/memory` command
- [x] Persistent notes with search
- [x] `/notes` command

### Phase 5 — Practical Tools ✅

- [x] Safe calculator (AST-based, no `eval`)
- [x] Web search (DuckDuckGo, no API key)
- [x] `/config` effective settings view
- [x] `/doctor` health diagnostics
- [x] Model fallback chain and `/models`

### Phase 6 — Conversation Quality ✅

- [x] Streaming responses (token by token)
- [x] Conversation summarization when context grows
- [x] Multiple named sessions (`/session`)
- [x] Memory search (retrieve relevant past facts with `/recall`)
- [x] `/remember` and `/forget` commands
- [x] Export / import conversation history (`/export`, `/import`)
- [x] Markdown rendering in terminal output (`cli/markdown.py`, stdlib ANSI)

### Phase 7 — Voice

- [ ] Speech-to-text
- [ ] Text-to-speech
- [ ] Voice interaction loop
- [ ] Wake word / activation ("Hey Liz")
- [ ] Voice activity detection
- [ ] Background listening daemon

### Phase 8 — Proactivity

- [ ] Event monitors (battery, network, builds)
- [ ] Scheduled tasks / cron-like routines
- [ ] Proactive notifications
- [ ] Rule engine ("when X happens, do Y")
- [ ] Priority and quiet hours

### Phase 9 — System Integration

- [ ] Application launching
- [ ] Process management
- [ ] System information
- [ ] Desktop notifications
- [ ] Clipboard access
- [ ] Screenshot capture
- [ ] Window management

### Phase 10 — Multimodal

- [ ] Screen reading / vision
- [ ] Image understanding
- [ ] Document parsing (PDF, Office)
- [ ] Web page reading

### Phase 11 — Automation & Workflows

- [ ] Multi-step workflow engine
- [ ] Scriptable routines
- [ ] Task chaining (tool → tool)
- [ ] Undo / rollback for actions
- [ ] Action history and audit log

### Phase 12 — Personalization

- [ ] User preference learning
- [ ] Custom system prompt per user
- [ ] Tone and style adaptation
- [ ] Frequently used tool shortcuts
- [ ] Per-project context profiles

### Phase 13 — Ecosystem

- [ ] Plugin / extension architecture
- [ ] Third-party tool registry
- [ ] MCP (Model Context Protocol) support
- [ ] REST API for external control
- [ ] Web dashboard

### Phase 14 — Reliability & Security

- [ ] Rate limit handling and retry policies
- [ ] Sandboxed command execution
- [ ] Secret redaction in logs
- [ ] Encrypted memory at rest
- [ ] Audit trail for tool calls
- [ ] Graceful offline mode

## Security

- AI commands require explicit permission
- Tool execution controlled by PermissionLayer
- No unrestricted system access
- User confirmation for sensitive operations

## License

See [`LICENSE`](LICENSE) for details.
