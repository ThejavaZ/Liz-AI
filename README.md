# Liz-AI

> Personal AI assistant for Linux, built with Python.

Liz-AI is a personal assistant designed to help with programming, system tasks, organization, and everyday computer interactions.

Built with **modularity, security, and practical AI tooling** in mind.

## Features

- Multi-provider AI support (Gemini, OpenAI, Claude, DeepSeek)
- Tool calling with permission control
- Internal CLI commands (`/help`, `/tools`, `/clear`, `/status`)
- Docker support
- 121+ automated tests

## Quick Start

### Installation

```bash
git clone git@github.com:ThejavaZ/Liz-AI.git
cd Liz-AI
python -m venv .venv
source .venv/bin/activate
pip install -e .
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

| Command  | Description                     |
|----------|---------------------------------|
| `/help`  | Show available commands         |
| `/tools` | List registered tools           |
| `/clear` | Clear conversation history      |
| `/status`| Show provider, model, and state |
| `/exit`  | Exit Liz AI                     |
| `/quit`  | Exit Liz AI                     |

Any input without `/` is sent to the AI provider.

## Supported Providers

| Provider   | Variable          | Default Model       |
|------------|-------------------|---------------------|
| Gemini     | `AI_PROVIDER=gemini` | `gemini-2.0-flash` |
| OpenAI     | `AI_PROVIDER=openai` | `gpt-4o-mini`      |
| Claude     | `AI_PROVIDER=claude` | `claude-sonnet-4-20250514` |
| DeepSeek   | `AI_PROVIDER=deepseek` | `deepseek-chat`  |

Set the provider and model in `.env`:

```env
AI_PROVIDER=openai
AI_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

## Tools

Liz has access to the following tools:

| Tool          | Description                    |
|---------------|--------------------------------|
| `list_directory` | List files in a directory   |
| `read_file`      | Read file contents          |
| `file_exists`    | Check if a file exists      |
| `git_status`     | Show git status             |
| `git_log`        | Show git log                |
| `git_branch`     | List git branches           |
| `run_command`    | Execute a shell command     |

Tools are controlled by a permission layer (ALLOW/DENY/ASK).

## Architecture

```text
CLI → CommandHandler (internal commands)
       ↓
     Agent → AIProvider → Tool Calling → ToolRegistry → PermissionLayer → Tool
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
│   └── models.py       # Domain models
├── infrastructure/
│   └── ai/
│       ├── base.py     # AIProvider interface
│       ├── factory.py  # Provider factory
│       ├── gemini.py   # Google Gemini
│       ├── openai.py   # OpenAI
│       ├── claude.py   # Anthropic Claude
│       └── deepseek.py # DeepSeek
└── tools/
    ├── base.py         # Tool interface
    ├── models.py       # ToolResult, ToolStatus
    ├── registry.py     # ToolRegistry
    ├── permissions.py  # PermissionLayer
    ├── filesystem.py   # File operations
    ├── git.py          # Git operations
    └── terminal.py     # Shell commands
```

## Development

### Run Tests

```bash
python -m pytest tests/ -v
```

### Test Coverage

```
tests/
├── test_context.py          # Context and conversation history
├── test_models.py           # Domain models
├── test_provider_factory.py # Provider selection
├── test_provider_gemini.py  # Gemini integration
├── test_provider_openai.py  # OpenAI integration
├── test_provider_claude.py  # Claude integration
├── test_provider_deepseek.py # DeepSeek integration
├── test_tool_filesystem.py  # File operations
├── test_tool_git.py         # Git operations
├── test_tool_terminal.py    # Terminal commands
├── test_tool_models.py      # Tool result models
├── test_tool_calling.py     # Agent-tool integration
└── test_commands.py         # CLI commands
```

## Roadmap

### Phase 1 - Foundation ✅

- [x] Project structure
- [x] Configuration layer
- [x] Agent structure
- [x] Prompt system
- [x] Basic CLI conversation
- [x] AI provider integration

### Phase 2 - Tools ✅

- [x] Filesystem tools
- [x] Git tools
- [x] Terminal tools
- [x] Permission system
- [x] Tool error handling
- [x] Multi-provider support
- [x] Tool calling integration

### Phase 3 - CLI ✅

- [x] Internal commands
- [x] Docker support
- [ ] Persistent memory
- [ ] Conversation history

### Phase 4 - Voice

- [ ] Speech-to-text
- [ ] Text-to-speech
- [ ] Voice interaction loop

### Phase 5 - System Integration

- [ ] Application launching
- [ ] Process management
- [ ] System information
- [ ] Desktop integration

## Security

- AI commands require explicit permission
- Tool execution controlled by PermissionLayer
- No unrestricted system access
- User confirmation for sensitive operations

## License

See [`LICENSE`](LICENSE) for details.
