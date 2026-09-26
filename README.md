# Liz-AI

> Personal AI assistant for Linux, built with Python.

Liz-AI is an experimental personal assistant designed to help with programming, system tasks, organization, information retrieval, and everyday computer interactions.

The project is being developed with a focus on **modularity, security, extensibility, and practical AI tooling**.

## Status

🚧 **Early development**

Liz is currently in the initial architecture and development stage. Features such as persistent memory, voice interaction, system automation, and advanced tools will be introduced progressively.

## Goals

Liz is intended to eventually provide:

- 💬 Natural language interaction
- 🧠 Persistent and contextual memory
- 💻 Linux system interaction
- 📂 File and filesystem operations
- 🔧 Git and development tools
- 🖥️ Application and system automation
- 🎙️ Speech-to-text
- 🔊 Text-to-speech
- 🤖 AI-powered programming assistance
- 🔐 Permission-controlled system actions

The primary goal is not to create an autonomous system with unrestricted access to the computer, but a **useful and controllable personal assistant**.

## Architecture

Liz follows a lightweight layered architecture inspired by **Clean Architecture and Hexagonal Architecture**.

```text
src/liz/
│
├── cli/
│   └── app.py
│
├── config/
│   └── settings.py
│
├── core/
│   ├── agent.py
│   ├── context.py
│   ├── memory.py
│   └── models.py
│
├── infrastructure/
│   ├── ai/
│   │   └── openai_client.py
│   │
│   ├── persistence/
│   │   └── memory_store.py
│   │
│   └── voice/
│       ├── speech_to_text.py
│       └── text_to_speech.py
│
└── tools/
    ├── filesystem.py
    ├── git.py
    └── terminal.py
```

### Core

Contains Liz's main domain logic.

The core should remain independent from external services whenever possible.

### Infrastructure

Contains implementations that interact with external systems and providers.

Examples:

- AI providers
- Persistence
- Speech recognition
- Text-to-speech

### Tools

Tools provide Liz with controlled capabilities.

Examples:

- Filesystem operations
- Git operations
- Terminal commands

The AI should **request a tool**, rather than directly executing arbitrary operations on the host system.

```text
User
  │
  ▼
Liz Agent
  │
  ▼
AI Model
  │
  ├── filesystem tool
  ├── git tool
  └── terminal tool
          │
          ▼
      Permission
        Layer
          │
          ▼
       System
```

This separation is intended to make Liz easier to extend and safer to operate.

## Project Structure

| Directory         | Responsibility                            |
| ----------------- | ----------------------------------------- |
| `core/`           | Agent, context, memory, and domain models |
| `tools/`          | Actions Liz can perform                   |
| `infrastructure/` | External services and implementations     |
| `config/`         | Application configuration                 |
| `cli/`            | Command-line interface                    |
| `prompts/`        | Liz's system instructions                 |
| `data/`           | Local application data                    |
| `tests/`          | Automated tests                           |

## Technology Stack

- **Python**
- **OpenAI API**
- **SQLite** — planned initial persistence layer
- **Linux**
- **Git**
- **Speech-to-Text / Text-to-Speech** — planned

The stack may evolve as the project grows.

## Development Roadmap

### Phase 1 — Foundation

- [x] Project structure
- [x] Configuration layer
- [x] Initial agent structure
- [ ] Prompt system
- [ ] Basic CLI conversation
- [ ] AI provider integration

### Phase 2 — Tools

- [ ] Filesystem tools
- [ ] Git tools
- [ ] Terminal tools
- [ ] Permission/confirmation system
- [ ] Tool error handling

### Phase 3 — Memory

- [ ] Conversation context
- [ ] Persistent memory
- [ ] Memory retrieval
- [ ] User preferences
- [ ] Memory management

### Phase 4 — Voice

- [ ] Speech-to-text
- [ ] Text-to-speech
- [ ] Voice interaction loop
- [ ] Wake word / activation system

### Phase 5 — System Integration

- [ ] Application launching
- [ ] Process management
- [ ] System information
- [ ] Controlled automation
- [ ] Desktop integration

### Phase 6 — Advanced Capabilities

- [ ] Screen/vision capabilities
- [ ] Development workflow assistance
- [ ] Advanced automation
- [ ] Additional tool providers
- [ ] Plugin/tool architecture

## Security Principles

Liz is intended to operate under explicit boundaries.

### No unrestricted execution

AI-generated commands should never be treated as automatically trusted.

Potentially destructive operations should require explicit user confirmation.

```text
Liz:
I need to execute:

git push origin main

Allow? [y/N]
```

### No fake actions

Liz must never claim that an action was completed unless the underlying tool actually executed it successfully.

### Least privilege

Tools should have only the permissions necessary to perform their intended operations.

## Installation

Clone the repository:

```bash
git clone git@github.com:ThejavaZ/Liz-AI.git
cd Liz-AI
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
pip install -e .
```

Create the environment file:

```bash
cp .env.example .env
```

Configure the required environment variables in `.env`.

## Running Liz

The CLI entry point is:

```bash
python main.py
```

> The project is currently under active development, so the exact interface and commands may change.

## Philosophy

Liz is being built around a simple idea:

> **An AI assistant should be capable, useful, transparent, and under the user's control.**

The project prioritizes understandable architecture and explicit tool boundaries over giving the AI unrestricted access to the host machine.

## License

See [`LICENSE`](LICENSE) for the license of this project.
