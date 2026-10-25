## Description

<!-- What does this PR change, and why? Link related issues: "Fixes #12". -->

## Type

- [ ] `feat:` new capability
- [ ] `fix:` bug fix
- [ ] `docs:` documentation only
- [ ] `test:` tests only
- [ ] `refactor:` no behavior change
- [ ] `ci:` workflows / pipelines
- [ ] `perf:` performance
- [ ] `chore:` tooling, dependencies

## Files affected

<!-- Files you created or modified and why they matter. -->

-
-

## How to test

<!-- Commands or steps a reviewer can repeat locally. -->

1.
2.

## Checklist

See [`AGENTS.md`](https://github.com/ThejavaZ/Liz-AI/blob/main/AGENTS.md) for the full contract.

- [ ] `python -m pytest tests/ -q` passes locally (all tests, not just the new ones)
- [ ] New behavior is covered: happy path **and** error path
- [ ] Tests do not hit the network, the microphone or a real provider
- [ ] Conventional commit message in English
- [ ] README / AGENTS.md updated if structure, counts or roadmap items changed
- [ ] No new dependency without justification and approval (rule 4)
- [ ] No heavy ML library; persistence still stdlib `sqlite3` (rules 1–2)
- [ ] Nothing sensitive committed (`.env`, `*.db`, `.venv`, `__pycache__`)

## Notes for the reviewer

<!-- Anything tricky: invariants kept, follow-ups, screenshots of the CLI. -->
