"""Security tests: permissions, confirmation, path sandbox, terminal args.

No destructive action really runs: confirmations are faked, subprocesses
mocked, and the sandbox is exercised against tmp_path trees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from liz.cli.app import (
    ALWAYS_ALLOWED_TOOLS,
    confirm_tool_request,
    create_tool_registry,
    parse_fs_roots,
)
from liz.config.settings import Settings
from liz.tools.filesystem import FileExists, ListDirectory, ReadFile
from liz.tools.models import ToolStatus
from liz.tools.permissions import Permission, PermissionLayer
from liz.tools.registry import ToolRegistry
from liz.tools.terminal import RunCommand
from liz.tools.sandbox import format_roots, resolve_roots, within_roots


def _settings(**overrides) -> Settings:
    return Settings(
        ai_provider="gemini",
        ai_model="test-model",
        gemini_api_key="test-key",
        **overrides,
    )


# --- permission rules ---------------------------------------------------


class TestActionRules:
    def test_action_rule_overrides_tool_rule(self) -> None:
        pl = PermissionLayer()
        pl.set_rule("notes", Permission.ALLOW)
        pl.set_rule("notes", Permission.ASK, action="delete")

        assert pl.check("notes", "add") == Permission.ALLOW
        assert pl.check("notes", "delete") == Permission.ASK
        assert pl.check("notes") == Permission.ALLOW  # no action

    def test_action_rule_overrides_default(self) -> None:
        pl = PermissionLayer()
        pl.set_rule("memory", Permission.ASK, action="forget")

        assert pl.check("memory", "forget") == Permission.ASK
        assert pl.check("memory", "remember") == Permission.ASK  # default
        assert pl.check("other") == Permission.ASK

    def test_action_helpers(self) -> None:
        pl = PermissionLayer()
        pl.set_rule("notes", Permission.ALLOW)
        pl.set_rule("notes", Permission.ASK, action="delete")

        assert pl.is_allowed("notes", "add")
        assert pl.requires_ask("notes", "delete")
        assert not pl.is_denied("notes", "delete")

    def test_action_rule_with_non_string_action_ignored(self) -> None:
        # registry passes kwargs straight through; only str actions match
        pl = PermissionLayer()
        pl.set_rule("notes", Permission.ASK, action="delete")
        assert pl.check("notes") == Permission.ASK  # default anyway
        assert pl.check("notes", None) == Permission.ASK


# --- confirmation flow --------------------------------------------------


class TestConfirmation:
    def _registry(self, confirm=None) -> ToolRegistry:
        from liz.cli.app import ALWAYS_ALLOWED_TOOLS

        permissions = PermissionLayer()
        permissions.set_default(Permission.ASK)
        for name in ALWAYS_ALLOWED_TOOLS:
            permissions.set_rule(name, Permission.ALLOW)
        return ToolRegistry(permissions=permissions, confirm=confirm)

    def test_ask_without_callback_denies(self) -> None:
        registry = self._registry(confirm=None)
        registry.register(RunCommand())
        result = registry.execute("run_command", command="echo hi")
        assert result.status == ToolStatus.DENIED

    def test_ask_with_approving_callback_runs(self, monkeypatch) -> None:
        from unittest.mock import MagicMock

        monkeypatch.setattr(
            "liz.tools.terminal.subprocess.run",
            lambda parts, **kwargs: MagicMock(
                returncode=0, stdout="hi\n", stderr=""
            ),
        )
        registry = self._registry(confirm=lambda name, preview: True)
        registry.register(RunCommand())
        result = registry.execute("run_command", command="echo hi")
        assert result.status == ToolStatus.SUCCESS

    def test_ask_with_declining_callback_denies(self) -> None:
        seen = []
        registry = self._registry(
            confirm=lambda name, preview: seen.append((name, preview)) or False
        )
        registry.register(RunCommand())

        result = registry.execute("run_command", command="echo hi")

        assert result.status == ToolStatus.DENIED
        assert seen and seen[0][0] == "run_command"
        assert "echo hi" in seen[0][1]

    def test_callback_exception_denies(self) -> None:
        def broken(name: str, preview: str) -> bool:
            raise RuntimeError("prompt exploded")

        registry = self._registry(confirm=broken)
        registry.register(RunCommand())

        result = registry.execute("run_command", command="echo hi")
        assert result.status == ToolStatus.DENIED

    def test_allowed_tool_never_asks(self, monkeypatch) -> None:
        from unittest.mock import MagicMock

        def explode(name: str, preview: str) -> bool:
            raise AssertionError("must not prompt for allowed tools")

        monkeypatch.setattr(
            "liz.tools.terminal.subprocess.run",
            lambda parts, **kwargs: MagicMock(
                returncode=0, stdout="hi\n", stderr=""
            ),
        )
        registry = self._registry(confirm=explode)
        registry.register(RunCommand())
        # calculator is allowed in the real policy; use a rule here
        registry.permissions.set_rule("run_command", Permission.ALLOW)

        result = registry.execute("run_command", command="echo hi")
        assert result.status == ToolStatus.SUCCESS

    def test_preview_is_truncated(self) -> None:
        preview = ToolRegistry._preview({"command": "x" * 200})
        assert len(preview) < 120
        assert preview.endswith("...'")


class TestConfirmPrompt:
    def test_accepts_yes(self, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda prompt="": "y")
        assert confirm_tool_request("run_command", "command='ls'") is True

    def test_accepts_spanish_si(self, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda prompt="": "Sí")
        assert confirm_tool_request("notes", "") is True

    def test_default_is_no(self, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        assert confirm_tool_request("run_command", "") is False

    def test_eof_denies(self, monkeypatch) -> None:
        def eof(prompt=""):
            raise EOFError

        monkeypatch.setattr("builtins.input", eof)
        assert confirm_tool_request("run_command", "") is False

    def test_keyboard_interrupt_denies(self, monkeypatch) -> None:
        def interrupted(prompt=""):
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", interrupted)
        assert confirm_tool_request("run_command", "") is False


class TestAppPolicy:
    def test_read_only_tools_allowed(self) -> None:
        registry = create_tool_registry()
        for name in ALWAYS_ALLOWED_TOOLS:
            assert registry.permissions.check(name) == Permission.ALLOW, name

    def test_sensitive_tools_ask(self) -> None:
        registry = create_tool_registry()
        assert registry.permissions.check("run_command") == Permission.ASK
        assert registry.permissions.check("notes", "delete") == Permission.ASK
        assert registry.permissions.check("memory", "forget") == Permission.ASK

    def test_notes_add_still_allowed(self) -> None:
        registry = create_tool_registry()
        assert registry.permissions.check("notes", "add") == Permission.ALLOW

    def test_unknown_tool_falls_back_to_ask(self) -> None:
        registry = create_tool_registry()
        assert registry.permissions.check("future_tool") == Permission.ASK

    def test_confirm_is_wired(self) -> None:
        registry = create_tool_registry(confirm=lambda n, p: False)
        assert registry.confirm is not None

    def test_parse_fs_roots_default(self) -> None:
        roots = parse_fs_roots(_settings())
        assert Path.home() in [r.resolve() for r in roots]
        assert Path.cwd() in [r.resolve() for r in roots]

    def test_parse_fs_roots_explicit(self) -> None:
        settings = _settings(fs_roots="/tmp/a, ~/proj ,")
        roots = parse_fs_roots(settings)
        assert [r.name for r in roots] == ["a", "proj"]


# --- path sandbox -------------------------------------------------------


class TestSandbox:
    def test_roots_none_is_unrestricted(self) -> None:
        allowed, resolved = within_roots("/etc/passwd", None)
        assert allowed is True
        assert resolved == Path("/etc/passwd")

    def test_inside_root(self, tmp_path) -> None:
        allowed, _ = within_roots(tmp_path / "file.txt", [tmp_path])
        assert allowed is True

    def test_root_itself_allowed(self, tmp_path) -> None:
        allowed, _ = within_roots(tmp_path, [tmp_path])
        assert allowed is True

    def test_outside_root_denied(self, tmp_path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        allowed, resolved = within_roots("/etc/passwd", [root])
        assert allowed is False
        assert resolved == Path("/etc/passwd")

    def test_traversal_denied(self, tmp_path) -> None:
        root = tmp_path / "root"
        (root / "sub").mkdir(parents=True)
        allowed, _ = within_roots(root / "sub" / ".." / ".." / "escape", [root])
        assert allowed is False

    def test_symlink_escape_denied(self, tmp_path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret")
        link = root / "link.txt"
        link.symlink_to(secret)

        allowed, resolved = within_roots(link, [root])

        assert allowed is False
        assert resolved == secret  # resolved through the symlink

    def test_format_roots(self, tmp_path) -> None:
        assert format_roots(None) == "unrestricted"
        assert str(tmp_path) in format_roots([tmp_path])

    def test_resolve_roots_normalizes(self, tmp_path) -> None:
        roots = resolve_roots([str(tmp_path / "..")])
        assert roots == [tmp_path.resolve().parent]


class TestFilesystemSandbox:
    def test_read_outside_denied(self, tmp_path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("secret")

        result = ReadFile([root]).execute(str(outside))

        assert result.status == ToolStatus.ERROR
        assert "outside allowed" in result.error

    def test_read_inside_works(self, tmp_path) -> None:
        (tmp_path / "ok.txt").write_text("hello")

        result = ReadFile([tmp_path]).execute(str(tmp_path / "ok.txt"))

        assert result.status == ToolStatus.SUCCESS
        assert result.data == "hello"

    def test_traversal_denied(self, tmp_path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("secret")

        result = ReadFile([root]).execute(
            str(root / ".." / "secret.txt")
        )

        assert result.status == ToolStatus.ERROR

    def test_symlink_denied(self, tmp_path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("secret")
        (root / "link").symlink_to(outside)

        result = ReadFile([root]).execute(str(root / "link"))

        assert result.status == ToolStatus.ERROR

    def test_list_outside_denied(self, tmp_path) -> None:
        root = tmp_path / "root"
        root.mkdir()

        result = ListDirectory([root]).execute(str(tmp_path))

        assert result.status == ToolStatus.ERROR

    def test_list_inside_works(self, tmp_path) -> None:
        (tmp_path / "a.txt").write_text("x")

        result = ListDirectory([tmp_path]).execute(str(tmp_path))

        assert result.status == ToolStatus.SUCCESS
        assert "a.txt" in result.data

    def test_exists_outside_denied(self, tmp_path) -> None:
        root = tmp_path / "root"
        root.mkdir()

        result = FileExists([root]).execute("/etc/passwd")

        assert result.status == ToolStatus.ERROR

    def test_unrestricted_without_roots(self, tmp_path) -> None:
        target = tmp_path / "f.txt"
        target.write_text("data")

        result = ReadFile().execute(str(target))

        assert result.status == ToolStatus.SUCCESS


class TestTerminalHardening:
    def test_absolute_path_outside_roots_denied(self, tmp_path) -> None:
        root = tmp_path / "root"
        root.mkdir()

        result = RunCommand(roots=[root]).execute("cat /etc/passwd")

        assert result.status == ToolStatus.ERROR
        assert "outside allowed" in result.error

    def test_home_path_outside_roots_denied(self, tmp_path, monkeypatch) -> None:
        root = tmp_path / "root"
        root.mkdir()
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        (home / ".bashrc").write_text("alias x=y")

        result = RunCommand(roots=[root]).execute("cat ~/.bashrc")

        assert result.status == ToolStatus.ERROR

    def test_path_inside_roots_allowed(self, tmp_path, monkeypatch) -> None:
        target = tmp_path / "note.txt"
        target.write_text("hola")

        def fake_run(parts, **kwargs):
            from unittest.mock import MagicMock

            return MagicMock(returncode=0, stdout="hola\n", stderr="")

        monkeypatch.setattr("liz.tools.terminal.subprocess.run", fake_run)

        result = RunCommand(roots=[tmp_path]).execute(f"cat {target}")

        assert result.status == ToolStatus.SUCCESS

    def test_find_exec_blocked(self, tmp_path) -> None:
        result = RunCommand(roots=[tmp_path]).execute(
            "find . -exec rm {} ;"
        )
        assert result.status == ToolStatus.ERROR
        assert "-exec" in result.error

    def test_find_delete_blocked(self, tmp_path) -> None:
        result = RunCommand(roots=[tmp_path]).execute("find . -delete")
        assert result.status == ToolStatus.ERROR
        assert "not allowed" in result.error

    def test_flags_are_not_treated_as_paths(self, monkeypatch) -> None:
        # "-name" would resolve as a relative path; flags must be skipped
        from unittest.mock import MagicMock

        monkeypatch.setattr(
            "liz.tools.terminal.subprocess.run",
            lambda parts, **kwargs: MagicMock(
                returncode=0, stdout="a.txt", stderr=""
            ),
        )
        result = RunCommand().execute("find . -name '*.txt'")
        assert result.status == ToolStatus.SUCCESS
        assert "outside allowed" not in (result.error or "")

    def test_command_not_in_allowlist_still_denied(self) -> None:
        result = RunCommand().execute("bash -c 'rm -rf /'")
        assert result.status == ToolStatus.ERROR
        assert "not allowed" in result.error

    def test_unrestricted_without_roots(self, tmp_path, monkeypatch) -> None:
        def fake_run(parts, **kwargs):
            from unittest.mock import MagicMock

            return MagicMock(returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr("liz.tools.terminal.subprocess.run", fake_run)

        result = RunCommand().execute("cat /etc/hostname")

        assert result.status == ToolStatus.SUCCESS
