from enum import Enum


class Permission(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionLayer:
    def __init__(self) -> None:
        self._rules: dict[str, Permission] = {}
        self._default: Permission = Permission.ASK

    def set_rule(
        self, tool_name: str, permission: Permission, action: str | None = None
    ) -> None:
        """Rule for a tool, optionally narrowed to one action.

        ``set_rule("notes", ALLOW)`` plus ``set_rule("notes", ASK,
        action="delete")`` allows every note operation except deletion.
        """
        key = f"{tool_name}:{action}" if action else tool_name
        self._rules[key] = permission

    def set_default(self, permission: Permission) -> None:
        self._default = permission

    def check(self, tool_name: str, action: str | None = None) -> Permission:
        if action:
            specific = self._rules.get(f"{tool_name}:{action}")
            if specific is not None:
                return specific
        return self._rules.get(tool_name, self._default)

    def is_allowed(self, tool_name: str, action: str | None = None) -> bool:
        return self.check(tool_name, action) == Permission.ALLOW

    def is_denied(self, tool_name: str, action: str | None = None) -> bool:
        return self.check(tool_name, action) == Permission.DENY

    def requires_ask(self, tool_name: str, action: str | None = None) -> bool:
        return self.check(tool_name, action) == Permission.ASK
