from enum import Enum


class Permission(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionLayer:
    def __init__(self) -> None:
        self._rules: dict[str, Permission] = {}
        self._default: Permission = Permission.ASK

    def set_rule(self, tool_name: str, permission: Permission) -> None:
        self._rules[tool_name] = permission

    def set_default(self, permission: Permission) -> None:
        self._default = permission

    def check(self, tool_name: str) -> Permission:
        return self._rules.get(tool_name, self._default)

    def is_allowed(self, tool_name: str) -> bool:
        return self.check(tool_name) == Permission.ALLOW

    def is_denied(self, tool_name: str) -> bool:
        return self.check(tool_name) == Permission.DENIED

    def requires_ask(self, tool_name: str) -> bool:
        return self.check(tool_name) == Permission.ASK
