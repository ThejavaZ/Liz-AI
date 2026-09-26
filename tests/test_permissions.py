from liz.tools.permissions import PermissionLayer, Permission


class TestPermissionLayer:
    def test_default_is_ask(self) -> None:
        pl = PermissionLayer()
        assert pl.check("any_tool") == Permission.ASK

    def test_set_rule_allow(self) -> None:
        pl = PermissionLayer()
        pl.set_rule("tool_a", Permission.ALLOW)
        assert pl.check("tool_a") == Permission.ALLOW
        assert pl.is_allowed("tool_a")

    def test_set_rule_deny(self) -> None:
        pl = PermissionLayer()
        pl.set_rule("tool_b", Permission.DENY)
        assert pl.check("tool_b") == Permission.DENY
        assert not pl.is_allowed("tool_b")
        assert not pl.requires_ask("tool_b")

    def test_set_rule_ask(self) -> None:
        pl = PermissionLayer()
        pl.set_rule("tool_c", Permission.ASK)
        assert pl.check("tool_c") == Permission.ASK
        assert pl.requires_ask("tool_c")

    def test_set_default(self) -> None:
        pl = PermissionLayer()
        pl.set_default(Permission.ALLOW)
        assert pl.check("unknown_tool") == Permission.ALLOW
        assert pl.is_allowed("unknown_tool")

    def test_rules_override_default(self) -> None:
        pl = PermissionLayer()
        pl.set_default(Permission.ALLOW)
        pl.set_rule("specific", Permission.DENY)
        assert pl.check("specific") == Permission.DENY
        assert pl.check("other") == Permission.ALLOW
