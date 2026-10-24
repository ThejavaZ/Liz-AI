import pytest

from liz.tools.calculator import Calculator, evaluate
from liz.tools.models import ToolStatus


class TestEvaluate:
    def test_addition(self) -> None:
        assert evaluate("2 + 2") == 4

    def test_precedence(self) -> None:
        assert evaluate("2 + 2 * 3") == 8

    def test_parentheses(self) -> None:
        assert evaluate("(2 + 2) * 3") == 12

    def test_division(self) -> None:
        assert evaluate("10 / 4") == 2.5

    def test_floor_division(self) -> None:
        assert evaluate("7 // 2") == 3

    def test_power(self) -> None:
        assert evaluate("2 ** 10") == 1024

    def test_modulo(self) -> None:
        assert evaluate("7 % 3") == 1

    def test_negative(self) -> None:
        assert evaluate("-5 + 3") == -2

    def test_unary_plus(self) -> None:
        assert evaluate("+5") == 5

    def test_sqrt(self) -> None:
        assert evaluate("sqrt(16)") == 4

    def test_nested_function(self) -> None:
        assert evaluate("sqrt(4 + 5)") == 3

    def test_abs(self) -> None:
        assert evaluate("abs(-7)") == 7

    def test_round(self) -> None:
        assert evaluate("round(3.6)") == 4

    def test_min(self) -> None:
        assert evaluate("min(3, 1, 2)") == 1

    def test_max(self) -> None:
        assert evaluate("max(3, 1, 2)") == 3

    def test_sum(self) -> None:
        assert evaluate("sum(1, 2, 3)") == 6

    def test_factorial(self) -> None:
        assert evaluate("factorial(5)") == 120

    def test_ceil(self) -> None:
        assert evaluate("ceil(3.2)") == 4

    def test_floor(self) -> None:
        assert evaluate("floor(3.8)") == 3

    def test_log(self) -> None:
        assert evaluate("log(1)") == 0

    def test_log10(self) -> None:
        assert evaluate("log10(1000)") == 3

    def test_log2(self) -> None:
        assert evaluate("log2(8)") == 3

    def test_exp(self) -> None:
        assert evaluate("exp(0)") == 1

    def test_constant_pi(self) -> None:
        assert evaluate("pi") == pytest.approx(3.141592653589793)

    def test_constant_e(self) -> None:
        assert evaluate("e") == pytest.approx(2.718281828459045)

    def test_constant_tau(self) -> None:
        assert evaluate("tau") == pytest.approx(6.283185307179586)

    def test_decimal(self) -> None:
        assert evaluate("1.5 * 2") == 3.0

    def test_whitespace(self) -> None:
        assert evaluate("  2 + 2  ") == 4


class TestEvaluateRejectsUnsafeInput:
    def test_import_rejected(self) -> None:
        with pytest.raises(ValueError):
            evaluate("__import__('os')")

    def test_attribute_access_rejected(self) -> None:
        with pytest.raises(ValueError):
            evaluate("(1).to_bytes")

    def test_name_rejected(self) -> None:
        with pytest.raises(ValueError):
            evaluate("os")

    def test_unknown_function_rejected(self) -> None:
        with pytest.raises(ValueError):
            evaluate("exec(1)")

    def test_string_literal_rejected(self) -> None:
        with pytest.raises(ValueError):
            evaluate("'hello'")

    def test_comparison_rejected(self) -> None:
        with pytest.raises(ValueError):
            evaluate("1 < 2")

    def test_bitwise_rejected(self) -> None:
        with pytest.raises(ValueError):
            evaluate("1 & 2")

    def test_division_by_zero(self) -> None:
        with pytest.raises(ZeroDivisionError):
            evaluate("1 / 0")


class TestCalculatorTool:
    def test_name(self) -> None:
        assert Calculator().name == "calculator"

    def test_description(self) -> None:
        assert "math" in Calculator().description.lower()

    def test_parameters_schema(self) -> None:
        schema = Calculator().parameters_schema
        assert "expression" in schema["properties"]
        assert "expression" in schema["required"]

    def test_execute_success(self) -> None:
        result = Calculator().execute(expression="2 + 2")
        assert result.status == ToolStatus.SUCCESS
        assert result.data == "4"

    def test_execute_integer_result(self) -> None:
        result = Calculator().execute(expression="10 / 2")
        assert result.data == "5"

    def test_execute_float_result(self) -> None:
        result = Calculator().execute(expression="10 / 4")
        assert result.data == "2.5"

    def test_execute_empty(self) -> None:
        result = Calculator().execute(expression="")
        assert result.status == ToolStatus.ERROR

    def test_execute_whitespace_only(self) -> None:
        result = Calculator().execute(expression="   ")
        assert result.status == ToolStatus.ERROR

    def test_execute_bad_expression(self) -> None:
        result = Calculator().execute(expression="2 +")
        assert result.status == ToolStatus.ERROR

    def test_execute_division_by_zero(self) -> None:
        result = Calculator().execute(expression="1 / 0")
        assert result.status == ToolStatus.ERROR
        assert "zero" in result.error.lower()

    def test_execute_rejects_code(self) -> None:
        result = Calculator().execute(expression="__import__('os').system('ls')")
        assert result.status == ToolStatus.ERROR
