"""Safe arithmetic evaluation.

Uses the AST instead of eval so arbitrary code can never run. Only
numeric literals, arithmetic operators and a small set of math
functions are allowed.
"""

import ast
import operator
from typing import Any

from liz.tools.base import Tool
from liz.tools.models import ToolResult

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": lambda *args: _safe_sum(args),
    "sqrt": lambda x: _safe_sqrt(x),
    "log": lambda x: _safe_log(x),
    "log10": lambda x: _safe_log10(x),
    "log2": lambda x: _safe_log2(x),
    "exp": lambda x: _safe_exp(x),
    "ceil": lambda x: _safe_ceil(x),
    "floor": lambda x: _safe_floor(x),
    "factorial": lambda x: _safe_factorial(x),
}

_CONSTANTS = {
    "pi": 3.141592653589793,
    "e": 2.718281828459045,
    "tau": 6.283185307179586,
    "inf": float("inf"),
}


def _import_math():
    import math

    return math


def _safe_sum(args):
    return sum(args)


def _safe_sqrt(x):
    return _import_math().sqrt(x)


def _safe_log(x):
    return _import_math().log(x)


def _safe_log10(x):
    return _import_math().log10(x)


def _safe_log2(x):
    return _import_math().log2(x)


def _safe_exp(x):
    return _import_math().exp(x)


def _safe_ceil(x):
    return _import_math().ceil(x)


def _safe_floor(x):
    return _import_math().floor(x)


def _safe_factorial(x):
    return _import_math().factorial(x)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported literal: {node.value!r}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _BIN_OPS[op_type](_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")
        name = node.func.id
        if name not in _FUNCTIONS:
            raise ValueError(f"Unknown function: {name}")
        args = [_eval_node(arg) for arg in node.args]
        return _FUNCTIONS[name](*args)

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"Unknown name: {node.id}")

    raise ValueError(f"Unsupported expression: {type(node).__name__}")


def evaluate(expression: str) -> float:
    """Evaluate a math expression safely. Raises ValueError on bad input."""
    tree = ast.parse(expression.strip(), mode="eval")
    return _eval_node(tree)


class Calculator(Tool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate a math expression safely (no code execution)"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g. '2 + 2 * sqrt(16)'",
                }
            },
            "required": ["expression"],
        }

    def execute(self, expression: str) -> ToolResult:
        if not expression or not expression.strip():
            return ToolResult.error("Expression is empty")

        try:
            result = evaluate(expression)
        except ZeroDivisionError:
            return ToolResult.error("Division by zero")
        except Exception as e:
            return ToolResult.error(f"Cannot evaluate expression: {e}")

        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return ToolResult.success(str(result))
