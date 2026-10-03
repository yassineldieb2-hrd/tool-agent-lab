"""Three small demo tools: calculator, current_time, convert_units."""
from __future__ import annotations

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .tools import ToolError, tool

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str) -> float:
    """Evaluate arithmetic only. `eval()` on model output would be remote code execution."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            left, right = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(right) > 100:
                raise ToolError("exponent too large")
            return _BIN[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        raise ToolError("only numbers and + - * / ** % ( ) are allowed")
    if len(expr) > 200:
        raise ToolError("expression too long")
    try:
        return ev(ast.parse(expr.strip(), mode="eval"))
    except SyntaxError:
        raise ToolError("invalid expression syntax") from None
    except ZeroDivisionError:
        raise ToolError("division by zero") from None


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly.

    Args:
        expression: e.g. '0.15 * 240 + 7'. Supports + - * / ** % and parentheses.
    """
    value = safe_eval(expression)
    return str(round(value, 10)).removesuffix(".0")


@tool
def current_time(timezone: str) -> str:
    """Get the current date and time in an IANA timezone.

    Args:
        timezone: IANA name such as 'Europe/Warsaw' or 'Asia/Dubai'
    """
    try:
        return datetime.now(ZoneInfo(timezone)).strftime("%A %Y-%m-%d %H:%M %Z")
    except (ZoneInfoNotFoundError, ValueError):
        raise ToolError(f"unknown timezone '{timezone}'") from None


_LENGTH = {"km": 1000.0, "m": 1.0, "cm": 0.01, "mi": 1609.344, "ft": 0.3048, "in": 0.0254}
_MASS = {"kg": 1.0, "g": 0.001, "lb": 0.45359237, "oz": 0.028349523125}


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between units of length (km m cm mi ft in), mass (kg g lb oz) or temperature (C F K).

    Args:
        value: the number to convert
        from_unit: unit symbol to convert from
        to_unit: unit symbol to convert to
    """
    a, b = from_unit.strip(), to_unit.strip()
    for table in (_LENGTH, _MASS):
        if a.lower() in table and b.lower() in table:
            return f"{value * table[a.lower()] / table[b.lower()]:.6g} {b.lower()}"
    temps = {"C", "F", "K"}
    if a.upper() in temps and b.upper() in temps:
        c = {"C": value, "F": (value - 32) * 5 / 9, "K": value - 273.15}[a.upper()]
        out = {"C": c, "F": c * 9 / 5 + 32, "K": c + 273.15}[b.upper()]
        return f"{out:.6g} {b.upper()}"
    raise ToolError(f"cannot convert '{a}' to '{b}' (different dimensions or unknown unit)")


ALL_TOOLS = [calculator, current_time, convert_units]
