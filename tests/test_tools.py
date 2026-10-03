import pytest
from tool_agent_lab.builtin_tools import calculator, convert_units, current_time, safe_eval
from tool_agent_lab.tools import ToolError, tool


def test_schema_generated_from_signature_and_docstring():
    s = convert_units.schema()["function"]
    assert s["name"] == "convert_units"
    assert s["parameters"]["required"] == ["value", "from_unit", "to_unit"]
    assert s["parameters"]["properties"]["value"] == {"type": "number", "description": "the number to convert"}
    assert "Args" not in s["description"] and s["description"].startswith("Convert")


def test_optional_params_not_required():
    @tool
    def f(a: int, b: str = "x") -> str:
        """Doc."""
        return b * a
    assert f.parameters["required"] == ["a"]
    assert f.run({"a": 2}) == "xx"


def test_decorator_rejects_missing_docstring_and_bad_types():
    with pytest.raises(ValueError):
        @tool
        def nodoc(a: int) -> str:
            return ""
    with pytest.raises(ValueError):
        @tool
        def bad(a: list) -> str:
            """Doc."""
            return ""


def test_validation_errors_are_readable_strings():
    assert "missing required" in calculator.run({})
    assert "unknown argument" in calculator.run({"expression": "1", "x": 1})
    assert "must be string" in calculator.run({"expression": 5})
    assert "not valid JSON" in calculator.run("{oops")
    assert "must be an object" in calculator.run([1])


def test_bool_is_not_accepted_as_number_and_int_promotes_to_float():
    assert "must be number" in convert_units.run({"value": True, "from_unit": "km", "to_unit": "m"})
    assert convert_units.run({"value": 1, "from_unit": "km", "to_unit": "m"}) == "1000 m"
    assert calculator.run('{"expression": "2+2"}') == "4"   # JSON-string arguments are accepted


@pytest.mark.parametrize("expr,expected", [("0.15 * 240 + 7", "43"), ("2 ** 10", "1024"), ("(1+2)*3", "9"),
                                           ("-4 + 10 % 4", "-2"), ("1/4", "0.25")])
def test_calculator(expr, expected):
    assert calculator.run({"expression": expr}) == expected


@pytest.mark.parametrize("evil", ["__import__('os').system('id')", "open('/etc/passwd')", "[1]", "'a'*5",
                                  "1/0", "9**9**9", "2 +", "x", "True", "x" * 300])
def test_calculator_refuses_non_arithmetic(evil):
    assert calculator.run({"expression": evil}).startswith("ERROR")


def test_safe_eval_raises_tool_error():
    with pytest.raises(ToolError):
        safe_eval("abs(-1)")


def test_units():
    assert convert_units.run({"value": 10, "from_unit": "km", "to_unit": "mi"}) == "6.21371 mi"
    assert convert_units.run({"value": 100, "from_unit": "C", "to_unit": "F"}) == "212 F"
    assert convert_units.run({"value": 1, "from_unit": "kg", "to_unit": "km"}).startswith("ERROR")


def test_current_time():
    out = current_time.run({"timezone": "Europe/Warsaw"})
    assert not out.startswith("ERROR") and out.split()[-1] in ("CET", "CEST")
    assert current_time.run({"timezone": "Mars/Base"}).startswith("ERROR")


def test_tool_bug_becomes_error_not_exception():
    @tool
    def boom(x: int) -> str:
        """Always fails."""
        raise RuntimeError("kaput")
    assert boom.run({"x": 1}) == "ERROR: tool 'boom' failed: RuntimeError: kaput"
