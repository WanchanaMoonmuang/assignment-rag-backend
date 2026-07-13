import ast
import math
import operator

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "abs": abs, "round": round, "sqrt": math.sqrt, "sin": math.sin,
    "cos": math.cos, "tan": math.tan, "log": math.log, "log10": math.log10,
    "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


class CalculatorError(ValueError):
    pass


def calculate(expression: str) -> float | int:
    if len(expression) > 200:
        raise CalculatorError("Expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError("Invalid expression") from exc
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise CalculatorError("Expression is too complex")
    result = _evaluate(tree.body)
    if isinstance(result, int) and abs(result) > 10**308:
        raise CalculatorError("Result is too large")
    if not isinstance(result, (int, float)) or not math.isfinite(result):
        raise CalculatorError("Result must be finite")
    return result


def _evaluate(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        if abs(node.value) > 10**12:
            raise CalculatorError("Number is too large")
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise CalculatorError("Exponent is too large")
        try:
            return _BINARY[type(node.op)](left, right)
        except (ArithmeticError, OverflowError, ValueError) as exc:
            raise CalculatorError("Calculation failed") from exc
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and not node.keywords:
        try:
            return _FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args])
        except (ArithmeticError, OverflowError, TypeError, ValueError) as exc:
            raise CalculatorError("Calculation failed") from exc
    raise CalculatorError("Unsupported expression")
