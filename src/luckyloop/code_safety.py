from __future__ import annotations

import ast
from pathlib import Path

from .schemas import CodeValidationResult


BLOCKED_IMPORT_ROOTS = {
    "os",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "httpx",
    "openai",
    "huggingface_hub",
    "datasets",
    "shutil",
}
BLOCKED_CALL_NAMES = {"eval", "exec", "compile", "__import__", "input", "open"}
ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "argparse",
    "json",
    "math",
    "time",
    "pathlib",
    "warnings",
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
}


def validate_generated_code(code: str) -> CodeValidationResult:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return CodeValidationResult(status="rejected", reason=f"syntax_error: {exc}")

    blocked_nodes: list[str] = []
    blocked_imports: list[str] = []
    warnings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if root in BLOCKED_IMPORT_ROOTS or root not in ALLOWED_IMPORT_ROOTS:
                    blocked_imports.append(name)
        if isinstance(node, ast.Call):
            func_name = _call_name(node.func)
            if func_name in BLOCKED_CALL_NAMES:
                blocked_nodes.append(f"call:{func_name}")
            if func_name in {"Path.write_text", "write_text", "to_pickle", "to_parquet"}:
                warnings.append(f"write call must stay inside provided output directory: {func_name}")
        if isinstance(node, ast.Attribute) and node.attr in {"system", "popen", "remove", "unlink", "rmtree", "open"}:
            blocked_nodes.append(f"attribute:{node.attr}")

    if blocked_imports or blocked_nodes:
        return CodeValidationResult(
            status="rejected",
            reason="blocked imports or calls in generated code",
            blocked_nodes=sorted(set(blocked_nodes)),
            blocked_imports=sorted(set(blocked_imports)),
            warnings=warnings,
        )
    return CodeValidationResult(status="accepted", reason="static validation passed", warnings=warnings)


def write_validated_code(code: str, path: Path) -> CodeValidationResult:
    validation = validate_generated_code(code)
    if validation.status == "accepted":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
    return validation


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""
