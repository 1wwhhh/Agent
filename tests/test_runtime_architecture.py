from __future__ import annotations

import ast
from pathlib import Path


RUNTIME_MODULE_PATH = Path("app/api/runtime.py")
BOOTSTRAP_MODULE_PATH = Path("app/api/bootstrap.py")


def _read_module(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _collect_imports(tree: ast.AST) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def test_runtime_module_stays_model_agnostic():
    source = _read_module(RUNTIME_MODULE_PATH)
    tree = ast.parse(source)
    imports = _collect_imports(tree)

    forbidden_import_prefixes = {
        "os",
        "app.adapters",
        "app.schemas.model",
    }
    assert not any(
        imported == forbidden or imported.startswith(f"{forbidden}.")
        for imported in imports
        for forbidden in forbidden_import_prefixes
    )

    forbidden_tokens = [
        "from_env(",
        "LLM_PROVIDER",
        "DEEPSEEK",
        "QWEN",
        "OPENAI",
        "ModelRouter",
        "RuntimeLLMConfig",
    ]
    for token in forbidden_tokens:
        assert token not in source


def test_runtime_builder_accepts_llm_client_only():
    source = _read_module(RUNTIME_MODULE_PATH)
    tree = ast.parse(source)

    build_runtime_components = next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "build_runtime_components"
    )
    parameter_names = [argument.arg for argument in build_runtime_components.args.kwonlyargs]

    assert "client" in parameter_names
    assert "config" not in parameter_names
    assert "provider" not in parameter_names


def test_bootstrap_is_the_only_env_and_adapter_aware_boundary():
    source = _read_module(BOOTSTRAP_MODULE_PATH)
    tree = ast.parse(source)
    imports = _collect_imports(tree)

    assert "app.adapters" in imports
    assert "app.schemas.model" in imports
    assert "RuntimeLLMConfig.from_env()" in source
    assert "ModelRouter().build_client(config)" in source
