# Testing

## Current Test Baseline

- Python baseline: `Python 3.10`
- Standard Conda environment: `rag310`
- Current supported runtime: `Python 3.10+`
- Production can evaluate `Python 3.11` later, but that is not the current requirement

## Current Verified Baseline

1. Current standard environment
   - Python `3.10`
   - Conda environment: `rag310`
2. Environment self-check passed
   - `python -c "import asyncio, random, secrets, ssl; print('ok')"`
3. Smoke test passed
   - `python -m pytest tests/test_simple_flow.py -q`
   - Result: `1 passed`
4. Core regression passed
   - `python -m pytest tests/test_observability.py tests/test_llm_layer.py tests/test_parser_repair.py tests/test_executor_stability.py tests/test_executor_failure.py tests/test_router_permissions.py tests/test_simple_flow.py tests/test_complex_dag.py tests/test_concurrency.py tests/test_runtime_architecture.py -q`
   - Result: `112 passed`
5. This is the known verified test baseline at `Runtime Alpha v0.6` freeze time.

## Recommended Conda Environment

Windows PowerShell or CMD:

```bat
conda create -n rag310 python=3.10 -y
conda activate rag310
```

## Environment Self-Check

Run this before tests:

```bat
python -c "import asyncio, random, secrets, ssl; print('ok')"
```

## Dependency Installation

If using editable install:

```bat
python -m pip install -U pip setuptools wheel
pip install -e ".[dev]"
```

If not using editable install:

```bat
python -m pip install -U pip setuptools wheel
pip install -r requirements-dev.txt
```

## Standard Test Commands

### Smoke Test

```bat
python -m pytest tests/test_simple_flow.py -q
```

### Core Regression

Windows PowerShell or CMD:

```bat
python -m pytest ^
  tests/test_observability.py ^
  tests/test_llm_layer.py ^
  tests/test_parser_repair.py ^
  tests/test_executor_stability.py ^
  tests/test_executor_failure.py ^
  tests/test_router_permissions.py ^
  tests/test_simple_flow.py ^
  tests/test_complex_dag.py ^
  tests/test_concurrency.py ^
  tests/test_runtime_architecture.py ^
  -q
```

Single-line equivalent:

```bat
python -m pytest tests/test_observability.py tests/test_llm_layer.py tests/test_parser_repair.py tests/test_executor_stability.py tests/test_executor_failure.py tests/test_router_permissions.py tests/test_simple_flow.py tests/test_complex_dag.py tests/test_concurrency.py tests/test_runtime_architecture.py -q
```

### Full Test

```bat
python -m pytest -q
```

### Single Module Debug

```bat
python -m pytest tests/test_executor_stability.py -q -vv
```

## Windows Notes

The previous `rag` environment showed:

```text
Fatal Python error:
_Py_HashRandomization_Init:
failed to get random numbers

Python runtime state:
preinitialized
```

This failure happened during Python interpreter pre-initialization, earlier than `pytest`, `asyncio`, project code, and test code.

The most likely explanation is Conda or Python environment corruption, or Windows `PATH` / DLL / random-source related problems. It should not be attributed to Runtime main code.

## Troubleshooting

- If `python -c "import asyncio, random, secrets, ssl; print('ok')"` fails, treat it as an environment problem first.
- If `pytest` fails before tests start collecting, verify that `python`, `pip`, and `pytest` all point to the active `rag310` environment.
- If you see behavior similar to the old `rag` environment, prefer creating a fresh environment instead of adding Runtime code workarounds.

## Recommendation

If similar failures appear again, create a clean environment first:

```bat
conda create -n rag310 python=3.10 -y
conda activate rag310
```

Do not work around a damaged Python or Conda environment inside Runtime main code.
