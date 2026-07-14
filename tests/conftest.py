from __future__ import annotations

import asyncio
import inspect

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "asyncio: run async test functions in an event loop")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    test_function = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_function):
        return None

    test_args = {
        name: pyfuncitem.funcargs[name]
        for name in pyfuncitem._fixtureinfo.argnames
        if name in pyfuncitem.funcargs
    }
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(test_function(**test_args))
    finally:
        asyncio.set_event_loop(None)
        loop.close()
    return True
