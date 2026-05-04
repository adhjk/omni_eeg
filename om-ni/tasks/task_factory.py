from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Task(Protocol):
    def wrap_console(self, console: Any) -> Any: ...
    def wrap_marker_backend(self, marker_backend: Any) -> Any: ...


def resolve_task_mode(config: dict[str, Any]) -> str:
    mode = config.get("task_mode", None)
    if mode is None:
        return "motor"
    if not isinstance(mode, str) or not mode.strip():
        raise RuntimeError(f"Invalid task_mode: expected non-empty string, got {mode!r}")
    return mode.strip()


def load_task(task_mode: str) -> Task:
    if task_mode == "motor":
        module_name = "tasks.task_motor"
    else:
        module_name = f"tasks.task_{task_mode}"

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Task module not found for task_mode={task_mode!r}. "
            f"Expected a Python module named {module_name!r} (file: task_{task_mode}.py under tasks/)."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import task module {module_name!r} for task_mode={task_mode!r}: {exc}"
        ) from exc

    task_cls = getattr(module, "Task", None)
    if task_cls is None:
        raise RuntimeError(
            f"Invalid task module {module_name!r}: missing 'Task' class. "
            "Define class Task with methods wrap_console(...) and wrap_marker_backend(...)."
        )
    try:
        task: Any = task_cls()
    except Exception as exc:
        raise RuntimeError(f"Failed to construct Task from {module_name!r}: {exc}") from exc

    if not isinstance(task, Task):
        raise RuntimeError(
            f"Invalid Task in {module_name!r}: object does not implement required protocol "
            "(wrap_console, wrap_marker_backend)."
        )
    return task


def load_task_from_config(config: dict[str, Any]) -> Task:
    return load_task(resolve_task_mode(config))
