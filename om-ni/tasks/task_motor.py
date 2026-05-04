from __future__ import annotations

from typing import Any


class Task:
    def wrap_console(self, console: Any) -> Any:
        return console

    def wrap_marker_backend(self, marker_backend: Any) -> Any:
        return marker_backend
