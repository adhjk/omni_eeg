from __future__ import annotations


def click_screen_position(x: int, y: int, *, duration_sec: float = 0.3) -> None:
    """Move the system mouse to a screen position and perform one click."""
    try:
        import pyautogui

        pyautogui.moveTo(int(x), int(y), duration=float(duration_sec))
        pyautogui.click()
    except Exception as exc:
        raise RuntimeError(f"自动鼠标点击失败，目标坐标=({int(x)}, {int(y)}): {exc}") from exc
