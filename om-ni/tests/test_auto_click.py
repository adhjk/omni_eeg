from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from tasks.mouse_auto_click import click_screen_position
from tasks.visual_exp_1_2 import _auto_click_selection_if_enabled
from tasks.visual_window import _top_left_widget_center


class _FakeWidget:
    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self._x = x
        self._y = y
        self._width = width
        self._height = height

    def winfo_rootx(self) -> int:
        return self._x

    def winfo_rooty(self) -> int:
        return self._y

    def winfo_width(self) -> int:
        return self._width

    def winfo_height(self) -> int:
        return self._height


class AutoClickTests(unittest.TestCase):
    def test_mouse_moves_for_point_three_seconds_then_clicks(self) -> None:
        fake_pyautogui = SimpleNamespace(moveTo=mock.Mock(), click=mock.Mock())
        with mock.patch.dict(sys.modules, {"pyautogui": fake_pyautogui}):
            click_screen_position(123, 456)

        fake_pyautogui.moveTo.assert_called_once_with(123, 456, duration=0.3)
        fake_pyautogui.click.assert_called_once_with()

    def test_mouse_failure_is_reported(self) -> None:
        fake_pyautogui = SimpleNamespace(
            moveTo=mock.Mock(side_effect=OSError("mouse unavailable")),
            click=mock.Mock(),
        )
        with mock.patch.dict(sys.modules, {"pyautogui": fake_pyautogui}):
            with self.assertRaisesRegex(RuntimeError, "自动鼠标点击失败"):
                click_screen_position(10, 20)

    def test_disabled_config_does_not_wait_or_click(self) -> None:
        window = mock.Mock()
        with mock.patch("tasks.visual_exp_1_2.click_screen_position") as click:
            _auto_click_selection_if_enabled({"auto_click": False}, window)

        window.wait_for_selection_target.assert_not_called()
        click.assert_not_called()

    def test_enabled_config_clicks_reported_target(self) -> None:
        window = mock.Mock()
        window.wait_for_selection_target.return_value = (321, 654)
        with mock.patch("tasks.visual_exp_1_2.click_screen_position") as click:
            _auto_click_selection_if_enabled({"auto_click": True}, window)

        window.wait_for_selection_target.assert_called_once_with()
        click.assert_called_once_with(321, 654, duration_sec=0.3)

    def test_finds_visible_top_left_widget_center(self) -> None:
        widgets = [
            _FakeWidget(300, 100, 80, 40),
            _FakeWidget(100, 100, 100, 60),
            _FakeWidget(50, 300, 120, 80),
        ]
        self.assertEqual(_top_left_widget_center(widgets), (150, 130))
        self.assertEqual(_top_left_widget_center(widgets[2:]), (110, 340))
        self.assertIsNone(_top_left_widget_center([]))


if __name__ == "__main__":
    unittest.main()
