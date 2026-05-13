"""Streamlit web interface for oi-mi — task dispatcher only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st

from cli import load_config as load_app_config, resolve_config_path
from tasks.task_factory import load_streamlit_renderer

# ---------- 页面基础设置 ----------
_GUI_ROOT = Path(__file__).resolve().parent
_PAGE_ICON_FILENAME = "OMNI_ICON.svg"

def _resolve_asset_path(filename: str) -> Path | None:
    candidates = (
        _GUI_ROOT / "assets" / filename,
        Path.cwd() / "assets" / filename,
        Path.cwd() / "oi-mi" / "assets" / filename,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

_PAGE_ICON_PATH = _resolve_asset_path(_PAGE_ICON_FILENAME)
st.set_page_config(
    page_title="oi-mi Control Panel",
    page_icon=str(_PAGE_ICON_PATH) if _PAGE_ICON_PATH is not None else None,
    layout="wide",
)

# ---------- 配置加载 ----------
def parse_config_path(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", dest="config_path", type=Path, default=None)
    args, _ = parser.parse_known_args(argv)
    return resolve_config_path(args.config_path)

CONFIG_PATH = parse_config_path(sys.argv[1:])

def load_config() -> dict:
    try:
        return load_app_config(CONFIG_PATH)
    except Exception as exc:
        st.error(f"加载配置文件失败: {exc}")
        return {}

# ---------- 主入口 ----------
def main() -> None:
    config = load_config()
    if not config:
        return

    # 尝试从任务工厂获取该模式的专用渲染器
    renderer = load_streamlit_renderer(config)
    if renderer is not None:
        renderer(config, CONFIG_PATH)
        st.stop()  # 阻断后续执行
    else:
        st.error("当前 task_mode 未提供可用的 Streamlit 界面，请检查配置文件。")

if __name__ == "__main__":
    main()