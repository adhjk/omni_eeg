from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 调试开关
DEBUG = True

def debug_log(msg: str) -> None:
    if DEBUG:
        print(f"[VisualWindow] {msg}")

@dataclass(slots=True)
class SelectionItem:
    item_id: int
    title: str
    image_path: Path


class VisualStimulusWindow:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *, title: str, default_image_path: Path) -> None:
        if self._initialized:
            self._title = title
            self._default_image_path = Path(default_image_path)
            debug_log("窗口已初始化，更新 title/default_image_path 并跳过重复初始化")
            return
        self._initialized = True
        debug_log("开始初始化窗口实例")
        self._title = title
        self._default_image_path = Path(default_image_path)
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._start_error: Exception | None = None
        self._fullscreen = True
        self._selection_queue: queue.Queue[int] = queue.Queue()
        self._root = None
        debug_log("窗口实例初始化完成")

    def start(self) -> None:
        debug_log("start() 被调用")
        if self._thread is not None and self._thread.is_alive():
            debug_log("窗口线程已在运行，显示窗口并返回")
            self.show()
            return
        self._stop.clear()
        self._ready.clear()
        self._start_error = None
        debug_log("创建新的窗口线程")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        debug_log("等待窗口准备就绪...")
        self._ready.wait(timeout=8.0)
        if self._start_error is not None:
            message = str(self._start_error).strip() or repr(self._start_error)
            lowered = message.lower()
            if "no display name" in lowered or "$display" in lowered:
                raise RuntimeError(
                    "视觉刺激窗口启动失败：当前运行环境没有可用的图形显示（$DISPLAY 未设置）。"
                    "请在有桌面环境的机器运行，或配置 X11 转发，或使用 xvfb 启动 Streamlit（例如 xvfb-run -a streamlit run gui.py）。"
                ) from self._start_error
            raise RuntimeError(f"视觉刺激窗口启动失败：{message}") from self._start_error
        if not self._ready.is_set():
            raise RuntimeError("视觉刺激窗口启动超时")
        debug_log("窗口已就绪")
        self.show()

    def close(self) -> None:
        """隐藏窗口（不销毁），Tkinter 主循环继续运行。"""
        debug_log("close() 被调用，隐藏窗口")
        self._queue.put(("hide", None))

    def show(self) -> None:
        """重新显示已隐藏的窗口。"""
        debug_log("show() 被调用，显示窗口")
        self._queue.put(("show", None))

    def destroy(self) -> None:
        """彻底销毁窗口并退出主循环（一般不建议调用，保留用于彻底清理）。"""
        debug_log("destroy() 被调用，将销毁窗口")
        self._stop.set()
        self._queue.put(("destroy", None))
        thread = self._thread
        self._thread = None
        if thread is not None:
            debug_log("等待窗口线程结束...")
            thread.join(timeout=2.0)
            debug_log("窗口线程已结束")
        # 重置单例标志，允许重新创建（谨慎使用）
        # type(self)._instance = None
        # type(self)._initialized = False

    def show_black(self, title: str, subtitle: str = "") -> None:
        debug_log(f"show_black: title={title}, subtitle={subtitle}")
        self._queue.put(("state", {"title": str(title), "subtitle": str(subtitle), "mode": "black"}))

    def show_image(self, title: str, subtitle: str = "", *, image_path: Path | None = None) -> None:
        path = self._default_image_path if image_path is None else Path(image_path)
        debug_log(f"show_image: title={title}, subtitle={subtitle}, path={path}")
        self._queue.put(("state", {"title": str(title), "subtitle": str(subtitle), "mode": "image", "image_path": path}))

    def show_selection(self, title: str, subtitle: str, items: list[SelectionItem]) -> None:
        debug_log(f"show_selection: title={title}, items count={len(items)}")
        self._queue.put(("selection", {"title": str(title), "subtitle": str(subtitle), "items": list(items)}))

    def poll_selection(self) -> int | None:
        try:
            val = int(self._selection_queue.get_nowait())
            debug_log(f"poll_selection 返回 {val}")
            return val
        except queue.Empty:
            return None

    def begin_pause(self, message: str) -> threading.Event:
        debug_log(f"begin_pause: {message}")
        event = threading.Event()
        self._queue.put(("pause", {"message": str(message), "event": event}))
        return event

    def _run(self) -> None:
        debug_log("_run: 窗口线程启动")
        try:
            import tkinter as tk
        except Exception as exc:
            debug_log(f"导入 tkinter 失败: {exc}")
            self._start_error = exc
            self._ready.set()
            return

        try:
            debug_log("创建 Tk 根窗口")
            root = tk.Tk()
            self._root = root
            root.title(self._title)
            root.configure(bg="#000000")
            root.attributes("-topmost", True)
            root.attributes("-fullscreen", True)

            def toggle_fullscreen(_event=None) -> None:
                self._fullscreen = not self._fullscreen
                root.attributes("-fullscreen", self._fullscreen)
                debug_log(f"全屏切换: {self._fullscreen}")

            root.bind("<Escape>", toggle_fullscreen)

            frame = tk.Frame(root, bg="#000000")
            frame.pack(fill="both", expand=True)

            title_label = tk.Label(frame, font=("Arial", 28), bg="#000000", fg="#ffffff", justify="center")
            title_label.pack(fill="x", pady=(26, 6))
            subtitle_label = tk.Label(frame, font=("Arial", 18), bg="#000000", fg="#cccccc", justify="center")
            subtitle_label.pack(fill="x", pady=(0, 18))

            image_label = tk.Label(frame, bg="#000000")
            selection_frame = tk.Frame(frame, bg="#000000")

            photo_cache: dict[str, Any] = {}

            def load_photo(path: Path) -> Any:
                key = str(path)
                debug_log(f"load_photo: {key}")
                if key in photo_cache:
                    debug_log(f"使用缓存的图片: {key}")
                    return photo_cache[key]
                try:
                    photo = tk.PhotoImage(file=key)
                    photo_cache[key] = photo
                    debug_log(f"图片加载成功: {key}")
                    return photo
                except Exception as e:
                    debug_log(f"图片加载失败 {key}: {e}\n{traceback.format_exc()}")
                    raise

            def clear_selection() -> None:
                debug_log("清除选择区域")
                for child in selection_frame.winfo_children():
                    child.destroy()

            pause_root: Any | None = None

            def show_pause(message: str, event: threading.Event) -> None:
                nonlocal pause_root
                debug_log(f"显示暂停窗口: {message}")
                try:
                    root.withdraw()
                except Exception as e:
                    debug_log(f"root.withdraw() 异常: {e}")
                pause_root = tk.Toplevel()
                pause_root.title(self._title)
                pause_root.configure(bg="#000000")
                pause_root.attributes("-topmost", True)
                pause_root.geometry("520x140")
                lbl = tk.Label(pause_root, text=message, font=("Arial", 16), bg="#000000", fg="#ffffff", justify="center")
                lbl.pack(fill="both", expand=True, padx=20, pady=20)

                def on_space(_evt=None) -> None:
                    nonlocal pause_root
                    debug_log("暂停窗口空格键被按下，继续")
                    try:
                        pause_root.destroy()
                    except Exception as e:
                        debug_log(f"销毁暂停窗口异常: {e}")
                    pause_root = None
                    try:
                        root.deiconify()
                    except Exception as e:
                        debug_log(f"root.deiconify() 异常: {e}")
                    event.set()

                pause_root.bind("<space>", on_space)
                pause_root.focus_force()

            def apply_state(payload: dict[str, Any]) -> None:
                debug_log(f"apply_state: mode={payload.get('mode')}")
                clear_selection()
                selection_frame.pack_forget()
                image_label.place_forget()
                # 释放图片引用
                image_label.config(image='')
                if hasattr(image_label, 'image'):
                    del image_label.image

                mode = str(payload.get("mode", "black"))
                title_label.config(text=str(payload.get("title", "")))
                subtitle_label.config(text=str(payload.get("subtitle", "")))

                if mode == "image":
                    path = payload.get("image_path", self._default_image_path)
                    debug_log(f"显示图片: {path}")
                    try:
                        photo = load_photo(Path(path))
                        image_label.config(image=photo)
                        image_label.image = photo
                        image_label.place(relx=0.5, rely=0.55, anchor=tk.CENTER)
                        debug_log("图片显示成功")
                    except Exception as e:
                        debug_log(f"显示图片失败: {e}\n{traceback.format_exc()}")
                        # 回退到黑屏
                        title_label.config(text="图片加载失败，请检查路径")
                        subtitle_label.config(text=str(path))

            def apply_selection(payload: dict[str, Any]) -> None:
                debug_log("apply_selection: 开始构建选择界面")
                image_label.place_forget()
                image_label.config(image='')
                if hasattr(image_label, 'image'):
                    del image_label.image
                clear_selection()
                title_label.config(text=str(payload.get("title", "")))
                subtitle_label.config(text=str(payload.get("subtitle", "")))
                items: list[SelectionItem] = list(payload.get("items") or [])
                debug_log(f"共有 {len(items)} 个选项")
                selection_frame.pack(fill="both", expand=True, pady=(10, 20))
                for idx, item in enumerate(items):
                    try:
                        photo = load_photo(item.image_path)
                        thumb = photo.subsample(5, 5) if hasattr(photo, "subsample") else photo
                    except Exception as e:
                        debug_log(f"加载选项图片失败 {item.image_path}: {e}")
                        continue
                    row = idx // 5
                    col = idx % 5

                    def on_click(chosen_id: int = int(item.item_id)) -> None:
                        debug_log(f"用户点击了选项 {chosen_id}")
                        self._selection_queue.put(chosen_id)

                    btn = tk.Button(
                        selection_frame,
                        text=str(item.title),
                        image=thumb,
                        compound="top",
                        command=on_click,
                        bg="#111111",
                        fg="#ffffff",
                        activebackground="#222222",
                        activeforeground="#ffffff",
                        relief="flat",
                        padx=10,
                        pady=10,
                    )
                    btn.image = thumb
                    btn.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

                for i in range(5):
                    selection_frame.grid_columnconfigure(i, weight=1)
                for i in range(max(1, (len(items) + 4) // 5)):
                    selection_frame.grid_rowconfigure(i, weight=1)
                debug_log("选择界面构建完成")

            def poll() -> None:
                if self._stop.is_set():
                    debug_log("轮询检测到停止标志，退出主循环")
                    try:
                        root.quit()
                    finally:
                        return
                try:
                    while True:
                        action, payload = self._queue.get_nowait()
                        debug_log(f"处理队列消息: {action}")
                        if action == "destroy":
                            debug_log("收到 destroy 命令，退出主循环")
                            root.quit()
                            return
                        if action == "hide":
                            debug_log("隐藏窗口")
                            root.withdraw()
                        elif action == "show":
                            debug_log("显示窗口")
                            root.deiconify()
                        elif action == "state":
                            apply_state(dict(payload))
                        elif action == "selection":
                            apply_selection(dict(payload))
                        elif action == "pause":
                            show_pause(str(payload.get("message", "")), payload.get("event"))
                except queue.Empty:
                    pass
                root.after(50, poll)

            apply_state({"title": "", "subtitle": "", "mode": "black"})
            self._ready.set()
            debug_log("窗口就绪，开始轮询")
            poll()
            debug_log("进入 Tk 主循环")
            root.mainloop()
            debug_log("Tk 主循环已退出")
        except Exception as exc:
            debug_log(f"窗口线程发生异常: {exc}\n{traceback.format_exc()}")
            self._start_error = exc
            self._ready.set()
