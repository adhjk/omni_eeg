from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SelectionItem:
    item_id: int
    title: str
    image_path: Path


class VisualStimulusWindow:
    def __init__(self, *, title: str, default_image_path: Path) -> None:
        self._title = title
        self._default_image_path = Path(default_image_path)
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._start_error: Exception | None = None
        self._fullscreen = True
        self._selection_queue: queue.Queue[int] = queue.Queue()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
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

    def close(self) -> None:
        self._stop.set()
        self._queue.put(("close", None))
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)

    def show_black(self, title: str, subtitle: str = "") -> None:
        self._queue.put(("state", {"title": str(title), "subtitle": str(subtitle), "mode": "black"}))

    def show_image(self, title: str, subtitle: str = "", *, image_path: Path | None = None) -> None:
        path = self._default_image_path if image_path is None else Path(image_path)
        self._queue.put(("state", {"title": str(title), "subtitle": str(subtitle), "mode": "image", "image_path": path}))

    def show_selection(self, title: str, subtitle: str, items: list[SelectionItem]) -> None:
        self._queue.put(("selection", {"title": str(title), "subtitle": str(subtitle), "items": list(items)}))

    def poll_selection(self) -> int | None:
        try:
            return int(self._selection_queue.get_nowait())
        except queue.Empty:
            return None

    def begin_pause(self, message: str) -> threading.Event:
        event = threading.Event()
        self._queue.put(("pause", {"message": str(message), "event": event}))
        return event

    def _run(self) -> None:
        try:
            import tkinter as tk
        except Exception as exc:
            self._start_error = exc
            self._ready.set()
            return

        try:
            root = tk.Tk()
            root.title(self._title)
            root.configure(bg="#000000")
            root.attributes("-topmost", True)
            root.attributes("-fullscreen", True)

            def toggle_fullscreen(_event=None) -> None:
                self._fullscreen = not self._fullscreen
                root.attributes("-fullscreen", self._fullscreen)

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
                if key in photo_cache:
                    return photo_cache[key]
                photo = tk.PhotoImage(file=key)
                photo_cache[key] = photo
                return photo

            def clear_selection() -> None:
                for child in selection_frame.winfo_children():
                    child.destroy()

            pause_root: Any | None = None

            def show_pause(message: str, event: threading.Event) -> None:
                nonlocal pause_root
                try:
                    root.withdraw()
                except Exception:
                    pass
                pause_root = tk.Toplevel()
                pause_root.title(self._title)
                pause_root.configure(bg="#000000")
                pause_root.attributes("-topmost", True)
                pause_root.geometry("520x140")
                lbl = tk.Label(pause_root, text=message, font=("Arial", 16), bg="#000000", fg="#ffffff", justify="center")
                lbl.pack(fill="both", expand=True, padx=20, pady=20)

                def on_space(_evt=None) -> None:
                    try:
                        pause_root.destroy()
                    except Exception:
                        pass
                    pause_root = None
                    try:
                        root.deiconify()
                    except Exception:
                        pass
                    event.set()

                pause_root.bind("<space>", on_space)
                pause_root.focus_force()

            def apply_state(payload: dict[str, Any]) -> None:
                clear_selection()
                selection_frame.pack_forget()
                image_label.place_forget()
                mode = str(payload.get("mode", "black"))
                title_label.config(text=str(payload.get("title", "")))
                subtitle_label.config(text=str(payload.get("subtitle", "")))
                if mode == "image":
                    path = payload.get("image_path", self._default_image_path)
                    photo = load_photo(Path(path))
                    image_label.config(image=photo)
                    image_label.image = photo
                    image_label.place(relx=0.5, rely=0.55, anchor=tk.CENTER)

            def apply_selection(payload: dict[str, Any]) -> None:
                clear_selection()
                image_label.place_forget()
                title_label.config(text=str(payload.get("title", "")))
                subtitle_label.config(text=str(payload.get("subtitle", "")))
                items: list[SelectionItem] = list(payload.get("items") or [])
                selection_frame.pack(fill="both", expand=True, pady=(10, 20))
                for idx, item in enumerate(items):
                    photo = load_photo(item.image_path)
                    thumb = photo.subsample(5, 5) if hasattr(photo, "subsample") else photo
                    row = idx // 5
                    col = idx % 5

                    def on_click(chosen_id: int = int(item.item_id)) -> None:
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

            def poll() -> None:
                if self._stop.is_set():
                    try:
                        root.quit()
                    finally:
                        return
                try:
                    while True:
                        action, payload = self._queue.get_nowait()
                        if action == "close":
                            root.quit()
                            return
                        if action == "state":
                            apply_state(dict(payload))
                        if action == "selection":
                            apply_selection(dict(payload))
                        if action == "pause":
                            show_pause(str(payload.get("message", "")), payload.get("event"))
                except queue.Empty:
                    pass
                root.after(50, poll)

            apply_state({"title": "", "subtitle": "", "mode": "black"})
            self._ready.set()
            poll()
            root.mainloop()
        except Exception as exc:
            self._start_error = exc
            self._ready.set()
