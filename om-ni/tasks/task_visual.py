from __future__ import annotations
import winsound
import queue
import threading
from pathlib import Path
from typing import Any

# 调试工具：打印当前线程ID
def _t():
    return f"[线程:{threading.get_ident()}]"

class _ConsoleProxy:
    def __init__(self, console: Any, task: "Task") -> None:
        self._console = console
        self._task = task

    def print(self, *args: Any, **kwargs: Any) -> None:
        self._console.print(*args, **kwargs)
        self._task.on_console_print(args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._console, name)

class _VisualGui:
    _singleton = None  # 全局单例：只创建一次GUI
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._singleton is None:
                cls._singleton = super().__new__(cls)
            return cls._singleton

    def __init__(self, *, title: str, image_path: Path):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        print(f"{_t()} VisualGui 初始化（仅一次）")
        self._title = title
        self._image_path = image_path
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread = None
        self._root = None
        self._label = None
        self._image_label = None
        self._photo = None
        self._stop = threading.Event()
        self._started = threading.Event()
        self._start_error = None
        self._is_fullscreen = True
        self._visible = False

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            self.show()
            return

        print(f"{_t()} 启动GUI线程（仅一次）")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._started.wait(timeout=5)
        if self._start_error:
            raise RuntimeError("GUI启动失败") from self._start_error
        self.show()

    def show(self):
        self._visible = True
        self._queue.put(("show", None))

    def hide(self):
        self._visible = False
        self._queue.put(("hide", None))

    def close(self):
        self.hide()  # 只隐藏，不销毁！

    def set_state(self, title: str, subtitle: str, *, show_image: bool):
        self._queue.put(("state", (title, subtitle, show_image)))

    def _run(self):
        print(f"{_t()} GUI线程运行中（永久运行）")
        try:
            import tkinter as tk
        except Exception as e:
            self._start_error = e
            self._started.set()
            return

        try:
            root = tk.Tk()
            self._root = root
            root.title(self._title)
            root.attributes("-topmost", True)
            root.attributes("-fullscreen", self._is_fullscreen)
            root.configure(bg="#000000")
            root.withdraw()

            def toggle_fullscreen(event=None):
                self._is_fullscreen = not self._is_fullscreen
                root.attributes("-fullscreen", self._is_fullscreen)
            root.bind("<Escape>", toggle_fullscreen)

            frame = tk.Frame(root, bg="#000000")
            frame.pack(fill="both", expand=True)

            self._label = tk.Label(frame, font=("Arial", 22), justify="center", bg="#000000", fg="#ffffff")
            self._label.pack(fill="x", pady=(30, 20))

            self._photo = tk.PhotoImage(file=str(self._image_path))
            self._image_label = tk.Label(frame, image=self._photo, bg="#000000")

            def poll():
                try:
                    while True:
                        act, data = self._queue.get_nowait()
                        if act == "show":
                            root.deiconify()
                        elif act == "hide":
                            root.withdraw()
                        elif act == "state":
                            t, s, show = data
                            self._label.config(text=f"{t}\n{s}")
                            if show:
                                self._image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                            else:
                                self._image_label.place_forget()
                except queue.Empty:
                    pass
                root.after(30, poll)

            poll()
            self._started.set()
            root.mainloop()

        except Exception as e:
            self._start_error = e
            print(f"{_t()} GUI异常: {e}")

class _MarkerBackendProxy:
    def __init__(self, backend: Any, task: "Task") -> None:
        self._backend = backend
        self._task = task

    def send(self, label: int, timestamp=None) -> None:
        self._backend.send(label, timestamp=timestamp)
        self._task.on_marker_label(label)

    def send_event(self, event_name: str, timestamp=None) -> None:
        if hasattr(self._backend, "send_event"):
            self._backend.send_event(event_name, timestamp=timestamp)
        self._task.on_protocol_event(event_name)

    def __getattr__(self, name):
        return getattr(self._backend, name)

class Task:
    def __init__(self) -> None:
        self._console = None
        self._gui = None
        self._audio_path = None

    def wrap_console(self, console: Any) -> Any:
        print(f"{_t()} wrap_console")
        self._console = console
        proxy = _ConsoleProxy(console, self)
        root = Path(__file__).resolve().parents[1]
        img = root / "assets" / "EEG.png"
        aud = root / "assets" / "guitar.wav"
        self._audio_path = aud
        self._gui = _VisualGui(title="OI-MI", image_path=img)
        self._gui.start()
        self._render_event(2)
        return proxy

    def wrap_marker_backend(self, backend):
        return _MarkerBackendProxy(backend, self)

    def on_protocol_event(self, e):
        if e in ("cue_left_on", "cue_right_on", "cue_idle_on"):
            self._render_event({"cue_left_on":0,"cue_right_on":1,"cue_idle_on":2}[e])
        elif e == "session_end":
            self._render_event(2)
            self.close()

    def on_marker_label(self, label):
        self._render_event(label)

    def close(self):
        print(f"{_t()} 任务结束 → 隐藏GUI")
        if self._gui:
            self._gui.hide()

    def on_console_print(self, args):
        txt = " ".join(map(str, args))
        if any(k in txt for k in ["停止实时解码","停止测试模式","测试数据已保存"]):
            self.close()

    def _render_event(self, label):
        if label == 0:
            self._gui.set_state("事件0","想象左方",show_image=True)
            self._play_guitar()
        elif label == 1:
            self._gui.set_state("事件1","想象右方",show_image=True)
        elif label == 2:
            self._gui.set_state("静息","保持静息",show_image=False)

    def _play_guitar(self):
        def p():
            try:
                winsound.PlaySound(
                    str(self._audio_path),
                    winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT
                )
            except Exception as e:
                print(f"{_t()} 播放失败: {e}")
        threading.Thread(target=p, daemon=False).start()
