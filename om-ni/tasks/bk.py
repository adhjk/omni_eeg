from __future__ import annotations
import winsound
import atexit
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
    def __init__(self, *, title: str, image_path: Path) -> None:
        print(f"{_t()} VisualGui 初始化")
        self._title = title
        self._image_path = image_path
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._root = None
        self._label = None
        self._image_label = None
        self._photo = None
        self._stop = threading.Event()
        self._started = threading.Event()
        self._start_error = None
        self._is_fullscreen = True  # 新增：追踪全屏状态

    def start(self) -> None:
        print(f"{_t()} 启动GUI线程")
        if self._thread:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._started.wait(timeout=5)
        if self._start_error:
            raise RuntimeError("GUI启动失败") from self._start_error

    def close(self) -> None:
        print(f"{_t()} 收到关闭指令")
        self._stop.set()
        self._queue.put(("close", None))
        if self._thread:
            self._thread.join(1)
        print(f"{_t()} GUI线程已关闭")

    def set_state(self, title: str, subtitle: str, *, show_image: bool) -> None:
        self._queue.put(("state", (title, subtitle, show_image)))

    def _run(self) -> None:
        print(f"{_t()} GUI线程运行中")
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

            # ========== 全局配置 ==========
            root.attributes("-topmost", True)
            root.attributes("-fullscreen", self._is_fullscreen)
            root.configure(bg="#000000")
            # ==============================

            # ========== ESC 切换全屏 ==========
            def toggle_fullscreen(event=None):
                self._is_fullscreen = not self._is_fullscreen
                root.attributes("-fullscreen", self._is_fullscreen)
            root.bind("<Escape>", toggle_fullscreen)
            # ===================================

            # 外层容器 纯黑
            frame = tk.Frame(root, bg="#000000")
            frame.pack(fill="both", expand=True)

            # 文字标签：黑底+白色字体，居中
            self._label = tk.Label(
                frame,
                font=("Arial", 22),
                justify="center",
                bg="#000000",
                fg="#ffffff"
            )
            self._label.pack(fill="x", pady=(30, 20))

            # 加载图片
            self._photo = tk.PhotoImage(file=str(self._image_path))
            print(f"{_t()} 图片加载成功")

            # 图片容器：居中、黑底
            self._image_label = tk.Label(frame, image=self._photo, bg="#000000")

            def poll():
                if self._stop.is_set():
                    root.quit()
                    return
                try:
                    while True:
                        act, data = self._queue.get_nowait()
                        if act == "close":
                            print(f"{_t()} 执行安全关闭")
                            root.quit()
                            return
                        if act == "state":
                            t, s, show = data
                            self._label.config(text=f"{t}\n{s}")
                            if show:
                                self._image_label.config(image=self._photo)
                                self._image_label.image = self._photo
                                self._image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                            else:
                                self._image_label.place_forget()
                except queue.Empty:
                    pass
                root.after(50, poll)

            poll()
            self._started.set()
            print(f"{_t()} GUI主循环启动")
            root.mainloop()
            print(f"{_t()} GUI主循环结束")

        except Exception as e:
            self._start_error = e
            print(f"{_t()} GUI异常：{e}")

        finally:
            self._image_label = None
            self._label = None
            self._root = None
            if self._photo is not None:
                try:
                    self._photo.tk.call('image', 'delete', self._photo.name)
                except Exception:
                    pass


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
        # atexit.register(self.close)
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
        print(f"{_t()} Task.close()")
        if self._gui:
            self._gui.close()
            self._gui = None

    def on_console_print(self, args):
        txt = " ".join(map(str, args))
        if any(k in txt for k in ["停止实时解码","停止测试模式","测试数据已保存"]):
            print(f"{_t()} 检测到关闭关键词")
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
                # ========== 修复音频播放 ==========
                winsound.PlaySound(
                    str(self._audio_path),
                    winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT
                )
            except Exception as e:
                print(f"{_t()} 播放失败: {e}")
        threading.Thread(target=p, daemon=False).start()
