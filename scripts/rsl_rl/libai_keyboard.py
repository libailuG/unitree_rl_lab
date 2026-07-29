import sys
import time
import threading
from collections import defaultdict
import termios
import tty
import select


class TermiosKeyMonitor:
    """基于 select + termios 的非阻塞按键检测 (纯事件模型)。

    局限: termios 无法检测按键释放, 因此只能上报"本帧读到了什么键"。
    每个按键只在本帧触发一次 just_pressed; is_pressed / was_just_released
    始终返回 False (如需长按/释放检测, 请使用 pynput 方案)。
    """

    def __init__(self):
        self.just_pressed: dict[str, bool] = {}
        self._old_settings = None
        self._running = False

    def start(self):
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._running = True

    def stop(self):
        self._running = False
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def frame(self):
        """非阻塞读取 stdin, 每读到一个字符即记录为本帧 just_pressed。"""
        self.just_pressed.clear()   # 上一帧的事件过期

        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':         # ESC 开头 (方向键等), 跳过
                continue
            self.just_pressed[ch] = True   # 只触发一次

    def is_pressed(self, key: str) -> bool:
        return False                         # termios 无法检测长按

    def was_just_pressed(self, key: str) -> bool:
        return self.just_pressed.get(key, False)

    def was_just_released(self, key: str) -> bool:
        return False   
