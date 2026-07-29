import sys
import time
import atexit
import termios
import tty
import select


class TermiosKeyMonitor:
    """基于 select + termios 的非阻塞按键检测 (纯事件模型)。

    局限: termios 无法检测按键释放, 因此只能上报"本帧读到了什么键"。
    每个按键只在本帧触发一次 just_pressed; is_pressed / was_just_released
    始终返回 False (如需长按/释放检测, 请使用 pynput 方案)。

    用法:
        # 方式 1: 手动 start/stop (需配合 try/finally)
        monitor = TermiosKeyMonitor()
        monitor.start()
        try:
            while True:
                monitor.frame()
                if monitor.was_just_pressed('q'):
                    break
        finally:
            monitor.stop()

        # 方式 2: context manager (推荐)
        with TermiosKeyMonitor() as monitor:
            while True:
                monitor.frame()
                if monitor.was_just_pressed('q'):
                    break
    """

    def __init__(self):
        self.just_pressed: dict[str, bool] = {}
        self._old_settings = None
        self._running = False
        self._atexit_registered = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        return False

    def start(self):
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._running = True
        # atexit 作为最后防线：即使忘记调用 stop()，解释器退出时也会恢复终端
        if not self._atexit_registered:
            atexit.register(self._safe_stop)
            self._atexit_registered = True

    def stop(self):
        """恢复终端设置。幂等：重复调用安全。"""
        self._running = False
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            except (termios.error, OSError, ValueError, TypeError):
                pass  # stdin 可能已被回收或不是 tty
            self._old_settings = None

    def _safe_stop(self):
        """atexit 回调：静默失败，不抛出异常打断解释器退出。"""
        try:
            self.stop()
        except Exception:
            pass

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


def main():

    monitor = TermiosKeyMonitor()

    monitor.start()
    print("按任意键查看响应... (按 'q' 退出)\n")

    try:
        while True:
            monitor.frame()

            # ---- just_pressed: 每帧只触发一次 (两种方案通用) ----
            for key in monitor.just_pressed:
                print(f"[按下]  '{key}'")

            # ---- 退出: 按 'q' ----
            if monitor.was_just_pressed('q'):
                print("\n再见!")
                break

            time.sleep(0.05)   # ~20 Hz 轮询

    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
