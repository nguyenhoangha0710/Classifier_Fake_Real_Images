"""Flushed stage logs and periodic heartbeats during slow I/O/native GPU calls."""
import faulthandler
import threading
import time


class Progress:
    def __init__(self, name, interval=30, stack_after=300):
        self.name, self.interval = name, interval
        self.started = self.changed = time.monotonic()
        self.current = 'starting'
        self.stop_event = threading.Event()
        self.thread = None
        self.stack_after = stack_after
        self.last_dump = 0

    def __enter__(self):
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        faulthandler.enable()
        return self

    def stage(self, text, emit=True):
        self.current, self.changed = text, time.monotonic()
        if emit:
            print(f'[{self.name}] {time.monotonic()-self.started:.1f}s | {text}', flush=True)

    def _heartbeat(self):
        while not self.stop_event.wait(self.interval):
            print(f'[{self.name} heartbeat] elapsed={time.monotonic()-self.started:.0f}s '
                  f'stage_age={time.monotonic()-self.changed:.0f}s | {self.current}', flush=True)
            now = time.monotonic()
            if self.stack_after and now-self.changed>=self.stack_after and now-self.last_dump>=self.stack_after:
                print(f'[{self.name}] Stage unchanged for {self.stack_after}s; Python stacks follow (diagnostic, not proof of deadlock)',flush=True)
                faulthandler.dump_traceback(all_threads=True)
                self.last_dump = now

    def __exit__(self, *exc):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=1)
