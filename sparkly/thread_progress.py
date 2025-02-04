from pyspark import Accumulator
from threading import Event, Thread
from sparkly.utils import get_logger
import time


class ThreadProgress:
    def __init__(self, purpose: str, total: int, acc: Accumulator):
        self.total = total
        self.purpose = purpose
        self.acc = acc
        self.thread = None
        self.event = None
        self.log = get_logger(__name__)

    def start(self):
        self.event = Event()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.event.set()
        self.thread.join()

    def _run(self):
        last_val = 0
        self.log.debug(f"Starting {self.purpose}...")
        while not self.event.is_set():
            current_val = self.acc.value
            delta = current_val - last_val
            if delta > self.total // 10:
                last_val = current_val
                self.log.debug(f"{self.purpose} progress updated: {current_val}/{self.total}")
            time.sleep(1)
        current_val = self.acc.value
        delta = current_val - last_val
        if delta:
            self.log.debug(f"Final progress: {current_val}/{self.total}")
        self.log.debug(f"{self.purpose} done.")
        self.stop()
