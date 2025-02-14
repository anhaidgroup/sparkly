from pyspark import Accumulator
from threading import Event, Thread
from sparkly.utils import get_logger
from tqdm import tqdm
import time

logger = get_logger(__name__)


class ThreadProgressBar:
    def __init__(self, purpose: str, total: int, acc: Accumulator, show_progress_bar: bool):
        self.total = total
        self.purpose = purpose
        self.acc = acc
        self.thread = None
        self.event = None
        self.show_progress_bar = show_progress_bar

    def __enter__(self):
        self.start()
        return

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return

    def start(self):
        self.event = Event()
        self.thread = Thread(target=self._run)
        self.thread.start()

    def stop(self):
        self.event.set()
        self.thread.join()

    def _run(self):
        last_val = 0
        logger.debug(f"Starting {self.purpose}...")
        if self.show_progress_bar:
            with tqdm(total=self.total, desc=self.purpose) as pbar:
                while not self.event.is_set():
                    current_val = self.acc.value
                    if type(current_val) is not int:
                        delta = current_val.iloc[0] - last_val
                        pbar.update(delta)
                        last_val = current_val.iloc[0]
                        logger.debug(f"{self.purpose} progress updated: {current_val.iloc[0]}/{self.total}")
                        time.sleep(5)
            logger.debug(f"{self.purpose} done.")
        else:
            while not self.event.is_set():
                current_val = self.acc.value
                if type(current_val) is not int:
                    logger.debug(f"{self.purpose} progress updated: {current_val.iloc[0]}/{self.total}")
                    time.sleep(5)
            logger.debug(f"{self.purpose} done.")
