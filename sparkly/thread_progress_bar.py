from pyspark import Accumulator
from threading import Event, Thread
from sparkly.utils import get_logger
from tqdm import tqdm
import time
import logging
import sys
import os
from tqdm.contrib.logging import tqdm_logging_redirect

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG
)
logger = get_logger(__name__)


class ThreadProgressBar:
    """
    progress tracking through logging only or logging and a tqdm progress bar

    Parameters
    ----------

    purpose : str
        function this progress bar is tracking

    total : int
        number of increments to take, e.g. number of tuples to iterate over

    acc : Accumulator
        accumulator object that is being updated in the calling function to track progress

    show_progress_bar: bool
        show the progress bar in addition to debug logs
    """
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
            with tqdm_logging_redirect(), tqdm(total=self.total, desc=self.purpose, leave=True, position=1) as pbar:
                while not self.event.is_set():
                    current_val = self.acc.value
                    if not isinstance(current_val, int):
                        delta = current_val.iloc[0] - last_val
                        logger.debug(f"{self.purpose} progress updated: {current_val.iloc[0]}/{self.total}")
                        pbar.update(delta)
                        last_val = current_val.iloc[0]
                        time.sleep(5)
                    else:
                        delta = current_val - last_val
                        logger.debug(f"{self.purpose} progress updated: {current_val}/{self.total}")
                        pbar.update(delta)
                        last_val = current_val
                        time.sleep(5)
                final_val = self.acc.value
                if not isinstance(final_val, int):
                    pbar.update(final_val.iloc[0] - last_val)
                else:
                    pbar.update(final_val - last_val)
            logger.debug(f"{self.purpose} done.")
            return
        else:
            while not self.event.is_set():
                current_val = self.acc.value
                if not isinstance(current_val, int):
                    logger.debug(f"{self.purpose} progress updated: {current_val.iloc[0]}/{self.total}")
                    time.sleep(5)
                else:
                    logger.debug(f"{self.purpose} progress updated: {current_val}/{self.total}")
                    time.sleep(5)         
            logger.debug(f"{self.purpose} done.")
