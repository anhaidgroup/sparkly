import requests
import time
from threading import Event, Thread
from tqdm import tqdm
from pyspark import Accumulator
from sparkly.utils import get_logger

logger = get_logger(__name__)


class ThreadProgress:
    def __init__(self, purpose: str, total: int, acc: Accumulator, show_progress_bar: bool, spark_ui_url: str, app_id: str, max_failures=3):
        self.total = total
        self.purpose = purpose
        self.acc = acc
        self.thread = None
        self.event = None
        self.show_progress_bar = show_progress_bar
        self.spark_ui_url = spark_ui_url
        self.app_id = app_id
        self.max_failures = max_failures
        self.failure_count = 0

    def start(self):
        self.event = Event()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.event.set()
        self.thread.join()

    def _check_spark_stage_status(self):
        url = f"{self.spark_ui_url}/api/v1/applications/{self.app_id}/stages"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                stages = response.json()
                active_tasks = 0
                failed_tasks = 0
                completed_stages = 0
                running_stages = 0
                for stage in stages:
                    if stage["status"] == "ACTIVE":
                        active_tasks += stage["numActiveTasks"]
                        running_stages += 1
                    elif stage["status"] == "COMPLETE":
                        completed_stages += 1
                    failed_tasks += stage["numFailedTasks"]
                if completed_stages == len(stages):
                    return False
                if active_tasks == 0 and running_stages > 0:
                    self.failure_count += 1
                else:
                    self.failure_count = 0
                if self.failure_count >= self.max_failures or failed_tasks > 0:
                    logger.error("Too many task failures. Stopping progress tracking.")
                    return True
            else:
                logger.error(f"Failed to fetch stage status: HTTP {response.status_code}")
        except requests.RequestException as e:
            logger.error(f"Error fetching Spark stage status: {e}")
        return False

    def _run(self):
        last_val = 0
        logger.debug(f"Starting {self.purpose}...")
        if self.show_progress_bar:
            with tqdm(total=self.total, desc=self.purpose, leave=False) as pbar:
                while not self.event.is_set():
                    if self._check_spark_stage_status():
                        self.event.set()
                        return
                    current_val = self.acc.value
                    delta = current_val - last_val
                    if delta:
                        pbar.update(delta)
                        last_val = current_val
                        logger.debug(f"{self.purpose} progress updated: {current_val}/{self.total}")
                    time.sleep(5)
                current_val = self.acc.value
                delta = current_val - last_val
                if delta:
                    pbar.update(delta)
                    logger.debug(f"Final progress: {current_val}/{self.total}")
                logger.debug(f"{self.purpose} done.")
            return
        else:
            while not self.event.is_set():
                if self._check_spark_stage_status():
                    self.event.set()
                    return
                current_val = self.acc.value
                delta = current_val - last_val
                if delta > self.total // 10:
                    last_val = current_val
                    logger.debug(f"{self.purpose} progress updated: {current_val}/{self.total}")
                time.sleep(5)
            current_val = self.acc.value
            delta = current_val - last_val
            if delta:
                logger.debug(f"Final progress: {current_val}/{self.total}")
            logger.debug(f"{self.purpose} done.")
            return
