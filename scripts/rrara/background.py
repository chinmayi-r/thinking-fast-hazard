"""One background job at a time, never waited for.

The RRARA agent starts the reflector's validate() here while HAZARD executes the reflex's action, and at the next decision
asks collect(): a finished job gives its result and wall time, an unfinished one gives None and keeps running. Nothing in
this file blocks. Python's ThreadPoolExecutor runs the function in a second thread; a Future is the handle to it.
"""
import time
from concurrent.futures import ThreadPoolExecutor


class BackgroundJob:
    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._future = None
        self.started_at = None
        self.tag = None  # whatever the caller wants to remember about the job (here: the decision number it checks)

    def start(self, fn, *args, tag=None):
        assert self._future is None, "a job is already pending; collect() or discard() it first"
        self.started_at = time.perf_counter()
        self.tag = tag
        self._future = self._pool.submit(fn, *args)

    def pending(self):
        return self._future is not None

    def ready(self):
        return self._future is not None and self._future.done()

    def collect(self):
        """Non-blocking. Finished: returns (result, wall_s, tag) and frees the slot. Not finished or empty: None."""
        if not self.ready():
            return None
        fut, tag = self._future, self.tag
        self._future = None
        wall = time.perf_counter() - self.started_at
        try:
            result = fut.result()
        except Exception as e:  # the reflector's API call failed after its retries: report, never crash the episode
            result = {"error": repr(e)}
        return result, wall, tag

    def discard(self):
        """Forget a pending job (its thread finishes on its own; the result is thrown away)."""
        self._future = None

    def shutdown(self):
        self._pool.shutdown(wait=False)
