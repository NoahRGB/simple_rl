# NoahRGB 11/2026
#
# a class to keep track of the runtime of various
# sections of the training process
#
#

import time
from collections import defaultdict
from contextlib import contextmanager
import torch

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class Timer:

    def __init__(self):
        self.stack = []
        self.reset()

    def reset(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.start = time.perf_counter()

    @contextmanager
    def section(self, name, sync=False):
        self.stack.append(name)
        path = "/".join(self.stack) # indented name e.g. timer/collect/policy

        if sync: cuda_sync()
        start_time = time.perf_counter()

        try: yield # this will run the actual code inside the section() indentation
        finally:
            if sync: cuda_sync()
            self.totals[path] += time.perf_counter() - start_time
            self.counts[path] += 1
            self.stack.pop()

    def stats(self, reset=True):
        total_time_elapsed = time.perf_counter() - self.start
        out = {}

        # for each timed section, output what fraction of the total time
        # was spent in that section as well as the milliseconds elapsed
        for path, total in self.totals.items():
            out[f"{path}/frac"] = total / total_time_elapsed
            out[f"{path}/ms"] = 1000 * (total / self.counts[path])

        if reset: self.reset()
        return out


# create global timer that is used everywhere
timer = Timer()