# NoahRGB 11/2026
#
# contains a logger class which holds keys/values and 
# aggregates across them using a variety of output options
#
#

import json
from pathlib import Path
import numpy as np

AGGREGATORS = {"mean": np.mean, "sum": np.sum, "max": np.max, "min": np.min, "last": lambda v: v[-1]}

class Logger:

    def __init__(self):
        self.setup()

    def setup(self, outputs=[]):
        self.outputs = outputs
        self.values = {}
        self.aggs = {}

    def add(self, name, value=None, agg="mean", prefix=None):

        # add a dict of values idividually
        if isinstance(name, dict):
            for key, val in name.items():
                self.add(key, val, agg=agg, prefix=prefix)
            return

        if prefix:
            name = f"{prefix}/{name}"

        # if value is a tensor then detach the graphh
        if hasattr(value, "detach"):
            value = value.detach()

        self.values.setdefault(name, []).append(value)
        self.aggs[name] = agg

    def write(self, step):
        metrics = {}

        # aggregate all logged values
        for name, values in self.values.items():
            metrics[name] = float(AGGREGATORS[self.aggs[name]](values))

        # output each metric using each selected output method
        for output in self.outputs:
            output(step, metrics)

        self.values = {}
        self.aggs = {}

    def close(self):
        # make sure any output methods are closed (e.g. files)
        for output in self.outputs:
            if hasattr(output, "close"):
                output.close()
        self.setup()

class TerminalOuptut:
    def __call__(self, step, metrics):
        for key, value in sorted(metrics.items()):
            print(f"[{step}] {key}: {round(value, 2)}")
        print()

class JSONOutput:
    def __init__(self, run_dir):
        self.file = open(Path(run_dir) / "metrics.jsonl", "a")

    def __call__(self, step, metrics):
        self.file.write(json.dumps({"step": step, **metrics}) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


# create a global logger that is used everywhere
logger = Logger()

# shortcuts
setup = logger.setup
add = logger.add
write = logger.write
close = logger.close