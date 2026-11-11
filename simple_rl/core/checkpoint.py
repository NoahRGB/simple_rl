# NoahRGB 11/2026
#
# a class that manages saving/loading algorithms componets
# e.g. networks, optimisers, step counts, etc.
#
#
# objects with a torch state_dict() can be passed in by same and are
# saved by reference
# e.g. checkpoint = Checkpoint(run.dir, network=network, optim=optim)
#
# plain values (floats, ints, tensors) should be passed into savee
# explicitly
# e.g. checkpoint.save(epsilon=epsilon)
#
#
#

from pathlib import Path
import torch

FILENAME = "checkpoint.pt"


class Checkpoint:

    def __init__(self, run_dir, every=None, **objects):
        # objects are the named things to checkpoint e.g. Checkpoint(run.dir, network=network, optim=optim)
        for name, obj in objects.items():
            if not hasattr(obj, "state_dict"):
                raise TypeError(f"'{name}' has no state_dict(). pass plain values to save() instead")

        self.path = Path(run_dir) / FILENAME
        self.every = every # how many timesteps between saves in maybe_save()
        self.objects = objects
        self.last_save = 0

    def save(self, **values):
        # values are plain values to store alongside the objects e.g. save(epsilon=epsilon)
        clashes = set(values) & set(self.objects)
        assert not clashes, f"{clashes} are already the names of checkpointed objects"

        state = dict(values)
        for name, obj in self.objects.items():
            state[name] = obj.state_dict()

        # write to a tmp file and then swap them
        tmp = self.path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(self.path)

    def maybe_save(self, step, **values):
        # save if at least "every" timesteps have passed since the last save
        if self.every is not None and step - self.last_save >= self.every:
            self.save(**values)
            self.last_save = step

    def load(self, load_dir):
        state = torch.load(Path(load_dir) / FILENAME, weights_only=True)
        for name, obj in self.objects.items():
            obj.load_state_dict(state[name])

        return {name: value for name, value in state.items() if name not in self.objects}
