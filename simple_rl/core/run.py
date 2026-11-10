# NoahRGB 11/2026
#
#
#
#
#
#

import random, time, importlib, json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import torch
import numpy as np

from simple_rl.core import logger as log


@dataclass
class Run:
    # this is what the algorithms get given
    cfg: object
    dir: Path

def now():
    return datetime.now().isoformat(timespec="seconds")

def write_status(run_dir, **fields):
    # write to a tmp file and then swap them
    path = run_dir / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    status.update(fields, updated=now())

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(path)

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def start(cfg, run_dir):
    run_dir = Path(run_dir)

    (run_dir / "notes.md").write_text(
        f"# {cfg.title}\n\n{cfg.notes}\n\n"
    )

    seed(cfg.seed)
    log.setup([])

    train = importlib.import_module(f"simple_rl.algos.{cfg.algo.name}.train").train
    write_status(run_dir, status="running", started=now())
    start_time = time.time()

    try:
        train(cfg, Run(cfg=cfg, dir=run_dir))
    except KeyboardInterrupt:
        write_status(run_dir, status="interrupted", duration_s=round(time.time() - start_time))
        raise
    except Exception as e:
        write_status(run_dir, status="crashed", error=f"{type(e).__name__}: {e}", duration_s=round(time.time() - start_time))
        raise
    finally:
        log.close()

    write_status(run_dir, status="finished", duration_s=round(time.time() - start_time))