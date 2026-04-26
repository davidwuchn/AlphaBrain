#!/usr/bin/env python
"""Combine: single-step inference + init_state_map override (force task 0 to
init_state_idx=41, the matched-to-training init state). Runs through the
standard eval framework so we get proper SR + video output.

Combines two monkey-patches:
  1. M1Inference.step → re-infer every env step, use chunk[0] only (single-step)
  2. task_suite.get_task_init_states → return only [init_states[idx_from_map]]
     for tasks in the map; forces num_trials_per_task=1
"""
import sys, os
for _p in [p for p in os.environ.get("VLA_EXTRA_SYSPATH", "").split(":") if p]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dataclasses
import json

import tyro
from libero.libero import benchmark

from benchmarks.LIBERO.eval import eval_libero as base
from benchmarks.LIBERO.model2libero_interface import M1Inference


@dataclasses.dataclass
class Args(base.Args):
    init_state_map_json: str = ""


# 1. single-step infer monkey-patch
_orig_step = M1Inference.step
def _singlestep_step(self, *args, **kwargs):
    self.cached_raw_actions = None
    self.step_counter = 0
    return _orig_step(self, *args, **kwargs)
M1Inference.step = _singlestep_step
print("[SINGLESTEP] M1Inference.step patched: re-infer every step, chunk[0] only")


def main(args: Args) -> None:
    assert args.init_state_map_json, "--args.init-state-map-json required"
    with open(args.init_state_map_json) as f:
        init_map = {int(k): int(v) for k, v in json.load(f).items()}
    args.num_trials_per_task = 1

    # 2. init_state monkey-patch
    suite_cls = benchmark.get_benchmark_dict()[args.task_suite_name]
    _orig_get = suite_cls.get_task_init_states

    def _patched_get(self, task_id):
        full = _orig_get(self, task_id)
        if task_id in init_map:
            return [full[init_map[task_id]]]
        raise RuntimeError(f"task_id={task_id} not in init_state_map")
    suite_cls.get_task_init_states = _patched_get
    print(f"[INIT MAP] {init_map} (num_trials=1)")

    base.eval_libero(args)


if __name__ == "__main__":
    tyro.cli(main)
