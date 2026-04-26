#!/usr/bin/env python
"""Run the actual training dataloader for one batch, dump what `lang` and
other fields contain as passed to the model. Answers: does training-time
prompt string match eval-time task.language, or is it something different
(empty, wrong task name, etc.)?
"""
import sys, os
from pathlib import Path
from omegaconf import OmegaConf

# mirror what train_alphabrain.py does
sys.path.insert(0, "/share/zhanghe/AlphaBrain-zh")

from AlphaBrain.dataloader.lerobot_datasets import get_vla_dataset

CFG_PATH = "/share/zhanghe/AlphaBrain-zh/configs/finetune_config.yaml"

# Load the pi05_goal_task0 mode's data_cfg
full = OmegaConf.load(CFG_PATH)
mode = full["modes"]["pi05_goal_task0"]
data_cfg = OmegaConf.merge(
    OmegaConf.load("/share/zhanghe/AlphaBrain-zh/configs/datasets/libero.yaml").datasets.vla_data,
    mode.datasets.vla_data,
)
# Some fields need expansion
data_cfg.data_root_dir = os.environ.get("LIBERO_DATA_ROOT", "/share/zhanghe/Datasets")

print(f"=== data_cfg.task_whitelist = {data_cfg.get('task_whitelist', None)}")
print(f"=== data_cfg.dataset_mix     = {data_cfg.dataset_mix}")
print(f"=== data_cfg.data_root_dir   = {data_cfg.data_root_dir}")
print()

ds = get_vla_dataset(data_cfg, mode="train")
print(f"dataset length (step count): {len(ds)}")
print()

# Dump 5 random samples' lang + task_index
import numpy as np
rng = np.random.RandomState(0)
indices = rng.choice(len(ds), size=5, replace=False).tolist()
for i in indices:
    sample = ds[i]
    print(f"--- sample idx={i} ---")
    print(f"  keys: {list(sample.keys())}")
    # Lang
    lang = sample.get("annotation.human.action.task_description")
    print(f"  annotation.human.action.task_description = {lang!r}")
    # Alt names (gr00t sometimes stores differently)
    for k in sample:
        if "annotation" in k or "lang" in k or "instruction" in k or "task" in k.lower():
            v = sample[k]
            if hasattr(v, "shape"): v = f"<ndarray shape={v.shape}>"
            print(f"  {k} = {v!r}")
    # Action first step (gripper)
    if "action" in sample:
        a = sample["action"]
        print(f"  action shape: {getattr(a, 'shape', type(a))}  first: {a[0] if len(a) else 'empty'}")
