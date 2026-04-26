#!/usr/bin/env python
"""Probe: feed training demo's actual first-frame inputs to the server and
compare server output against the demo's recorded action[0:H].

If model truly memorized the training data (loss=0.006 says so), output
should match the recorded action chunk closely.

Run from repo root, vla env (no proxy):
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
    export NO_PROXY=127.0.0.1,localhost
    python benchmarks/LIBERO/eval/probe_demo_replay_zhanghe.py \
        --lerobot_dataset_path /share/zhanghe/Datasets/libero_goal_no_noops_1.0.0_lerobot \
        --episode_index 0 \
        --task_description "put the bowl on the plate" \
        --host 127.0.0.1 --port 5795
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import imageio.v3 as iio

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def _read_first_frame(ds_path: Path, episode_index: int):
    """Return (primary_img, wrist_img, state8, action_chunk) for episode's t=0..H-1."""
    chunk = episode_index // 1000
    pqf = ds_path / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    tbl = pq.read_table(str(pqf), columns=["observation.state", "action"])
    state0 = np.asarray(tbl["observation.state"][0].as_py(), dtype=np.float32)
    action_chunk = np.asarray([tbl["action"][i].as_py() for i in range(min(8, len(tbl)))],
                              dtype=np.float32)

    # Read first frame from each video
    primary_video = ds_path / f"videos/chunk-{chunk:03d}/observation.images.image/episode_{episode_index:06d}.mp4"
    wrist_video   = ds_path / f"videos/chunk-{chunk:03d}/observation.images.wrist_image/episode_{episode_index:06d}.mp4"
    primary = iio.imread(str(primary_video), index=0)  # (H, W, 3) uint8
    wrist   = iio.imread(str(wrist_video),   index=0)
    return primary, wrist, state0, action_chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot_dataset_path", required=True)
    ap.add_argument("--episode_index", type=int, required=True)
    ap.add_argument("--task_description", required=True,
                    help="Must exactly match the demo's task string")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5795)
    ap.add_argument("--state_history_len", type=int, default=16,
                    help="Match eval_libero.py's n=16 state tile")
    args = ap.parse_args()

    ds = Path(args.lerobot_dataset_path)
    primary, wrist, state0, demo_action = _read_first_frame(ds, args.episode_index)

    print(f"primary: shape={primary.shape}, dtype={primary.dtype}")
    print(f"wrist:   shape={wrist.shape}, dtype={wrist.dtype}")
    print(f"state0:  {state0}")
    print(f"\ndemo action[0:H] (recorded ground truth, shape {demo_action.shape}):")
    np.set_printoptions(precision=4, suppress=True)
    print(demo_action)

    # Tile state to match eval's (n, 8) shape
    states = np.tile(state0, (args.state_history_len, 1)).astype(np.float32)

    client = WebsocketClientPolicy(args.host, args.port)
    payload = {
        "batch_images": [[primary, wrist]],
        "instructions": [args.task_description],
        "states": [states.tolist()],
        "do_sample": False,
        "use_ddim": False,
        "num_ddim_steps": 10,
    }
    resp = client.infer(payload)
    if resp.get("status") == "error" or "data" not in resp:
        raise RuntimeError(f"server error: {resp}")
    out = np.asarray(resp["data"]["normalized_actions"], dtype=np.float32)
    if out.ndim == 3:
        out = out[0]

    print(f"\nserver output (chunk {out.shape}):")
    print(out)

    H = min(out.shape[0], demo_action.shape[0])
    diff = out[:H] - demo_action[:H]
    print(f"\nper-step L2 diff (first {H} steps):")
    for t in range(H):
        print(f"  t={t}: L2={np.linalg.norm(diff[t]):.4f}  diff={diff[t]}")
    print(f"\noverall  L2={np.linalg.norm(diff):.4f}  max-abs={np.abs(diff).max():.4f}")
    print(f"per-dim  max-abs across chunk: {np.abs(diff).max(axis=0)}")


if __name__ == "__main__":
    main()
