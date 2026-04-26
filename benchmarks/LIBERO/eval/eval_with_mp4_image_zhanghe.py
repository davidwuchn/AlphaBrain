#!/usr/bin/env python
"""Closed-loop eval where the IMAGE input to the model comes from the demo
MP4 (same timestep), but the env steps with model's output and the model
sees env's real state.

Tests: if the only difference between training and eval is the image source,
swapping eval's image to MP4 should restore success.

Run (libero env, server up):
    ${LIBERO_PYTHON} benchmarks/LIBERO/eval/eval_with_mp4_image_zhanghe.py \
        --lerobot_dataset_path /share/zhanghe/Datasets/libero_goal_no_noops_1.0.0_lerobot \
        --episode_index 20 \
        --task_description "open the middle drawer of the cabinet" \
        --task_suite_name libero_goal --task_id 0 --init_state_idx 41 \
        --port 5795 \
        --video_out /tmp/eval_mp4image_ep20.mp4
"""
import argparse
import sys, os
for _p in [p for p in os.environ.get("VLA_EXTRA_SYSPATH", "").split(":") if p]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import imageio
import imageio.v3 as iio
from pathlib import Path

from libero.libero import benchmark


def _binarize(g_dataset: float) -> float:
    return -1.0 if float(g_dataset) > 0.5 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot_dataset_path", required=True)
    ap.add_argument("--episode_index", type=int, required=True)
    ap.add_argument("--task_description", required=True)
    ap.add_argument("--task_suite_name", default="libero_goal")
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--init_state_idx", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--state_history_len", type=int, default=16)
    ap.add_argument("--state_source", choices=["env", "mp4"], default="env",
                    help="env=use real env state per step (cleanest); "
                         "mp4=use demo state at corresponding step (tests state OOD too)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_extra_steps", type=int, default=50,
                    help="If MP4 ends before env succeeds, repeat last MP4 frame this many extra steps")
    ap.add_argument("--video_out", required=True)
    args = ap.parse_args()

    from benchmarks.LIBERO.eval.eval_libero import _get_libero_env, LIBERO_ENV_RESOLUTION, LIBERO_DUMMY_ACTION, _quat2axisangle
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    # Load demo MP4 + state
    ds = Path(args.lerobot_dataset_path)
    chunk = args.episode_index // 1000
    mp4_primary = list(iio.imiter(str(ds / f"videos/chunk-{chunk:03d}/observation.images.image/episode_{args.episode_index:06d}.mp4")))
    mp4_wrist   = list(iio.imiter(str(ds / f"videos/chunk-{chunk:03d}/observation.images.wrist_image/episode_{args.episode_index:06d}.mp4")))
    print(f"loaded {len(mp4_primary)} MP4 frames")

    if args.state_source == "mp4":
        import pyarrow.parquet as pq
        pqf = ds / f"data/chunk-{chunk:03d}/episode_{args.episode_index:06d}.parquet"
        tbl = pq.read_table(str(pqf), columns=["observation.state"])
        mp4_states = np.array([tbl["observation.state"][i].as_py() for i in range(len(tbl))], dtype=np.float32)

    # Setup env
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed=args.seed)
    env.reset()
    obs = env.set_init_state(init_states[args.init_state_idx])
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    client = WebsocketClientPolicy(args.host, args.port)
    frames = []
    done = False
    total_steps = len(mp4_primary) + args.max_extra_steps
    for t in range(total_steps):
        # Image: MP4 frame at this step (clamp to last)
        idx = min(t, len(mp4_primary) - 1)
        img_primary = mp4_primary[idx]
        img_wrist   = mp4_wrist[idx]

        # State: env's real state OR mp4 state
        if args.state_source == "env":
            st8 = np.concatenate([
                np.asarray(obs["robot0_eef_pos"]),
                _quat2axisangle(obs["robot0_eef_quat"]),
                np.asarray(obs["robot0_gripper_qpos"]),
            ]).astype(np.float32)
        else:
            st8 = mp4_states[idx]
        states = np.tile(st8, (args.state_history_len, 1)).astype(np.float32)

        payload = {
            "batch_images": [[img_primary, img_wrist]],
            "instructions": [args.task_description],
            "states": [states.tolist()],
            "do_sample": False,
            "use_ddim": False,
            "num_ddim_steps": 10,
        }
        resp = client.infer(payload)
        if resp.get("status") == "error" or "data" not in resp:
            raise RuntimeError(f"server error at t={t}: {resp}")
        a = np.asarray(resp["data"]["normalized_actions"], dtype=np.float32)
        if a.ndim == 3: a = a[0]
        first = a[0]
        # Server output gripper: [+1,-1] → dataset [0,1] → robosuite via binarize
        grip_dataset = (1.0 - first[6]) / 2.0
        env_action = np.concatenate([first[:6], [_binarize(grip_dataset)]]).astype(np.float32)

        obs, reward, done, info = env.step(env_action.tolist())
        # Save env render frame for video (so user can see what env is doing)
        env_frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        frames.append(env_frame)
        if t % 10 == 0:
            print(f"  t={t:3d}  env step OK  done={done}  grip(server)={first[6]:+.3f}→ds={grip_dataset:.3f}→ros={env_action[6]:.0f}")
        if done:
            print(f"\n✓ SUCCESS at step {t}/{total_steps} (state_source={args.state_source})")
            break
    else:
        print(f"\n✗ FAILED — episode ran all {total_steps} steps without success")

    Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.video_out, frames, fps=20, codec="libx264")
    print(f"wrote {args.video_out}  ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
