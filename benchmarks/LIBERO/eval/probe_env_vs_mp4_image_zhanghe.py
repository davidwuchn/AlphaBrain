#!/usr/bin/env python
"""Side-by-side probe: at each step of a gt-action replay, query the model
on (env-rendered frame) vs (demo MP4 frame) and compare action outputs to
ground truth.

Cleanest demonstration of image-OOD: same model, same prompt, same state,
same robot configuration — only the image source differs.

Two phases:

    # Phase A (libero env): replay gt actions, save env frames + states
    ${LIBERO_PYTHON} benchmarks/LIBERO/eval/probe_env_vs_mp4_image_zhanghe.py \
        --phase replay_dump \
        --lerobot_dataset_path /share/zhanghe/Datasets/libero_goal_no_noops_1.0.0_lerobot \
        --episode_index 20 \
        --task_suite_name libero_goal --task_id 0 --init_state_idx 41 \
        --output_npz /tmp/envframes_ep20.npz

    # Phase B (vla env): for each step, probe model with both image sources
    ${ALPHABRAIN_PYTHON} benchmarks/LIBERO/eval/probe_env_vs_mp4_image_zhanghe.py \
        --phase probe \
        --lerobot_dataset_path /share/zhanghe/Datasets/libero_goal_no_noops_1.0.0_lerobot \
        --episode_index 20 \
        --task_description "open the middle drawer of the cabinet" \
        --env_frames_npz /tmp/envframes_ep20.npz \
        --port 5795 \
        --output_jsonl /tmp/probe_envmp4_ep20.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np


# ===== Phase A =====
def _binarize_gripper_dataset_to_robosuite(g_dataset: float) -> float:
    return -1.0 if float(g_dataset) > 0.5 else 1.0


def run_replay_dump(args):
    import pyarrow  # noqa: F401  (this phase runs in libero env if it has pyarrow)
    # Actually pyarrow may not be present in libero env. Fall back to reading
    # actions from a pre-dumped npz if present.
    pass  # below


def run_replay_dump_lib(args):
    """Replay in libero env. Reads actions from a precomputed npz to avoid
    needing pyarrow inside libero env."""
    import imageio.v3 as iio
    from libero.libero import benchmark
    from benchmarks.LIBERO.eval.eval_libero import _get_libero_env, LIBERO_ENV_RESOLUTION, LIBERO_DUMMY_ACTION, _quat2axisangle

    actions_path = args.actions_npz
    if not actions_path:
        # fall back: assume actions already dumped at default location
        actions_path = f"/tmp/replay_ep{args.episode_index}_actions.npz"
    data = np.load(actions_path)
    actions = data["actions"]
    print(f"loaded {len(actions)} gt actions from {actions_path}")

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed=args.seed)
    env.reset()
    obs = env.set_init_state(init_states[args.init_state_idx])
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    primaries, wrists, states_8d = [], [], []
    # Capture frame 0 (initial) + after each gt action
    def snapshot(obs):
        primaries.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.uint8))
        wrists.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).astype(np.uint8))
        st = np.concatenate([
            np.asarray(obs["robot0_eef_pos"]),
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"]),
        ]).astype(np.float32)
        states_8d.append(st)
    snapshot(obs)
    for t, action in enumerate(actions[:-1]):  # last action's resulting frame not needed for query
        env_action = np.concatenate([action[:6], [_binarize_gripper_dataset_to_robosuite(action[6])]]).astype(np.float32)
        obs, _, done, _ = env.step(env_action.tolist())
        snapshot(obs)
        if done:
            print(f"env reached done at step {t}")
            break
    env.close()
    primaries = np.stack(primaries)
    wrists = np.stack(wrists)
    states_8d = np.stack(states_8d)
    print(f"captured {len(primaries)} env frames (primary+wrist 256x256, state 8-d)")
    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, primary=primaries, wrist=wrists, states=states_8d)
    print(f"wrote {args.output_npz}")


# ===== Phase B =====
def run_probe(args):
    import pyarrow.parquet as pq
    import imageio.v3 as iio
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    # Load env frames
    env_data = np.load(args.env_frames_npz)
    env_primary = env_data["primary"]   # (T, 256, 256, 3) uint8
    env_wrist   = env_data["wrist"]
    env_states  = env_data["states"]    # (T, 8) float32
    T = len(env_primary)
    print(f"loaded {T} env frames from {args.env_frames_npz}")

    # Load demo MP4 frames + gt action
    ds = Path(args.lerobot_dataset_path)
    chunk = args.episode_index // 1000
    primary_video = ds / f"videos/chunk-{chunk:03d}/observation.images.image/episode_{args.episode_index:06d}.mp4"
    wrist_video   = ds / f"videos/chunk-{chunk:03d}/observation.images.wrist_image/episode_{args.episode_index:06d}.mp4"
    pqf = ds / f"data/chunk-{chunk:03d}/episode_{args.episode_index:06d}.parquet"
    mp4_primary = list(iio.imiter(str(primary_video)))
    mp4_wrist   = list(iio.imiter(str(wrist_video)))
    tbl = pq.read_table(str(pqf), columns=["action", "observation.state"])
    gt_actions = np.array([tbl["action"][i].as_py() for i in range(len(tbl))], dtype=np.float32)
    mp4_states = np.array([tbl["observation.state"][i].as_py() for i in range(len(tbl))], dtype=np.float32)

    n = min(T, len(mp4_primary), len(gt_actions))
    print(f"will probe {n} steps (env={T}, mp4={len(mp4_primary)}, gt={len(gt_actions)})")

    client = WebsocketClientPolicy(args.host, args.port)

    def query(primary, wrist, state8):
        states = np.tile(state8, (args.state_history_len, 1)).astype(np.float32)
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
        a = np.asarray(resp["data"]["normalized_actions"], dtype=np.float32)
        if a.ndim == 3: a = a[0]
        first = a[0].copy()
        first[6] = (1.0 - first[6]) / 2.0  # server [+1,-1] → dataset [0,1]
        return first

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w") as out:
        for t in range(n):
            # Use env state for env query, mp4 state for mp4 query (matches what
            # each pipeline would actually send)
            a_env = query(env_primary[t], env_wrist[t], env_states[t])
            a_mp4 = query(mp4_primary[t], mp4_wrist[t], mp4_states[t])
            gt = gt_actions[t]
            out.write(json.dumps({
                "step": t,
                "gt_action": gt.tolist(),
                "pred_env": a_env.tolist(),
                "pred_mp4": a_mp4.tolist(),
            }) + "\n")
            if t % 10 == 0:
                d_env = float(np.linalg.norm(a_env[:6] - gt[:6]))
                d_mp4 = float(np.linalg.norm(a_mp4[:6] - gt[:6]))
                print(f"  t={t:3d}  arm L2  env={d_env:.4f}  mp4={d_mp4:.4f}  "
                      f"grip env={a_env[6]:.3f} mp4={a_mp4[6]:.3f} gt={gt[6]:.3f}")
    print(f"wrote {args.output_jsonl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["replay_dump", "probe"], required=True)
    ap.add_argument("--lerobot_dataset_path", default="")
    ap.add_argument("--episode_index", type=int, default=-1)
    # Phase A
    ap.add_argument("--actions_npz", default="",
                    help="Optional: pre-dumped gt actions npz (avoid pyarrow in libero env)")
    ap.add_argument("--task_suite_name", default="libero_goal")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--init_state_idx", type=int, default=0)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output_npz", default="")
    # Phase B
    ap.add_argument("--env_frames_npz", default="")
    ap.add_argument("--task_description", default="")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--state_history_len", type=int, default=16)
    ap.add_argument("--output_jsonl", default="")
    args = ap.parse_args()

    if args.phase == "replay_dump":
        assert args.output_npz and args.lerobot_dataset_path and args.episode_index >= 0
        run_replay_dump_lib(args)
    else:
        assert args.env_frames_npz and args.output_jsonl and args.port and args.task_description
        run_probe(args)


if __name__ == "__main__":
    main()
