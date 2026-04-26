#!/usr/bin/env python
"""Visual + numeric diff between the image the MODEL SEES during training
(decoded from the lerobot MP4) vs the image the MODEL SEES during eval
(rendered by LIBERO env at the matched init_state).

If they're identical pixels → my "pixel OOD" claim was bullshit, need to
look elsewhere. If meaningfully different → quantify how different and
why.

Run (libero env):
    ${LIBERO_PYTHON} benchmarks/LIBERO/eval/compare_train_vs_eval_image_zhanghe.py \
        --lerobot_dataset_path /share/zhanghe/Datasets/libero_goal_no_noops_1.0.0_lerobot \
        --episode_index 20 \
        --task_suite_name libero_goal \
        --task_id 0 \
        --init_state_idx 41 \
        --output_png /tmp/compare_train_vs_eval_ep20.png
"""
import argparse
from pathlib import Path

import numpy as np
import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from libero.libero import benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot_dataset_path", required=True)
    ap.add_argument("--episode_index", type=int, required=True)
    ap.add_argument("--task_suite_name", default="libero_goal")
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--init_state_idx", type=int, required=True)
    ap.add_argument("--num_steps_wait", type=int, default=10,
                    help="Steps to let objects settle after env reset")
    ap.add_argument("--frame_idx", type=int, default=0,
                    help="Which frame of the demo MP4 to compare against")
    ap.add_argument("--output_png", required=True)
    args = ap.parse_args()

    ds = Path(args.lerobot_dataset_path)
    chunk = args.episode_index // 1000
    primary_video = ds / f"videos/chunk-{chunk:03d}/observation.images.image/episode_{args.episode_index:06d}.mp4"
    wrist_video   = ds / f"videos/chunk-{chunk:03d}/observation.images.wrist_image/episode_{args.episode_index:06d}.mp4"

    # --- training-side image: decode from MP4 ---
    train_primary = iio.imread(str(primary_video), index=args.frame_idx)
    train_wrist   = iio.imread(str(wrist_video),   index=args.frame_idx)

    # --- eval-side image: render from env at matched init_state ---
    from benchmarks.LIBERO.eval.eval_libero import _get_libero_env, LIBERO_ENV_RESOLUTION, LIBERO_DUMMY_ACTION
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed=0)
    env.reset()
    obs = env.set_init_state(init_states[args.init_state_idx])
    # Match eval_libero.py's 10-step settle
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    # Env images are upside-down in robosuite convention; eval rotates 180 to
    # match training preprocessing. Do the same here so we compare in the
    # "what the model actually sees" orientation.
    eval_primary = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    eval_wrist   = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    env.close()

    # Shape check
    print(f"train primary: shape={train_primary.shape}, dtype={train_primary.dtype}")
    print(f"eval  primary: shape={eval_primary.shape}, dtype={eval_primary.dtype}")
    print(f"train wrist:   shape={train_wrist.shape},   dtype={train_wrist.dtype}")
    print(f"eval  wrist:   shape={eval_wrist.shape},   dtype={eval_wrist.dtype}")

    # Resize eval to match train if needed (robosuite at 256, eval_libero stores at 256)
    if eval_primary.shape != train_primary.shape:
        print(f"  shape mismatch → resizing eval to {train_primary.shape}")
        from PIL import Image
        eval_primary = np.array(Image.fromarray(eval_primary).resize((train_primary.shape[1], train_primary.shape[0]), Image.BILINEAR))
        eval_wrist   = np.array(Image.fromarray(eval_wrist).resize((train_wrist.shape[1], train_wrist.shape[0]), Image.BILINEAR))

    # Pixel stats
    def stats(name, a, b):
        d = np.abs(a.astype(np.int32) - b.astype(np.int32))
        print(f"{name}: mean|diff|={d.mean():.2f}/255  max={d.max()}  std={d.std():.2f}  "
              f"pixels-differ-by>10: {(d.max(axis=-1) > 10).sum()}/{d.shape[0]*d.shape[1]} "
              f"({100*(d.max(axis=-1) > 10).sum()/(d.shape[0]*d.shape[1]):.1f}%)")
    stats("primary", train_primary, eval_primary)
    stats("wrist  ", train_wrist,   eval_wrist)

    # Plot 3×2 grid: [train, eval, diff] × [primary, wrist]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row, (name, ti, ei) in enumerate([("primary", train_primary, eval_primary),
                                          ("wrist",   train_wrist,   eval_wrist)]):
        d = np.abs(ti.astype(np.int32) - ei.astype(np.int32)).astype(np.uint8)
        axes[row][0].imshow(ti);     axes[row][0].set_title(f"{name} TRAIN (MP4 frame {args.frame_idx})")
        axes[row][1].imshow(ei);     axes[row][1].set_title(f"{name} EVAL (env render after {args.num_steps_wait} settle steps)")
        axes[row][2].imshow(d, vmin=0, vmax=50)
        axes[row][2].set_title(f"{name} |TRAIN - EVAL| (clip 0-50)")
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(f"ep{args.episode_index} task{args.task_id} init_idx={args.init_state_idx}", fontsize=14)
    fig.tight_layout()
    Path(args.output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=120, bbox_inches="tight")
    print(f"\nwrote {args.output_png}")

    # Also dump individual images (full resolution + 224x224 model-input size)
    out_dir = Path(args.output_png).parent
    stem = Path(args.output_png).stem
    from PIL import Image
    for name, img in [("train_primary", train_primary), ("train_wrist", train_wrist),
                      ("eval_primary",  eval_primary),  ("eval_wrist",  eval_wrist)]:
        Image.fromarray(img).save(out_dir / f"{stem}__{name}_256.png")
        Image.fromarray(img).resize((224, 224), Image.BILINEAR).save(out_dir / f"{stem}__{name}_224.png")
    print(f"wrote 8 individual PNGs to {out_dir}/{stem}__*.png")


if __name__ == "__main__":
    main()
