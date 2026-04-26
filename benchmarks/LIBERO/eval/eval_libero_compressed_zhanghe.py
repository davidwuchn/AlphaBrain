#!/usr/bin/env python
"""Drop-in eval client wrapper that runs each env-rendered image through a
codec round-trip BEFORE sending to the model — to mimic the av1 compression
artifacts the model saw at training time (when it consumed lerobot MP4 frames).

Hypothesis being tested: the only train→eval distribution gap is image
compression artifacts. If this fixes SR from 0% to high, the hypothesis is
confirmed and no retraining is needed — eval just needs to match training's
input distribution.

Codec options (per-frame, real-time):
  jpeg  — fast (~5 ms/frame at 256×256), DCT-based artifacts approximate av1
  av1   — slow (~hundreds of ms/frame), exact match to lerobot MP4 codec
  none  — disable, identical to vanilla eval_libero.py (sanity check)

Run (libero env, server up):
    ${LIBERO_PYTHON} benchmarks/LIBERO/eval/eval_libero_compressed_zhanghe.py \
        --args.pretrained-path ./results/training/Pi05-goal-task0/checkpoints/steps_32500 \
        --args.host 127.0.0.1 --args.port 5795 \
        --args.task-suite-name libero_goal \
        --args.num-trials-per-task 10 \
        --args.video-out-path /tmp/eval_compressed/videos \
        --args.image-codec jpeg \
        --args.jpeg-quality 75
"""
import sys, os
for _p in [p for p in os.environ.get("VLA_EXTRA_SYSPATH", "").split(":") if p]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dataclasses
import io
import time
import tempfile
import numpy as np

import tyro

from benchmarks.LIBERO.eval import eval_libero as base
from benchmarks.LIBERO.model2libero_interface import M1Inference


@dataclasses.dataclass
class Args(base.Args):
    image_codec: str = "jpeg"     # jpeg | h264 | av1 | blur | resize | none
    jpeg_quality: int = 75
    av1_quality: int = 8
    h264_quality: int = 8
    blur_sigma: float = 1.0       # gaussian blur sigma (when codec=blur)
    resize_intermediate: int = 128  # resize-down then resize-up size (when codec=resize)


def _jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality, subsampling=2)  # 4:2:0 like av1
    buf.seek(0)
    return np.array(Image.open(buf))


def _video_codec_roundtrip(img: np.ndarray, codec: str, quality: int) -> np.ndarray:
    import imageio
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        imageio.mimwrite(f.name, [img], codec=codec, fps=20, quality=quality)
        return next(iter(imageio.imiter(f.name)))


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        out[..., c] = gaussian_filter(img[..., c], sigma=sigma)
    return out


def _resize_roundtrip(img: np.ndarray, intermediate: int) -> np.ndarray:
    from PIL import Image
    h, w = img.shape[:2]
    pil = Image.fromarray(img)
    return np.array(pil.resize((intermediate, intermediate), Image.BILINEAR).resize((w, h), Image.BILINEAR))


def main(args: Args) -> None:
    if args.image_codec == "jpeg":
        compress = lambda img: _jpeg_roundtrip(img, args.jpeg_quality)
        codec_desc = f"jpeg q={args.jpeg_quality}"
    elif args.image_codec == "av1":
        compress = lambda img: _video_codec_roundtrip(img, "libaom-av1", args.av1_quality)
        codec_desc = f"av1 q={args.av1_quality}"
    elif args.image_codec == "h264":
        compress = lambda img: _video_codec_roundtrip(img, "h264", args.h264_quality)
        codec_desc = f"h264 q={args.h264_quality}"
    elif args.image_codec == "blur":
        compress = lambda img: _gaussian_blur(img, args.blur_sigma)
        codec_desc = f"gaussian_blur sigma={args.blur_sigma}"
    elif args.image_codec == "resize":
        compress = lambda img: _resize_roundtrip(img, args.resize_intermediate)
        codec_desc = f"resize_roundtrip intermediate={args.resize_intermediate}"
    elif args.image_codec == "none":
        compress = lambda img: img
        codec_desc = "none (identical to vanilla eval)"
    else:
        raise ValueError(f"unknown image_codec={args.image_codec!r}")

    # Per-frame latency probe (1× warmup + 5× timing) so user knows the cost
    dummy = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    compress(dummy)  # warmup
    t0 = time.time()
    for _ in range(5):
        compress(dummy)
    avg_ms = (time.time() - t0) / 5 * 1000
    print(f"[image-codec] {codec_desc} — {avg_ms:.1f} ms/frame (avg of 5)")

    # Monkey-patch M1Inference.step to compress images first.
    # Signature: step(self, images, task_description=None, **kwargs)
    orig_step = M1Inference.step

    def patched_step(self, images, *a, **kw):
        compressed = [compress(np.ascontiguousarray(img)) for img in images]
        return orig_step(self, compressed, *a, **kw)

    M1Inference.step = patched_step
    print(f"[image-codec] M1Inference.step patched — every image goes through {codec_desc} round-trip")

    base.eval_libero(args)


if __name__ == "__main__":
    tyro.cli(main)
