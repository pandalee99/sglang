# SPDX-License-Identifier: Apache-2.0
"""Output-quality verification for LingBot-Video (GPU, run manually).

Two layers of verification, per the add-model rule that a noisy/garbled
output means the implementation is wrong:

1. Non-noise heuristics on the SGLang output (always run): spatial std,
   spatial gradient, and consecutive-frame correlation cleanly separate
   natural video from noise/collapsed output. Failing any check means the
   implementation is almost certainly broken; passing is necessary but not
   sufficient — still eyeball the video.
2. Optional reference comparison (--reference-repo): runs the official
   diffusers path (`scripts/inference.py --backend diffusers`) with the
   same inputs and reports per-frame PSNR between the two outputs. NOTE:
   noise streams are not guaranteed identical across the two
   implementations even with the same seed, so PSNR is informational —
   judge structural similarity, not bitwise parity.

Usage (single GPU):

    python lingbot_video_quality_check.py \
        --model-path robbyant/lingbot-video-moe-30b-a3b \
        --output-dir /tmp/lingbot_quality

    # With the official repo for a side-by-side reference run:
    python lingbot_video_quality_check.py \
        --model-path <model_dir> \
        --reference-repo <path_to_lingbot-video_checkout> \
        --prompt-json <path_to_prompt.json> \
        --output-dir /tmp/lingbot_quality

T2I: pass --num-frames 1 (saves a .png; temporal check is skipped).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import imageio.v3 as iio
import numpy as np

# Schema-faithful compact caption (the DiT consumes the serialized caption
# JSON as its prompt; raw free-text is out-of-distribution).
COMPACT_T2V_CAPTION = {
    "comprehensive_description": {
        "scene_content_description": (
            "A small silver robot arm on a white table slowly picks up a red "
            "cube and places it into a blue bowl. The background is a plain, "
            "softly lit laboratory wall."
        ),
        "camera_movement_description": (
            "The camera is static at eye level, medium shot, with the robot "
            "arm centered and in sharp focus."
        ),
    },
    "camera_info": {
        "color": "Neutral",
        "frame_size": "Medium",
        "shot_type_angle": "Eye level",
        "lens_size": "Medium",
        "composition": "Center",
        "lighting": "Soft light",
        "lighting_type": "Artificial light",
    },
    "world_knowledge": [],
    "prominent_elements": [
        {
            "name": "robot arm",
            "description": "A small silver robot arm with a two-finger gripper.",
            "actions": [
                {
                    "timestamp": "[0.0s - 2.5s]",
                    "action": "reaches toward the red cube and grips it",
                },
                {
                    "timestamp": "[2.5s - 5.0s]",
                    "action": "lifts the cube and places it into the blue bowl",
                },
            ],
            "location": "center of the frame",
            "relative_size": "dominant",
            "shape_and_color": "articulated silver metal arm",
            "texture": "brushed metal",
            "appearance_details": "two-finger gripper, visible joints",
            "relationship": "manipulating the red cube on the table",
            "orientation": "upright, base on the table",
            "pose": "reaching and lifting",
            "expression": "",
            "clothing": "",
            "gender": "",
            "skin_tone_and_texture": "",
        }
    ],
}


def load_frames(path: str) -> np.ndarray:
    """Load a saved output as float32 [T, H, W, C] in [0, 255]."""
    if path.endswith((".png", ".jpg", ".jpeg", ".webp")):
        frame = iio.imread(path)
        return np.asarray(frame, dtype=np.float32)[None, ...]
    return np.asarray(iio.imread(path, plugin="pyav"), dtype=np.float32)


def non_noise_report(frames: np.ndarray) -> tuple[bool, list[str]]:
    """Heuristic gates separating natural output from noise/collapse."""
    lines: list[str] = []
    ok = True

    spatial_std = float(frames.std())
    passed = spatial_std > 8.0
    ok &= passed
    lines.append(
        f"  spatial_std={spatial_std:.1f} (>8 expected; ~0 means "
        f"black/collapsed output) -> {'PASS' if passed else 'FAIL'}"
    )

    grad = float(np.abs(np.diff(frames, axis=2)).mean())
    passed = grad < 35.0
    ok &= passed
    lines.append(
        f"  spatial_grad={grad:.1f} (<35 expected; pure noise is ~60+) "
        f"-> {'PASS' if passed else 'FAIL'}"
    )

    if frames.shape[0] > 1:
        flat = frames.reshape(frames.shape[0], -1)
        corrs = []
        for t in range(flat.shape[0] - 1):
            a = flat[t] - flat[t].mean()
            b = flat[t + 1] - flat[t + 1].mean()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            corrs.append(float(a @ b) / denom if denom > 0 else 0.0)
        corr = float(np.mean(corrs))
        passed = corr > 0.5
        ok &= passed
        lines.append(
            f"  temporal_corr={corr:.3f} (>0.5 expected; noise is ~0) "
            f"-> {'PASS' if passed else 'FAIL'}"
        )
    return ok, lines


def psnr_report(a: np.ndarray, b: np.ndarray) -> list[str]:
    frames = min(a.shape[0], b.shape[0])
    if a.shape[1:] != b.shape[1:]:
        return [
            f"  shape mismatch {a.shape} vs {b.shape}: resize one side or "
            "match --height/--width; skipping PSNR."
        ]
    psnrs = []
    for t in range(frames):
        mse = float(np.mean((a[t] - b[t]) ** 2))
        psnrs.append(99.0 if mse == 0 else 10.0 * np.log10(255.0**2 / mse))
    return [
        f"  frames_compared={frames} mean_psnr={np.mean(psnrs):.2f}dB "
        f"min_psnr={np.min(psnrs):.2f}dB",
        "  (informational: noise streams may differ across implementations "
        "even with the same seed — judge structure, not bitwise parity)",
    ]


def run_sglang(args: argparse.Namespace, prompt: str) -> str:
    from sglang import DiffGenerator

    generator = DiffGenerator.from_pretrained(model_path=args.model_path)
    result = generator.generate(
        sampling_params_kwargs=dict(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            fps=args.fps,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            flow_shift=args.shift,
            seed=args.seed,
            save_output=True,
            output_path=args.output_dir,
            output_file_name="sglang_output",
        ),
    )
    if result is None or result.output_file_path is None:
        raise RuntimeError("SGLang generation returned no output file.")
    return result.output_file_path


def run_reference(args: argparse.Namespace, prompt_json_path: str) -> str:
    ref_output = os.path.join(
        args.output_dir, "reference_output.png" if args.num_frames == 1 else "reference_output.mp4"
    )
    mode = "t2i" if args.num_frames == 1 else "t2v"
    cmd = [
        sys.executable,
        os.path.join(args.reference_repo, "scripts", "inference.py"),
        "--backend", "diffusers",
        "--model_dir", args.model_path,
        "--mode", mode,
        "--prompt_json", prompt_json_path,
        "--output", ref_output,
        "--height", str(args.height),
        "--width", str(args.width),
        "--steps", str(args.steps),
        "--guidance_scale", str(args.guidance_scale),
        "--shift", str(args.shift),
        "--seed", str(args.seed),
        "--fps", str(args.fps),
        "--transformer_dtype", "bf16",
        "--text_encoder_dtype", "bf16",
        "--vae_dtype", "fp32",
    ]
    if mode != "t2i":
        cmd += ["--num_frames", str(args.num_frames)]
    env = dict(os.environ)
    env["PYTHONPATH"] = args.reference_repo + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env, cwd=args.reference_repo)
    return ref_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="lingbot_quality_out")
    parser.add_argument("--prompt-json", default=None,
                        help="reference-style prompt.json ({caption, duration}); "
                        "defaults to a built-in compact caption")
    parser.add_argument("--reference-repo", default=None,
                        help="path to a lingbot-video checkout to also run the "
                        "official diffusers path and compare")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.prompt_json:
        with open(args.prompt_json, encoding="utf-8") as f:
            caption = json.load(f)["caption"]
        prompt_json_path = os.path.abspath(args.prompt_json)
    else:
        caption = COMPACT_T2V_CAPTION
        prompt_json_path = os.path.join(args.output_dir, "prompt.json")
        duration = None if args.num_frames == 1 else round(args.num_frames / args.fps)
        with open(prompt_json_path, "w", encoding="utf-8") as f:
            json.dump({"caption": caption, "duration": duration}, f)
    prompt = json.dumps(caption, ensure_ascii=False, separators=(",", ":"))

    print("== SGLang generation ==")
    sglang_path = run_sglang(args, prompt)
    print(f"saved: {sglang_path}")
    frames = load_frames(sglang_path)

    print("== Non-noise heuristics ==")
    ok, lines = non_noise_report(frames)
    print("\n".join(lines))
    print(f"heuristics verdict: {'PASS' if ok else 'FAIL'}")

    if args.reference_repo:
        print("== Reference (diffusers) generation ==")
        ref_path = run_reference(args, prompt_json_path)
        print(f"saved: {ref_path}")
        print("== SGLang vs reference ==")
        print("\n".join(psnr_report(frames, load_frames(ref_path))))

    print(
        "\nFinal call is visual: open the saved output(s) and confirm the "
        "content matches the caption."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
