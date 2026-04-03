#!/usr/bin/env python3
"""
Debug script to capture intermediate values from sglang LTX-2.3 pipeline.
Outputs: text embeddings, sigmas, latents at various steps.

Run with: /workspace/lab/sglang/.venv/bin/python debug_sglang.py
"""

import os
import sys
import numpy as np

# Set environment to capture debug info
os.environ["SGLANG_LTX_DEBUG"] = "1"

sys.path.insert(0, "python")

import torch
from sglang.multimodal_gen import DiffGenerator

# Config - must match native script
MODEL_PATH = "/workspace/lab/models/ltx-2.3-diffusers"
PROMPT = "A curious raccoon exploring a garden at sunset"
SEED = 42
HEIGHT = 384
WIDTH = 512
NUM_FRAMES = 41
NUM_INFERENCE_STEPS = 30

OUTPUT_FILE = "/tmp/sglang_ltx_debug.npz"


def main():
    print("=" * 60)
    print("sglang LTX-2.3 Debug Script")
    print("=" * 60)

    print("\n[1/3] Loading model...")
    gen = DiffGenerator.from_pretrained(model_path=MODEL_PATH)

    print("\n[2/3] Running generation with debug hooks...")

    # Use the generate API
    result = gen.generate(
        sampling_params_kwargs={
            "prompt": PROMPT,
            "seed": SEED,
            "height": HEIGHT,
            "width": WIDTH,
            "num_frames": NUM_FRAMES,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": 3.0,  # Match native CFG
            "save_output": False,
            "return_frames": True,
        }
    )

    print("\n[3/3] Extracting debug info...")

    # The debug info should be captured in the result or we need to modify the pipeline
    # For now, let's just capture what we can from the result
    if hasattr(result, 'latents'):
        final_latent = result.latents
        print(f"  Final latent shape: {final_latent.shape}")
        print(f"  Final latent stats: mean={final_latent.float().mean():.6f}, std={final_latent.float().std():.6f}")
    else:
        print("  Warning: latents not available in result")
        final_latent = None

    # Try to get scheduler sigmas
    # This requires internal access which may not be available
    sigmas = None

    print(f"\n[Done] Saving results to {OUTPUT_FILE}")

    save_dict = {
        "prompt": PROMPT,
        "seed": SEED,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": NUM_INFERENCE_STEPS,
    }

    if final_latent is not None:
        save_dict["final_video_latent"] = final_latent.cpu().float().numpy() if isinstance(final_latent, torch.Tensor) else final_latent

    np.savez(OUTPUT_FILE, **save_dict)

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"  Prompt: {PROMPT}")
    print(f"  Seed: {SEED}")
    print(f"  Resolution: {WIDTH}x{HEIGHT}, {NUM_FRAMES} frames")
    print(f"  Steps: {NUM_INFERENCE_STEPS}")

    # Shutdown
    gen.shutdown()


if __name__ == "__main__":
    main()
