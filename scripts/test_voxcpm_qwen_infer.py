#!/usr/bin/env python3
"""
Inference test script for a VoxCPM2 + Qwen-backbone ("voxcpm2-qwen")
checkpoint produced by scripts/train_voxcpm_finetune.py using
conf/voxcpm_v2_qwen/voxcpm_poc_hi_en.yaml.

Handles both save shapes scripts/train_voxcpm_finetune.py can produce:
  - LoRA run (lora: {...} set in the config): step_XXXXXXX/ contains
    lora_weights.safetensors + lora_config.json (no standalone config.json).
    --ckpt_dir should be that step_XXXXXXX folder; --base_model should be
    your init_voxcpm_qwen_poc.py output directory (also read automatically
    from lora_config.json if you built it with this repo's training script).
  - Full-finetune run: step_XXXXXXX/ is a standalone checkpoint (config.json,
    tokenizer files, and audiovae files are copied in automatically). Pass
    only --ckpt_dir; --base_model is not needed.

This intentionally does NOT reuse scripts/test_voxcpm_lora_infer.py: that
script imports voxcpm.model.voxcpm.LoRAConfig (the VoxCPM v1 field shape:
`target_modules`, no enable_lm/enable_dit/enable_proj split), which does not
match the fields voxcpm2 / voxcpm2-qwen's LoRA application expects
(voxcpm.model.voxcpm2.LoRAConfig). Using it against a voxcpm2-qwen LoRA
checkpoint would construct the wrong config shape. This script uses the v2
LoRAConfig directly, which is what VoxCPMQwenModel actually inherits.

Usage (LoRA):
    python scripts/test_voxcpm_qwen_infer.py \\
        --ckpt_dir checkpoints/voxcpm2_qwen_poc_hi_en/step_0002000 \\
        --text "यह एक परीक्षण वाक्य है।" \\
        --output qwen_poc_test.wav

Usage (full finetune):
    python scripts/test_voxcpm_qwen_infer.py \\
        --ckpt_dir checkpoints/voxcpm2_qwen_poc_hi_en_full/step_0002000 \\
        --text "This is a test sentence." \\
        --output qwen_poc_test.wav

With voice cloning:
    python scripts/test_voxcpm_qwen_infer.py \\
        --ckpt_dir <...> \\
        --text "..." \\
        --prompt_audio path/to/ref.wav \\
        --prompt_text "Reference audio transcript" \\
        --seed 42 --output clone.wav
"""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm2 import LoRAConfig  # v2 shape: enable_lm/enable_dit/enable_proj -- NOT voxcpm.model.voxcpm's v1 LoRAConfig


def parse_args():
    parser = argparse.ArgumentParser("VoxCPM2 (Qwen backbone) inference test", formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("--ckpt_dir", type=str, required=True, help="step_XXXXXXX checkpoint directory")
    parser.add_argument("--base_model", type=str, default="", help="Required for LoRA checkpoints unless lora_config.json has base_model set; the init_voxcpm_qwen_poc.py output dir")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--prompt_audio", type=str, default="")
    parser.add_argument("--prompt_text", type=str, default="")
    parser.add_argument("--output", type=str, default="qwen_poc_test.wav")
    parser.add_argument("--cfg_value", type=float, default=2.0)
    parser.add_argument("--inference_timesteps", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=600)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

    lora_config_path = ckpt_dir / "lora_config.json"
    is_lora = lora_config_path.exists()

    if is_lora:
        with open(lora_config_path, "r", encoding="utf-8") as f:
            lora_info = json.load(f)
        pretrained_path = args.base_model or lora_info.get("base_model")
        if not pretrained_path:
            raise ValueError(
                "This is a LoRA checkpoint but no base model path is available "
                "(pass --base_model, or make sure lora_config.json has 'base_model' set)."
            )
        lora_cfg_dict = lora_info.get("lora_config", {})
        lora_cfg = LoRAConfig(**lora_cfg_dict) if lora_cfg_dict else None
        print(f"[Qwen PoC Inference] LoRA checkpoint. base_model={pretrained_path}", file=sys.stderr)
        if lora_cfg:
            print(f"  r={lora_cfg.r} alpha={lora_cfg.alpha} enable_lm={lora_cfg.enable_lm} enable_dit={lora_cfg.enable_dit}", file=sys.stderr)
        model = VoxCPM.from_pretrained(
            hf_model_id=pretrained_path,
            load_denoiser=False,
            optimize=True,
            lora_config=lora_cfg,
            lora_weights_path=str(ckpt_dir),
        )
    else:
        print(f"[Qwen PoC Inference] Full-finetune checkpoint: {ckpt_dir}", file=sys.stderr)
        model = VoxCPM.from_pretrained(hf_model_id=str(ckpt_dir), load_denoiser=False, optimize=True)

    prompt_wav_path = args.prompt_audio or None
    prompt_text = args.prompt_text or None

    print(f"[Qwen PoC Inference] Synthesizing: '{args.text}'", file=sys.stderr)
    audio_np = model.generate(
        text=args.text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        normalize=args.normalize,
        denoise=False,
        seed=args.seed,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio_np, model.tts_model.sample_rate)
    print(
        f"[Qwen PoC Inference] Saved: {out_path}, duration: {len(audio_np) / model.tts_model.sample_rate:.2f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
