#!/usr/bin/env python3
"""
Build a fresh VoxCPM2-with-Qwen-backbone checkpoint directory, ready to be
used as `pretrained_path` by scripts/train_voxcpm_finetune.py.

This does NOT train anything. It:
  1. Loads the Qwen backbone (default Qwen2.5-0.5B) from HuggingFace.
  2. Adds VoxCPM's audio marker special tokens to Qwen's tokenizer and
     resizes the backbone's embedding table to match.
  3. Randomly initializes the residual acoustic LM, LocEnc, LocDiT, and all
     projection/stop-head layers (these were never part of any pretrained
     LLM -- they're VoxCPM-specific and always start from scratch).
  4. Reuses AudioVAE V2's weights from an existing VoxCPM2 checkpoint
     (frozen) -- pass --reference-voxcpm2-path pointing at a local snapshot
     of openbmb/VoxCPM2 (e.g. what `huggingface-cli download openbmb/VoxCPM2`
     or `VoxCPM.from_pretrained("openbmb/VoxCPM2")` produces).
  5. Saves config.json + tokenizer + the non-Qwen weights to --save-path.

Example:
    python scripts/init_voxcpm_qwen_poc.py \\
        --save-path ./pretrained_models/VoxCPM2-Qwen2.5-0.5B-poc \\
        --qwen-model Qwen/Qwen2.5-0.5B \\
        --reference-voxcpm2-path ./pretrained_models/VoxCPM2

Then point conf/voxcpm_v2_qwen/voxcpm_poc_hi_en.yaml's `pretrained_path` at
--save-path and run scripts/train_voxcpm_finetune.py against it.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from voxcpm.model.voxcpm2_qwen import VoxCPMQwenModel  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-path", required=True, help="Directory to write the PoC checkpoint into.")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-0.5B", help="HF model id or local path for the backbone.")
    parser.add_argument(
        "--reference-voxcpm2-path",
        default=None,
        help="Local path to an existing VoxCPM2 checkpoint to borrow AudioVAE V2 weights from. "
        "Strongly recommended -- without it AudioVAE is randomly initialized and won't produce usable audio.",
    )
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--device", default="cpu", help="Device to build the model on (cpu is fine for init).")
    args = parser.parse_args()

    if args.reference_voxcpm2_path is None:
        print(
            "WARNING: --reference-voxcpm2-path not given. AudioVAE will be randomly initialized. "
            "Pass a local openbmb/VoxCPM2 snapshot path to reuse its trained AudioVAE, "
            "e.g. the directory VoxCPM.from_pretrained('openbmb/VoxCPM2') downloads to.",
            file=sys.stderr,
        )

    VoxCPMQwenModel.init_poc_checkpoint(
        save_path=args.save_path,
        qwen_model_name_or_path=args.qwen_model,
        reference_voxcpm2_path=args.reference_voxcpm2_path,
        max_length=args.max_length,
        device=args.device,
    )
    print(f"Done. Point pretrained_path at: {args.save_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
