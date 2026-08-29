"""
VoxCPMQwenModel: a proof-of-concept variant of VoxCPM2Model whose
text-semantic backbone is a pretrained Qwen2-family LLM (default:
Qwen/Qwen2.5-0.5B) instead of MiniCPM-4.

Scope of this PoC (see POC_QWEN_BACKBONE.md at the repo root for the full
write-up):
  - Only `base_lm` (the text-semantic backbone) is replaced. The residual
    acoustic LM, local encoder (LocEnc) and local DiT (LocDiT) stay
    MiniCPM-style modules -- they are small, randomly-initialized,
    task-specific components that were never "MiniCPM-4" in the sense of a
    pretrained LLM, so there's nothing meaningful to swap there.
  - AudioVAE V2 is reused as-is (frozen) from an existing VoxCPM2 checkpoint.
  - Everything downstream of the backbone (LocEnc, residual LM, LocDiT,
    projections, stop head) is randomly initialized and MUST be trained --
    this PoC does not produce speech until you fine-tune it on paired
    text+audio data (see init_poc_checkpoint() below and
    scripts/init_voxcpm_qwen_poc.py).
  - Restricting the PoC to Hindi + English is a *data* decision (what you
    put in your training manifest / how you filter examples), not an
    architecture one -- nothing here hardcodes a language allowlist.

Two Qwen-tokenizer-specific corrections vs. base VoxCPM2Model:
  1. VoxCPM2's audio marker tokens (audio_start/end, ref_audio_start/end) are
     hardcoded ids (101-104) into MiniCPM's tokenizer, which happen to be
     otherwise-unused slots there. Those ids ARE real BPE tokens in Qwen's
     vocabulary, so reusing them would silently corrupt text. This module
     instead adds 4 new special tokens to the Qwen tokenizer and resizes the
     backbone's embedding table to match (see
     prepare_qwen_tokenizer_and_vocab_size()).
  2. `from_local()` loads the tokenizer with `AutoTokenizer` instead of the
     hardcoded `LlamaTokenizerFast` VoxCPM2Model.from_local() uses, since
     Qwen ships a BPE tokenizer, not MiniCPM's.
"""

import json
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

from ..modules.audiovae import AudioVAEV2
from ..modules.minicpm4.config import MiniCPM4Config, RopeScalingConfig
from ..modules.qwen_backbone import QwenLMBackbone
from .utils import get_dtype
from .voxcpm2 import LoRAConfig, VoxCPMConfig, VoxCPM2Model

try:
    from safetensors.torch import load_file

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


# New special tokens standing in for VoxCPM2's hardcoded 101-104 marker ids,
# which are unsafe to reuse against a real (non-MiniCPM) BPE vocabulary.
QWEN_AUDIO_SPECIAL_TOKENS = [
    "<|voxcpm_audio_start|>",
    "<|voxcpm_audio_end|>",
    "<|voxcpm_ref_audio_start|>",
    "<|voxcpm_ref_audio_end|>",
]


class VoxCPMQwenConfig(VoxCPMConfig):
    architecture: str = "voxcpm2-qwen"
    qwen_model_name_or_path: str = "Qwen/Qwen2.5-0.5B"
    freeze_qwen_backbone: bool = False


VoxCPMQwenConfig.model_rebuild()


def prepare_qwen_tokenizer_and_vocab_size(
    qwen_model_name_or_path: str,
    trust_remote_code: bool = False,
) -> Tuple[AutoTokenizer, int]:
    """Load Qwen's own tokenizer and add VoxCPM's audio marker special
    tokens to it. Returns (tokenizer, resized_vocab_size) -- pass
    resized_vocab_size to QwenLMBackbone(vocab_size_override=...) so the
    backbone's embedding table has rows for the new tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_name_or_path, trust_remote_code=trust_remote_code)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": QWEN_AUDIO_SPECIAL_TOKENS})
    if num_added != len(QWEN_AUDIO_SPECIAL_TOKENS):
        # Tokens may already exist if this tokenizer was prepared before --
        # that's fine, just confirm none of them alias a real token below.
        pass
    return tokenizer, len(tokenizer)


def build_default_lm_config_for_qwen(qwen_model_name_or_path: str, max_length: int = 8192) -> MiniCPM4Config:
    """Build the MiniCPM4Config "shape template" used to size VoxCPM2's
    auxiliary MiniCPM-style modules (residual acoustic LM, LocEnc, LocDiT)
    and every projection layer around the backbone.

    IMPORTANT: `hidden_size` here must equal the Qwen backbone's actual
    hidden size, because VoxCPM2Model's projections (enc_to_lm_proj,
    fusion_concat_proj, stop_proj, the FSQ layer, ...) and the text/audio
    embedding fusion in forward() are all sized off config.lm_config.
    hidden_size. This function reads Qwen's real hidden_size/num_attention_
    heads/num_key_value_heads so the residual LM inherits a config that at
    least divides evenly; its depth (num_hidden_layers) is overridden
    separately by VoxCPM2Model.__init__ via config.residual_lm_num_layers.
    """
    qwen_cfg = AutoConfig.from_pretrained(qwen_model_name_or_path)
    head_dim = getattr(qwen_cfg, "head_dim", qwen_cfg.hidden_size // qwen_cfg.num_attention_heads)

    # use_mup=False -> MiniCPMDecoderLayer falls back to plain residual adds
    # for the auxiliary modules (they're trained from scratch here, not
    # inheriting MiniCPM's muP-tuned pretraining, so there's no reason to
    # opt into that parametrization).
    rope_scaling = RopeScalingConfig(
        type="default",
        short_factor=[1.0] * (head_dim // 2),
        long_factor=[1.0] * (head_dim // 2),
        original_max_position_embeddings=max_length,
    )

    eos_token_id = qwen_cfg.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    return MiniCPM4Config(
        bos_token_id=getattr(qwen_cfg, "bos_token_id", 0) or 0,
        eos_token_id=eos_token_id or 0,
        hidden_size=qwen_cfg.hidden_size,
        intermediate_size=qwen_cfg.intermediate_size,
        max_position_embeddings=max_length,
        num_attention_heads=qwen_cfg.num_attention_heads,
        num_hidden_layers=qwen_cfg.num_hidden_layers,  # overridden per-submodule by callers
        num_key_value_heads=getattr(qwen_cfg, "num_key_value_heads", qwen_cfg.num_attention_heads),
        rms_norm_eps=getattr(qwen_cfg, "rms_norm_eps", 1e-6),
        rope_scaling=rope_scaling,
        vocab_size=qwen_cfg.vocab_size,  # informational only for the Qwen path; aux modules set vocab_size=0
        use_mup=False,
        scale_emb=1.0,
        dim_model_base=256,
        scale_depth=1.0,
        rope_theta=getattr(qwen_cfg, "rope_theta", 10000.0),
        no_rope=False,
    )


class VoxCPMQwenModel(VoxCPM2Model):
    """VoxCPM2Model with a pluggable Qwen backbone. See module docstring."""

    def __init__(
        self,
        config: VoxCPMQwenConfig,
        tokenizer,
        audio_vae: AudioVAEV2,
        lora_config: Optional[LoRAConfig] = None,
        device: Optional[str] = None,
    ):
        super().__init__(config, tokenizer, audio_vae, lora_config, device=device)

        # Re-point the audio marker token ids at the special tokens added by
        # prepare_qwen_tokenizer_and_vocab_size(), instead of the base
        # class's hardcoded 101-104 (which collide with real Qwen BPE
        # tokens -- see module docstring).
        ids = self.text_tokenizer.convert_tokens_to_ids(QWEN_AUDIO_SPECIAL_TOKENS)
        unk_id = getattr(self.text_tokenizer, "unk_token_id", None)
        if any(i is None or i == unk_id for i in ids):
            raise ValueError(
                "Qwen tokenizer is missing VoxCPM's audio marker special tokens "
                f"{QWEN_AUDIO_SPECIAL_TOKENS}. Build the tokenizer with "
                "prepare_qwen_tokenizer_and_vocab_size() before constructing VoxCPMQwenModel, "
                "or use init_poc_checkpoint() which does this for you."
            )
        (
            self.audio_start_token,
            self.audio_end_token,
            self.ref_audio_start_token,
            self.ref_audio_end_token,
        ) = ids

    def _build_base_lm(self, config: VoxCPMQwenConfig) -> nn.Module:
        return QwenLMBackbone(
            model_name_or_path=config.qwen_model_name_or_path,
            freeze=config.freeze_qwen_backbone,
            vocab_size_override=config.lm_config.vocab_size,
        )

    # ------------------------------------------------------------------ #
    # Checkpoint I/O -- overridden because the tokenizer is Qwen's BPE
    # tokenizer (AutoTokenizer), not MiniCPM's LlamaTokenizerFast, and
    # because the config schema is VoxCPMQwenConfig, not VoxCPMConfig.
    # ------------------------------------------------------------------ #
    @classmethod
    def from_local(
        cls,
        path: str,
        optimize: bool = True,
        training: bool = False,
        device: Optional[str] = None,
        lora_config: Optional[LoRAConfig] = None,
    ):
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            config = VoxCPMQwenConfig.model_validate_json(f.read())
        tokenizer = AutoTokenizer.from_pretrained(path)

        audio_vae_config = getattr(config, "audio_vae_config", None)
        audio_vae = AudioVAEV2(config=audio_vae_config) if audio_vae_config else AudioVAEV2()
        audiovae_safetensors_path = os.path.join(path, "audiovae.safetensors")
        audiovae_pth_path = os.path.join(path, "audiovae.pth")
        if os.path.exists(audiovae_safetensors_path) and SAFETENSORS_AVAILABLE:
            vae_state_dict = load_file(audiovae_safetensors_path, device="cpu")
        elif os.path.exists(audiovae_pth_path):
            checkpoint = torch.load(audiovae_pth_path, map_location="cpu", weights_only=True)
            vae_state_dict = checkpoint.get("state_dict", checkpoint)
        else:
            raise FileNotFoundError(
                f"AudioVAE checkpoint not found. Expected either {audiovae_safetensors_path} or {audiovae_pth_path}"
            )

        model = cls(config, tokenizer, audio_vae, lora_config, device=device)
        if not training:
            model = model.to(get_dtype(model.config.dtype))
        else:
            for name, param in model.named_parameters():
                if "audio_vae" in name:
                    param.requires_grad = False
                    continue
                if lora_config is not None and "lora" not in name:
                    param.requires_grad = False
        model.audio_vae = model.audio_vae.to(torch.float32)

        safetensors_path = os.path.join(path, "model.safetensors")
        pytorch_model_path = os.path.join(path, "pytorch_model.bin")
        model_state_dict = {}
        if os.path.exists(safetensors_path) and SAFETENSORS_AVAILABLE:
            model_state_dict = load_file(safetensors_path)
        elif os.path.exists(pytorch_model_path):
            checkpoint = torch.load(pytorch_model_path, map_location="cpu", weights_only=True)
            model_state_dict = checkpoint.get("state_dict", checkpoint)
        # NOTE: unlike VoxCPM2Model, a missing model.safetensors is NOT fatal
        # here -- a freshly init_poc_checkpoint()'d directory only ships
        # weights for the non-Qwen parts (residual LM, LocEnc, LocDiT,
        # projections); the Qwen backbone's own pretrained weights are
        # already loaded inside _build_base_lm() via AutoModelForCausalLM.

        for kw, val in vae_state_dict.items():
            model_state_dict[f"audio_vae.{kw}"] = val

        model.load_state_dict(model_state_dict, strict=False)
        if training:
            return model
        return model.to(model.device).eval().optimize(disable=not optimize)

    @classmethod
    def init_poc_checkpoint(
        cls,
        save_path: str,
        qwen_model_name_or_path: str = "Qwen/Qwen2.5-0.5B",
        reference_voxcpm2_path: Optional[str] = None,
        max_length: int = 8192,
        device: str = "cpu",
    ) -> "VoxCPMQwenModel":
        """Build and save a fresh (untrained apart from the Qwen backbone)
        VoxCPM2-with-Qwen-backbone checkpoint directory, ready to be pointed
        at by a training config's `pretrained_path`.

        reference_voxcpm2_path: path to an existing VoxCPM2 checkpoint (e.g.
        a downloaded openbmb/VoxCPM2 snapshot) to borrow the AudioVAE V2
        weights + config from. AudioVAE is reused frozen -- see module
        docstring for why it doesn't need to change.
        """
        os.makedirs(save_path, exist_ok=True)

        tokenizer, vocab_size = prepare_qwen_tokenizer_and_vocab_size(qwen_model_name_or_path)
        lm_config = build_default_lm_config_for_qwen(qwen_model_name_or_path, max_length=max_length)
        lm_config.vocab_size = vocab_size

        audio_vae_config = None
        if reference_voxcpm2_path is not None:
            with open(os.path.join(reference_voxcpm2_path, "config.json"), "r", encoding="utf-8") as f:
                ref_config = json.load(f)
            audio_vae_config = ref_config.get("audio_vae_config")

        from .voxcpm2 import VoxCPMDitConfig, VoxCPMEncoderConfig, CfmConfig  # local import, avoids cycles at module load

        config_kwargs = dict(
            lm_config=lm_config,
            encoder_config=VoxCPMEncoderConfig(),
            dit_config=VoxCPMDitConfig(cfm_config=CfmConfig()),
            max_length=max_length,
            device=device,
            dtype="float32" if device == "cpu" else "bfloat16",
            qwen_model_name_or_path=qwen_model_name_or_path,
            freeze_qwen_backbone=False,
        )
        if audio_vae_config is not None:
            config_kwargs["audio_vae_config"] = audio_vae_config
        config = VoxCPMQwenConfig(**config_kwargs)

        audio_vae = AudioVAEV2(config=config.audio_vae_config) if config.audio_vae_config else AudioVAEV2()
        if reference_voxcpm2_path is not None:
            vae_path = os.path.join(reference_voxcpm2_path, "audiovae.safetensors")
            if os.path.exists(vae_path) and SAFETENSORS_AVAILABLE:
                audio_vae.load_state_dict(load_file(vae_path, device="cpu"), strict=False)
                print(f"Loaded AudioVAE weights from {vae_path}", file=sys.stderr)
            else:
                print(
                    f"WARNING: no audiovae.safetensors found under {reference_voxcpm2_path}; "
                    "AudioVAE is randomly initialized (won't produce usable audio).",
                    file=sys.stderr,
                )
        else:
            print(
                "WARNING: no reference_voxcpm2_path given; AudioVAE is randomly initialized. "
                "Pass a downloaded openbmb/VoxCPM2 snapshot path to reuse its trained AudioVAE.",
                file=sys.stderr,
            )

        model = cls(config, tokenizer, audio_vae, lora_config=None, device=device)

        # Save everything a from_local() call needs, EXCEPT the Qwen
        # backbone's own weights (those come from HF on every load via
        # _build_base_lm -> AutoModelForCausalLM.from_pretrained).
        state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith("base_lm.")}
        if SAFETENSORS_AVAILABLE:
            from safetensors.torch import save_file

            audio_vae_sd = {k[len("audio_vae."):]: v for k, v in state_dict.items() if k.startswith("audio_vae.")}
            non_vae_sd = {k: v for k, v in state_dict.items() if not k.startswith("audio_vae.")}
            save_file(non_vae_sd, os.path.join(save_path, "model.safetensors"))
            save_file(audio_vae_sd, os.path.join(save_path, "audiovae.safetensors"))
        else:
            torch.save(state_dict, os.path.join(save_path, "pytorch_model.bin"))

        with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
            f.write(config.model_dump_json(indent=2))
        tokenizer.save_pretrained(save_path)

        print(f"Initialized VoxCPMQwenModel PoC checkpoint at {save_path}", file=sys.stderr)
        return model
