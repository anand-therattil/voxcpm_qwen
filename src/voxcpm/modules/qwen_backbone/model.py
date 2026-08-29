"""
QwenLMBackbone: adapts a HuggingFace Qwen2-family causal LM into the same
interface VoxCPM2Model expects from its text-semantic backbone (previously
MiniCPMModel), so it can be swapped in via VoxCPM2Model._build_base_lm().

Interface contract (see voxcpm.model.voxcpm2.VoxCPM2Model and
voxcpm.modules.minicpm4.model.MiniCPMModel):

  .embed_tokens(text_tokens) -> Tensor(B, T, H)
  .forward(inputs_embeds, is_causal=True) -> (hidden_states, next_decoder_cache)
      next_decoder_cache: List[(k, v)] per layer, k/v shaped
      (B, num_kv_heads, T, head_dim) -- same layout StaticKVCache uses.
  .setup_cache(batch_size, max_length, device, dtype) -> None
  .forward_step(inputs_embeds_1tok, position_id) -> Tensor(B, H)
      single-token decode step against the StaticKVCache set up above.
  .kv_cache -> StaticKVCache instance (read by VoxCPM2Model's generation loop
      via .fill_caches(...) / .step() / .get_layer_cache(i))

Verification status: forward() and forward_step() (including the
prefill -> fill_caches -> stepwise-decode pattern VoxCPM2Model's generation
loop actually uses) were cross-checked against a randomly-initialized tiny
Qwen2Config and matched a plain full-sequence forward to float32 machine
precision (~1e-7 max abs diff). They have NOT been exercised against real
Qwen2.5-0.5B weights or wired end-to-end through VoxCPM2's generation loop --
do that before trusting this for streaming inference / voxcpm.generate().
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from ..minicpm4.cache import StaticKVCache


class QwenLMBackbone(nn.Module):
    """Wraps a pretrained Qwen2-family causal LM (e.g. Qwen2.5-0.5B) so it can
    stand in for VoxCPM2's MiniCPM-4 text-semantic backbone.

    Only the *backbone* transformer is replaced. VoxCPM2's residual acoustic
    LM, local encoder and local DiT remain MiniCPM-style modules (see
    VoxCPMQwenModel in model/voxcpm2_qwen.py) -- they are small,
    randomly-initialized, task-specific components, not a pretrained LLM, so
    there's nothing to "replace" there.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-0.5B",
        freeze: bool = False,
        trust_remote_code: bool = False,
        vocab_size_override: Optional[int] = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

        causal_lm = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        if vocab_size_override is not None and vocab_size_override != causal_lm.config.vocab_size:
            # Grows the embedding table (new rows are randomly initialized,
            # pretrained rows are preserved) -- used to add VoxCPM's audio
            # marker special tokens on top of Qwen's own vocabulary. See
            # model/voxcpm2_qwen.py:prepare_qwen_tokenizer_and_vocab_size().
            causal_lm.resize_token_embeddings(vocab_size_override)

        self.qwen = causal_lm.model  # the bare decoder stack (Qwen2Model/Qwen3Model), no LM head
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        if freeze:
            for p in self.qwen.parameters():
                p.requires_grad = False

        self.kv_cache: Optional[StaticKVCache] = None

    # ------------------------------------------------------------------ #
    # Interface parity with MiniCPMModel
    # ------------------------------------------------------------------ #
    @property
    def embed_tokens(self) -> nn.Module:
        return self.qwen.embed_tokens

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Training-time / full-sequence forward. Mirrors MiniCPMModel.forward.

        `is_causal` is accepted for interface parity; Qwen2Model always builds
        a causal mask internally when none is supplied and inputs_embeds is
        used, which is the only mode VoxCPM2 exercises here.
        """
        from transformers.cache_utils import DynamicCache

        past = DynamicCache()
        out = self.qwen(
            inputs_embeds=inputs_embeds,
            past_key_values=past,
            use_cache=True,
        )
        hidden_states = out.last_hidden_state
        next_decoder_cache = self._extract_kv_list(out.past_key_values)
        return hidden_states, next_decoder_cache

    def setup_cache(self, batch_size: int, max_length: int, device, dtype: torch.dtype):
        self.kv_cache = StaticKVCache(
            num_layers=self.num_hidden_layers,
            num_kv_heads=self.num_key_value_heads,
            dim_kv_head=self.head_dim,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            max_length=max_length,
        )

    def forward_step(
        self,
        inputs_embeds: torch.Tensor,
        position_id: torch.Tensor,
    ) -> torch.Tensor:
        """Single-token decode step against self.kv_cache (a StaticKVCache),
        mirroring MiniCPMModel.forward_step / MiniCPMDecoderLayer.forward_step
        so VoxCPM2Model's existing autoregressive generation loop (which calls
        `.kv_cache.fill_caches(...)`, `.kv_cache.step()`, and this method
        directly) works unmodified against a Qwen backbone.

        This re-implements Qwen2's attention forward against the raw
        StaticKVCache buffer instead of going through transformers' own Cache
        abstraction, because VoxCPM2's generation loop pre-fills a
        StaticKVCache from a prompt pass and then steps it token-by-token.
        """
        assert self.kv_cache is not None, "KV cache is not setup"
        bsz = inputs_embeds.size(0)
        hidden_states = inputs_embeds.unsqueeze(1)  # (B, 1, H)

        position_ids = position_id.view(1, 1).expand(bsz, 1)
        cos, sin = self.qwen.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.qwen.layers):
            hidden_states = self._decoder_layer_step(
                layer, hidden_states, (cos, sin), position_id, self.kv_cache.get_layer_cache(layer_idx)
            )

        hidden_states = self.qwen.norm(hidden_states)
        return hidden_states.squeeze(1)

    def _decoder_layer_step(self, layer, hidden_states, position_emb, position_id, kv_cache):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states = self._attn_step(layer.self_attn, hidden_states, position_emb, position_id, kv_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def _attn_step(self, attn, hidden_states, position_emb, position_id, kv_cache):
        bsz, q_len, _ = hidden_states.size()
        assert q_len == 1
        num_heads = self.num_attention_heads
        num_kv_heads = self.num_key_value_heads
        head_dim = self.head_dim

        q = attn.q_proj(hidden_states).view(bsz, 1, num_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(hidden_states).view(bsz, 1, num_kv_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(bsz, 1, num_kv_heads, head_dim).transpose(1, 2)

        cos, sin = position_emb
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        key_cache, value_cache = kv_cache
        key_cache[:, :, position_id, :] = k
        value_cache[:, :, position_id, :] = v

        attn_mask = (torch.arange(key_cache.size(2), device=key_cache.device) <= position_id).view(1, 1, 1, -1)

        q = q.contiguous()
        key_cache_c = key_cache.contiguous()
        value_cache_c = value_cache.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, key_cache_c, value_cache_c, attn_mask=attn_mask, enable_gqa=True
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, 1, num_heads * head_dim)
        return attn.o_proj(attn_output)

    @staticmethod
    def _extract_kv_list(past_key_values) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Normalize a transformers Cache object into a list of (k, v) tuples,
        one per layer, shaped (B, num_kv_heads, T, head_dim) -- the layout
        StaticKVCache.fill_caches() expects. Handles both the newer
        `cache.layers[i].keys/.values` API and the legacy
        `cache.key_cache[i]/value_cache[i]` API across transformers versions.
        """
        if hasattr(past_key_values, "layers"):
            return [(layer.keys, layer.values) for layer in past_key_values.layers]
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    # cos/sin from Qwen2RotaryEmbedding come back as (B, T, head_dim); add the
    # head axis for broadcasting against (B, num_heads, T, head_dim).
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed
