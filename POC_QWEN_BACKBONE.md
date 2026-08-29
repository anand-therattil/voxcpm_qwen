# PoC: VoxCPM2 with a Qwen2.5-0.5B backbone (Hindi + English)

Scaffold for replacing VoxCPM2's MiniCPM-4 text-semantic backbone with a
pretrained Qwen2-family LLM, scoped to a Hindi+English proof of concept.
**No training has been run.** This is architecture + tooling only.

## What changed

| Component | Status |
|---|---|
| `src/voxcpm/modules/qwen_backbone/model.py` | New. `QwenLMBackbone` wraps `AutoModelForCausalLM` (default `Qwen/Qwen2.5-0.5B`) behind the same interface `MiniCPMModel` exposes (`embed_tokens`, `forward`, `setup_cache`, `forward_step`, `.kv_cache`). |
| `src/voxcpm/model/voxcpm2.py` | One hook added: `VoxCPM2Model._build_base_lm(config)`, called from `__init__` instead of hardcoding `MiniCPMModel(config.lm_config)`. Default behavior unchanged. |
| `src/voxcpm/model/voxcpm2_qwen.py` | New. `VoxCPMQwenModel(VoxCPM2Model)` overrides `_build_base_lm`, fixes the audio-marker-token/vocab collision (see below), overrides `from_local` (Qwen's tokenizer instead of `LlamaTokenizerFast`), and adds `init_poc_checkpoint()` to build a fresh checkpoint directory. |
| `src/voxcpm/core.py`, `scripts/train_voxcpm_finetune.py` | Both now recognize `"architecture": "voxcpm2-qwen"` and route to `VoxCPMQwenModel`. |
| `scripts/init_voxcpm_qwen_poc.py` | New CLI: builds the initial (untrained-except-Qwen) checkpoint directory. |
| `conf/voxcpm_v2_qwen/voxcpm_poc_hi_en.yaml` | New training config (LoRA by default) pointing at the above. |

## What's reused vs. replaced vs. fresh

- **Reused, frozen**: AudioVAE V2 (from an existing VoxCPM2 checkpoint you point `init_voxcpm_qwen_poc.py` at). It's a pure audio codec, independent of the backbone.
- **Replaced**: only `base_lm`, the text-semantic backbone. That's the one component actually carrying pretrained-LLM knowledge, so it's the one meaningfully "swappable."
- **Fresh / randomly initialized, must be trained**: residual acoustic LM, LocEnc, LocDiT, all projection layers (`enc_to_lm_proj`, `lm_to_dit_proj`, `res_to_dit_proj`, `fusion_concat_proj`, FSQ layer, stop head). These were never pretrained-LLM components even in stock VoxCPM2 -- they're small MiniCPM-shaped modules trained end-to-end with the backbone. Swapping the backbone doesn't change that they still need training.

## Why the audio marker tokens had to change

Stock VoxCPM2 hardcodes `audio_start_token=101`, `audio_end_token=102`,
`ref_audio_start_token=103`, `ref_audio_end_token=104` -- ids that happen to
be unused slots in MiniCPM's own tokenizer. In Qwen's vocabulary those ids
are real BPE tokens, so reusing them would silently corrupt text. Instead,
`prepare_qwen_tokenizer_and_vocab_size()` adds 4 new special tokens to
Qwen's tokenizer and resizes the backbone's embedding table (new rows
random-init, pretrained rows untouched); `VoxCPMQwenModel.__init__`
re-points the marker attributes at their real ids.

## Hindi + English scope

Nothing in the architecture enforces a language allowlist -- restricting to
Hindi/English is purely a data decision: filter `train_manifest`/
`val_manifest` (same JSONL format as `examples/train_data_example.jsonl`,
i.e. `{"audio": "...", "text": "..."}` per line) to hi/en examples before
training. Qwen2.5's tokenizer has solid Hindi (Devanagari) coverage, which
was the reasoning for choosing it as the backbone for this PoC.

## Verification done so far (no GPU/data available in this session)

`QwenLMBackbone.forward()` and `.forward_step()` (including the exact
prefill -> `kv_cache.fill_caches()` -> stepwise-decode pattern
`VoxCPM2Model`'s generation loop uses) were tested against a randomly
initialized tiny `Qwen2Config` in isolation: stepwise decode matched a full
non-cached forward pass to ~1e-7 max abs difference in float32. That
confirms the custom attention/cache/RoPE re-implementation in
`forward_step` is numerically correct in isolation.

**Not yet verified:**
- End-to-end: `VoxCPMQwenModel.init_poc_checkpoint()` -> `train_voxcpm_finetune.py`
  actually running a training step (needs a Python env with `torch`+`transformers`+
  `librosa` etc. and, ideally, real audio data -- this session couldn't install/run
  the full stack against your local audio libraries).
- Real Qwen2.5-0.5B weights specifically (only tested against a random tiny config).
- LoRA application to `QwenLMBackbone` via `apply_lora_to_named_linear_modules`
  (should work -- Qwen2Attention uses the same `q_proj/k_proj/v_proj/o_proj`
  attribute names MiniCPM does -- but not exercised here).
- `torch.compile` in `VoxCPM2Model.optimize()` against `QwenLMBackbone.forward_step`
  (stock code path, untouched, but worth a smoke test since it's a different module).

## Next steps to actually run this

1. `pip install -e .` in this repo (transformers is already a dependency).
2. Get a local VoxCPM2 snapshot for its AudioVAE weights, e.g.:
   ```python
   from voxcpm import VoxCPM
   VoxCPM.from_pretrained("openbmb/VoxCPM2")  # downloads to a local cache dir
   ```
3. Build the PoC checkpoint:
   ```bash
   python scripts/init_voxcpm_qwen_poc.py \
       --save-path ./pretrained_models/VoxCPM2-Qwen2.5-0.5B-poc \
       --qwen-model Qwen/Qwen2.5-0.5B \
       --reference-voxcpm2-path <path to the VoxCPM2 snapshot from step 2>
   ```
4. Prepare a Hindi+English manifest (JSONL, `{"audio": ..., "text": ...}` per
   line) -- candidate sources: IndicTTS/Shrutilipi (Hindi), LibriTTS/
   Multilingual LibriSpeech (English), or your own recordings.
5. Edit `conf/voxcpm_v2_qwen/voxcpm_poc_hi_en.yaml`'s paths, then:
   ```bash
   python scripts/train_voxcpm_finetune.py --args.load conf/voxcpm_v2_qwen/voxcpm_poc_hi_en.yaml
   ```
   (matching how the existing `conf/voxcpm_v2/*.yaml` configs are invoked --
   check `scripts/train_voxcpm_finetune.py`'s `argbind` usage / any existing
   run instructions in `README.md` for the exact CLI invocation your version expects.)
6. Once a checkpoint exists, `voxcpm.generate()`/the CLI should load it via
   `architecture: "voxcpm2-qwen"` in its `config.json` -- but re-verify the
   `forward_step`/streaming path against the trained checkpoint before
   trusting generated audio; see "Not yet verified" above.
