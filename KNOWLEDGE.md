# Stable Audio 3 — Colab Notebook: Knowledge Transfer

> Repo: `hashachar/stableaudio-colab` · branch `master`  
> Based on Stability AI's official [`stable-audio-3`](https://github.com/Stability-AI/stable-audio-3) package

---

## 1. Project Context

This is a Google Colab notebook that wraps Stability AI's Stable Audio 3 open-weight models in an interactive UI. The goal is to give musicians and producers access to three generative audio modes without writing code:

| Section | Feature |
|---|---|
| 5 | Text-to-audio generation |
| 7 | Inpainting (replace a time segment) |
| 8 | Audio-to-audio (variations from an existing clip) |
| 9 | RF-Inversion (structure-preserving re-stylization) |

---

## 2. Model Variants

Both models are gated on Hugging Face and require license acceptance before use.

| Model | Params | Max duration | GPU required | VRAM |
|---|---|---|---|---|
| `small-music` | 433 M | 120 s | No (CPU-ok) | ~1.7 GB |
| `medium` | 1.4 B | 285 s | Yes | ~6–7 GB |

**Architecture:** Latent diffusion — T5-Gemma text encoder → DiT (Diffusion Transformer) → SAME autoencoder. Objective: **rectified flow** (timestep `t ∈ [0,1]`, not sigma-based). Output: stereo 44.1 kHz 16-bit PCM.

**Flash Attention 2:** Required by `medium` for full performance. Only available on compute capability ≥ 8.0 (A100, L4, H100). The install cell detects the GPU and installs `flash-attn` automatically; on T4 (cc 7.5) it is skipped and the model falls back to standard attention (slower but correct).

---

## 3. Python Package

The notebook installs `stable-audio-3` directly from GitHub with `--no-deps` to avoid the `torch==2.7.1` pin conflicting with Colab's torch:

```
pip install git+https://github.com/Stability-AI/stable-audio-3.git --no-deps
```

The public API surface used by the notebook is a single method:

```python
from stable_audio_3 import StableAudioModel

model = StableAudioModel.from_pretrained("small-music", device="cuda")

output = model.generate(
    prompt             = "128 BPM tech house drum loop",
    negative_prompt    = None,               # optional
    duration           = 30.0,              # seconds
    steps              = 8,                 # diffusion steps
    cfg_scale          = 1.0,              # classifier-free guidance
    seed               = -1,               # -1 = random
    chunked_decode     = True,             # saves VRAM during decode

    # ── Audio-to-Audio ─────────────────────────────────────────────
    init_audio         = (44100, tensor),   # (sample_rate, Tensor[C,T])
    init_noise_level   = 0.45,             # 0.0–1.0

    # ── Inpainting ─────────────────────────────────────────────────
    inpaint_audio      = (44100, tensor),
    inpaint_mask_start_seconds = 5.0,
    inpaint_mask_end_seconds   = 15.0,
)
# output: Tensor[B, C, T] at 44100 Hz
```

`from_pretrained` accepts `model_half=True` to load weights in fp16 and halve VRAM.

---

## 4. Generation Modes — Technical Detail

### 4.1 Text-to-Audio

Pure generation from random Gaussian noise. Only `prompt`, `duration`, `steps`, `cfg_scale`, and `seed` matter. CFG default is 1.0 (unconditional); raise to 3–7 for clear prompt adherence.

### 4.2 Inpainting

Replaces a time segment with new content while keeping the rest unchanged.

**Mechanism:**
1. The source clip is encoded to latents.
2. A binary mask covers `[start, end]` seconds.
3. Unmasked latent regions are copied exactly into every denoising step → timbre outside the mask is structurally preserved.
4. The masked region is regenerated from noise guided by the prompt.
5. A progressive mask hardening is applied across steps: `strength = (step+1)/total_steps` — early steps preserve more, later steps allow more change.

**Parameter guidance (CFG for inpainting):**

| CFG | Effect |
|---|---|
| 0.0 | Ignores prompt; fills from musical context only |
| 1.0 ★ | Default. Mild prompt influence, good seam continuity |
| 3–5 | Noticeable prompt; slight seam risk |
| 7–10 | Strong prompt; may feel disconnected |
| 12–15 | Harsh / incoherent |

Sweet spot: **1.5–4.0**.

**Known limitation:** End-of-clip inpaints fail for melodic material (one-sided context). Mid-clip positions with bilateral context work well.

### 4.3 Audio-to-Audio (Variations)

Encodes source audio to latent space, blends noise at level `init_noise_level`, then denoises.

For rectified flow the blending is:
```
starting_latent = audio_latent × (1 − noise_level) + random_noise × noise_level
```

The denoising schedule then runs from `sigma_max = noise_level` down to 0.

**Noise level guidance:**

| Level | Effect |
|---|---|
| 0.1–0.2 | Near-identical; subtle timbre shift |
| 0.3–0.45 ★ | Sweet spot — harmonic context preserved, light drift |
| 0.5–0.65 | Creative; instrumentation may shift |
| 0.7–0.85 | Heavy; needs a prompt for direction |
| 0.9–1.0 | Approaches pure generation |

**Prompting with A2A:** With `cfg_scale=1.0` the prompt is ignored (free variation). Raise CFG to 2–4 to steer toward the description. Optimal `cfg_scale` for A2A is lower than for inpainting because there is no structural mask to absorb the guidance energy.

### 4.4 RF-Inversion (Re-Stylization)

Based on *"Semantic Image Inversion and Editing using Rectified SDEs"*. Instead of blending random noise into the audio latent, it runs a **forward ODE** to find the latent "noise fingerprint" of the audio — the noise vector that the model would denoise back into the source. Generation then denoises from that fingerprint with a new prompt, preserving structural identity while adopting the prompt's character.

This is implemented for real — **both** RF-Inversion controllers run directly on the rectified-flow model (not a substitute). Velocities come from `sa3.dit` (`= sa3.model.model`, the `DiTWrapper` the library's own samplers call), and conditioning, padding mask, and the timestep schedule are built the same way `generate()` builds them. Convention (verified against `sample_discrete_euler`): **t=1 is pure noise, t=0 is data**, and the model returns velocity `v` with `denoised = x − t·v`, i.e. `v = noise − data`.

**Two phases:**
1. **Inversion — forward controlled ODE (t: 0→1):** integrate `y₀` (the encoded source latent) to a Gaussian "fingerprint" using the model's *unconditional* velocity blended with a controller toward a fixed target noise `y₁`:
   ```
   v = (1−γ)·v_uncond + γ·(y₁ − y)/(1 − t)
   ```
2. **Generation — reverse controlled ODE (t: 1→0):** denoise from the fingerprint with the *new* prompt's velocity (with CFG), blended (strength η, active while `t ≥ eta_stop`) with a controller pulling back toward the source latent `y₀`:
   ```
   v = (1−η)·v_edit + η·(x − y₀)/t          # η applied only while t ≥ eta_stop
   ```
   The η controller is what actually preserves structural identity — without it (η=0) Phase 2 is *identical* to the stock Euler sampler used by `model.generate()`, i.e. plain "denoise from inverted noise" (DDIM-style inversion).

**Gamma (γ) guidance** — forward inversion strength:

| γ | Use case |
|---|---|
| 0.0 | Pure model drift — most semantic/faithful fingerprint, best for style transfer with strong prompt |
| 0.2–0.35 ★ | Default. Balanced inversion |
| 0.4–0.6 | Stronger push to the chosen Gaussian; more editing freedom |
| 0.7–1.0 | Heavy noise; approaches A2A behavior |

**Eta (η) guidance** — reverse structure-preservation strength (default 0.0):

| η | Use case |
|---|---|
| 0.0 | Off. Free re-style from the inverted fingerprint (Phase 2 = stock Euler sampler) |
| 0.3–0.7 ★ | Locks the source's structure while still adopting the prompt |
| 0.8–1.0 | Near-exact reconstruction; prompt has little effect |

`eta_stop` (default 0.30) releases the structure controller once `t < eta_stop`, so the final denoising steps always follow the prompt; lower it to hold structure longer.

**Implementation note:** The forward/reverse loops call `sa3.dit(x, t, cfg_scale=…, batch_cfg=True, rescale_cfg=True, padding_mask=…, apg_scale=1.0, **cond_inputs)`. `cond_inputs` is the flattened dict from `inner.get_conditioning_inputs(...)` and **must** include zero `inpaint_mask` / `inpaint_masked_input` input-concat channels (generate() always injects these — omitting them gives the DiT the wrong input channel count). A broad `try/except` still falls back to guided A2A on a genuine API break, but it now surfaces the real exception instead of claiming inversion is unavailable.

**Recommended steps:** 50 (used for both phases). 20–30 for fast previews. Cost ≈ 3× a normal generate of the same step count (Phase 1 is unconditional 1×/step, Phase 2 is CFG 2×/step).

---

## 5. Notebook Architecture

### Cell dependency chain

```
Cell 0  Style (CSS)
Cell 3  Install       ← pip installs SA3, deps, flash-attn
Cell 4  Drive Cache   ← sets HF_HOME to Google Drive path
Cell 5  HF Auth       ← login + HF_TOKEN env var
Cell 6  Utilities     ← get_model(), MODEL_CONFIGS, _session_audio, SPINNER_HTML
Cell 8  Example Prompts ← defines widgets, Layout, Audio, clear_output
Cell 9  Generate      ← SPINNER_HTML defined here
Cell 12 Inpainting    ← defines _limit_note_html()
Cell 13 A2A           ← depends on: get_model, widgets, Layout, Audio, SPINNER_HTML
Cell 14 RF-Inversion  ← same deps as A2A
```

Cells 13 and 14 can be run independently as long as cells 3–9 have run first. They do not depend on cell 12 (inpainting).

### Key shared globals (defined in Utilities cell)

| Name | Type | Purpose |
|---|---|---|
| `SAMPLE_RATE` | `int` | 44100 — canonical sample rate throughout |
| `MODEL_CONFIGS` | `dict` | Per-model metadata (max duration, defaults, GPU flag) |
| `_model_cache` | `dict` | `model_name → StableAudioModel` — keeps model in memory |
| `_session_audio` | `dict` | `title → /tmp/*.wav` — audio saved from Generate cell |
| `get_model(name)` | `fn` | Load or return cached model; evicts other cached models first |
| `_is_model_on_disk(name)` | `fn` | Checks HF cache for model checkpoint |

### Model caching pattern

`get_model()` caches one model at a time. Loading a different model evicts the current one (`del _model_cache[other]` + `gc.collect()` + `torch.cuda.empty_cache()`).

**Known pitfall:** If a model is loaded before GPU is connected (CPU run), the cache holds a CPU model. Cells 13 and 14 detect this via `next(model.parameters()).device` and reload on CUDA if needed.

### Widget pattern

Each feature cell is self-contained:
- State variable (`_a2a_audio`, `_rfi_audio`, `_inpaint_audio`) — `None` until audio is loaded
- Drive folder scan (`/content/drive/MyDrive/Inpainting-Test`) + Upload button + Session dropdown
- All widget names prefixed to avoid collisions (`a2a_*`, `rfi_*`)
- `with output_widget: clear_output(wait=True); display(...)` for live result updates

### RAM / VRAM optimizations

The Utilities cell patches `load_diffusion_cond` to load safetensors directly onto the GPU (`device=str(device)`) instead of CPU first, cutting the peak system-RAM spike by ~5 GB when loading `medium`.

---

## 6. Monkey-Patches in Utilities Cell

Two patches are applied at import time and are idempotent (re-running the cell is safe):

**1. HF token injection**
```python
_sa3_cfg.hf_hub_download = _sa3_hf_hub_download_authed
```
`stable_audio_3` calls `hf_hub_download` via a module-level local reference that bypasses Colab's token auto-detection. The wrapper injects `HF_TOKEN` from the environment when no token is provided.

**2. Low-RAM model loading**
```python
_sa3_lu.load_diffusion_cond = _load_diffusion_cond_low_ram
```
Loads safetensors directly onto the target device instead of CPU, avoiding a full-model system-RAM spike.

---

## 7. Known Issues and Fixes

### Triton flex_attention OOM warnings during inpainting
```
OutOfMemoryError: out of resource: triton_flex_attention Required: 65536 Hardware limit: 65536
```
**Status:** Benign. The Triton autotuner tries block sizes that exceed GPU shared memory (64 KB on T4), logs the failure, and falls back to `BLOCK_M=64, BLOCK_N=64`. Audio is generated correctly; only the larger configs are skipped.

**Root cause:** T4 has 64 KB shared memory per SM. Configs requiring 65536–98304 bytes don't fit.

**Fix:** None needed. On A100 (192 KB shared memory per SM) these configs succeed and the warning disappears.

### "No module named 'flash_attn'" in Utilities cell
**Status:** Fixed (install cell now installs `flash-attn` on cc ≥ 8.0).

**If still seen:** Re-run cell 1 (Install) to pick up the new flash-attn step, then re-run cell 6 (Utilities).

### A2A shows GPU RAM at 0
**Status:** Fixed. The A2A callback now checks the cached model's device and reloads on GPU if it was cached during a CPU session.

**Manual workaround if needed:**
```python
del _model_cache["small-music"]  # or "medium"
import gc, torch
gc.collect()
torch.cuda.empty_cache()
```
Then re-run the A2A cell.

### RecursionError during inpainting (historical)
`stable_audio_3` triggers deep Python recursion during model loading. Fixed by raising the recursion limit around the `get_model()` call:
```python
_sys.setrecursionlimit(max(_sys.getrecursionlimit(), 4000))
```
This is already in all three feature cells (inpainting, A2A, RF-Inversion).

---

## 8. The maxgraf96 Plugin Fork

`maxgraf96/stable-audio-3` branch `sa3-variations` turns the model into an AU/VST3/Standalone DAW plugin. Separate from the Colab notebook — included here for context.

**Architecture:**
- **C++ JUCE plugin** with an HTML5 UI (WebBrowserComponent serving embedded assets)
- **MLX inference pipeline** (Python, Apple Silicon only) — weights stay loaded in a resident worker thread; only prompt/seed/noise/mask vary per call
- **Five variation slots** with crossfade playback and drag-to-DAW export

**Model used:** `dit_medium_f16.safetensors` (Medium DiT) + `same_l_encoder/decoder_f32.safetensors` (Large autoencoder). ~6.4 GB total.

**Hard requirements:** Apple Silicon (M1+), macOS 13.5+. MLX is arm64-only; no Intel Mac fallback.

**Variation modes:**
- *Free variation*: 5 × audio-to-audio, noise=0.45, cfg=1.0 (unconditioned)
- *Preserve Sound*: 3 × a2a + 2 × inpainting, cfg=4.0 (anchors timbre)

---

## 9. SA3 Internal API (for RF-Inversion implementation)

These are the internal model attributes accessed by the RF-Inversion cell. They may break on future package versions:

```python
sa3   = get_model("medium")    # StableAudioModel
inner = sa3.model              # ConditionedDiffusionModelWrapper  (NOT the velocity field)
dit   = sa3.dit                # = sa3.model.model — DiTWrapper, the velocity field samplers call
pt    = inner.pretransform     # SAME autoencoder

inner.sample_rate              # int: 44100
inner.io_channels              # latent channel count (== C_lat of encoded latents)
pt.io_channels                 # int: 2 (stereo audio I/O)
pt.downsampling_ratio          # audio_samples → latent frames
pt.encode(audio)               # Tensor[B,C,T] → Tensor[B,C_lat,T_lat]
pt.decode(latent, chunked=True)# Tensor[B,C_lat,T_lat] → Tensor[B,C,T]

# Conditioning — positional (metadata_list, device); MUST add zero inpaint channels:
ct = inner.conditioner([{"prompt": "...", "seconds_total": dur}], str(device))
ct["inpaint_mask"]         = [torch.zeros((1,1,T_lat), ...)]
ct["inpaint_masked_input"] = [torch.zeros((1, inner.io_channels, T_lat), ...)]
cond_inputs = inner.get_conditioning_inputs(ct)               # positive (flattened dict)
neg_inputs  = inner.get_conditioning_inputs(ct, negative=True)# negative_* keys

# Velocity field (the call the library's samplers make on model.model):
dit(x, t_batch, cfg_scale=cfg, batch_cfg=True, rescale_cfg=True,
    padding_mask=pm, apg_scale=1.0, **cond_inputs)   # → velocity Tensor
# cfg_scale==1.0 → single conditional pass; cfg_scale>1.0 → internal batched CFG
# (needs negative_* keys in cond_inputs, else a null embedding is used).

# Schedule + padding mask (build them like sample_diffusion/generate do):
from stable_audio_3.inference.sampling import build_schedule         # descending 1.0→0.0
from stable_audio_3.data.utils import (
    compute_effective_seq_len_from_conditioning, create_padding_mask_from_lengths)
```

The RF-Inversion cell hand-rolls both ODE loops directly on `dit` rather than calling `sample_diffusion`, because the reverse η-controller is not expressible through the stock sampler. At η=0 the reverse loop is byte-for-byte equivalent to `sample_discrete_euler`, which is the correctness anchor.

---

## 10. Future Work / Open Items

- [ ] **RF-Inversion robustness:** Now runs real two-controller inversion on `sa3.dit` (no silent A2A substitution). May still break if `inner.conditioner`, `get_conditioning_inputs`, `build_schedule`, or the DiT forward signature are renamed in a future SA3 release; the guarded fallback surfaces the real exception.
- [ ] **RF-Inversion tuning:** η (structure preservation) defaults to 0.0 (off). Good faithful-edit presets (γ≈0.3–0.5, η≈0.3–0.6, eta_stop≈0.3) are untested on hardware — needs a GPU pass to lock in defaults.
- [ ] **Steps slider for A2A:** Currently hardcoded to 8. Exposing it (range 4–50) would let users trade speed for quality.
- [ ] **Session audio persistence:** Audio saved to `_session_audio` is lost on kernel restart. A Drive-backed save option would improve the batch workflow.
- [ ] **Intro cell:** Section references in the intro HTML still say "section 5 & 7" — should be updated to mention sections 8 and 9.
- [ ] **RF-Inversion for Colab notebook:** The maxgraf96 plugin has battle-tested A2A/inpaint presets. Those noise/cfg defaults could be cross-referenced to tune the notebook defaults.
