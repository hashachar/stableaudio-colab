# Decision Log

Running log of every meaningful design decision in the project, oldest first. Each entry: what changed, why, expected benefit, possible downside. (Routine fixes and cosmetic tweaks are in `git log`; this file records the choices that shaped the notebook.)

---

### 2026-05-20 — Use the `stable-audio-3` package, not `stable-audio-tools`
- **What:** Built on Stability AI's newer `stable-audio-3` inference package (installed from GitHub — it is not on PyPI) with `StableAudioModel.from_pretrained(...)`.
- **Why:** Cleaner high-level API (`generate()` with inpainting built in) versus the research-grade `stable-audio-tools`.
- **Benefit:** Far less glue code; inpainting for free.
- **Downside:** GitHub install is slower and pins us to an unversioned main branch; internal APIs (used later by RF-Inversion) can shift under us.

### 2026-05-20 — Two-model registry: `small-music` + `medium`
- **What:** `MODEL_CONFIGS` registry exposing small-music (433 M, CPU-ok, ≤120 s) and medium (1.4 B, GPU, ≤285 s), with per-model defaults (steps 8, CFG 1.0).
- **Why:** small-music makes the notebook usable on free/CPU runtimes; medium is the quality option.
- **Benefit:** One dropdown covers sketch-fast and final-quality workflows.
- **Downside:** Two gated repos to authorize; switching models mid-session costs a load.

### 2026-05-20 — Session audio registry shared across tools
- **What:** `_session_audio` dict (title → wav path) auto-populated on every generation, consumed by the source dropdowns of all audio-input tools, with a live-update hook.
- **Why:** The tools are most powerful chained (generate → inpaint → vary → morph); forcing download/re-upload between them would kill that.
- **Benefit:** Zero-friction chaining.
- **Downside:** Lives in `/tmp` — lost on runtime recycle (open item: Drive-backed persistence).

### 2026-05-21 — Persist model weights on Google Drive via `HF_HOME`
- **What:** Mount Drive and point `HF_HOME` at `MyDrive/stable-audio-cache` *before any `huggingface_hub` import* (cell ordering is load-bearing).
- **Why:** Colab disks are ephemeral; multi-GB downloads every session are unacceptable.
- **Benefit:** Weights download once per Google account, ever.
- **Downside:** Consumes user Drive quota; wrong cell order silently caches to ephemeral disk.

### 2026-05-27 — Auth via `HF_TOKEN` Colab secret + `hf_hub_download` monkey-patch
- **What:** Read the token from Colab's secrets manager into the environment, and patch the `hf_hub_download` name inside `stable_audio_3.model_configs` to inject it.
- **Why:** The package imports `hf_hub_download` by value, bypassing `huggingface_hub`'s own token auto-detection on Colab → spurious 401s.
- **Benefit:** Auth works with the standard Colab secrets UX; no token pasted into cells.
- **Downside:** Monkey-patch can break if the package restructures its imports. (2026-05-28 follow-up: patch made idempotent — always sources the *real* function from `huggingface_hub` — after a re-run-induced RecursionError.)

### 2026-05-27 — `#@title` on every cell; one cell per tool
- **What:** All code collapsed behind Colab form titles; each tool is a single self-contained cell with its own widgets, handlers, and layout.
- **Why:** The brief's UX bar: a polished product, not a research notebook. Fewer, bigger cells = less scrolling and no "run these 6 cells in order" per feature.
- **Benefit:** Open → run setup → generate; code invisible unless wanted.
- **Downside:** Big cells are harder to diff/review; widget-name prefixes (`a2a_*`, `rfi_*`, `morph_*`) needed to avoid global collisions.

### 2026-05-28 — Matplotlib waveform + zoom for inpainting (JS player reverted)
- **What:** Replaced an interactive JS waveform player with a static matplotlib render, zoom range-slider, precise mask start/end number inputs, and a "Preview mask" audition button.
- **Why:** The JS player fought Colab's output sandboxing and torchaudio/BytesIO backend quirks; reliability beat interactivity.
- **Benefit:** Deterministic rendering everywhere; sample-accurate mask entry.
- **Downside:** No click-drag mask selection; a re-render per adjustment.

### 2026-05-31 — Inpainting result: no full-song splice; click markers instead
- **What:** Reverted auto-splicing the inpainted window back into songs longer than the model max; instead surface the truncation limit and offer a download with metronome clicks at the mask edges.
- **Why:** The splice produced boundary artifacts that misrepresented model quality; the honest output is the model-length clip plus tools to align it in a DAW.
- **Benefit:** No silent quality lies; DAW-friendly workflow.
- **Downside:** Users with long songs must do the final splice themselves.

### 2026-06-01 — Add Audio-to-Audio (§8) and RF-Inversion (§9)
- **What:** Two new tools: A2A variations via `generate(init_audio=…, init_noise_level=…)`, and prompt-driven re-styling via rectified-flow inversion.
- **Why:** The model architecture supports both; they cover the "remix" space between generate-from-scratch and surgical inpainting.
- **Benefit:** The notebook becomes an editing suite, not a demo.
- **Downside:** RF-Inversion depends on package internals (see 2026-06-08 entry).

### 2026-06-02 — Install Flash Attention 2 only on capable GPUs
- **What:** Setup cell detects compute capability and installs FA2 only on ≥8.0 (A100/L4/H100); elsewhere a non-blocking warning explains the slowdown.
- **Why:** FA2 wheels fail on T4 (7.5); a hard dependency would break the free tier.
- **Benefit:** Fast path where possible, working path everywhere.
- **Downside:** Two performance profiles to document.

### 2026-06-07 — Fix the silent 2-minute load: CPU-side state-dict load + fp16
- **What:** Patched `load_diffusion_cond` to load the safetensors state dict on CPU (then one batched `.to(device)`) and to force fp16 on CUDA.
- **Why:** Upstream loaded weights to GPU then copied into a CPU model — hundreds of tiny GPU→CPU syncs = a silent multi-minute stall. fp16 is 2–3× faster on tensor-core GPUs.
- **Benefit:** Load time cut dramatically; inference 2–3× faster; halved VRAM.
- **Downside:** Another internals patch; fp16 is a (inaudible in practice) precision reduction.

### 2026-06-07 — Loading UX: phase messages + 30 s heartbeat ticker
- **What:** `get_model()` prints the two loading phases and a background thread prints elapsed time every 30 s.
- **Why:** The VAE/T5 init phase has no progress bar; users assumed a hang and restarted runtimes.
- **Benefit:** No more panic-restarts during a healthy load.
- **Downside:** A daemon thread that must be stopped on completion/failure (`finally`).

### 2026-06-08 — RF-Inversion made *real* (velocity field = `sa3.dit`)
- **What:** Reimplemented §9 as true two-controller RF-Inversion (Rout et al.): forward ODE recovers the noise fingerprint, reverse ODE regenerates with γ/η controllers. Key fixes: drive `sa3.dit` (the `DiTWrapper` the samplers call), not `sa3.model`; inject zero `inpaint_mask`/`inpaint_masked_input` channels exactly as `generate()` does.
- **Why:** The original attempt hit a DiT channel-count exception and silently fell back to plain A2A — the tool wasn't doing what its UI claimed.
- **Benefit:** Honest, structure-preserving re-styling; at η=0 the reverse pass is provably the stock sampler (correctness anchor).
- **Downside:** Deepest dependency on package internals in the notebook; γ/η defaults still untuned on hardware (math verified against source, not GPU-tested).

### 2026-06-08 — Audio Transition (§10) reuses the loaded model's autoencoder
- **What:** Morph = per-frame temporal **slerp** crossfade of A/B latents (via `sa3.model.pretransform.encode/decode`) + optional low-noise re-coherence pass; added a toggle to do that re-coherence with RF-Inversion instead of blind A2A.
- **Why slerp:** linear interpolation of high-dimensional latents collapses magnitude (mush); slerp follows the latent sphere. **Why pretransform reuse:** a separate `AutoencoderModel.from_pretrained("same-l")` API is unconfirmed in-package, and reuse means one model load.
- **Benefit:** A genuinely new creative tool with zero extra VRAM; two quality tiers for the seam repair.
- **Downside:** Re-coherence at noise > ~0.4 starts rewriting the source clips; RFI mode requires §9's cell to have run.

### 2026-06-08 — torch.compile OFF by default; warmup generation ON
- **What:** `ENABLE_TORCH_COMPILE = False` (was unconditional): compile JIT-stalled the *first* generation for minutes and recompiled on every duration change. When enabled, uses `dynamic=True` + a Drive-persisted inductor cache. `WARMUP_ON_LOAD = True`: a throwaway 2 s/4-step generate right after load pays first-call CUDA/cuDNN/SDPA/T5/VAE costs behind the expected loading wait.
- **Why:** For an 8-step distilled sampler already on Flash/SDPA attention, compile's ~20–40 % step speedup is rarely recouped interactively; the cold-start stall was the single worst UX moment in the notebook.
- **Benefit:** First real prompt runs at full speed; power users can still opt into compile for batch runs.
- **Downside:** Warmup adds ~seconds to load; compile users pay one JIT (once ever, thanks to the Drive cache).

### 2026-07-06 — Session-clip chaining: every tool registers its result; live dropdown refresh everywhere
- **What:** Utilities' single-callback `_on_session_audio_updated` hook became a tagged listener registry (`_add_session_listener`), plus a `_register_session_clip()` helper that copies a result to a unique `/tmp` path (fixed result paths get overwritten by the next run) and notifies all listeners. Inpainting, A2A, RF-Inversion, morph — and the new §11 — now register their outputs and refresh their source dropdowns the moment any tool saves a clip.
- **Why:** Only §5 registered results and only inpainting listened for changes, so chaining tool outputs (the notebook's core workflow) silently required download/re-upload or cell re-runs.
- **Benefit:** Generate → inpaint → vary → extend → morph in any order, zero friction; tags make cell re-runs replace their listener instead of stacking duplicates.
- **Downside:** Each registered clip is a wav copy in `/tmp` (a few MB each; ephemeral disk, so acceptable).

### 2026-07-06 — §5 multi-take generation with always-visible seeds
- **What:** A **Takes** slider (1–4) in the Advanced accordion generates N takes of the prompt in one click, each from its own seed, with per-take players and download buttons. Seeds are now always drawn explicitly (never left to the library's internal RNG) so every result panel displays the seed that produced it.
- **Why:** Diffusion output quality varies take to take; the fastest intention→output path is hearing several seeds at once. And a result whose seed is unknown is unreproducible by construction.
- **Benefit:** One click replaces four generate-listen-regenerate cycles; any take can be reproduced or refined (fix the seed, tweak one parameter).
- **Downside:** N takes cost N generations of wall-clock time; capped at 4 to keep the button honest.

### 2026-07-06 — §11 Extend & Loop (outpainting via the inpainting engine)
- **What:** New tool with three modes built on the public `generate(inpaint_audio=…)` API: **Continue end** (append silence, mask it, let the model fill), **Extend intro** (the mirror), and **Seamless loop** (`torch.roll` the clip half-way so the start/end joint sits mid-clip, inpaint a seam window around it, roll back; a seam-check player plays the loop ×2). Every joint re-writes ~0.5 s of existing audio (`_EXT_SEAM_OVERLAP`) so splices are composed rather than butt-joined.
- **Why:** Continuation/extension and loop-making are the two most-requested creative operations the architecture supports that weren't exposed; both are pure applications of masking, needing no model internals (unlike RF-Inversion).
- **Benefit:** Clip extension and DAW-ready loops with the reliability profile of §7; works on small-music (CPU) too.
- **Downside:** Total output is still capped at the model max, so extension can't exceed it; loop mode assumes `generate()` returns sample counts matching the request (guarded by trim/pad before the roll-back).

### 2026-07-06 — §5 Advanced accordion; A2A steps slider; docs split
- **What:** (1) §5 gains a collapsed **Advanced options** accordion — negative prompt, steps 4–50, CFG 0–15, seed — and `generate_audio()` gains `negative_prompt`. (2) §8's hardcoded `steps=8` becomes a 4–50 slider. (3) Intro cell finally lists all five tools. (4) Documentation split: lean `README.md` landing page, comprehensive `GUIDE.md` user guide, this `DECISIONS.md`.
- **Why:** §5 silently ignored seed/CFG/steps that the README claimed it had; "hide advanced options unless needed" (the brief) is an accordion, not omission. The A2A steps exposure was a logged open item. One README can't be both a 30-second pitch and a reference manual.
- **Benefit:** Reproducible §5 results (fixed seed) and prompt-adherence control without cluttering the default view; honest intro; docs that serve both audiences.
- **Downside:** More surface area to keep consistent between UI and docs; defaults must stay obviously "fine untouched" so the accordion never becomes required reading.
