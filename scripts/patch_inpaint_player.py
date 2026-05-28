#!/usr/bin/env python3
"""Replace cell-11-inpainting-stub with interactive JS waveform player."""
import json
from pathlib import Path

REPO = Path(__file__).parent.parent
NB   = REPO / "stable_audio_3_demo.ipynb"

# ── Player HTML (structure only, no <script>) ─────────────────────────────────
PLAYER_HTML = (
    '<div id="sa3pw" style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;'
    'max-width:700px;background:#F5F3FF;border-radius:12px;padding:12px;margin:6px 0;">'
    '<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap;">'
    '<button id="sa3-play" style="background:#7C3AED;color:#fff;border:none;border-radius:6px;'
    'padding:5px 14px;cursor:pointer;font-size:13px;min-width:80px;">&#9654; Play</button>'
    '<button id="sa3-prev" style="background:#EDE9FE;color:#5B21B6;border:1px solid #DDD6FE;'
    'border-radius:6px;padding:5px 12px;cursor:pointer;font-size:12px;" '
    'title="Play only the masked region">&#9986; Mask</button>'
    '<span id="sa3-time" style="font-size:12px;color:#374151;min-width:90px;'
    'font-variant-numeric:tabular-nums;">0:00&thinsp;/&thinsp;0:00</span>'
    '<span style="font-size:11px;color:#6B7280;margin-left:auto;">Zoom</span>'
    '<input id="sa3-zoom" type="range" min="0" max="500" value="0" '
    'style="width:110px;accent-color:#7C3AED;">'
    '<span id="sa3-zl" style="font-size:11px;color:#6B7280;min-width:32px;">Fit</span>'
    '</div>'
    '<div style="border-radius:8px;overflow:hidden;background:#EDE9FE;">'
    '<canvas id="sa3-cv" style="display:block;width:100%;height:110px;cursor:crosshair;"></canvas>'
    '</div>'
    '<input id="sa3-sc" type="range" min="0" max="0" value="0" '
    'style="width:100%;margin:3px 0 0;accent-color:#9CA3AF;opacity:0.5;">'
    '<div id="sa3-mi" style="font-size:12px;color:#92400E;background:#FEF3C7;'
    'border:1px solid #FCD34D;border-radius:6px;padding:5px 10px;margin-top:6px;">'
    'Drag on the waveform to set mask region</div>'
    '</div>'
)

# ── Player JS ─────────────────────────────────────────────────────────────────
PLAYER_JS = r"""
(function () {
'use strict';
const cv    = document.getElementById('sa3-cv');
const cx    = cv.getContext('2d');
const playB = document.getElementById('sa3-play');
const prevB = document.getElementById('sa3-prev');
const timeD = document.getElementById('sa3-time');
const zoomS = document.getElementById('sa3-zoom');
const zoomL = document.getElementById('sa3-zl');
const scrS  = document.getElementById('sa3-sc');
const mInfo = document.getElementById('sa3-mi');

/* ── State ────────────────────────────────────────────────────────────────── */
let _samples=null, _buf=null, _dur=0;
let _pps=0, _scroll=0;        // zoom (px/s, 0=fit) and scroll (px)
let _ms=0, _me=0;             // mask start / end (seconds)
let _drag=null;               // null | 'new' | 'left' | 'right' | 'move'
let _dAncT=0, _dAncMs=0, _dAncMe=0;
let _actx=null, _src=null, _startAt=0, _pausedAt=0, _playing=false, _raf=null;

/* ── Sizing helpers ───────────────────────────────────────────────────────── */
function eff()    { return _pps || (cv.width / (_dur || 1)); }
function scrMax() { return Math.max(0, eff() * _dur - cv.width); }
function clampS() { _scroll = Math.max(0, Math.min(_scroll, scrMax())); }
function t2x(t)   { return t * eff() - _scroll; }
function x2t(x)   { return (x + _scroll) / eff(); }

function canvasX(e) {
  const r = cv.getBoundingClientRect();
  return (e.clientX - r.left) * (cv.width / r.width);
}

/* ── Draw ─────────────────────────────────────────────────────────────────── */
function draw(t) {
  const W = cv.width, H = cv.height;
  cx.clearRect(0, 0, W, H);
  cx.fillStyle = '#F5F3FF'; cx.fillRect(0, 0, W, H);

  if (!_samples) {
    cx.fillStyle = '#9CA3AF'; cx.font = '13px sans-serif'; cx.textAlign = 'center';
    cx.fillText('Load an audio file to display the waveform', W/2, H/2);
    return;
  }

  const pps = eff(), mid = H / 2;

  /* Mask region */
  const maskL = Math.max(0, t2x(_ms)), maskR = Math.min(W, t2x(_me));
  if (maskR > maskL) {
    cx.fillStyle = 'rgba(252,211,77,0.45)';
    cx.fillRect(maskL, 0, maskR - maskL, H);
  }

  /* Waveform: min-max bars per canvas column */
  cx.beginPath(); cx.strokeStyle = '#7C3AED'; cx.lineWidth = 1;
  for (let px = 0; px < W; px++) {
    const t0 = (_scroll + px) / pps, t1 = (_scroll + px + 1) / pps;
    const i0 = Math.max(0,             Math.floor((t0 / _dur) * _samples.length));
    const i1 = Math.min(_samples.length, Math.ceil((t1 / _dur) * _samples.length));
    if (i0 >= i1) continue;
    let mn = 0, mx = 0;
    for (let i = i0; i < i1; i++) {
      if (_samples[i] < mn) mn = _samples[i];
      if (_samples[i] > mx) mx = _samples[i];
    }
    const yT = mid - mx * mid * 0.88, yB = mid - mn * mid * 0.88;
    cx.moveTo(px + 0.5, yT);
    cx.lineTo(px + 0.5, yB <= yT ? yT + 1 : yB);
  }
  cx.stroke();

  /* Mask boundary lines */
  for (const bx of [t2x(_ms), t2x(_me)]) {
    if (bx < 0 || bx > W) continue;
    cx.strokeStyle = '#D97706'; cx.lineWidth = 2;
    cx.beginPath(); cx.moveTo(bx, 0); cx.lineTo(bx, H); cx.stroke();
  }

  /* Playhead */
  if (t !== undefined) {
    const px = t2x(t);
    if (px >= 0 && px <= W) {
      cx.strokeStyle = '#EF4444'; cx.lineWidth = 1.5;
      cx.beginPath(); cx.moveTo(px, 0); cx.lineTo(px, H); cx.stroke();
    }
  }

  /* Time ticks */
  const intervals = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120];
  const intv = intervals.find(i => i * pps >= 50) || 120;
  cx.save();
  cx.strokeStyle = 'rgba(107,114,128,0.3)'; cx.lineWidth = 1;
  cx.fillStyle = '#6B7280'; cx.font = '10px sans-serif'; cx.textAlign = 'center';
  const firstT = Math.ceil((_scroll / pps) / intv) * intv;
  for (let tick = firstT; tick <= _dur; tick += intv) {
    const x = t2x(tick);
    if (x < 0 || x > W) continue;
    cx.beginPath(); cx.moveTo(x, H-13); cx.lineTo(x, H); cx.stroke();
    cx.fillText(fmt(tick), x, H - 15);
  }
  cx.restore();
}

function fmt(s) {
  const m = Math.floor(s / 60), ss = Math.floor(s % 60);
  return m + ':' + String(ss).padStart(2, '0');
}

/* ── Playback ─────────────────────────────────────────────────────────────── */
function curT() {
  return (_playing && _actx) ? (_actx.currentTime - _startAt + _pausedAt) : _pausedAt;
}

function animate() {
  const t = curT();
  if (_playing && t >= _dur) { stopPlay(); return; }
  if (_playing && _pps !== 0) {
    const px = t2x(t);
    if (px > cv.width * 0.8) {
      _scroll = Math.max(0, Math.min(t * eff() - cv.width * 0.35, scrMax()));
      syncSc();
    }
  }
  draw(t);
  timeD.textContent = fmt(t) + ' / ' + fmt(_dur);
  _raf = requestAnimationFrame(animate);
}

function startPlay(offset) {
  if (!_buf) return;
  if (!_actx) _actx = new (window.AudioContext || window.webkitAudioContext)();
  if (_actx.state === 'suspended') _actx.resume();
  _src = _actx.createBufferSource();
  _src.buffer = _buf; _src.connect(_actx.destination);
  _startAt = _actx.currentTime - offset;
  _src.start(0, offset);
  _playing = true; playB.textContent = '⏸ Pause';
  animate();
}

function pausePlay() {
  if (!_playing) return;
  _pausedAt = curT();
  try { _src.stop(); } catch(e) {}
  _playing = false; playB.textContent = '▶ Play';
  cancelAnimationFrame(_raf); draw(_pausedAt);
  timeD.textContent = fmt(_pausedAt) + ' / ' + fmt(_dur);
}

function stopPlay() {
  try { if (_playing) _src.stop(); } catch(e) {}
  _pausedAt = 0; _playing = false; playB.textContent = '▶ Play';
  cancelAnimationFrame(_raf); draw(0);
  timeD.textContent = '0:00 / ' + fmt(_dur);
}

playB.addEventListener('click', () => { if (_playing) pausePlay(); else startPlay(_pausedAt); });

prevB.addEventListener('click', () => {
  if (!_buf) return;
  if (_playing) pausePlay();
  if (!_actx) _actx = new (window.AudioContext || window.webkitAudioContext)();
  if (_actx.state === 'suspended') _actx.resume();
  const sr = _buf.sampleRate;
  const s = Math.floor(_ms * sr), len = Math.max(1, Math.ceil((_me - _ms) * sr));
  const tmp = _actx.createBuffer(_buf.numberOfChannels, len, sr);
  for (let c = 0; c < _buf.numberOfChannels; c++)
    tmp.getChannelData(c).set(_buf.getChannelData(c).subarray(s, s + len));
  const tsrc = _actx.createBufferSource();
  tsrc.buffer = tmp; tsrc.connect(_actx.destination); tsrc.start();
});

/* ── Zoom & scroll ────────────────────────────────────────────────────────── */
function syncSc() {
  const mx = scrMax();
  scrS.max   = Math.max(1, Math.round(mx));
  scrS.value = Math.round(_scroll);
  scrS.style.opacity = mx > 0 ? '1' : '0.3';
}

zoomS.addEventListener('input', () => {
  const v = +zoomS.value;
  if (v === 0) { _pps = 0; _scroll = 0; zoomL.textContent = 'Fit'; }
  else {
    const minP = cv.width / (_dur || 1);
    _pps = minP + (600 - minP) * (v / 500);
    zoomL.textContent = _pps.toFixed(0) + 'px/s';
  }
  clampS(); syncSc(); draw(curT());
});

scrS.addEventListener('input', () => { _scroll = +scrS.value; draw(curT()); });

cv.addEventListener('wheel', e => {
  e.preventDefault();
  if (!_samples) return;
  const x = canvasX(e), pivotT = x2t(x);
  const factor = e.deltaY > 0 ? 0.8 : 1.25;
  const minP = cv.width / (_dur || 1);
  const np = eff() * factor;
  if (np < minP * 1.02) {
    _pps = 0; _scroll = 0; zoomS.value = 0; zoomL.textContent = 'Fit';
  } else {
    _pps = Math.min(600, np);
    _scroll = pivotT * _pps - x;
    const pct = Math.round(Math.min(500, (_pps - minP) / (600 - minP) * 500));
    zoomS.value = pct; zoomL.textContent = _pps.toFixed(0) + 'px/s';
  }
  clampS(); syncSc(); draw(curT());
}, { passive: false });

/* ── Mouse drag for mask ──────────────────────────────────────────────────── */
const HTHR = 10;
function hitTest(x) {
  const lx = t2x(_ms), rx = t2x(_me);
  if (Math.abs(x - lx) < HTHR) return 'left';
  if (Math.abs(x - rx) < HTHR) return 'right';
  if (x > lx && x < rx)        return 'move';
  return 'new';
}

cv.addEventListener('mousedown', e => {
  if (!_samples) return;
  const x = canvasX(e), t = Math.max(0, Math.min(_dur, x2t(x)));
  _drag = hitTest(x); _dAncT = t; _dAncMs = _ms; _dAncMe = _me;
  if (_drag === 'new') _ms = _me = t;
  e.preventDefault();
});

window.addEventListener('mousemove', e => {
  if (!_drag) return;
  const x = canvasX(e), t = Math.max(0, Math.min(_dur, x2t(x)));
  const dt = t - _dAncT;
  if      (_drag === 'new')   { _ms = Math.min(_dAncT,t); _me = Math.max(_dAncT,t); }
  else if (_drag === 'left')  { _ms = Math.min(t, _me - 0.1); }
  else if (_drag === 'right') { _me = Math.max(t, _ms + 0.1); }
  else {
    const len = _dAncMe - _dAncMs;
    let ns = _dAncMs + dt, ne = _dAncMe + dt;
    if (ns < 0) { ns = 0; ne = len; }
    if (ne > _dur) { ne = _dur; ns = ne - len; }
    _ms = ns; _me = ne;
  }
  updMask(); draw(curT());
});

window.addEventListener('mouseup', () => {
  if (!_drag) return;
  const wasDrag = _drag; _drag = null;
  if (wasDrag === 'new' && Math.abs(_me - _ms) < 0.1) {
    /* single click = seek */
    const seekT = _dAncT; _ms = _dAncMs; _me = _dAncMe;
    if (_playing) { pausePlay(); startPlay(seekT); }
    else { _pausedAt = seekT; draw(seekT); timeD.textContent = fmt(seekT) + ' / ' + fmt(_dur); }
  } else {
    _ms = +_ms.toFixed(2); _me = +_me.toFixed(2);
    if (typeof google !== 'undefined' && google.colab)
      google.colab.kernel.invokeFunction('nb_update_mask', [_ms, _me], {});
  }
  updMask(); draw(curT());
});

cv.addEventListener('mousemove', e => {
  if (_drag || !_samples) return;
  const h = hitTest(canvasX(e));
  cv.style.cursor = (h==='left'||h==='right') ? 'ew-resize' : h==='move' ? 'grab' : 'crosshair';
});

/* ── Mask label ───────────────────────────────────────────────────────────── */
function updMask() {
  mInfo.textContent = 'Mask: ' + _ms.toFixed(1) + 's – ' + _me.toFixed(1) +
    's  (' + (_me - _ms).toFixed(1) + 's)  ·  drag to adjust';
}

/* ── Resize ───────────────────────────────────────────────────────────────── */
new ResizeObserver(() => {
  cv.width  = cv.clientWidth  || 700;
  cv.height = cv.clientHeight || 110;
  clampS(); syncSc(); draw(curT());
}).observe(cv);

/* ── Public API ───────────────────────────────────────────────────────────── */
window.sa3LoadAudio = async function (dataUrl, ms, me) {
  stopPlay(); _ms = ms; _me = me; updMask();
  const ab = await (await fetch(dataUrl)).arrayBuffer();
  if (!_actx) _actx = new (window.AudioContext || window.webkitAudioContext)();
  _buf = await _actx.decodeAudioData(ab);
  _dur = _buf.duration;
  const ch = _buf.getChannelData(0);
  const tgt = Math.min(ch.length, 12000), step = Math.floor(ch.length / tgt) || 1;
  _samples = new Float32Array(tgt);
  for (let i = 0; i < tgt; i++) _samples[i] = ch[i * step];
  _pps = 0; _scroll = 0; zoomS.value = 0; zoomL.textContent = 'Fit';
  syncSc(); draw(); timeD.textContent = '0:00 / ' + fmt(_dur);
};

/* initial empty draw */
cv.width = cv.clientWidth || 700; cv.height = cv.clientHeight || 110;
draw();
})();
"""

# ── New cell source ───────────────────────────────────────────────────────────
NEW_SOURCE = r'''#@title 7 — Inpainting
import base64
from io import BytesIO
from IPython.display import Javascript
from google.colab import files as colab_files
from google.colab import output as _colab_output

# All other names: widgets, Layout, Audio, clear_output, HTML, display,
# torchaudio, torch, SAMPLE_RATE, time, tempfile, get_model, SPINNER_HTML,
# _model_cache, _is_model_on_disk, MODEL_CONFIGS, _session_audio


# ── Player HTML/JS ────────────────────────────────────────────────────────────
_PLAYER_HTML = PLAYER_HTML_PLACEHOLDER

_PLAYER_JS = PLAYER_JS_PLACEHOLDER


# ── State ─────────────────────────────────────────────────────────────────────
_inpaint_audio = None   # torch.Tensor (channels, samples) at SAMPLE_RATE
_mask_start    = 5.0
_mask_end      = 15.0


# ── JS→Python callback ────────────────────────────────────────────────────────
def _nb_update_mask(start, end):
    global _mask_start, _mask_end
    _mask_start = float(start)
    _mask_end   = float(end)

_colab_output.register_callback('nb_update_mask', _nb_update_mask)


# ── Widgets ───────────────────────────────────────────────────────────────────
_UPLOAD_OPT = '— Upload new file —'

inpaint_model_dropdown = widgets.Dropdown(
    options=[(MODEL_CONFIGS[k]["label"], k) for k in MODEL_CONFIGS],
    value="small-music",
    description="Model:",
    style={"description_width": "52px"},
    layout=Layout(width="500px"),
)

inpaint_gpu_warning = widgets.HTML(
    value='<div class="sa3-warn">'
          '&#9888;&#65039;&nbsp; <b>medium</b> requires a GPU with compute capability ≥8.0 '
          '(A100 or L4). No compatible GPU detected. '
          'Go to <b>Runtime &rarr; Change runtime type &rarr; GPU</b>.'
          '</div>',
    layout=Layout(display="none", width="100%"),
)

source_dropdown = widgets.Dropdown(
    options=[_UPLOAD_OPT],
    description='Source:',
    style={'description_width': '60px'},
    layout=Layout(width='440px'),
)

upload_btn = widgets.Button(
    description='Upload Audio', icon='upload',
    layout=Layout(width='155px', height='36px'),
)
upload_btn.style.button_color = '#EDE9FE'

upload_status = widgets.HTML(
    value='<div class="sa3-info" style="margin:4px 0;">No file loaded yet.</div>'
)
upload_row = widgets.HBox(
    [upload_btn, upload_status],
    layout=Layout(align_items='center', gap='10px', margin='4px 0 0'),
)

player_out = widgets.Output()

inpaint_prompt = widgets.Textarea(
    value='Replace this region with a gentle acoustic guitar melody',
    placeholder='Describe what should replace the masked region…',
    layout=Layout(width='100%', height='64px'),
    style={'description_width': '0px'},
)

inpaint_btn = widgets.Button(
    description='  Inpaint', icon='magic', button_style='',
    layout=Layout(width='150px', height='42px'),
)
inpaint_btn.style.button_color = '#7C3AED'
inpaint_btn.style.font_weight  = '700'

inpaint_out = widgets.Output(layout=Layout(min_height='60px', margin='6px 0 0'))


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_audio_into_state(audio_tensor, label):
    global _inpaint_audio, _mask_start, _mask_end
    _inpaint_audio = audio_tensor
    duration       = audio_tensor.shape[-1] / SAMPLE_RATE
    _mask_start    = round(min(5.0,  duration * 0.15), 1)
    _mask_end      = round(min(15.0, duration * 0.40), 1)
    upload_status.value = (
        f'<div class="sa3-info" style="margin:4px 0;">'
        f'&#x2705;&nbsp; <strong>{label}</strong>'
        f' &nbsp;|&nbsp; {duration:.1f}&thinsp;s'
        f' &nbsp;|&nbsp; {audio_tensor.shape[0]} ch'
        f'</div>'
    )
    buf = BytesIO()
    torchaudio.save(buf, audio_tensor.cpu(), SAMPLE_RATE, format='wav')
    b64      = base64.b64encode(buf.getvalue()).decode('ascii')
    data_url = f'data:audio/wav;base64,{b64}'
    display(Javascript(
        f'if(window.sa3LoadAudio)'
        f'sa3LoadAudio({repr(data_url)},{_mask_start},{_mask_end});'
    ))


# ── Callbacks ─────────────────────────────────────────────────────────────────
def on_inpaint_model_change(change):
    new_model = change["new"]
    cfg       = MODEL_CONFIGS[new_model]
    needs_gpu = cfg["needs_gpu"] and not torch.cuda.is_available()
    inpaint_gpu_warning.layout.display = "" if needs_gpu else "none"

inpaint_model_dropdown.observe(on_inpaint_model_change, names="value")


def on_source_change(change):
    title = change['new']
    if title == _UPLOAD_OPT:
        upload_row.layout.display = ''
    else:
        upload_row.layout.display = 'none'
        wav_path = _session_audio[title]
        audio, sr = torchaudio.load(wav_path)
        _load_audio_into_state(audio, title)


def on_upload(b):
    uploaded = colab_files.upload()
    if not uploaded:
        return
    for fname, data in uploaded.items():
        tmp = f'/tmp/{fname}'
        with open(tmp, 'wb') as f:
            f.write(data)
        audio, sr = torchaudio.load(tmp)
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        _load_audio_into_state(audio, fname)


def on_inpaint(b):
    if _inpaint_audio is None:
        with inpaint_out:
            clear_output(wait=True)
            display(HTML('<div class="sa3-error">Select or upload an audio file first.</div>'))
        return
    prompt = inpaint_prompt.value.strip()
    if not prompt:
        with inpaint_out:
            clear_output(wait=True)
            display(HTML('<div class="sa3-error">Please enter a prompt.</div>'))
        return

    model_name = inpaint_model_dropdown.value
    start, end = _mask_start, _mask_end
    duration   = _inpaint_audio.shape[-1] / SAMPLE_RATE
    inpaint_btn.disabled    = True
    inpaint_btn.description = '  Inpainting…'

    if model_name in _model_cache:
        _spin_status    = "Running inpainting diffusion…"
        _spin_substatus = f"&#x26A1;&nbsp; <strong>{model_name}</strong> already loaded in memory."
    elif _is_model_on_disk(model_name):
        _spin_status    = f"Loading <strong>{model_name}</strong> from Drive cache…"
        _spin_substatus = "&#x1F4BE;&nbsp; Weights found on Drive — no download needed."
    else:
        _spin_status    = f"Downloading <strong>{model_name}</strong> weights…"
        _spin_substatus = "&#x2B07;&nbsp; First run — may take 1&ndash;3&thinsp;min."
    with inpaint_out:
        clear_output(wait=True)
        display(HTML(SPINNER_HTML.format(status=_spin_status, substatus=_spin_substatus)))

    try:
        model = get_model(model_name)
        t0    = time.perf_counter()
        audio_out = model.generate(
            prompt=prompt,
            inpaint_audio=(SAMPLE_RATE, _inpaint_audio),
            inpaint_mask_start_seconds=start,
            inpaint_mask_end_seconds=end,
            duration=duration,
            steps=8, cfg_scale=1.0,
        )
        elapsed = time.perf_counter() - t0
        short   = prompt[:90] + ('…' if len(prompt) > 90 else '')
        tmp     = tempfile.mktemp(suffix='.wav', dir='/tmp')
        torchaudio.save(tmp, audio_out[0].cpu(), SAMPLE_RATE)
        with inpaint_out:
            clear_output(wait=True)
            display(HTML(f"""
            <div class="sa3-result">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
                <span class="sa3-badge">{model_name}</span>
                <span class="sa3-stat">mask {start:.1f}s–{end:.1f}s</span>
                <span class="sa3-stat">generated in {elapsed:.1f}&thinsp;s</span>
              </div>
              <div style="font-size:13px;color:#374151;font-style:italic;line-height:1.5;">
                &ldquo;{short}&rdquo;
              </div>
            </div>
            """))
            display(Audio(tmp, autoplay=False))
    except RuntimeError as exc:
        with inpaint_out:
            clear_output(wait=True)
            display(HTML(f'<div class="sa3-error"><strong>Error:</strong> {exc}</div>'))
    finally:
        inpaint_btn.disabled    = False
        inpaint_btn.description = '  Inpaint'


# ── Connect observers ─────────────────────────────────────────────────────────
source_dropdown.observe(on_source_change, names='value')
upload_btn.on_click(on_upload)
inpaint_btn.on_click(on_inpaint)

def _on_session_audio_updated():
    prev = source_dropdown.value
    opts = [_UPLOAD_OPT] + list(_session_audio.keys())
    source_dropdown.options = opts
    if prev in opts:
        source_dropdown.value = prev


# ── Inject player then display layout ─────────────────────────────────────────
if _session_audio:
    source_dropdown.options = [_UPLOAD_OPT] + list(_session_audio.keys())

with player_out:
    display(HTML(_PLAYER_HTML))
    display(Javascript(_PLAYER_JS))

display(widgets.VBox([
    widgets.HTML('<div style="font-size:12px;color:#6B7280;margin:0 0 3px;">Model:</div>'),
    inpaint_model_dropdown,
    inpaint_gpu_warning,
    widgets.HTML('<div style="font-size:12px;color:#6B7280;margin:8px 0 3px;">Audio source:</div>'),
    source_dropdown,
    upload_row,
    player_out,
    widgets.HTML('<div style="font-size:12px;color:#6B7280;margin:10px 0 2px;">'
                 'Prompt for the replacement region:</div>'),
    inpaint_prompt,
    inpaint_btn,
    inpaint_out,
], layout=Layout(width='100%', max_width='700px')))
'''

# Substitute the HTML and JS placeholders into the source
NEW_SOURCE = NEW_SOURCE.replace("PLAYER_HTML_PLACEHOLDER", repr(PLAYER_HTML))
NEW_SOURCE = NEW_SOURCE.replace("PLAYER_JS_PLACEHOLDER",   repr(PLAYER_JS))

# ── Patch notebook ────────────────────────────────────────────────────────────
nb = json.loads(NB.read_text(encoding="utf-8"))
for cell in nb["cells"]:
    if cell.get("id") == "cell-11-inpainting-stub":
        cell["source"] = NEW_SOURCE
        break
else:
    raise RuntimeError("cell-11-inpainting-stub not found in notebook")

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("patch_inpaint_player: OK")
