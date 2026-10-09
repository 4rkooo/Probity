"""Provenance Canvas component (Person 3).

Self-contained HTML/JS component rendered through ``streamlit.components.v1.html``:

- Synchronized original / recompressed-preview players driven by one master clock
  (play, pause, seek, and frame-step are mirrored; the selected PTS stays visible).
- Lossless still comparison (target vs. result) with a shared pixel-exact zoom.
- Pixel provenance overlay from the authoritative ``provenance.npz`` arrays: gray for
  ORIGINAL, cyan for BORROWED, magenta hatching for GENERATED_BLEND. The legend stays
  visible when the overlay is toggled off.
- Hover/click inspector resolving class, target coordinate, source frame/PTS/coordinate,
  donor crop, transform, color coefficients, and decision reason codes.
- **Open source** seeks both players to the donor frame and outlines the source region
  on the donor still.
- The preview player carries an unhideable "recompressed preview - not the canonical
  result" label.

Only non-identity pixels are shipped to the browser (``[x, y, class, source_index,
source_x, source_y]`` rows); every other pixel is ORIGINAL at its own coordinate, exactly
as in the NPZ. :func:`resolve_from_exceptions` mirrors the browser lookup for tests.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

PROVENANCE_CLASS_NAMES = ("ORIGINAL", "BORROWED", "GENERATED_BLEND")


def load_b64(path: Path | str) -> str:
    """Read a PNG/JPEG/MP4 file and encode it as a data URI."""
    p = Path(path)
    if not p.exists():
        return ""
    mime = {".png": "image/png", ".mp4": "video/mp4"}.get(p.suffix.lower(), "image/jpeg")
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"


# Backwards-compatible alias.
load_image_b64 = load_b64


def resolve_from_exceptions(
    x: int, y: int, exceptions: Sequence[Sequence[float]]
) -> tuple[int, int, float, float]:
    """Python twin of the canvas lookup: returns (class, source_index, source_x, source_y)."""
    for row in exceptions:
        if int(row[0]) == x and int(row[1]) == y:
            return int(row[2]), int(row[3]), float(row[4]), float(row[5])
    return 0, 0, float(x), float(y)


def build_canvas_html(
    *,
    target_img_uri: str,
    result_img_uri: str,
    donor_img_uris: dict[int, str],
    source_video_uri: str,
    lut_entries: list[dict[str, Any]],
    exceptions: list[list[float]],
    frame_size: tuple[int, int],
    subject_bbox: Sequence[int],
    decision_codes: dict[str, str],
    target_pts_us: int,
    nominal_fps: float,
    initial_pixel: tuple[int, int],
) -> str:
    data = {
        "lut": lut_entries,
        "exceptions": exceptions,
        "W": frame_size[0],
        "H": frame_size[1],
        "bbox": list(subject_bbox),
        "decisions": decision_codes,
        "targetPtsUs": target_pts_us,
        "fps": nominal_fps,
        "initial": list(initial_pixel),
        "donors": {str(k): v for k, v in donor_img_uris.items()},
        "classNames": list(PROVENANCE_CLASS_NAMES),
    }
    return (
        _TEMPLATE.replace("__TARGET__", target_img_uri)
        .replace("__RESULT__", result_img_uri)
        .replace("__VIDEO__", source_video_uri)
        .replace("__DATA__", json.dumps(data))
    )


def render_provenance_canvas(height: int = 1000, **kwargs: Any) -> None:
    """Render the synchronized players, lossless stills, overlay, and pixel inspector."""
    page = build_canvas_html(**kwargs)
    if hasattr(st, "iframe"):
        # Trusted, locally generated HTML only: no user-supplied markup reaches this page.
        st.iframe(page, height=height)
    else:  # Streamlit < 1.50
        components.html(page, height=height, scrolling=True)


_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  * { box-sizing: border-box; }
  body { margin:0; padding:8px; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:#0f172a; color:#f8fafc; }
  h4 { margin:6px 0; font-size:13px; color:#cbd5e1; text-transform:uppercase; letter-spacing:.5px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .panel { background:#1e293b; border:1px solid #334155; border-radius:6px; padding:8px; }
  .hdr { display:flex; justify-content:space-between; font-size:12px; font-weight:600; margin-bottom:6px; gap:6px; }
  .warn { background:#78350f; color:#fef3c7; border-left:4px solid #f59e0b; padding:4px 8px; font-size:11px; font-weight:700; border-radius:3px; margin-bottom:6px; }
  video, canvas { width:100%; display:block; background:#000; border-radius:4px; }
  .stage { position:relative; }
  .stage img.result-at-pts { position:absolute; inset:0; width:100%; height:100%; display:none; image-rendering:pixelated; }
  .controls { display:flex; align-items:center; gap:8px; margin:8px 0; font-size:12px; flex-wrap:wrap; }
  button { background:#0284c7; color:#fff; border:none; padding:5px 10px; border-radius:4px; font-size:11px; font-weight:600; cursor:pointer; }
  button.secondary { background:#334155; }
  input[type=range] { flex:1; min-width:120px; }
  .legend { display:flex; gap:14px; align-items:center; font-size:11px; flex-wrap:wrap; padding:6px 10px; background:#1e293b; border-radius:6px; margin:8px 0; }
  .sw { width:12px; height:12px; display:inline-block; border-radius:2px; vertical-align:middle; margin-right:4px; }
  .insp { display:grid; grid-template-columns:repeat(4,1fr) 90px; gap:10px; padding:10px; background:#1e293b; border:1px solid #0284c7; border-radius:6px; font-size:11px; align-items:start; }
  .lbl { color:#94a3b8; text-transform:uppercase; font-size:9px; font-weight:700; }
  .val { font-family:monospace; font-size:12px; margin-top:2px; word-break:break-all; }
  canvas.still { cursor:crosshair; image-rendering:pixelated; }
  #crop { width:90px; height:90px; image-rendering:pixelated; border:1px solid #334155; }
</style></head>
<body>
<h4>Synchronized playback (one master clock)</h4>
<div class="grid">
  <div class="panel">
    <div class="hdr"><span style="color:#94a3b8;">ORIGINAL SOURCE (canonical bytes, browser decode)</span><span id="ptsA">0.000s</span></div>
    <div class="stage"><video id="vA" src="__VIDEO__" muted playsinline preload="auto"></video></div>
  </div>
  <div class="panel">
    <div class="warn">RECOMPRESSED PREVIEW - NOT THE CANONICAL RESULT</div>
    <div class="hdr"><span style="color:#38bdf8;">DERIVED PREVIEW (result shown at target PTS)</span><span id="ptsB">0.000s</span></div>
    <div class="stage"><video id="vB" src="__VIDEO__" muted playsinline preload="auto"></video>
      <img class="result-at-pts" id="resAtPts" src="__RESULT__" alt="Result at target PTS"></div>
  </div>
</div>
<div class="controls">
  <button id="play">▶ Play</button>
  <button class="secondary" id="stepBack">◀ 1 frame</button>
  <button class="secondary" id="stepFwd">1 frame ▶</button>
  <button class="secondary" id="toTarget">Seek target PTS</button>
  <input type="range" id="seek" min="0" max="90" step="0.001" value="0">
  <span id="clock" style="font-family:monospace;">master 0.000s</span>
</div>

<h4>Lossless stills (canonical PNGs) and provenance overlay</h4>
<div class="legend">
  <strong style="color:#94a3b8;">LEGEND</strong>
  <span><span class="sw" style="background:#94a3b8;"></span>ORIGINAL</span>
  <span><span class="sw" style="background:#00ffff;"></span>BORROWED</span>
  <span><span class="sw" style="background:repeating-linear-gradient(45deg,#ff00ff,#ff00ff 3px,transparent 3px,transparent 6px);"></span>GENERATED_BLEND (hatched)</span>
  <label style="margin-left:auto;"><input type="checkbox" id="ovl" checked> Overlay</label>
  <label><input type="checkbox" id="zoom" checked> Zoom to subject</label>
</div>
<div class="grid">
  <div class="panel">
    <div class="hdr"><span id="leftTitle" style="color:#94a3b8;">TARGET FRAME (lossless PNG)</span><span id="leftPts"></span></div>
    <canvas class="still" id="cL" width="640" height="360"></canvas>
  </div>
  <div class="panel">
    <div class="hdr"><span style="color:#38bdf8;">PROBITY RESULT (lossless PNG) + overlay</span><span id="cursor">hover a pixel</span></div>
    <canvas class="still" id="cR" width="640" height="360"></canvas>
  </div>
</div>

<div style="height:8px"></div>
<div class="insp" id="inspector">
  <div><div class="lbl">Class / target (x, y)</div><div class="val" id="iClass">-</div><div class="val" id="iXY">-</div></div>
  <div><div class="lbl">Source frame / PTS</div><div class="val" id="iFrame">-</div><div class="val" id="iPts">-</div></div>
  <div><div class="lbl">Source (x, y) / transform</div><div class="val" id="iSrc">-</div><div class="val" id="iXf">-</div></div>
  <div><div class="lbl">Color / reasons</div><div class="val" id="iColor">-</div><div class="val" id="iReasons">-</div>
    <div style="margin-top:6px;"><button id="openSrc">🔍 Open source</button> <button class="secondary" id="backTarget">Back to target</button></div></div>
  <div><div class="lbl">Donor crop</div><canvas id="crop" width="90" height="90"></canvas></div>
</div>

<script>
const D = __DATA__;
const W = D.W, H = D.H;
const exc = new Map();
for (const r of D.exceptions) exc.set(r[1] * W + r[0], r);
function resolve(x, y) {
  const r = exc.get(y * W + x);
  return r ? {cls:r[2], idx:r[3], sx:r[4], sy:r[5]} : {cls:0, idx:0, sx:x, sy:y};
}

// ---------------- synchronized players ----------------
const vA = document.getElementById('vA'), vB = document.getElementById('vB');
const seek = document.getElementById('seek'), resAtPts = document.getElementById('resAtPts');
const targetS = D.targetPtsUs / 1e6, frameS = 1 / D.fps;
function syncFrom(master) {
  const t = master.currentTime;
  const other = master === vA ? vB : vA;
  if (Math.abs(other.currentTime - t) > frameS / 2) other.currentTime = t;
  document.getElementById('ptsA').textContent = vA.currentTime.toFixed(3) + 's';
  document.getElementById('ptsB').textContent = t.toFixed(3) + 's';
  document.getElementById('clock').textContent = 'master ' + t.toFixed(3) + 's';
  seek.value = t;
  resAtPts.style.display = Math.abs(t - targetS) <= frameS / 2 ? 'block' : 'none';
}
function seekTo(t) { vA.currentTime = t; vB.currentTime = t; syncFrom(vA); }
vA.addEventListener('loadedmetadata', () => { seek.max = vA.duration; seekTo(targetS); });
vA.addEventListener('timeupdate', () => syncFrom(vA));
vA.addEventListener('seeked', () => syncFrom(vA));
document.getElementById('play').onclick = (e) => {
  if (vA.paused) { vA.play(); vB.play(); e.target.textContent = '❚❚ Pause'; }
  else { vA.pause(); vB.pause(); seekTo(vA.currentTime); e.target.textContent = '▶ Play'; }
};
vA.addEventListener('pause', () => vB.pause());
document.getElementById('stepBack').onclick = () => seekTo(Math.max(0, vA.currentTime - frameS));
document.getElementById('stepFwd').onclick = () => seekTo(vA.currentTime + frameS);
document.getElementById('toTarget').onclick = () => seekTo(targetS);
seek.oninput = () => seekTo(parseFloat(seek.value));

// ---------------- lossless stills ----------------
const imgT = new Image(), imgR = new Image();
imgT.src = "__TARGET__"; imgR.src = "__RESULT__";
const donorImgs = {};
for (const k in D.donors) { const im = new Image(); im.src = D.donors[k]; im.onload = refresh; donorImgs[k] = im; }
const cL = document.getElementById('cL'), cR = document.getElementById('cR');
let showOverlay = true, zoomed = true, leftIdx = 0, pinned = null, hover = null;

function viewport() {
  if (!zoomed) return {x0:0, y0:0, w:W, h:H};
  const [x1, y1, x2, y2] = D.bbox;
  let w = Math.max((x2 - x1) * 2.2, 160), h = w * 9 / 16;
  if (h < (y2 - y1) * 2.2) { h = (y2 - y1) * 2.2; w = h * 16 / 9; }
  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  return {x0: Math.max(0, Math.min(W - w, cx - w / 2)), y0: Math.max(0, Math.min(H - h, cy - h / 2)), w, h};
}
function toCanvas(v, x, y, c) { return [(x - v.x0) * c.width / v.w, (y - v.y0) * c.height / v.h]; }

function drawStill(c, img, v) {
  const ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, c.width, c.height);
  if (img.complete && img.naturalWidth) ctx.drawImage(img, v.x0, v.y0, v.w, v.h, 0, 0, c.width, c.height);
  return ctx;
}
function drawOverlay(ctx, v) {
  if (!showOverlay) return;
  ctx.fillStyle = 'rgba(148,163,184,0.18)';
  ctx.fillRect(0, 0, cR.width, cR.height);
  const px = cR.width / v.w;
  for (const r of D.exceptions) {
    const [x, y, cls] = r;
    if (x < v.x0 - 1 || x > v.x0 + v.w || y < v.y0 - 1 || y > v.y0 + v.h) continue;
    const [cx, cy] = toCanvas(v, x, y, cR);
    if (cls === 1) { ctx.fillStyle = 'rgba(0,255,255,0.55)'; ctx.fillRect(cx, cy, Math.ceil(px), Math.ceil(px)); }
    else if (cls === 2 && (x + y) % 4 < 2) { ctx.fillStyle = 'rgba(255,0,255,0.8)'; ctx.fillRect(cx, cy, Math.ceil(px), Math.ceil(px)); }
  }
}
function mark(ctx, c, v, x, y, color, size) {
  const [cx, cy] = toCanvas(v, x - size / 2, y - size / 2, c);
  ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.strokeRect(cx, cy, size * c.width / v.w, size * c.height / v.h);
}
function drawAll() {
  const v = viewport();
  const left = leftIdx === 0 ? imgT : (donorImgs[leftIdx] || imgT);
  const ctxL = drawStill(cL, left, v);
  const ctxR = drawStill(cR, imgR, v);
  drawOverlay(ctxR, v);
  const p = pinned || hover;
  if (p) {
    mark(ctxR, cR, v, p[0] + 0.5, p[1] + 0.5, '#f59e0b', 3);
    if (leftIdx !== 0) {
      const o = resolve(p[0], p[1]);
      mark(ctxL, cL, v, o.sx + 0.5, o.sy + 0.5, '#f59e0b', 8);
      // outline the whole source region this donor contributed
      let mnx = 1e9, mny = 1e9, mxx = -1, mxy = -1;
      for (const r of D.exceptions) if (r[3] === leftIdx) { mnx = Math.min(mnx, r[4]); mny = Math.min(mny, r[5]); mxx = Math.max(mxx, r[4]); mxy = Math.max(mxy, r[5]); }
      if (mxx >= 0) {
        const [ax, ay] = toCanvas(v, mnx, mny, cL), [bx, by] = toCanvas(v, mxx + 1, mxy + 1, cL);
        ctxL.setLineDash([4, 3]); ctxL.strokeStyle = '#00ffff'; ctxL.strokeRect(ax, ay, bx - ax, by - ay); ctxL.setLineDash([]);
      }
    }
  }
}
function refresh() { const p = pinned || hover; if (p) inspect(p[0], p[1]); drawAll(); }
imgT.onload = refresh; imgR.onload = refresh;

function fmtPts(us) { return (us / 1e6).toFixed(3) + 's (' + us + ' µs)'; }
function inspect(x, y) {
  const o = resolve(x, y), e = D.lut[o.idx];
  const name = D.classNames[o.cls];
  const colors = ['#94a3b8', '#00ffff', '#ff00ff'];
  document.getElementById('iClass').innerHTML = '<span style="color:' + colors[o.cls] + '">' + name + '</span>';
  document.getElementById('iXY').textContent = '(' + x + ', ' + y + ')';
  document.getElementById('iFrame').textContent = 'f' + e.frame_number + ' [' + e.role + ']';
  document.getElementById('iPts').textContent = fmtPts(e.pts_us);
  document.getElementById('iSrc').textContent = '(' + o.sx.toFixed(1) + ', ' + o.sy.toFixed(1) + ')';
  document.getElementById('iXf').textContent = (e.transform_id || 'identity') + ' · ' + e.alignment_method +
    (e.matrix ? ' [' + e.matrix.map(n => +n.toFixed(3)).join(',') + ']' : '');
  document.getElementById('iColor').textContent = e.color_gain ? 'gain ' + e.color_gain.join('/') + ' bias ' + e.color_bias.join('/') : 'no color transform';
  document.getElementById('iReasons').textContent = e.decision_ids.length ? e.decision_ids.map(id => D.decisions[id] || id.slice(0, 8)).join(', ') : 'n/a (target pixel)';
  document.getElementById('openSrc').disabled = o.idx === 0;
  const crop = document.getElementById('crop').getContext('2d');
  crop.imageSmoothingEnabled = false;
  crop.clearRect(0, 0, 90, 90);
  const src = o.idx === 0 ? imgT : donorImgs[o.idx];
  if (src && src.complete) crop.drawImage(src, o.sx - 7, o.sy - 7, 15, 15, 0, 0, 90, 90);
  crop.strokeStyle = '#f59e0b'; crop.strokeRect(42, 42, 6, 6);
}
function pixelAt(ev) {
  const rect = cR.getBoundingClientRect(), v = viewport();
  const x = Math.floor(v.x0 + (ev.clientX - rect.left) * v.w / rect.width);
  const y = Math.floor(v.y0 + (ev.clientY - rect.top) * v.h / rect.height);
  return [Math.max(0, Math.min(W - 1, x)), Math.max(0, Math.min(H - 1, y))];
}
cR.addEventListener('mousemove', (ev) => {
  hover = pixelAt(ev);
  document.getElementById('cursor').textContent = '(' + hover[0] + ', ' + hover[1] + ')' + (pinned ? ' · pinned' : '');
  if (!pinned) inspect(hover[0], hover[1]);
  drawAll();
});
cR.addEventListener('click', (ev) => { pinned = pinned ? null : pixelAt(ev); if (pinned) inspect(pinned[0], pinned[1]); drawAll(); });
cR.addEventListener('mouseleave', () => { hover = null; drawAll(); });

document.getElementById('openSrc').onclick = () => {
  const p = pinned || hover || D.initial;
  const o = resolve(p[0], p[1]);
  if (o.idx === 0) return;
  pinned = p;
  const e = D.lut[o.idx];
  leftIdx = o.idx;
  document.getElementById('leftTitle').textContent = 'SOURCE DONOR f' + e.frame_number + ' (lossless PNG)';
  document.getElementById('leftTitle').style.color = '#f59e0b';
  document.getElementById('leftPts').textContent = 'PTS ' + fmtPts(e.pts_us);
  seekTo(e.pts_us / 1e6);
  inspect(p[0], p[1]);
  drawAll();
};
document.getElementById('backTarget').onclick = () => {
  leftIdx = 0;
  document.getElementById('leftTitle').textContent = 'TARGET FRAME (lossless PNG)';
  document.getElementById('leftTitle').style.color = '#94a3b8';
  document.getElementById('leftPts').textContent = 'PTS ' + fmtPts(D.targetPtsUs);
  seekTo(targetS); drawAll();
};
document.getElementById('ovl').onchange = (e) => { showOverlay = e.target.checked; drawAll(); };
document.getElementById('zoom').onchange = (e) => { zoomed = e.target.checked; drawAll(); };

document.getElementById('leftPts').textContent = 'PTS ' + fmtPts(D.targetPtsUs);
pinned = D.initial; inspect(D.initial[0], D.initial[1]);
drawAll();
</script>
</body></html>
"""
