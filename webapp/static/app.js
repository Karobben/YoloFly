const runSelect = document.getElementById("runSelect");
const refreshRunsBtn = document.getElementById("refreshRuns");
const imageSelect = document.getElementById("imageSelect");
const videoSelect = document.getElementById("videoSelect");
const frameInput = document.getElementById("frameInput");
const openImageBtn = document.getElementById("openImage");
const openFrameBtn = document.getElementById("openFrame");
const labelPathInput = document.getElementById("labelPath");
const loadLabelBtn = document.getElementById("loadLabel");
const saveLabelBtn = document.getElementById("saveLabel");
const rawText = document.getElementById("rawText");
const applyRawBtn = document.getElementById("applyRaw");
const classFiltersEl = document.getElementById("classFilters");
const boxListEl = document.getElementById("boxList");
const addClassInput = document.getElementById("addClassInput");
const toggleAddModeBtn = document.getElementById("toggleAddMode");
const classFiltersCsvEl = document.getElementById("classFiltersCsv");
const boxListCsvEl = document.getElementById("boxListCsv");
const addClassInputCsv = document.getElementById("addClassInputCsv");
const toggleAddModeCsvBtn = document.getElementById("toggleAddModeCsv");
const zoomOutBtn = document.getElementById("zoomOut");
const zoomInBtn = document.getElementById("zoomIn");
const zoomRange = document.getElementById("zoomRange");
const zoomText = document.getElementById("zoomText");
const classStatsEl = document.getElementById("classStats");
const statusEl = document.getElementById("status");
const tabRuns = document.getElementById("tabRuns");
const tabCsv = document.getElementById("tabCsv");
const paneRuns = document.getElementById("paneRuns");
const paneCsv = document.getElementById("paneCsv");
const csvSelect = document.getElementById("csvSelect");
const refreshCsvBtn = document.getElementById("refreshCsv");
const loadCsvBtn = document.getElementById("loadCsv");
const csvPreviewEl = document.getElementById("csvPreview");
const csvFullStatsEl = document.getElementById("csvFullStats");
const csvStatusEl = document.getElementById("csvStatus");
const videoViewInfoText = document.getElementById("videoViewInfoText");
const jsonSelect = document.getElementById("jsonSelect");
const refreshJsonBtn = document.getElementById("refreshJson");
const loadJsonBtn = document.getElementById("loadJson");
const showTracking = document.getElementById("showTracking");
const showTrackIds = document.getElementById("showTrackIds");
const showTrackHistory = document.getElementById("showTrackHistory");
const trackHistoryLen = document.getElementById("trackHistoryLen");
const trackingInfoText = document.getElementById("trackingInfoText");
const videoPathInput = document.getElementById("videoPathInput");
const videoPathFrameInput = document.getElementById("videoPathFrameInput");
const openVideoPathFrameBtn = document.getElementById("openVideoPathFrame");
const prevFrameBtn = document.getElementById("prevFrameBtn");
const nextFrameBtn = document.getElementById("nextFrameBtn");
const frameSlider = document.getElementById("frameSlider");
const playPauseBtn = document.getElementById("playPauseBtn");
const videoInfoText = document.getElementById("videoInfoText");
const trackingPathText = document.getElementById("trackingPathText");
const videoViewLoadModal = document.getElementById("videoViewLoadModal");
const videoViewLoadStatus = document.getElementById("videoViewLoadStatus");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const videoPlayer = document.getElementById("videoPlayer");
const mediaStage = document.getElementById("mediaStage");

let mediaImg = null;
let boxes = [];
let selected = -1;
let dragStart = null;
let visibleClasses = new Set();
let addMode = false;
let zoomScale = 1.0;
let csvFrameMin = 1;
let csvFrameMax = 1;
let loadedVideoFrameCount = 1;
/** When set, frame↔time mapping uses this (media-timeline span) instead of OpenCV probe count. */
let loadedVideoFrameCountTimeline = null;
let classUniverse = [];
/** Active FPS for frame↔time mapping in the HTML5 video element. */
let loadedVideoFps = 10;
/** OpenCV / container FPS from `/api/video_info_by_path` (not overwritten by refine). */
let loadedVideoFpsReported = 0;
let playTimer = null;
/** `requestVideoFrameCallback` handle while playing (must cancel on stop). */
let playbackVfcHandle = null;
/** True while Video View playback is active (RVFC path does not use `playTimer`). */
let playbackActive = false;
/** Bumped on each new seek/overlay load so stale async results are ignored. */
let overlayLoadGeneration = 0;
/** Debounce rapid frame-slider scrubbing (full seek + API per tick is expensive). */
let frameSliderDebounceTimer = null;
const FRAME_SLIDER_DEBOUNCE_MS = 90;
let mediaMode = "image";
let currentVideoPath = "";
let currentCsvPath = "";
let currentJsonPath = "";
let trackingCurrent = {};
let trackingHistory = [];
let trackingFrameMin = 1;
let trackingFrameMax = 1;
let frameViewMin = 1;
let frameViewMax = 1;
const pageQs = new URLSearchParams(window.location.search);
const initialRunParam = pageQs.get("run");
const initialSnapshotDir = (pageQs.get("snapshot_dir") || "").trim();
const initialTrackingDir = (pageQs.get("tracking_dir") || "").trim();
const initialTrackingVideoPath = (pageQs.get("video_path") || "").trim();

/** True when opened via ?snapshot_dir=… (label load/save use absolute paths). */
let exploreSnapshotActive = false;
const CLASS_COLORS = [
  "#3d8bfd", "#ff6b6b", "#51cf66", "#ffd43b", "#845ef7",
  "#22b8cf", "#f06595", "#ffa94d", "#94d82d", "#e599f7",
];

function setStatus(msg) { statusEl.textContent = msg || ""; }
function setCsvStatus(msg) { csvStatusEl.textContent = msg || ""; }

function showVideoViewLoad(msg) {
  if (!videoViewLoadModal || !videoViewLoadStatus) return;
  videoViewLoadStatus.textContent = msg || "Loading…";
  videoViewLoadModal.classList.remove("hidden");
}

function updateVideoViewLoad(msg) {
  if (!videoViewLoadStatus) return;
  videoViewLoadStatus.textContent = msg || "Loading…";
}

function hideVideoViewLoad() {
  if (!videoViewLoadModal) return;
  videoViewLoadModal.classList.add("hidden");
}

function q(url) { return fetch(url).then((r) => r.json()); }

function fillSelect(el, items) {
  el.innerHTML = "";
  for (const it of items) {
    const o = document.createElement("option");
    o.value = it;
    o.textContent = it;
    el.appendChild(o);
  }
}

function setTab(tab) {
  const isRuns = tab === "runs";
  paneRuns.classList.toggle("hidden", !isRuns);
  paneCsv.classList.toggle("hidden", isRuns);
  tabRuns.classList.toggle("active", isRuns);
  tabCsv.classList.toggle("active", !isRuns);
}

function parseYolo(text) {
  const out = [];
  for (const ln of text.split(/\r?\n/)) {
    const t = ln.trim();
    if (!t) continue;
    const p = t.split(/\s+/).map(Number);
    if (p.length < 5 || p.some(Number.isNaN)) continue;
    out.push({ cls: Math.round(p[0]), xc: p[1], yc: p[2], w: p[3], h: p[4], conf: p[5] });
  }
  return out;
}

function yoloText() {
  return boxes.map((b) => [b.cls, b.xc, b.yc, b.w, b.h, b.conf]
    .filter((x, i) => i < 5 || x != null)
    .map((x) => typeof x === "number" ? Number(x).toFixed(6).replace(/\.?0+$/, "") : x)
    .join(" ")).join("\n");
}

function classesPresent() {
  return [...new Set(boxes.map((b) => b.cls))].sort((a, b) => a - b);
}

function setClassUniverse(classes) {
  classUniverse = [...new Set(classes)].sort((a, b) => a - b);
}

function colorForClass(cls) {
  return CLASS_COLORS[Math.abs((cls || 0) % CLASS_COLORS.length)];
}

function colorForFly(id) {
  const n = String(id).split("").reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
  return CLASS_COLORS[n % CLASS_COLORS.length];
}

function ensureClassVisibility() {
  for (const c of classesPresent()) {
    if (!visibleClasses.has(c)) visibleClasses.add(c);
  }
}

function syncRawFromBoxes() {
  rawText.value = yoloText();
}

function hasMedia() {
  return mediaMode === "video" ? videoPlayer.videoWidth > 0 : !!mediaImg;
}

function applyZoom() {
  if (!hasMedia()) return;
  canvas.style.width = `${Math.round(canvas.width * zoomScale)}px`;
  canvas.style.height = `${Math.round(canvas.height * zoomScale)}px`;
  if (mediaMode === "video") {
    videoPlayer.style.width = canvas.style.width;
    videoPlayer.style.height = canvas.style.height;
  }
  zoomText.textContent = `${Math.round(zoomScale * 100)}%`;
  zoomRange.value = String(Math.round(zoomScale * 100));
}

function renderClassFilters() {
  const targets = [classFiltersEl, classFiltersCsvEl].filter(Boolean);
  const classes = classUniverse.length ? classUniverse : classesPresent();
  for (const t of targets) t.innerHTML = "";
  for (const cls of classes) {
    for (const target of targets) {
      const chip = document.createElement("label");
      chip.className = "class-chip";
      chip.style.borderColor = colorForClass(cls);
      const ck = document.createElement("input");
      ck.type = "checkbox";
      ck.checked = visibleClasses.has(cls);
      ck.onchange = () => {
        if (ck.checked) visibleClasses.add(cls);
        else visibleClasses.delete(cls);
        draw();
        renderBoxList();
        renderClassFilters();
      };
      const dot = document.createElement("span");
      dot.style.display = "inline-block";
      dot.style.width = "8px";
      dot.style.height = "8px";
      dot.style.borderRadius = "50%";
      dot.style.background = colorForClass(cls);
      const txt = document.createElement("span");
      txt.textContent = `c${cls}`;
      chip.appendChild(ck);
      chip.appendChild(dot);
      chip.appendChild(txt);
      target.appendChild(chip);
    }
  }
}

function selectBox(idx) {
  selected = idx;
  draw();
  renderBoxList();
  const node = boxListEl.querySelector(`[data-idx="${idx}"]`);
  if (node) node.scrollIntoView({ block: "nearest" });
}

function updateAddModeUI() {
  toggleAddModeBtn.textContent = addMode ? "Add: ON" : "Add: OFF";
  if (toggleAddModeCsvBtn) toggleAddModeCsvBtn.textContent = addMode ? "Add: ON" : "Add: OFF";
  canvas.classList.toggle("adding", addMode);
}

function renderClassStats() {
  const counts = {};
  for (const b of boxes) counts[b.cls] = (counts[b.cls] || 0) + 1;
  const keys = Object.keys(counts).map(Number).sort((a, b) => a - b);
  classStatsEl.innerHTML = "";
  if (!keys.length) {
    classStatsEl.textContent = "No boxes loaded.";
    return;
  }
  for (const cls of keys) {
    const row = document.createElement("div");
    row.className = "stat-row";
    const left = document.createElement("span");
    left.textContent = `Class ${cls}`;
    left.style.color = colorForClass(cls);
    const right = document.createElement("span");
    right.textContent = String(counts[cls]);
    row.appendChild(left);
    row.appendChild(right);
    classStatsEl.appendChild(row);
  }
}

function renderBoxList() {
  const targets = [boxListEl, boxListCsvEl].filter(Boolean);
  for (const t of targets) t.innerHTML = "";
  const order = boxes.map((_, i) => i);
  if (selected >= 0 && selected < order.length) {
    const p = order.indexOf(selected);
    if (p > 0) {
      order.splice(p, 1);
      order.unshift(selected);
    }
  }
  const buildRow = (idx) => {
    const b = boxes[idx];
    const row = document.createElement("div");
    row.className = "box-row" + (idx === selected ? " selected" : "");
    row.dataset.idx = String(idx);
    row.onclick = () => selectBox(idx);

    const grid = document.createElement("div");
    grid.className = "box-grid";

    const mkInput = (value, step = "0.000001") => {
      const i = document.createElement("input");
      i.type = "number";
      i.step = step;
      i.value = String(value ?? "");
      return i;
    };

    const cls = mkInput(b.cls, "1");
    const xc = mkInput(b.xc);
    const yc = mkInput(b.yc);
    const w = mkInput(b.w);
    const h = mkInput(b.h);
    const del = document.createElement("button");
    del.className = "del-btn";
    del.textContent = "Del";
    del.onclick = (e) => {
      e.stopPropagation();
      boxes.splice(idx, 1);
      if (selected >= boxes.length) selected = boxes.length - 1;
      syncRawFromBoxes();
      renderBoxList();
      renderClassStats();
      draw();
    };

    const apply = () => {
      b.cls = Math.round(parseFloat(cls.value) || 0);
      b.xc = Math.min(1, Math.max(0, parseFloat(xc.value) || 0));
      b.yc = Math.min(1, Math.max(0, parseFloat(yc.value) || 0));
      b.w = Math.min(1, Math.max(1e-6, parseFloat(w.value) || 1e-6));
      b.h = Math.min(1, Math.max(1e-6, parseFloat(h.value) || 1e-6));
      syncRawFromBoxes();
      renderClassStats();
      draw();
    };
    [cls, xc, yc, w, h].forEach((i) => i.onchange = apply);

    grid.appendChild(cls);
    grid.appendChild(xc);
    grid.appendChild(yc);
    grid.appendChild(w);
    grid.appendChild(h);
    grid.appendChild(del);
    row.appendChild(grid);
    return row;
  };
  order.forEach((idx) => {
    for (const t of targets) t.appendChild(buildRow(idx));
  });
}

function trackingCenter(fly, key) {
  const v = fly && fly[key];
  if (!Array.isArray(v) || v.length < 2) return null;
  return { x: Number(v[0]) * canvas.width, y: Number(v[1]) * canvas.height };
}

function nearestHeadCenterFromCsv(body) {
  if (!body || !Array.isArray(boxes) || !boxes.length) return null;
  let best = null;
  let bestD2 = Infinity;
  for (const b of boxes) {
    if (!b || Number(b.cls) !== 1) continue;
    const hx = Number(b.xc) * canvas.width;
    const hy = Number(b.yc) * canvas.height;
    if (!Number.isFinite(hx) || !Number.isFinite(hy)) continue;
    const dx = hx - body.x;
    const dy = hy - body.y;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD2) {
      bestD2 = d2;
      best = { x: hx, y: hy };
    }
  }
  return best;
}

function drawArrow(x1, y1, x2, y2, color, alpha = 1) {
  const oldAlpha = ctx.globalAlpha;
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const size = 8;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - size * Math.cos(angle - Math.PI / 6), y2 - size * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(x2 - size * Math.cos(angle + Math.PI / 6), y2 - size * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = oldAlpha;
}

function drawTrackingOverlay() {
  const wantArrows = !!(showTracking && showTracking.checked);
  const wantIds = !!(showTrackIds && showTrackIds.checked);
  if (!wantArrows && !wantIds) return;

  if (wantArrows && showTrackHistory && showTrackHistory.checked && trackingHistory.length) {
    const nFrames = trackingHistory.length;
    for (let i = 0; i < nFrames; i++) {
      const fr = trackingHistory[i];
      const ageAlpha = Math.max(0.08, ((i + 1) / nFrames) * 0.7);
      const radius = 2 + ((i + 1) / nFrames) * 2;
      for (const [id, fly] of Object.entries(fr.flies || {})) {
        const p = trackingCenter(fly, "body");
        if (!p) continue;
        const oldAlpha = ctx.globalAlpha;
        ctx.globalAlpha = ageAlpha;
        ctx.fillStyle = colorForFly(id);
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = oldAlpha;
      }
    }
  }

  for (const [id, fly] of Object.entries(trackingCurrent || {})) {
    const body = trackingCenter(fly, "body");
    const head = trackingCenter(fly, "head") || nearestHeadCenterFromCsv(body);
    if (!body) continue;
    const color = colorForFly(id);
    if (wantArrows && head) {
      drawArrow(body.x, body.y, head.x, head.y, color, 0.95);
    }
    if (wantArrows || wantIds) {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(body.x, body.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
    if (wantIds) {
      const label = String(id).replace(/^fly_/, "");
      ctx.font = "14px sans-serif";
      const tw = ctx.measureText(label).width;
      const x = body.x + 5;
      const y = Math.max(14, body.y - 6);
      ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
      ctx.fillRect(x - 2, y - 13, tw + 6, 17);
      ctx.fillStyle = color;
      ctx.fillText(label, x + 1, y);
    }
  }
}

function draw() {
  if (!hasMedia()) return;
  if (mediaMode === "video") {
    canvas.width = videoPlayer.videoWidth;
    canvas.height = videoPlayer.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  } else {
    canvas.width = mediaImg.naturalWidth;
    canvas.height = mediaImg.naturalHeight;
    ctx.drawImage(mediaImg, 0, 0);
  }
  boxes.forEach((b, i) => {
    if (!visibleClasses.has(b.cls)) return;
    const x1 = (b.xc - b.w / 2) * canvas.width;
    const y1 = (b.yc - b.h / 2) * canvas.height;
    const w = b.w * canvas.width;
    const h = b.h * canvas.height;
    const classColor = colorForClass(b.cls);
    ctx.strokeStyle = i === selected ? "#ffffff" : classColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, w, h);
    // Draw class tag with class-specific color.
    const clsTxt = `c${b.cls}`;
    ctx.font = "14px sans-serif";
    const tw = ctx.measureText(clsTxt).width;
    ctx.fillStyle = classColor;
    ctx.fillRect(x1, Math.max(0, y1 - 18), tw + 10, 18);
    ctx.fillStyle = "#000";
    ctx.fillText(clsTxt, x1 + 5, Math.max(13, y1 - 5));
  });
  drawTrackingOverlay();
  applyZoom();
}

function hit(x, y) {
  for (let i = boxes.length - 1; i >= 0; i--) {
    const b = boxes[i];
    if (!visibleClasses.has(b.cls)) continue;
    const x1 = (b.xc - b.w / 2) * canvas.width;
    const y1 = (b.yc - b.h / 2) * canvas.height;
    const w = b.w * canvas.width;
    const h = b.h * canvas.height;
    if (x >= x1 && x <= x1 + w && y >= y1 && y <= y1 + h) return i;
  }
  return -1;
}

function loadImageURL(url) {
  const img = new Image();
  img.onload = () => {
    mediaMode = "image";
    videoPlayer.pause();
    videoPlayer.classList.add("hidden");
    canvas.classList.remove("overlay");
    mediaImg = img;
    draw();
  };
  img.onerror = () => setStatus("Failed to load media.");
  img.src = url;
}

/** 1-based frame count used for seek, playback sync, and FPS-vs-duration checks. */
function effectiveTimelineFrameCount() {
  const t = loadedVideoFrameCountTimeline;
  if (t != null && t >= 1) return Math.max(1, t);
  return Math.max(1, loadedVideoFrameCount || 1);
}

/**
 * When browser `duration` is much shorter than OpenCV nominal `(N−1)/fps`, the probe
 * frame count is usually inflated (short MP4s). Mapping with that N seeks too early,
 * so overlays look ahead of the decoded frame. Fit N to `duration × fps`.
 */
function recomputeLoadedVideoFrameCountTimeline() {
  loadedVideoFrameCountTimeline = null;
  const nProbe = Math.max(1, loadedVideoFrameCount || 1);
  if (nProbe < 2) return;
  const dur = videoPlayer.duration;
  const fpsRep = loadedVideoFpsReported > 0 ? loadedVideoFpsReported : 0;
  if (!(Number.isFinite(dur) && dur > 0) || fpsRep < 0.25) return;
  const nominalDur = (nProbe - 1) / fpsRep;
  if (!(nominalDur > 0)) return;
  if (dur < nominalDur * 0.93) {
    const nFit = Math.max(2, Math.min(nProbe, Math.floor(dur * fpsRep + 1e-9) + 1));
    if (nFit + 1 < nProbe) {
      loadedVideoFrameCountTimeline = nFit;
    }
  }
}

/**
 * Effective FPS for mapping frame index ↔ HTML5 timeline.
 * - Prefer OpenCV probe when sane and consistent with browser duration.
 * - If probe is missing or disagrees (>3%) with (N−1)/duration, use timeline FPS
 *   so seeks line up with what the decoder actually presents (fixes mid-clip drift).
 */
function effectiveFpsForFrameMapping() {
  const n = effectiveTimelineFrameCount();
  const dur = videoPlayer.duration;
  let fdur = 0;
  if (Number.isFinite(dur) && dur > 0 && n > 1) {
    fdur = (n - 1) / dur;
  }
  const rep = loadedVideoFpsReported > 0 && loadedVideoFpsReported < 480 ? loadedVideoFpsReported : 0;
  if (fdur >= 0.25 && fdur < 480 && rep >= 0.25 && rep < 480) {
    const relDiff = Math.abs(fdur - rep) / Math.max(rep, fdur, 1e-6);
    if (relDiff > 0.03) {
      return fdur;
    }
    return rep;
  }
  if (rep >= 0.25 && rep < 480) {
    return rep;
  }
  if (fdur >= 0.25 && fdur < 480) {
    return fdur;
  }
  const f = loadedVideoFps > 0 ? loadedVideoFps : 0;
  return f >= 0.25 && f < 480 ? f : 10;
}

/**
 * Use CFR t=(frame−1)/fps when it matches the HTML5 timeline; otherwise linear across duration.
 * Short MP4s often have probe FPS × frame count that disagrees with `video.duration` → mid-clip drift.
 */
function useFpsBasedFrameTimeMapping() {
  const eff = effectiveFpsForFrameMapping();
  if (eff < 0.25 || eff >= 480) return false;
  const n = effectiveTimelineFrameCount();
  const dur = videoPlayer.duration;
  if (!(Number.isFinite(dur) && dur > 0) || n < 2) return true;
  const cfrLen = (n - 1) / eff;
  const relMis = Math.abs(cfrLen - dur) / Math.max(dur, 1e-6);
  return relMis <= 0.012;
}

/**
 * During play, RVFC `mediaTime` (and sometimes `currentTime`) can sit slightly ahead of the
 * composited frame — overlays then look "ahead" of the fly. When paused, use the exact time.
 */
function playbackTimeForOverlay(optMediaTime) {
  const ct = videoPlayer.currentTime;
  const tm =
    typeof optMediaTime === "number" && Number.isFinite(optMediaTime) && optMediaTime >= 0
      ? optMediaTime
      : ct;
  if (videoPlayer.paused) return Math.max(0, tm);
  let base = tm;
  if (Number.isFinite(ct)) base = Math.min(tm, ct);
  let slack = 0.015;
  if (Number.isFinite(ct) && tm > ct) slack += tm - ct;
  const fp = 1 / Math.max(0.25, effectiveFpsForFrameMapping());
  slack += Math.min(0.25 * fp, 0.03);
  if (!useFpsBasedFrameTimeMapping()) {
    const dur = videoPlayer.duration;
    const n = effectiveTimelineFrameCount();
    if (Number.isFinite(dur) && dur > 0 && n > 1) {
      slack += Math.min(0.03, dur / (300 * (n - 1)));
    }
  }
  return Math.max(0, base - slack);
}

/**
 * Refine `loadedVideoFps` for display and for **fallback** seeking when
 * `video.duration` is not yet available (metadata still loading).
 *
 * OpenCV/container FPS often disagrees with the HTML5 media timeline (VFR,
 * bad metadata, short clips). When the server reports probe FPS, scrubbing uses
 * CFR in `frameIndexToVideoTime`; this refinement mainly affects the FPS label,
 * `videoInfoText`, and the pre-metadata fallback path (no probe FPS → linear).
 *
 * We prefer FPS implied by browser duration and OpenCV frame count when the
 * browser's total duration is **not** suspiciously shorter than OpenCV's
 * nominal length `(frame_count-1)/fps`. Many short MP4s report a rounded
 * `video.duration` that is too small; that inflates `(n-1)/duration` and FPS
 * derived from it becomes unreliable. In that case fall back to the
 * server-reported FPS.
 */
function refineLoadedVideoFpsFromMediaDuration() {
  const dur = videoPlayer.duration;
  const n = Math.max(1, loadedVideoFrameCount || 1);
  if (!Number.isFinite(dur) || dur <= 0 || n < 2) return;
  const fpsFromDuration = (n - 1) / dur;
  const fpsRep = loadedVideoFpsReported > 0 ? loadedVideoFpsReported : 0;
  let chosen = fpsFromDuration;
  if (fpsRep > 0) {
    const nominalDur = (n - 1) / fpsRep;
    if (nominalDur > 0 && dur < nominalDur * 0.97) {
      chosen = fpsRep;
    }
  }
  if (chosen < 0.25 || chosen >= 480) return;
  loadedVideoFps = chosen;
  recomputeLoadedVideoFrameCountTimeline();
  if (videoInfoText) {
    const nProbe = n;
    const nMap = effectiveTimelineFrameCount();
    const eff = effectiveFpsForFrameMapping();
    const syncMode = useFpsBasedFrameTimeMapping()
      ? `overlay sync: CFR @ ${eff.toFixed(2)} fps`
      : "overlay sync: linear timeline (CFR vs duration mismatch or no probe)";
    const src =
      chosen === fpsRep && fpsRep > 0 && dur < ((nProbe - 1) / fpsRep) * 0.97
        ? "OpenCV FPS (browser duration was short vs nominal)"
        : "from video length";
    const frameLine =
      nMap < nProbe
        ? `Frames: ${nMap} for sync (OpenCV probe ${nProbe})`
        : `Frames: ${nProbe}`;
    videoInfoText.textContent =
      `${frameLine}, FPS: ${chosen.toFixed(2)} (${src}) · ${syncMode}`;
  }
  applyFrameViewLimits();
}

/**
 * Map 1-based frame index to HTML5 `currentTime`.
 *
 * When CFR is consistent with the browser timeline (`(N-1)/fps ≈ duration`),
 * use **constant frame rate**: `t = (frame-1)/fps` clamped to duration.
 * Otherwise use **linear** spread across `[0,duration]` so seeks match the
 * decoder (avoids middle drift when probe metadata disagrees with the file).
 *
 * Before metadata loads or without a sane FPS, fall back to FPS estimate or linear.
 */
function frameIndexToVideoTime(frameNo) {
  const ff = clampFrameToView(frameNo);
  const n = effectiveTimelineFrameCount();
  const dur = videoPlayer.duration;
  if (useFpsBasedFrameTimeMapping()) {
    const fps = effectiveFpsForFrameMapping();
    const t = (ff - 1) / fps;
    if (Number.isFinite(dur) && dur > 0) {
      return Math.min(Math.max(0, t), Math.max(0, dur - 1e-4));
    }
    return Math.max(0, t);
  }
  if (Number.isFinite(dur) && dur > 0 && n > 1) {
    return ((ff - 1) / (n - 1)) * Math.max(0, dur - 1e-4);
  }
  const fps = loadedVideoFps || 10;
  return Math.max(0, (ff - 1) / fps);
}

/** Inverse of frameIndexToVideoTime for playback sync (clamped to current frame view). */
function videoTimeToFrameIndex(t) {
  const n = effectiveTimelineFrameCount();
  const dur = videoPlayer.duration;
  if (useFpsBasedFrameTimeMapping()) {
    const fps = effectiveFpsForFrameMapping();
    const tt = Math.max(0, t);
    let f = 1 + Math.floor(tt * fps + 1e-6);
    if (Number.isFinite(dur) && dur > 0 && tt >= dur - 1e-3) {
      f = n;
    }
    f = Math.max(1, Math.min(f, n));
    return clampFrameToView(f);
  }
  let abs;
  if (Number.isFinite(dur) && dur > 0 && n > 1) {
    const r = Math.max(0, Math.min(1, t / dur));
    abs = 1 + Math.floor(r * (n - 1) + 1e-12);
  } else {
    abs = Math.floor(t * (loadedVideoFps || 10)) + 1;
  }
  return clampFrameToView(abs);
}

/**
 * HTML5 video seeks asynchronously. If we load CSV/JSON and draw immediately after
 * assigning currentTime, the decoder often still shows the previous frame — overlays
 * then look "ahead" of the pixels. Wait for seeked (with timeout) before fetching overlays.
 */
function waitForVideoSeeked(targetTime) {
  return new Promise((resolve) => {
    const dur = videoPlayer.duration;
    let clamped = Math.max(0, targetTime);
    if (Number.isFinite(dur) && dur > 0) {
      clamped = Math.min(clamped, Math.max(0, dur - 1e-4));
    }
    const n = effectiveTimelineFrameCount();
    let eps = 0.02;
    if (Number.isFinite(dur) && dur > 0 && n > 1) {
      eps = Math.min(0.05, dur / (200 * Math.max(1, n - 1)));
    } else {
      const fps = loadedVideoFps || 10;
      eps = Math.min(0.02, 0.25 / fps);
    }
    const cur = videoPlayer.currentTime;
    if (Number.isFinite(cur) && Math.abs(cur - clamped) <= eps) {
      requestAnimationFrame(() => resolve());
      return;
    }
    let finished = false;
    const done = () => {
      if (finished) return;
      finished = true;
      clearTimeout(tmr);
      videoPlayer.removeEventListener("seeked", onSeeked);
      resolve();
    };
    const tmr = setTimeout(done, 8000);
    const onSeeked = () => done();
    videoPlayer.addEventListener("seeked", onSeeked, { once: true });
    videoPlayer.currentTime = clamped;
  });
}

/**
 * After seeked, wait for the next presented frame so overlays match decoded pixels
 * (seeked can fire before the compositor shows the target frame).
 */
function waitForVideoFrameComposite(timeoutMs = 1500) {
  if (typeof videoPlayer.requestVideoFrameCallback !== "function") {
    return new Promise((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    });
  }
  return new Promise((resolve) => {
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      resolve();
    };
    const tmr = setTimeout(finish, timeoutMs);
    try {
      videoPlayer.requestVideoFrameCallback(() => {
        clearTimeout(tmr);
        finish();
      });
    } catch (e) {
      clearTimeout(tmr);
      finish();
    }
  });
}

function loadVideoURL(path) {
  return new Promise((resolve, reject) => {
    mediaMode = "video";
    mediaImg = null;
    const src = `/api/video_file_by_path?video_path=${encodeURIComponent(path)}`;
    const done = () => {
      videoPlayer.classList.remove("hidden");
      canvas.classList.add("overlay");
      canvas.width = videoPlayer.videoWidth;
      canvas.height = videoPlayer.videoHeight;
      refineLoadedVideoFpsFromMediaDuration();
      applyZoom();
      draw();
      resolve();
    };
    if (currentVideoPath === path && videoPlayer.videoWidth > 0) {
      done();
      return;
    }
    currentVideoPath = path;
    videoPlayer.onloadedmetadata = done;
    videoPlayer.onerror = () => reject(new Error("Failed to load video."));
    videoPlayer.src = src;
    videoPlayer.load();
  });
}

function renderCsvPreview(rows) {
  if (!rows || !rows.length) {
    csvPreviewEl.textContent = "No rows.";
    return;
  }
  const table = document.createElement("table");
  table.className = "csv-table";
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    r.forEach((c) => {
      const td = document.createElement("td");
      td.textContent = c;
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  csvPreviewEl.innerHTML = "";
  csvPreviewEl.appendChild(table);
}

function parseCsvDetections(rows) {
  const out = [];
  for (const row of rows || []) {
    // Expected: frame class x y w h [conf]
    if (!row || row.length < 6) continue;
    const nums = row.map((x) => Number(x));
    if (nums.slice(0, 6).some((n) => Number.isNaN(n))) continue;
    out.push({
      frame: Math.round(nums[0]),
      cls: Math.round(nums[1]),
      xc: nums[2],
      yc: nums[3],
      w: nums[4],
      h: nums[5],
      conf: Number.isFinite(nums[6]) ? nums[6] : undefined,
    });
  }
  return out;
}

function renderCsvFullStats(counts) {
  csvFullStatsEl.innerHTML = "";
  const keys = Object.keys(counts || {}).map(Number).sort((a, b) => a - b);
  if (!keys.length) {
    csvFullStatsEl.textContent = "No detections indexed.";
    return;
  }
  for (const cls of keys) {
    const row = document.createElement("div");
    row.className = "stat-row";
    const left = document.createElement("span");
    left.textContent = `Class ${cls}`;
    left.style.color = colorForClass(cls);
    const right = document.createElement("span");
    right.textContent = String(counts[cls]);
    row.appendChild(left);
    row.appendChild(right);
    csvFullStatsEl.appendChild(row);
  }
}

function currentFrameInputValue() {
  return Math.max(1, parseInt(videoPathFrameInput.value || frameSlider.value || "1", 10));
}

function clampFrameToView(frameNo) {
  const lo = Math.max(1, frameViewMin || 1);
  const hiBase = effectiveTimelineFrameCount();
  const hi = Math.max(lo, Math.min(hiBase, frameViewMax || hiBase));
  return Math.max(lo, Math.min(frameNo, hi));
}

function applyFrameViewLimits() {
  frameViewMin = Math.max(1, frameViewMin || 1);
  const hiBase = effectiveTimelineFrameCount();
  frameViewMax = Math.max(frameViewMin, Math.min(frameViewMax || hiBase, hiBase));
  frameSlider.min = String(frameViewMin);
  frameSlider.max = String(frameViewMax);
  const cur = clampFrameToView(currentFrameInputValue());
  videoPathFrameInput.value = String(cur);
  frameSlider.value = String(cur);
}

async function loadVideoInfo(path) {
  const r = await q(`/api/video_info_by_path?video_path=${encodeURIComponent(path)}`);
  if (r.error) {
    setCsvStatus(r.error);
    return false;
  }
  loadedVideoFrameCount = Math.max(1, r.frame_count || 1);
  loadedVideoFrameCountTimeline = null;
  loadedVideoFpsReported = (r.fps && r.fps > 0) ? r.fps : 0;
  loadedVideoFps = loadedVideoFpsReported > 0 ? loadedVideoFpsReported : 10;
  if (!currentJsonPath) {
    frameViewMin = 1;
    frameViewMax = loadedVideoFrameCount;
  }
  applyFrameViewLimits();
  videoInfoText.textContent =
    `Frames: ${loadedVideoFrameCount}, container FPS: ${(r.fps || 0).toFixed(2)} (refined when video loads)`;
  return true;
}

async function applyCsvBoxesForFrame(frameNo, gen) {
  if (gen != null && gen !== overlayLoadGeneration) return;
  if (!currentCsvPath) {
    boxes = [];
  } else {
    const r = await q(`/api/csv_frame_boxes?path=${encodeURIComponent(currentCsvPath)}&frame=${frameNo}`);
    if (gen != null && gen !== overlayLoadGeneration) return;
    if (r.error) {
      setCsvStatus(r.error);
      boxes = [];
      selected = -1;
      renderClassStats();
      renderBoxList();
      draw();
      return;
    }
    boxes = (r.boxes || []).map((d) => ({ cls: d.cls, xc: d.xc, yc: d.yc, w: d.w, h: d.h, conf: d.conf }));
  }
  selected = -1;
  renderClassStats();
  renderBoxList();
  draw();
}

async function applyTrackingForFrame(frameNo, gen) {
  if (gen != null && gen !== overlayLoadGeneration) return;
  if (!currentJsonPath) {
    trackingCurrent = {};
    trackingHistory = [];
    draw();
    return;
  }
  const history = (showTrackHistory && showTrackHistory.checked)
    ? Math.max(1, Math.min(500, parseInt(trackHistoryLen.value || "30", 10)))
    : 0;
  const r = await q(`/api/tracking_frame?path=${encodeURIComponent(currentJsonPath)}&frame=${frameNo}&history=${history}`);
  if (gen != null && gen !== overlayLoadGeneration) return;
  if (r.error) {
    setCsvStatus(r.error);
    trackingCurrent = {};
    trackingHistory = [];
    draw();
    return;
  }
  trackingCurrent = r.flies || {};
  trackingHistory = r.history || [];
  const n = Object.keys(trackingCurrent).length;
  if (trackingInfoText) {
    const range = currentJsonPath ? `frames ${trackingFrameMin}-${trackingFrameMax}` : "no JSON";
    trackingInfoText.textContent = n
      ? `${currentJsonPath}: frame ${frameNo}, ${n} tracked flies (${range})`
      : `${currentJsonPath}: no tracking for frame ${frameNo} (${range})`;
  }
  if (trackingPathText && currentJsonPath) {
    trackingPathText.textContent = currentJsonPath;
  }
  draw();
}

/** During play, align CSV/JSON to the frame the browser actually presented (RVFC mediaTime). */
async function playbackSyncOverlays(optMediaTime) {
  const tSrc = playbackTimeForOverlay(optMediaTime);
  const frame = videoTimeToFrameIndex(tSrc);
  const prev = parseInt(videoPathFrameInput.value || "1", 10);
  if (frame !== prev) {
    const gen = ++overlayLoadGeneration;
    videoPathFrameInput.value = String(frame);
    frameSlider.value = String(frame);
    await applyCsvBoxesForFrame(frame, gen);
    await applyTrackingForFrame(frame, gen);
    if (gen !== overlayLoadGeneration) return;
    setCsvStatus(`Frame ${frame}: ${boxes.length} boxes, ${Object.keys(trackingCurrent || {}).length} tracked flies`);
  }
}

async function openCsvVideoFrame(frameNo) {
  const vp = (videoPathInput.value || "").trim();
  if (!vp) {
    setCsvStatus("Set video path first.");
    return;
  }
  const gen = ++overlayLoadGeneration;
  const f = clampFrameToView(frameNo);
  videoPathFrameInput.value = String(f);
  frameSlider.value = String(f);
  await loadVideoURL(vp);
  if (gen !== overlayLoadGeneration) return;
  const t = frameIndexToVideoTime(f);
  await waitForVideoSeeked(t);
  if (gen !== overlayLoadGeneration) return;
  await waitForVideoFrameComposite();
  if (gen !== overlayLoadGeneration) return;
  await applyCsvBoxesForFrame(f, gen);
  await applyTrackingForFrame(f, gen);
  if (gen !== overlayLoadGeneration) return;
  setCsvStatus(`Frame ${f}: ${boxes.length} boxes, ${Object.keys(trackingCurrent || {}).length} tracked flies`);
}

function stopPlayback() {
  playbackActive = false;
  if (playTimer) cancelAnimationFrame(playTimer);
  playTimer = null;
  if (playbackVfcHandle != null && typeof videoPlayer.cancelVideoFrameCallback === "function") {
    try {
      videoPlayer.cancelVideoFrameCallback(playbackVfcHandle);
    } catch (e) {
      /* ignore */
    }
    playbackVfcHandle = null;
  }
  videoPlayer.pause();
  playPauseBtn.textContent = "Play";
}

/** RAF fallback when `requestVideoFrameCallback` is unavailable. */
function startPlaybackLegacyRaf() {
  const maxPlayFrame = clampFrameToView(loadedVideoFrameCount || 1);
  const tick = async () => {
    const frame = videoTimeToFrameIndex(playbackTimeForOverlay(undefined));
    if (frame !== parseInt(videoPathFrameInput.value || "1", 10)) {
      const gen = ++overlayLoadGeneration;
      videoPathFrameInput.value = String(frame);
      frameSlider.value = String(frame);
      await applyCsvBoxesForFrame(frame, gen);
      await applyTrackingForFrame(frame, gen);
      if (gen !== overlayLoadGeneration) return;
      setCsvStatus(`Frame ${frame}: ${boxes.length} boxes, ${Object.keys(trackingCurrent || {}).length} tracked flies`);
    }
    if (videoPlayer.paused || frame >= maxPlayFrame) {
      stopPlayback();
      return;
    }
    playTimer = requestAnimationFrame(tick);
  };
  playTimer = requestAnimationFrame(tick);
}

function startPlayback() {
  stopPlayback();
  const maxPlayFrame = clampFrameToView(loadedVideoFrameCount || 1);
  playPauseBtn.textContent = "Pause";
  videoPlayer.play();
  playbackActive = true;
  if (typeof videoPlayer.requestVideoFrameCallback !== "function") {
    startPlaybackLegacyRaf();
    return;
  }
  const step = (_now, metadata) => {
    playbackVfcHandle = null;
    if (videoPlayer.paused || videoPlayer.ended) {
      stopPlayback();
      return;
    }
    let mt = null;
    if (
      metadata &&
      typeof metadata.mediaTime === "number" &&
      Number.isFinite(metadata.mediaTime) &&
      metadata.mediaTime >= 0
    ) {
      mt = metadata.mediaTime;
    }
    playbackSyncOverlays(mt)
      .catch((err) => {
        const msg = err && err.message ? err.message : String(err);
        setCsvStatus(`Playback overlay sync failed: ${msg}`);
      })
      .then(() => {
        if (videoPlayer.paused || videoPlayer.ended) {
          stopPlayback();
          return;
        }
        const curF = parseInt(videoPathFrameInput.value || "1", 10);
        if (curF >= maxPlayFrame) {
          stopPlayback();
          return;
        }
        try {
          playbackVfcHandle = videoPlayer.requestVideoFrameCallback(step);
        } catch (e) {
          startPlaybackLegacyRaf();
        }
      });
  };
  try {
    playbackVfcHandle = videoPlayer.requestVideoFrameCallback(step);
  } catch (e) {
    startPlaybackLegacyRaf();
  }
}

async function refreshRuns() {
  const r = await q("/api/runs");
  fillSelect(runSelect, r.runs || []);
  if (initialRunParam && (r.runs || []).includes(initialRunParam)) {
    runSelect.value = initialRunParam;
  }
  await refreshAssets();
}

async function refreshAssets() {
  const run = runSelect.value;
  if (!run) return;
  const r = await q(`/api/run_assets?run=${encodeURIComponent(run)}`);
  fillSelect(imageSelect, r.images || []);
  fillSelect(videoSelect, r.videos || []);
  if ((r.labels || []).length) labelPathInput.value = r.labels[0];
}

async function refreshCsvFiles() {
  const r = await q("/api/csv_files");
  fillSelect(csvSelect, r.csv_files || []);
}

async function refreshJsonFiles() {
  const r = await q("/api/json_files");
  fillSelect(jsonSelect, r.json_files || []);
}

openImageBtn.onclick = () => {
  const run = runSelect.value;
  const p = imageSelect.value;
  if (!run || !p) return;
  loadImageURL(`/api/media?run=${encodeURIComponent(run)}&path=${encodeURIComponent(p)}`);
  if (!labelPathInput.value) labelPathInput.value = `labels/${p.replace(/\.[^.]+$/, ".txt")}`;
  setStatus(`Opened image: ${p}`);
};

openFrameBtn.onclick = () => {
  const run = runSelect.value;
  const p = videoSelect.value;
  const f = Math.max(1, parseInt(frameInput.value || "1", 10));
  if (!run || !p) return;
  loadImageURL(`/api/video_frame?run=${encodeURIComponent(run)}&path=${encodeURIComponent(p)}&frame=${f}`);
  if (!labelPathInput.value) {
    const stem = p.replace(/\.[^.]+$/, "");
    labelPathInput.value = `labels/${stem}_${f}.txt`;
  }
  setStatus(`Opened frame ${f}: ${p}`);
};

openVideoPathFrameBtn.onclick = () => {
  const vp = (videoPathInput.value || "").trim();
  const f = Math.max(1, parseInt(videoPathFrameInput.value || "1", 10));
  if (!vp) {
    setCsvStatus("Set video path first.");
    return;
  }
  loadVideoInfo(vp).then((ok) => {
    if (!ok) return;
    openCsvVideoFrame(f);
  });
};

function applyLoadedLabel(content, exists, displayPath) {
  boxes = parseYolo(content || "");
  setClassUniverse(boxes.map((b) => b.cls));
  visibleClasses = new Set(classUniverse);
  syncRawFromBoxes();
  selected = -1;
  renderClassFilters();
  renderBoxList();
  renderClassStats();
  draw();
  setStatus(exists ? `Loaded label: ${displayPath}` : `Label not found (new file): ${displayPath}`);
}

loadLabelBtn.onclick = async () => {
  if (exploreSnapshotActive) {
    const lp = labelPathInput.value.trim();
    if (!lp) return;
    const resp = await fetch(`/api/label_abs?path=${encodeURIComponent(lp)}`);
    const r = await resp.json();
    if (r.error) {
      setStatus(r.error);
      return;
    }
    applyLoadedLabel(r.content || "", r.exists, lp);
    return;
  }
  const run = runSelect.value;
  const lp = labelPathInput.value.trim();
  if (!run || !lp) return;
  const r = await q(`/api/label?run=${encodeURIComponent(run)}&path=${encodeURIComponent(lp)}`);
  applyLoadedLabel(r.content || "", r.exists, lp);
};

saveLabelBtn.onclick = async () => {
  syncRawFromBoxes();
  if (exploreSnapshotActive) {
    const lp = labelPathInput.value.trim();
    if (!lp) return;
    const resp = await fetch("/api/label_abs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        path: lp,
        content: rawText.value + (rawText.value ? "\n" : ""),
      }),
    });
    const r = await resp.json();
    if (r.error) {
      setStatus(r.error);
      return;
    }
    setStatus(`Saved: ${r.saved}`);
    return;
  }
  const run = runSelect.value;
  const lp = labelPathInput.value.trim();
  if (!run || !lp) return;
  const resp = await fetch("/api/label", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run, path: lp, content: rawText.value + (rawText.value ? "\n" : "") }),
  });
  const r = await resp.json();
  setStatus(`Saved: ${r.saved}`);
};

applyRawBtn.onclick = () => {
  boxes = parseYolo(rawText.value);
  setClassUniverse(boxes.map((b) => b.cls));
  visibleClasses = new Set(classUniverse);
  selected = -1;
  renderClassFilters();
  renderBoxList();
  renderClassStats();
  draw();
};

async function loadCsvByPath(p) {
  if (!p) return false;
  // Keep table preview lightweight, but index detections from full csv.
  const preview = await q(`/api/csv_preview?path=${encodeURIComponent(p)}&limit=300`);
  if (preview.error) {
    setCsvStatus(preview.error);
    return false;
  }
  renderCsvPreview(preview.rows || []);

  const idx = await q(`/api/csv_index?path=${encodeURIComponent(p)}`);
  if (idx.error) {
    setCsvStatus(idx.error);
    return false;
  }
  currentCsvPath = p;
  csvFrameMin = idx.frame_min || 1;
  csvFrameMax = idx.frame_max || 1;
  setClassUniverse(idx.classes || []);
  visibleClasses = new Set(classUniverse);
  renderClassFilters();
  renderCsvFullStats(idx.class_counts || {});
  setCsvStatus(
    `Indexed CSV: ${idx.path} (${idx.count} detections, frames ${csvFrameMin}-${csvFrameMax}, preview 300 rows)`
  );
  return true;
}

loadCsvBtn.onclick = async () => {
  const p = csvSelect.value;
  if (!p) return;
  await loadCsvByPath(p);
};

async function loadJsonByPath(p) {
  if (!p) return false;
  const idx = await q(`/api/tracking_index?path=${encodeURIComponent(p)}`);
  if (idx.error) {
    setCsvStatus(idx.error);
    return false;
  }
  currentJsonPath = p;
  trackingFrameMin = idx.frame_min || 1;
  trackingFrameMax = idx.frame_max || 1;
  trackingInfoText.textContent = `${idx.path}: ${idx.count} frames (${trackingFrameMin}-${trackingFrameMax})`;
  if (trackingPathText) trackingPathText.textContent = idx.path;
  const currentFrame = currentFrameInputValue();
  const targetFrame = (currentFrame < trackingFrameMin || currentFrame > trackingFrameMax)
    ? trackingFrameMin
    : currentFrame;
  const vp = (videoPathInput.value || "").trim();
  videoPathFrameInput.value = String(targetFrame);
  frameSlider.value = String(targetFrame);
  if (vp) {
    const ok = await loadVideoInfo(vp);
    if (!ok) return false;
    // Full video timeline for scrubbing; tracking span is still shown in trackingInfoText.
    frameViewMin = 1;
    frameViewMax = Math.max(trackingFrameMax, loadedVideoFrameCount, csvFrameMax || 1);
    applyFrameViewLimits();
    await openCsvVideoFrame(targetFrame);
  } else {
    frameViewMin = trackingFrameMin;
    frameViewMax = trackingFrameMax;
    applyFrameViewLimits();
    await applyTrackingForFrame(targetFrame);
  }
  const jumped = targetFrame !== currentFrame ? `; jumped to first tracked frame ${targetFrame}` : "";
  setCsvStatus(`Loaded tracking JSON: ${idx.path}${jumped}`);
  return true;
}

loadJsonBtn.onclick = async () => {
  const p = jsonSelect.value;
  if (!p) return;
  await loadJsonByPath(p);
};

refreshJsonBtn.onclick = refreshJsonFiles;

for (const el of [showTracking, showTrackIds]) {
  if (el) el.onchange = draw;
}
if (showTrackHistory) {
  showTrackHistory.onchange = () => {
    const f = currentFrameInputValue();
    applyTrackingForFrame(f);
  };
}
if (trackHistoryLen) {
  trackHistoryLen.onchange = () => {
    const f = currentFrameInputValue();
    applyTrackingForFrame(f);
  };
}

frameSlider.addEventListener("input", () => {
  const f = parseInt(frameSlider.value || "1", 10);
  if (frameSliderDebounceTimer) clearTimeout(frameSliderDebounceTimer);
  frameSliderDebounceTimer = setTimeout(() => {
    frameSliderDebounceTimer = null;
    openCsvVideoFrame(f);
  }, FRAME_SLIDER_DEBOUNCE_MS);
});
frameSlider.addEventListener("change", () => {
  if (frameSliderDebounceTimer) {
    clearTimeout(frameSliderDebounceTimer);
    frameSliderDebounceTimer = null;
  }
  const f = parseInt(frameSlider.value || "1", 10);
  openCsvVideoFrame(f);
});
prevFrameBtn.onclick = () => {
  const f = clampFrameToView(parseInt(videoPathFrameInput.value || "1", 10) - 1);
  openCsvVideoFrame(f);
};
nextFrameBtn.onclick = () => {
  const f = clampFrameToView(parseInt(videoPathFrameInput.value || "1", 10) + 1);
  openCsvVideoFrame(f);
};
playPauseBtn.onclick = () => {
  if (playbackActive) stopPlayback();
  else startPlayback();
};

runSelect.onchange = refreshAssets;
refreshRunsBtn.onclick = refreshRuns;

canvas.onmousedown = (e) => {
  if (!hasMedia()) return;
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) * (canvas.width / r.width);
  const y = (e.clientY - r.top) * (canvas.height / r.height);
  if (addMode || e.shiftKey) {
    dragStart = { x, y };
    return;
  }
  selected = hit(x, y);
  renderBoxList();
  draw();
};

canvas.onmousemove = (e) => {
  if (!dragStart || !hasMedia()) return;
  draw();
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) * (canvas.width / r.width);
  const y = (e.clientY - r.top) * (canvas.height / r.height);
  const x1 = Math.min(dragStart.x, x), y1 = Math.min(dragStart.y, y);
  const w = Math.abs(x - dragStart.x), h = Math.abs(y - dragStart.y);
  ctx.strokeStyle = "#55ff55";
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, w, h);
};

canvas.onmouseup = (e) => {
  if (!dragStart || !hasMedia()) return;
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) * (canvas.width / r.width);
  const y = (e.clientY - r.top) * (canvas.height / r.height);
  const x1 = Math.min(dragStart.x, x), y1 = Math.min(dragStart.y, y);
  const x2 = Math.max(dragStart.x, x), y2 = Math.max(dragStart.y, y);
  dragStart = null;
  if (x2 - x1 < 3 || y2 - y1 < 3) return;
  const addCls = Math.max(0, Math.round(parseFloat(addClassInput.value || "0")));
  if (addClassInputCsv) addClassInputCsv.value = String(addCls);
  boxes.push({
    cls: addCls,
    xc: ((x1 + x2) / 2) / canvas.width,
    yc: ((y1 + y2) / 2) / canvas.height,
    w: (x2 - x1) / canvas.width,
    h: (y2 - y1) / canvas.height,
    conf: 1.0,
  });
  // Keep class visibility status unchanged on add.
  syncRawFromBoxes();
  renderBoxList();
  renderClassStats();
  draw();
};

toggleAddModeBtn.onclick = () => {
  addMode = !addMode;
  updateAddModeUI();
  setStatus(addMode ? "Add mode ON: drag on image to add box." : "Add mode OFF.");
};
if (toggleAddModeCsvBtn) {
  toggleAddModeCsvBtn.onclick = () => {
    addMode = !addMode;
    updateAddModeUI();
    setCsvStatus(addMode ? "Add mode ON: drag on image to add box." : "Add mode OFF.");
  };
}
if (addClassInputCsv) {
  addClassInputCsv.oninput = () => {
    addClassInput.value = addClassInputCsv.value || "0";
  };
}
addClassInput.oninput = () => {
  if (addClassInputCsv) addClassInputCsv.value = addClassInput.value || "0";
};

window.onkeydown = (e) => {
  if ((e.key === "Delete" || e.key === "Backspace") && selected >= 0) {
    if (document.activeElement === rawText || document.activeElement === labelPathInput) return;
    boxes.splice(selected, 1);
    selected = -1;
    syncRawFromBoxes();
    renderBoxList();
    renderClassStats();
    draw();
  }
};

zoomRange.oninput = () => {
  zoomScale = Math.max(0.1, Math.min(4, parseInt(zoomRange.value || "100", 10) / 100));
  applyZoom();
};
zoomInBtn.onclick = () => {
  zoomRange.value = String(Math.min(400, parseInt(zoomRange.value || "100", 10) + 10));
  zoomRange.oninput();
};
zoomOutBtn.onclick = () => {
  zoomRange.value = String(Math.max(10, parseInt(zoomRange.value || "100", 10) - 10));
  zoomRange.oninput();
};

canvas.addEventListener("wheel", (e) => {
  if (!e.ctrlKey) return;
  e.preventDefault();
  const delta = e.deltaY < 0 ? 0.05 : -0.05;
  zoomScale = Math.max(0.1, Math.min(4, zoomScale + delta));
  applyZoom();
}, { passive: false });

tabRuns.onclick = () => setTab("runs");
tabCsv.onclick = () => setTab("csv");
refreshCsvBtn.onclick = refreshCsvFiles;

async function loadSnapshotExplore(dir) {
  exploreSnapshotActive = false;
  const resp = await fetch(`/api/snapshot_explore/manifest?dir=${encodeURIComponent(dir)}`);
  const r = await resp.json();
  if (!resp.ok || r.error) {
    setStatus(r.error || `Could not open snapshot folder (${resp.status}).`);
    return;
  }
  if (!r.raw_image_abs) {
    setStatus("No *_snapshot_raw.png in this folder (expected detect_2 snapshot output).");
    return;
  }
  exploreSnapshotActive = true;
  labelPathInput.value = r.label_abs_path || "";
  loadImageURL(`/api/local_file?path=${encodeURIComponent(r.raw_image_abs)}`);
  setTab("runs");
  const lp = r.label_abs_path;
  if (lp) {
    const lrResp = await fetch(`/api/label_abs?path=${encodeURIComponent(lp)}`);
    const lr = await lrResp.json();
    if (lr.error) {
      setStatus(lr.error);
      exploreSnapshotActive = false;
      return;
    }
    applyLoadedLabel(lr.content || "", lr.exists, lp);
  } else {
    boxes = [];
    syncRawFromBoxes();
    renderClassFilters();
    renderBoxList();
    renderClassStats();
    draw();
    setStatus("Snapshot folder opened; no matching label path derived from raw PNG name.");
  }
}

async function loadTrackingExplore(dir, videoPathHint = "") {
  showVideoViewLoad("Resolving tracking result folder…");
  try {
    const qsp = new URLSearchParams({ dir });
    if (videoPathHint) qsp.set("video_path", videoPathHint);
    const resp = await fetch(`/api/tracking_explore/manifest?${qsp.toString()}`);
    const r = await resp.json();
    if (!resp.ok || r.error) {
      setCsvStatus(r.error || `Could not open tracking folder (${resp.status}).`);
      return;
    }
    if (!r.video_path || !r.csv_abs_path || !r.json_abs_path) {
      setCsvStatus("Tracking folder is missing required video/csv/json assets.");
      return;
    }

    const hint = (videoPathHint || "").trim();
    const apiVideo = (r.video_path || "").trim();
    const playVideo = hint || apiVideo;
    if (!playVideo) {
      setCsvStatus("Tracking folder resolved no video path.");
      return;
    }
    let mismatchNote = "";
    if (hint && apiVideo && hint !== apiVideo) {
      mismatchNote = " (using video_path from link, not manifest default)";
    }

    setTab("csv");
    if (videoViewInfoText) {
      videoViewInfoText.textContent = `Source: ${r.save_dir}`;
    }
    videoPathInput.value = playVideo;
    if (trackingPathText) trackingPathText.textContent = r.json_abs_path;

    updateVideoViewLoad("Loading video metadata…");
    currentJsonPath = "";
    const okVideo = await loadVideoInfo(playVideo);
    if (!okVideo) return;

    const firstFrame = Math.max(1, Number(r.frame_min || 1));
    videoPathFrameInput.value = String(firstFrame);
    frameSlider.value = String(firstFrame);

    updateVideoViewLoad("Loading detection CSV…");
    const okCsv = await loadCsvByPath(r.csv_abs_path);
    if (!okCsv) return;

    updateVideoViewLoad("Loading tracking JSON…");
    const okJson = await loadJsonByPath(r.json_abs_path);
    if (!okJson) return;

    setCsvStatus(`Video View ready: ${playVideo}${mismatchNote}`);
  } catch (e) {
    setCsvStatus(e.message || "Failed to load Video View.");
  } finally {
    hideVideoViewLoad();
  }
}

async function boot() {
  await refreshRuns();
  refreshCsvFiles();
  refreshJsonFiles();
  updateAddModeUI();
  renderClassStats();
  setTab("runs");
  if (initialSnapshotDir) {
    await loadSnapshotExplore(initialSnapshotDir);
  } else if (initialTrackingDir) {
    await loadTrackingExplore(initialTrackingDir, initialTrackingVideoPath);
  }
}

videoPlayer.addEventListener("durationchange", () => {
  if (mediaMode !== "video") return;
  if ((loadedVideoFrameCount || 0) < 2) return;
  refineLoadedVideoFpsFromMediaDuration();
});

boot();
