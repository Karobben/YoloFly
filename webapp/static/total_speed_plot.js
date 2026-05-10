const qs = new URLSearchParams(window.location.search);
const filePath = (qs.get("path") || "").trim();
const returnVideoPath = (qs.get("video_path") || "").trim();
const returnProject = (qs.get("project") || "").trim();
const returnClipId = (qs.get("clip_id") || "").trim();

const tspPath = document.getElementById("tspPath");
const tspMeta = document.getElementById("tspMeta");
const tspError = document.getElementById("tspError");
const tspPlot = document.getElementById("tspPlot");
const tspBackTable = document.getElementById("tspBackTable");
const tspBackResults = document.getElementById("tspBackResults");
const tspClipToolbar = document.getElementById("tspClipToolbar");
const tspFps = document.getElementById("tspFps");
const tspWinMinutes = document.getElementById("tspWinMinutes");
const tspAddClip = document.getElementById("tspAddClip");
const tspPreviewPanel = document.getElementById("tspPreviewPanel");
const tspClipSlider = document.getElementById("tspClipSlider");
const tspClipSliderDetail = document.getElementById("tspClipSliderDetail");
const tspSaveClip = document.getElementById("tspSaveClip");
const tspCancelPreview = document.getElementById("tspCancelPreview");
const tspClipsSection = document.getElementById("tspClipsSection");
const tspClipsList = document.getElementById("tspClipsList");

/** @type {{ frameMin: number, frameMax: number } | null} */
let gBounds = null;
/** @type {{ start: number, end: number, windowFrames: number } | null} */
let previewClip = null;
/** @type {{ id: number, start: number, end: number, colorIdx: number, name?: string }[]} */
let savedClips = [];
/** When set, that clip row shows the edit form */
let editingClipId = null;
/** Live band position on the plot while editing a clip (slider); cleared on Apply/Cancel */
let editPreview = null;

const CLIP_PALETTE = [
  { fill: "rgba(90, 170, 255, 0.22)", line: "#5aaaff" },
  { fill: "rgba(255, 140, 90, 0.22)", line: "#ff8c5a" },
  { fill: "rgba(160, 230, 120, 0.22)", line: "#a0e678" },
  { fill: "rgba(220, 120, 255, 0.2)", line: "#dc78ff" },
  { fill: "rgba(120, 220, 210, 0.22)", line: "#78dcd2" },
];

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function buildBackTableHref() {
  const q = new URLSearchParams({ path: filePath });
  if (returnVideoPath) q.set("video_path", returnVideoPath);
  if (returnProject) q.set("project", returnProject);
  if (returnClipId) q.set("clip_id", returnClipId);
  return `/csv-table?${q.toString()}`;
}

function buildBackResultsHref() {
  if (!returnVideoPath) return "";
  const q = new URLSearchParams({ video_path: returnVideoPath });
  if (returnProject) q.set("project", returnProject);
  if (returnClipId) q.set("clip_id", returnClipId);
  return `/video-results?${q.toString()}`;
}

function normalizeHeader(h) {
  return String(h || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_");
}

function pickSpeedColumns(headers) {
  const norm = headers.map(normalizeHeader);
  let frameIdx = norm.findIndex((h) => h === "frame" || h === "frames");
  if (frameIdx < 0) frameIdx = 0;

  const smoothIdx = norm.findIndex(
    (h) => h === "speed_smooth" || h === "speedsmooth",
  );
  const rawIdx = norm.findIndex((h) => h === "speed");
  return { frameIdx, rawIdx, smoothIdx, norm };
}

function parseNumericRows(headers, rows, cols) {
  const { frameIdx, rawIdx, smoothIdx } = cols;
  const xf = [];
  const yr = [];
  const ys = [];
  const need = Math.max(
    frameIdx,
    rawIdx >= 0 ? rawIdx : -1,
    smoothIdx >= 0 ? smoothIdx : -1,
    0,
  );
  for (const row of rows) {
    if (!row || row.length <= need) continue;
    const fx = parseFloat(String(row[frameIdx]).replace(",", "."));
    if (!Number.isFinite(fx)) continue;
    xf.push(fx);
    if (rawIdx >= 0) {
      const v = parseFloat(String(row[rawIdx]).replace(",", "."));
      yr.push(Number.isFinite(v) ? v : null);
    }
    if (smoothIdx >= 0) {
      const v = parseFloat(String(row[smoothIdx]).replace(",", "."));
      ys.push(Number.isFinite(v) ? v : null);
    }
  }
  return { xf, yr, ys };
}

function plotData(headers, rows) {
  const cols = pickSpeedColumns(headers);
  const labels = {
    frame: headers[cols.frameIdx] || "frame",
    raw: cols.rawIdx >= 0 ? headers[cols.rawIdx] || "speed" : null,
    smooth:
      cols.smoothIdx >= 0
        ? headers[cols.smoothIdx] || "speed_smooth"
        : null,
  };

  const traces = [];
  const { xf, yr, ys } = parseNumericRows(headers, rows, cols);

  if (!xf.length) {
    throw new Error("No numeric rows could be parsed for plotting.");
  }

  if (cols.rawIdx >= 0 && yr.length === xf.length) {
    traces.push({
      type: "scatter",
      mode: "lines",
      name: labels.raw || "speed",
      x: xf,
      y: yr,
      line: { color: "#7eb6ff", width: 1 },
      hovertemplate: "%{x:.0f}<br>%{y:.4f}<extra></extra>",
    });
  }
  if (cols.smoothIdx >= 0 && ys.length === xf.length) {
    traces.push({
      type: "scatter",
      mode: "lines",
      name: labels.smooth || "speed_smooth",
      x: xf,
      y: ys,
      line: { color: "#54d169", width: 2 },
      hovertemplate: "%{x:.0f}<br>%{y:.4f}<extra></extra>",
    });
  }

  if (!traces.length) {
    let yi = -1;
    for (let i = 0; i < headers.length; i++) {
      if (i !== cols.frameIdx) {
        yi = i;
        break;
      }
    }
    if (yi < 0) {
      throw new Error("Need at least two columns (frame + values) to plot.");
    }
    const xf2 = [];
    const yv = [];
    for (const row of rows) {
      if (!row || row.length <= Math.max(cols.frameIdx, yi)) continue;
      const fx = parseFloat(String(row[cols.frameIdx]).replace(",", "."));
      const v = parseFloat(String(row[yi]).replace(",", "."));
      if (!Number.isFinite(fx) || !Number.isFinite(v)) continue;
      xf2.push(fx);
      yv.push(v);
    }
    if (!xf2.length) {
      throw new Error("No numeric rows for fallback plot.");
    }
    traces.push({
      type: "scatter",
      mode: "lines",
      name: headers[yi] || "value",
      x: xf2,
      y: yv,
      line: { color: "#54d169", width: 2 },
      hovertemplate: "%{x:.0f}<br>%{y:.4f}<extra></extra>",
    });
  }

  return { traces, labels };
}

function readFps() {
  const v = parseFloat(String(tspFps.value));
  return Number.isFinite(v) && v > 0 ? v : 30;
}

function readWindowMinutes() {
  const v = parseFloat(String(tspWinMinutes.value));
  return Number.isFinite(v) && v > 0 ? v : 10;
}

function computeWindowFrames(fps, windowMinutes) {
  return Math.max(1, Math.round(fps * 60 * windowMinutes));
}

function sliderStepForSpan(spanFrames) {
  return Math.max(1, Math.floor(spanFrames / 1200));
}

function fmtFrames(a, b) {
  return `${Math.round(a)}–${Math.round(b)}`;
}

function fmtTimeRange(startF, endF, fps) {
  const s = (endF - startF) / fps;
  if (s >= 3600) {
    return `${(s / 3600).toFixed(2)} h span`;
  }
  if (s >= 60) {
    return `${(s / 60).toFixed(2)} min span`;
  }
  return `${s.toFixed(2)} s span`;
}

function fmtClock(startF, endF, fps) {
  const t0 = startF / fps;
  const t1 = endF / fps;
  const fmt = (t) => {
    if (t >= 3600) return `${Math.floor(t / 3600)}h ${Math.floor((t % 3600) / 60)}m ${(t % 60).toFixed(1)}s`;
    if (t >= 60) return `${Math.floor(t / 60)}m ${(t % 60).toFixed(1)}s`;
    return `${t.toFixed(2)}s`;
  };
  return `${fmt(t0)} → ${fmt(t1)}`;
}

function rectBand(x0, x1, fill, line, dash) {
  return {
    type: "rect",
    xref: "x",
    yref: "paper",
    x0,
    x1,
    y0: 0,
    y1: 1,
    fillcolor: fill,
    line: { color: line, width: 2, dash: dash || "solid" },
    layer: "below",
  };
}

function buildAllShapes() {
  const shapes = [];
  for (const c of savedClips) {
    const pal = CLIP_PALETTE[c.colorIdx % CLIP_PALETTE.length];
    let x0 = c.start;
    let x1 = c.end;
    if (editPreview && editPreview.clipId === c.id) {
      x0 = editPreview.start;
      x1 = editPreview.end;
    }
    shapes.push(rectBand(x0, x1, pal.fill, pal.line, "solid"));
  }
  if (previewClip) {
    shapes.push(
      rectBand(
        previewClip.start,
        previewClip.end,
        "rgba(255, 230, 120, 0.18)",
        "#ffd966",
        "dash",
      ),
    );
  }
  return shapes;
}

function relayoutShapes() {
  if (typeof Plotly === "undefined") return;
  try {
    if (!tspPlot || !tspPlot.data || !tspPlot.data.length) return;
  } catch {
    return;
  }
  Plotly.relayout(tspPlot, { shapes: buildAllShapes() });
}

function displayClipName(c, indexZeroBased) {
  if (c.name != null && String(c.name).trim() !== "") {
    return String(c.name).trim();
  }
  return `Clip ${indexZeroBased + 1}`;
}

function clampClipEndpoints(start, end, frameMin, frameMax) {
  let a = Number(start);
  let b = Number(end);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  a = Math.min(Math.max(a, frameMin), frameMax);
  b = Math.min(Math.max(b, frameMin), frameMax);
  if (b <= a) {
    b = Math.min(frameMax, a + 1);
  }
  if (b <= a) {
    a = Math.max(frameMin, b - 1);
  }
  return { start: a, end: b };
}

/** Move a clip window along the axis; window length in frames is preserved (capped by data span). */
function clampClipWindowStart(rawStart, windowLen, frameMin, frameMax) {
  const span = frameMax - frameMin;
  const w = Math.min(Math.max(1, Number(windowLen) || 1), span);
  const maxStart = frameMin + span - w;
  let s = Number(rawStart);
  if (!Number.isFinite(s)) s = frameMin;
  s = Math.min(Math.max(frameMin, s), maxStart);
  return { start: s, end: s + w };
}

function clampPreviewStart(start, windowFrames, frameMin, frameMax) {
  const span = frameMax - frameMin;
  const w = Math.min(windowFrames, span);
  const maxStart = frameMin + span - w;
  return Math.min(Math.max(frameMin, start), maxStart);
}

function beginPreview() {
  if (!gBounds) return;
  editingClipId = null;
  editPreview = null;
  const fps = readFps();
  const wm = readWindowMinutes();
  const want = computeWindowFrames(fps, wm);
  const { frameMin, frameMax } = gBounds;
  const span = frameMax - frameMin;
  if (span <= 0) {
    tspError.textContent = "Cannot clip: invalid frame range in data.";
    tspError.classList.remove("hidden");
    return;
  }
  const windowFrames = Math.min(want, span);
  const start = frameMin;
  previewClip = {
    start,
    end: start + windowFrames,
    windowFrames,
  };
  tspPreviewPanel.classList.remove("hidden");
  tspError.classList.add("hidden");

  const step = sliderStepForSpan(span);
  tspClipSlider.min = String(frameMin);
  tspClipSlider.max = String(frameMin + span - windowFrames);
  tspClipSlider.step = String(step);
  tspClipSlider.value = String(previewClip.start);

  updateSliderDetail();
  relayoutShapes();
}

function updatePreviewFromSlider() {
  if (!previewClip || !gBounds) return;
  const start = parseFloat(tspClipSlider.value);
  if (!Number.isFinite(start)) return;
  const { frameMin, frameMax } = gBounds;
  previewClip.start = clampPreviewStart(
    start,
    previewClip.windowFrames,
    frameMin,
    frameMax,
  );
  previewClip.end = previewClip.start + previewClip.windowFrames;
  updateSliderDetail();
  relayoutShapes();
}

function updateSliderDetail() {
  if (!previewClip) return;
  const fps = readFps();
  const { start, end } = previewClip;
  tspClipSliderDetail.innerHTML = `
    <strong>Frames:</strong> ${esc(fmtFrames(start, end))}
    · <strong>Time:</strong> ${esc(fmtClock(start, end, fps))}
    · ${esc(fmtTimeRange(start, end, fps))}
  `;
}

function cancelPreview() {
  previewClip = null;
  tspPreviewPanel.classList.add("hidden");
  relayoutShapes();
}

async function fetchClipsFromApi() {
  const r = await fetch(
    `/api/total_speed_clips?path=${encodeURIComponent(filePath)}`,
  );
  const d = await r.json();
  if (!r.ok || d.error) {
    throw new Error(d.error || `HTTP ${r.status}`);
  }
  savedClips = Array.isArray(d.clips) ? d.clips : [];
}

async function saveCurrentClip() {
  if (!previewClip || !gBounds) return;
  const fps = readFps();
  const ord = savedClips.length + 1;
  const name = `Clip ${ord}`;
  const body = {
    path: filePath,
    name,
    start: previewClip.start,
    end: previewClip.end,
    color_idx: savedClips.length % CLIP_PALETTE.length,
  };
  try {
    tspError.classList.add("hidden");
    const r = await fetch("/api/total_speed_clips", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
    await fetchClipsFromApi();
    previewClip = null;
    tspPreviewPanel.classList.add("hidden");
    tspClipSlider.value = "0";
    relayoutShapes();
    renderClipsList(fps);
  } catch (e) {
    tspError.textContent = e.message || String(e);
    tspError.classList.remove("hidden");
  }
}

async function deleteClip(id) {
  try {
    tspError.classList.add("hidden");
    const r = await fetch(
      `/api/total_speed_clips/${encodeURIComponent(id)}?path=${encodeURIComponent(filePath)}`,
      { method: "DELETE" },
    );
    const d = await r.json();
    if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
    await fetchClipsFromApi();
    if (editingClipId === id) editingClipId = null;
    if (editPreview && editPreview.clipId === id) editPreview = null;
    relayoutShapes();
    renderClipsList(readFps());
  } catch (e) {
    tspError.textContent = e.message || String(e);
    tspError.classList.remove("hidden");
  }
}

async function applyClipEdit(clipId, nameRaw, startRaw, endRaw) {
  const clip = savedClips.find((x) => x.id === clipId);
  if (!clip || !gBounds) return;
  const { frameMin, frameMax } = gBounds;
  const pair = clampClipEndpoints(startRaw, endRaw, frameMin, frameMax);
  if (!pair) {
    tspError.textContent = "Start and end frames must be valid numbers.";
    tspError.classList.remove("hidden");
    return;
  }
  let label = String(nameRaw ?? "").trim();
  if (!label) label = `Clip ${savedClips.indexOf(clip) + 1}`;
  if (label.length > 200) label = label.slice(0, 200);
  try {
    tspError.classList.add("hidden");
    const r = await fetch(`/api/total_speed_clips/${encodeURIComponent(clipId)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        path: filePath,
        name: label,
        start: pair.start,
        end: pair.end,
      }),
    });
    const d = await r.json();
    if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
    editPreview = null;
    await fetchClipsFromApi();
    editingClipId = null;
    relayoutShapes();
    renderClipsList(readFps());
  } catch (e) {
    tspError.textContent = e.message || String(e);
    tspError.classList.remove("hidden");
  }
}

function renderClipEditPanel(c, fps) {
  const panel = document.createElement("div");
  panel.className = "tsp-clip-edit-panel";

  const title = document.createElement("div");
  title.className = "tsp-clip-edit-title";
  title.textContent = "Edit clip";
  panel.appendChild(title);

  const nameLab = document.createElement("label");
  nameLab.className = "tsp-clip-edit-field";
  const nameLbl = document.createElement("span");
  nameLbl.className = "tsp-clip-edit-label";
  nameLbl.textContent = "Name";
  const nameIn = document.createElement("input");
  nameIn.type = "text";
  nameIn.className = "tsp-clip-edit-input";
  nameIn.maxLength = 200;
  nameIn.value = displayClipName(c, savedClips.indexOf(c));
  nameLab.appendChild(nameLbl);
  nameLab.appendChild(nameIn);
  panel.appendChild(nameLab);

  const rawWindowLen = Math.max(1, c.end - c.start);
  let effWindowLen = rawWindowLen;
  if (gBounds) {
    const span = gBounds.frameMax - gBounds.frameMin;
    effWindowLen = Math.min(rawWindowLen, Math.max(1, span));
  }

  const winInfo = document.createElement("p");
  winInfo.className = "tsp-clip-edit-hint qr-muted";
  winInfo.textContent = gBounds
    ? `Window length: ${Math.round(effWindowLen)} frames (${fmtTimeRange(0, effWindowLen, fps)} at ${fps} FPS). Slide to move along the plot — same as adding a new clip.`
    : "";
  panel.appendChild(winInfo);

  const sliderWrap = document.createElement("div");
  sliderWrap.className = "tsp-clip-edit-slider-wrap";

  const sliderLbl = document.createElement("label");
  sliderLbl.className = "tsp-slider-label";
  sliderLbl.setAttribute("for", `tsp-edit-clip-slider-${c.id}`);
  sliderLbl.textContent = "Slide to move clip window";

  const slider = document.createElement("input");
  slider.type = "range";
  slider.id = `tsp-edit-clip-slider-${c.id}`;
  slider.className = "tsp-clip-slider tsp-clip-edit-slider";

  if (gBounds) {
    const { frameMin, frameMax } = gBounds;
    const span = frameMax - frameMin;
    const step = sliderStepForSpan(span);
    const maxStart = frameMin + span - effWindowLen;
    slider.min = String(frameMin);
    slider.max = String(maxStart);
    slider.step = String(step);
    const initial = clampClipWindowStart(c.start, effWindowLen, frameMin, frameMax);
    slider.value = String(initial.start);
  } else {
    slider.disabled = true;
  }

  sliderWrap.appendChild(sliderLbl);
  sliderWrap.appendChild(slider);
  panel.appendChild(sliderWrap);

  if (gBounds) {
    const hint = document.createElement("p");
    hint.className = "tsp-clip-edit-hint qr-muted";
    hint.textContent = `Data spans frames ${gBounds.frameMin}–${gBounds.frameMax}.`;
    panel.appendChild(hint);
  }

  const detail = document.createElement("p");
  detail.className = "tsp-clip-edit-detail qr-muted";
  const updDetail = () => {
    if (!gBounds) {
      detail.textContent = "";
      editPreview = null;
      relayoutShapes();
      return;
    }
    const pos = clampClipWindowStart(
      slider.value,
      effWindowLen,
      gBounds.frameMin,
      gBounds.frameMax,
    );
    if (String(pos.start) !== slider.value) {
      slider.value = String(pos.start);
    }
    detail.innerHTML = `
      <strong>Frames:</strong> ${esc(fmtFrames(pos.start, pos.end))}
      · <strong>Time:</strong> ${esc(fmtClock(pos.start, pos.end, fps))}
      · ${esc(fmtTimeRange(pos.start, pos.end, fps))}
    `;
    editPreview = { clipId: c.id, start: pos.start, end: pos.end };
    relayoutShapes();
  };
  slider.addEventListener("input", updDetail);
  updDetail();
  panel.appendChild(detail);

  const actions = document.createElement("div");
  actions.className = "tsp-clip-edit-actions";

  const applyBtn = document.createElement("button");
  applyBtn.type = "button";
  applyBtn.className = "tsp-btn tsp-btn-save tsp-btn-small";
  applyBtn.textContent = "Apply";
  applyBtn.addEventListener("click", () => {
    if (!gBounds) return;
    const pos = clampClipWindowStart(
      slider.value,
      effWindowLen,
      gBounds.frameMin,
      gBounds.frameMax,
    );
    applyClipEdit(c.id, nameIn.value, pos.start, pos.end);
  });

  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "tsp-btn tsp-btn-ghost tsp-btn-small";
  cancelBtn.textContent = "Cancel";
  cancelBtn.addEventListener("click", () => {
    editPreview = null;
    editingClipId = null;
    tspError.classList.add("hidden");
    relayoutShapes();
    renderClipsList(readFps());
  });

  actions.appendChild(applyBtn);
  actions.appendChild(cancelBtn);
  panel.appendChild(actions);

  return panel;
}

function renderClipsList(fps) {
  tspClipsList.innerHTML = "";
  if (!savedClips.length) {
    const li = document.createElement("li");
    li.className = "tsp-clips-empty qr-muted";
    li.textContent =
      "No clips saved yet. Set FPS and window length, click Add Clip, slide to position, then Save clip.";
    tspClipsList.appendChild(li);
    return;
  }
  savedClips.forEach((c, idx) => {
    const li = document.createElement("li");
    li.className = "tsp-clip-item";
    const pal = CLIP_PALETTE[c.colorIdx % CLIP_PALETTE.length];
    li.style.borderLeftColor = pal.line;

    if (editingClipId === c.id) {
      li.appendChild(renderClipEditPanel(c, fps));
      tspClipsList.appendChild(li);
      return;
    }

    const main = document.createElement("div");
    main.className = "tsp-clip-item-main";

    const body = document.createElement("div");
    body.className = "tsp-clip-item-body";

    const nameRow = document.createElement("div");
    nameRow.className = "tsp-clip-item-name";
    const nameStrong = document.createElement("strong");
    nameStrong.textContent = displayClipName(c, idx);
    nameRow.appendChild(nameStrong);
    body.appendChild(nameRow);

    const stats = document.createElement("div");
    stats.className = "tsp-clip-item-stats";
    const rSpan = document.createElement("span");
    rSpan.className = "tsp-clip-item-range";
    const rStrong = document.createElement("strong");
    rStrong.textContent = fmtFrames(c.start, c.end);
    rSpan.appendChild(rStrong);
    const tSpan = document.createElement("span");
    tSpan.className = "tsp-clip-item-time";
    tSpan.textContent = fmtClock(c.start, c.end, fps);
    const mSpan = document.createElement("span");
    mSpan.className = "tsp-clip-item-meta";
    mSpan.textContent = fmtTimeRange(c.start, c.end, fps);
    stats.appendChild(rSpan);
    stats.appendChild(tSpan);
    stats.appendChild(mSpan);
    body.appendChild(stats);

    const actions = document.createElement("div");
    actions.className = "tsp-clip-item-actions";

    const editBtn = document.createElement("button");
    editBtn.type = "button";
    editBtn.className = "tsp-btn tsp-btn-small tsp-btn-edit";
    editBtn.textContent = "Edit";
    editBtn.addEventListener("click", () => {
      previewClip = null;
      tspPreviewPanel.classList.add("hidden");
      editingClipId = c.id;
      tspError.classList.add("hidden");
      renderClipsList(readFps());
    });

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "tsp-btn tsp-btn-danger tsp-btn-small";
    delBtn.textContent = "Delete";
    delBtn.addEventListener("click", () => deleteClip(c.id));

    actions.appendChild(editBtn);
    actions.appendChild(delBtn);

    main.appendChild(body);
    main.appendChild(actions);
    li.appendChild(main);
    tspClipsList.appendChild(li);
  });
}

function onFpsOrWindowChangeWhilePreview() {
  if (!previewClip || !gBounds) return;
  const fps = readFps();
  const wm = readWindowMinutes();
  const want = computeWindowFrames(fps, wm);
  const { frameMin, frameMax } = gBounds;
  const span = frameMax - frameMin;
  previewClip.windowFrames = Math.min(want, span);
  previewClip.start = clampPreviewStart(
    previewClip.start,
    previewClip.windowFrames,
    frameMin,
    frameMax,
  );
  previewClip.end = previewClip.start + previewClip.windowFrames;

  tspClipSlider.min = String(frameMin);
  tspClipSlider.max = String(frameMin + span - previewClip.windowFrames);
  tspClipSlider.step = String(sliderStepForSpan(span));
  tspClipSlider.value = String(previewClip.start);

  updateSliderDetail();
  relayoutShapes();
}

function initClipControls() {
  tspAddClip.addEventListener("click", () => beginPreview());
  tspClipSlider.addEventListener("input", () => updatePreviewFromSlider());
  tspSaveClip.addEventListener("click", () => {
    saveCurrentClip();
  });
  tspCancelPreview.addEventListener("click", () => cancelPreview());
  tspFps.addEventListener("input", () => {
    if (previewClip) updateSliderDetail();
    if (savedClips.length) renderClipsList(readFps());
  });
  tspFps.addEventListener("change", () => {
    if (previewClip) onFpsOrWindowChangeWhilePreview();
    if (savedClips.length) renderClipsList(readFps());
  });
  tspWinMinutes.addEventListener("change", () => {
    if (previewClip) onFpsOrWindowChangeWhilePreview();
  });
}

async function main() {
  tspBackTable.href = buildBackTableHref();
  const br = buildBackResultsHref();
  if (br) {
    tspBackResults.href = br;
    tspBackResults.classList.remove("hidden");
  }

  if (!filePath) {
    tspError.textContent = "Missing path query parameter.";
    tspError.classList.remove("hidden");
    return;
  }

  if (typeof Plotly === "undefined") {
    tspError.textContent =
      "Plot library failed to load. Check network access to the Plotly CDN.";
    tspError.classList.remove("hidden");
    return;
  }

  tspPath.innerHTML = `<code>${esc(filePath)}</code>`;

  try {
    const url = `/api/csv_table?path=${encodeURIComponent(filePath)}&max_rows=100000`;
    const r = await fetch(url);
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`);
    if (!data.ok) throw new Error("Unexpected response");

    const headers = data.headers || [];
    const rows = data.rows || [];
    if (!headers.length || !rows.length) {
      throw new Error("CSV has no data to plot.");
    }

    tspMeta.textContent = `${rows.length} row(s) · columns: ${headers.join(", ")}${data.truncated ? " · truncated" : ""}`;

    const colsOnce = pickSpeedColumns(headers);
    const { traces } = plotData(headers, rows);

    const xf = traces[0].x;
    const frameMin = Math.min(...xf);
    const frameMax = Math.max(...xf);
    gBounds = { frameMin, frameMax };

    const layout = {
      autosize: true,
      paper_bgcolor: "#111111",
      plot_bgcolor: "#151515",
      font: { color: "#ddd", size: 12 },
      title: {
        text: "Total moving speed (from CSV)",
        font: { size: 15, color: "#eee" },
      },
      xaxis: {
        title: headers[colsOnce.frameIdx] || "frame",
        gridcolor: "#2a2a2a",
        zerolinecolor: "#444",
      },
      yaxis: {
        title: "speed",
        gridcolor: "#2a2a2a",
        zerolinecolor: "#444",
      },
      legend: {
        orientation: "h",
        yanchor: "bottom",
        y: 1.02,
        xanchor: "right",
        x: 1,
        bgcolor: "rgba(0,0,0,0)",
      },
      margin: { l: 56, r: 24, t: 56, b: 48 },
      hovermode: "x unified",
      shapes: [],
    };

    const config = {
      responsive: true,
      scrollZoom: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
      displaylogo: false,
      toImageButtonOptions: {
        format: "png",
        filename: "total_speed_plot",
      },
    };

    await Plotly.newPlot(tspPlot, traces, layout, config);

    try {
      await fetchClipsFromApi();
    } catch (e) {
      savedClips = [];
      tspMeta.textContent += ` · clips: ${e.message || String(e)}`;
    }
    relayoutShapes();

    tspClipToolbar.classList.remove("hidden");
    tspClipsSection.classList.remove("hidden");
    renderClipsList(readFps());
    initClipControls();

    window.addEventListener(
      "resize",
      () => {
        Plotly.Plots.resize(tspPlot);
      },
      { passive: true },
    );
  } catch (e) {
    tspError.textContent = e.message || String(e);
    tspError.classList.remove("hidden");
  }
}

main();
