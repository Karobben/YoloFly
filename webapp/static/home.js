const projectNameInput = document.getElementById("projectNameInput");
const labInfoInput = document.getElementById("labInfoInput");
const createProjectBtn = document.getElementById("createProjectBtn");
const refreshProjectsBtn = document.getElementById("refreshProjectsBtn");
const homeStatus = document.getElementById("homeStatus");
const projectList = document.getElementById("projectList");
const detailTitle = document.getElementById("detailTitle");
const detailLabInfo = document.getElementById("detailLabInfo");
const saveLabInfoBtn = document.getElementById("saveLabInfoBtn");
const videoList = document.getElementById("videoList");
const videoListEmpty = document.getElementById("videoListEmpty");
const videoTableWrap = document.getElementById("videoTableWrap");
const videoDirInput = document.getElementById("videoDirInput");
const addDirBtn = document.getElementById("addDirBtn");
const videoPathsInput = document.getElementById("videoPathsInput");
const addPathsBtn = document.getElementById("addPathsBtn");
const videoModal = document.getElementById("videoModal");
const videoModalTitle = document.getElementById("videoModalTitle");
const videoModalStage = document.getElementById("videoModalStage");
const videoModalPlayer = document.getElementById("videoModalPlayer");
const videoMeasureCanvas = document.getElementById("videoMeasureCanvas");
const videoMeasureToggleBtn = document.getElementById("videoMeasureToggleBtn");
const videoMeasureClearBtn = document.getElementById("videoMeasureClearBtn");
const videoMeasureHint = document.getElementById("videoMeasureHint");
const closeVideoModalBtn = document.getElementById("closeVideoModalBtn");
const videoFullscreenBtn = document.getElementById("videoFullscreenBtn");
const videoSubclipSelectAll = document.getElementById("videoSubclipSelectAll");
const videoColHeader = document.getElementById("videoColHeader");
const videoFoldAllBtn = document.getElementById("videoFoldAllBtn");

const MEASURE_HIT_PX = 14;
/** Pauses at end of meta/subclip frame window (see openVideoModal). */
let videoPlaybackTimeHandler = null;

let videoMeasureMode = false;
let videoMeasurePoints = [];
let videoMeasureDragIdx = -1;

function getVideoContentMetrics() {
  const v = videoModalPlayer;
  const vw = v.videoWidth;
  const vh = v.videoHeight;
  const ew = v.clientWidth;
  const eh = v.clientHeight;
  if (!vw || !vh || !ew || !eh) return null;
  const scale = Math.min(ew / vw, eh / vh);
  const cw = vw * scale;
  const ch = vh * scale;
  const ox = (ew - cw) / 2;
  const oy = (eh - ch) / 2;
  return { vw, vh, scale, ox, oy, cw, ch, ew, eh };
}

function clientToNativeMeasure(clientX, clientY) {
  const v = videoModalPlayer;
  const r = v.getBoundingClientRect();
  const lx = clientX - r.left;
  const ly = clientY - r.top;
  const m = getVideoContentMetrics();
  if (!m) return null;
  const rx = lx - m.ox;
  const ry = ly - m.oy;
  if (rx < 0 || ry < 0 || rx > m.cw || ry > m.ch) return null;
  return { nx: rx / m.scale, ny: ry / m.scale };
}

function nativeToLocalVideo(nx, ny) {
  const m = getVideoContentMetrics();
  if (!m) return null;
  return { x: m.ox + nx * m.scale, y: m.oy + ny * m.scale };
}

function segmentDistPxNative(a, b) {
  const dx = b.nx - a.nx;
  const dy = b.ny - a.ny;
  return Math.sqrt(dx * dx + dy * dy);
}

function canvasLocalFromClient(clientX, clientY) {
  const c = videoMeasureCanvas.getBoundingClientRect();
  return { x: clientX - c.left, y: clientY - c.top };
}

function findMeasureHitIndex(clientX, clientY) {
  const { x: cx, y: cy } = canvasLocalFromClient(clientX, clientY);
  const vr = videoModalPlayer.getBoundingClientRect();
  const cr = videoMeasureCanvas.getBoundingClientRect();
  const offX = vr.left - cr.left;
  const offY = vr.top - cr.top;
  for (let i = videoMeasurePoints.length - 1; i >= 0; i--) {
    const p = videoMeasurePoints[i];
    const loc = nativeToLocalVideo(p.nx, p.ny);
    if (!loc) continue;
    const px = offX + loc.x;
    const py = offY + loc.y;
    const d = Math.hypot(cx - px, cy - py);
    if (d <= MEASURE_HIT_PX) return i;
  }
  return -1;
}

function drawVideoMeasureOverlay() {
  const canvas = videoMeasureCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const vr = videoModalPlayer.getBoundingClientRect();
  const cr = canvas.getBoundingClientRect();
  const offX = vr.left - cr.left;
  const offY = vr.top - cr.top;

  const screenPts = [];
  for (let i = 0; i < videoMeasurePoints.length; i++) {
    const loc = nativeToLocalVideo(videoMeasurePoints[i].nx, videoMeasurePoints[i].ny);
    if (!loc) return;
    screenPts.push({ x: offX + loc.x, y: offY + loc.y });
  }

  ctx.strokeStyle = "#55ff88";
  ctx.lineWidth = 2;
  ctx.font = "13px sans-serif";

  if (screenPts.length >= 2) {
    ctx.beginPath();
    ctx.moveTo(screenPts[0].x, screenPts[0].y);
    for (let i = 1; i < screenPts.length; i++) ctx.lineTo(screenPts[i].x, screenPts[i].y);
    ctx.stroke();

    let total = 0;
    for (let i = 0; i < videoMeasurePoints.length - 1; i++) {
      const d = segmentDistPxNative(videoMeasurePoints[i], videoMeasurePoints[i + 1]);
      total += d;
      const mx = (screenPts[i].x + screenPts[i + 1].x) / 2;
      const my = (screenPts[i].y + screenPts[i + 1].y) / 2;
      const label = `${d.toFixed(1)} px`;
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(0,0,0,0.75)";
      ctx.fillRect(mx - tw / 2 - 4, my - 11, tw + 8, 18);
      ctx.fillStyle = "#a8ffc8";
      ctx.fillText(label, mx - tw / 2, my + 4);
    }
    const sum = `Total: ${total.toFixed(1)} px`;
    const twSum = ctx.measureText(sum).width;
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(8, 8, twSum + 12, 22);
    ctx.fillStyle = "#9ec3ff";
    ctx.fillText(sum, 14, 24);
  }

  ctx.strokeStyle = "#ffffff";
  ctx.fillStyle = "#55ff88";
  for (const q of screenPts) {
    ctx.beginPath();
    ctx.arc(q.x, q.y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

function resizeVideoMeasureCanvas() {
  const stage = videoModalStage;
  const canvas = videoMeasureCanvas;
  if (!stage || !canvas) return;
  const cw = stage.clientWidth;
  const ch = stage.clientHeight;
  if (!cw || !ch) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cw * dpr);
  canvas.height = Math.round(ch * dpr);
  canvas.style.width = `${cw}px`;
  canvas.style.height = `${ch}px`;
  drawVideoMeasureOverlay();
}

function resetVideoMeasure() {
  videoMeasureMode = false;
  videoMeasurePoints = [];
  videoMeasureDragIdx = -1;
  if (videoModalStage) videoModalStage.classList.remove("measure-active");
  if (videoMeasureHint) videoMeasureHint.classList.add("hidden");
  if (videoMeasureClearBtn) videoMeasureClearBtn.classList.add("hidden");
  if (videoMeasureToggleBtn) videoMeasureToggleBtn.textContent = "Measure distance";
  drawVideoMeasureOverlay();
}

function setVideoMeasureMode(on) {
  videoMeasureMode = !!on;
  if (!videoMeasureMode) videoMeasureDragIdx = -1;
  if (videoModalStage) videoModalStage.classList.toggle("measure-active", videoMeasureMode);
  if (videoMeasureHint) videoMeasureHint.classList.toggle("hidden", !videoMeasureMode);
  if (videoMeasureClearBtn) videoMeasureClearBtn.classList.toggle("hidden", !videoMeasureMode);
  if (videoMeasureToggleBtn) {
    videoMeasureToggleBtn.textContent = videoMeasureMode ? "Stop measuring" : "Measure distance";
  }
  resizeVideoMeasureCanvas();
}
const videoMetaModal = document.getElementById("videoMetaModal");
const videoMetaModalTitle = document.getElementById("videoMetaModalTitle");
const videoMetaPathLabel = document.getElementById("videoMetaPathLabel");
const closeVideoMetaModalBtn = document.getElementById("closeVideoMetaModalBtn");
const metaDiskPixel = document.getElementById("metaDiskPixel");
const metaDiskRadiusMm = document.getElementById("metaDiskRadiusMm");
const metaFrameStart = document.getElementById("metaFrameStart");
const metaFrameEnd = document.getElementById("metaFrameEnd");
const metaFlyCount = document.getElementById("metaFlyCount");
const metaSplitX = document.getElementById("metaSplitX");
const metaSplitY = document.getElementById("metaSplitY");
const metaDetailedLocation = document.getElementById("metaDetailedLocation");
const metaTotalFrames = document.getElementById("metaTotalFrames");
const metaVideoWidth = document.getElementById("metaVideoWidth");
const metaVideoHeight = document.getElementById("metaVideoHeight");
const metaDetectFramesBtn = document.getElementById("metaDetectFramesBtn");
const metaSaveBtn = document.getElementById("metaSaveBtn");
const metaModalStatus = document.getElementById("metaModalStatus");
const projectMetaModal = document.getElementById("projectMetaModal");
const projectMetaModalTitle = document.getElementById("projectMetaModalTitle");
const projectMetaNameLabel = document.getElementById("projectMetaNameLabel");
const projectCurrently = document.getElementById("projectCurrently");
const projectAbstract = document.getElementById("projectAbstract");
const projectQuickrunOutput = document.getElementById("projectQuickrunOutput");
const projectSnapshotOutput = document.getElementById("projectSnapshotOutput");
const projectTrackingOutput = document.getElementById("projectTrackingOutput");
const closeProjectMetaModalBtn = document.getElementById("closeProjectMetaModalBtn");
const projectMetaSaveBtn = document.getElementById("projectMetaSaveBtn");
const projectMetaModalStatus = document.getElementById("projectMetaModalStatus");
const videoBatchBar = document.getElementById("videoBatchBar");
const videoSelectAll = document.getElementById("videoSelectAll");
const batchDeleteVideosBtn = document.getElementById("batchDeleteVideosBtn");
const quickRunFastviewBtn = document.getElementById("quickRunFastviewBtn");
const snapshotBatchBtn = document.getElementById("snapshotBatchBtn");
const trackingBatchBtn = document.getElementById("trackingBatchBtn");
const quickRunModal = document.getElementById("quickRunModal");
const quickRunScopeText = document.getElementById("quickRunScopeText");
const closeQuickRunModalBtn = document.getElementById("closeQuickRunModalBtn");
const quickRunStartBtn = document.getElementById("quickRunStartBtn");
const quickRunResetDefaultsBtn = document.getElementById("quickRunResetDefaultsBtn");
const quickRunModalStatus = document.getElementById("quickRunModalStatus");
const trackingBatchModal = document.getElementById("trackingBatchModal");
const trackingBatchScopeText = document.getElementById("trackingBatchScopeText");
const closeTrackingBatchModalBtn = document.getElementById("closeTrackingBatchModalBtn");
const trackingBatchStartBtn = document.getElementById("trackingBatchStartBtn");
const trackingBatchResetDefaultsBtn = document.getElementById("trackingBatchResetDefaultsBtn");
const trackingBatchModalStatus = document.getElementById("trackingBatchModalStatus");
const trackingOpenModal = document.getElementById("trackingOpenModal");
const trackingOpenModalParams = document.getElementById("trackingOpenModalParams");
const trackingOpenModalManifest = document.getElementById("trackingOpenModalManifest");
const closeTrackingOpenModalBtn = document.getElementById("closeTrackingOpenModalBtn");
const trackingOpenConfirmBtn = document.getElementById("trackingOpenConfirmBtn");
const quickRunWorkers = document.getElementById("quickRunWorkers");
const quickRunWindowOverlap = document.getElementById("quickRunWindowOverlap");
const quickRunSpeedWindow = document.getElementById("quickRunSpeedWindow");
const quickRunGamSplines = document.getElementById("quickRunGamSplines");
const quickRunGamGridPoints = document.getElementById("quickRunGamGridPoints");
const quickRunPeakPromFactor = document.getElementById("quickRunPeakPromFactor");
const quickRunPeakMinDistance = document.getElementById("quickRunPeakMinDistance");
const quickRunClipDurationMin = document.getElementById("quickRunClipDurationMin");
const quickRunClipFps = document.getElementById("quickRunClipFps");
const quickRunOutputDir = document.getElementById("quickRunOutputDir");
const quickRunWeights = document.getElementById("quickRunWeights");
const quickRunFrameSkip = document.getElementById("quickRunFrameSkip");
const quickRunConfThres = document.getElementById("quickRunConfThres");
const quickRunIouThres = document.getElementById("quickRunIouThres");
const quickRunImgsz = document.getElementById("quickRunImgsz");
const quickRunDevice = document.getElementById("quickRunDevice");
const quickRunLimit = document.getElementById("quickRunLimit");
const quickRunSkipDetect = document.getElementById("quickRunSkipDetect");
const quickRunSkipTrack = document.getElementById("quickRunSkipTrack");
const quickRunSkipViz = document.getElementById("quickRunSkipViz");
const quickRunRerun = document.getElementById("quickRunRerun");
const trackingBatchOutputDir = document.getElementById("trackingBatchOutputDir");
const trackingBatchWeights = document.getElementById("trackingBatchWeights");
const trackingBatchDevice = document.getElementById("trackingBatchDevice");
const trackingBatchConfThres = document.getElementById("trackingBatchConfThres");
const trackingBatchImgSize = document.getElementById("trackingBatchImgSize");
const trackingBatchNameOverride = document.getElementById("trackingBatchNameOverride");
const trackingBatchDirOverride = document.getElementById("trackingBatchDirOverride");
const trackingBatchTarTrStart = document.getElementById("trackingBatchTarTrStart");
const trackingBatchFrameStart = document.getElementById("trackingBatchFrameStart");
const trackingBatchFrameEnd = document.getElementById("trackingBatchFrameEnd");
const trackingBatchQuiet = document.getElementById("trackingBatchQuiet");
const trackingBatchExistOk = document.getElementById("trackingBatchExistOk");
const trackingBatchDetectFlagsSection = document.getElementById("trackingBatchDetectFlagsSection");
const trackingBatchInitLabelPath = document.getElementById("trackingBatchInitLabelPath");
const trackingBatchUseSnapshotInit = document.getElementById("trackingBatchUseSnapshotInit");
const trackingBatchAllowMissingInit = document.getElementById("trackingBatchAllowMissingInit");
const trackingBatchRerun = document.getElementById("trackingBatchRerun");
const trackingBatchSourcePreview = document.getElementById("trackingBatchSourcePreview");
const trackingBatchDerivedPreview = document.getElementById("trackingBatchDerivedPreview");
const videoMetaTsvPath = document.getElementById("videoMetaTsvPath");
const importVideoMetaTsvBtn = document.getElementById("importVideoMetaTsvBtn");
const exportVideoMetaTsvBtn = document.getElementById("exportVideoMetaTsvBtn");

let projects = [];
/** Set while dragging a project row by handle (HTML5 DnD). */
let dragProjectName = null;
let selectedProject = "";
let editingVideoPath = "";
/** When non-null, QuickRun uses only these paths; null means all project videos. */
let quickRunVideoPaths = null;
/** Selected videos/subclips captured when the Tracking Batch modal opens. */
let trackingBatchItems = null;
/** Last full project payload from the server (for QuickRun default output dir, etc.). */
let cachedProjectDetail = null;
/** Registered video table sort: `key` null = server order. */
let videoTableSort = { key: null, dir: "asc" };
/** Snapshot API counts (for sorting Detected Flies before cells are filled). */
let gSnapshotCountsForSort = { counts: {}, clip_counts: {} };
/** Per main-row video path: clip count from /api/project/video_subclips (for Clips column sort). Cleared when switching project. */
let gClipCountsForSort = Object.create(null);
let gClipCountsForSortProject = null;
/** Full URL path for Detect explorer after Tracking review modal (path + query only). */
let trackingOpenPendingUrl = "";

const QUICK_RUN_DEFAULTS = {
  workers: 64,
  windowOverlap: 200,
  speedWindow: 300,
  gamSplines: 25,
  gamGridPoints: 200,
  peakProminenceFactor: 0.2,
  peakMinDistanceFrames: 0,
  autoClipDurationMin: 10,
  autoClipFps: 30,
  outputDir: "QuickTestForAging",
  weights: "YoloFly/runs/train/2022_05_11_p633_1280_5l_e700_b128/weights/best.pt",
  frameSkip: 30,
  confThres: 0.3,
  iouThres: 0.45,
  imgsz: 640,
  device: "",
  limit: 0,
  skipDetect: false,
  skipTrack: false,
  skipVisualize: false,
  rerun: false,
};

const TRACKING_BATCH_DEFAULTS = {
  outputDir: "traking",
  weights: QUICK_RUN_DEFAULTS.weights,
  device: "",
  confThres: 0.4,
  imgSize: 1280,
  nameOverride: "",
  dirOverride: "",
  tarTrStart: "",
  frameStart: "",
  frameEnd: "",
  quiet: true,
  existOk: true,
  detectFlagsSection: false,
  initLabelPath: "",
  useSnapshotInit: true,
  allowMissingInit: false,
  rerun: false,
};

function applyQuickRunDefaultsToForm() {
  if (!quickRunWorkers) return;
  quickRunWorkers.value = String(QUICK_RUN_DEFAULTS.workers);
  quickRunWindowOverlap.value = String(QUICK_RUN_DEFAULTS.windowOverlap);
  quickRunSpeedWindow.value = String(QUICK_RUN_DEFAULTS.speedWindow);
  if (quickRunGamSplines) quickRunGamSplines.value = String(QUICK_RUN_DEFAULTS.gamSplines);
  if (quickRunGamGridPoints) quickRunGamGridPoints.value = String(QUICK_RUN_DEFAULTS.gamGridPoints);
  if (quickRunPeakPromFactor) quickRunPeakPromFactor.value = String(QUICK_RUN_DEFAULTS.peakProminenceFactor);
  if (quickRunPeakMinDistance) quickRunPeakMinDistance.value = String(QUICK_RUN_DEFAULTS.peakMinDistanceFrames);
  if (quickRunClipDurationMin) quickRunClipDurationMin.value = String(QUICK_RUN_DEFAULTS.autoClipDurationMin);
  if (quickRunClipFps) quickRunClipFps.value = String(QUICK_RUN_DEFAULTS.autoClipFps);
  quickRunOutputDir.value = QUICK_RUN_DEFAULTS.outputDir;
  quickRunWeights.value = QUICK_RUN_DEFAULTS.weights;
  quickRunFrameSkip.value = String(QUICK_RUN_DEFAULTS.frameSkip);
  quickRunConfThres.value = String(QUICK_RUN_DEFAULTS.confThres);
  quickRunIouThres.value = String(QUICK_RUN_DEFAULTS.iouThres);
  quickRunImgsz.value = String(QUICK_RUN_DEFAULTS.imgsz);
  quickRunDevice.value = QUICK_RUN_DEFAULTS.device;
  quickRunLimit.value = String(QUICK_RUN_DEFAULTS.limit);
  quickRunSkipDetect.checked = QUICK_RUN_DEFAULTS.skipDetect;
  quickRunSkipTrack.checked = QUICK_RUN_DEFAULTS.skipTrack;
  quickRunSkipViz.checked = QUICK_RUN_DEFAULTS.skipVisualize;
  quickRunRerun.checked = QUICK_RUN_DEFAULTS.rerun;
}

function applyTrackingBatchDefaultsToForm() {
  if (!trackingBatchOutputDir) return;
  trackingBatchOutputDir.value = TRACKING_BATCH_DEFAULTS.outputDir;
  trackingBatchWeights.value = TRACKING_BATCH_DEFAULTS.weights;
  if (trackingBatchDevice) trackingBatchDevice.value = TRACKING_BATCH_DEFAULTS.device;
  trackingBatchConfThres.value = String(TRACKING_BATCH_DEFAULTS.confThres);
  trackingBatchImgSize.value = String(TRACKING_BATCH_DEFAULTS.imgSize);
  trackingBatchNameOverride.value = TRACKING_BATCH_DEFAULTS.nameOverride;
  trackingBatchDirOverride.value = TRACKING_BATCH_DEFAULTS.dirOverride;
  trackingBatchTarTrStart.value = TRACKING_BATCH_DEFAULTS.tarTrStart;
  trackingBatchFrameStart.value = TRACKING_BATCH_DEFAULTS.frameStart;
  trackingBatchFrameEnd.value = TRACKING_BATCH_DEFAULTS.frameEnd;
  trackingBatchQuiet.checked = TRACKING_BATCH_DEFAULTS.quiet;
  trackingBatchExistOk.checked = TRACKING_BATCH_DEFAULTS.existOk;
  if (trackingBatchDetectFlagsSection) {
    trackingBatchDetectFlagsSection.checked = TRACKING_BATCH_DEFAULTS.detectFlagsSection;
  }
  trackingBatchInitLabelPath.value = TRACKING_BATCH_DEFAULTS.initLabelPath;
  trackingBatchUseSnapshotInit.checked = TRACKING_BATCH_DEFAULTS.useSnapshotInit;
  trackingBatchAllowMissingInit.checked = TRACKING_BATCH_DEFAULTS.allowMissingInit;
  trackingBatchRerun.checked = TRACKING_BATCH_DEFAULTS.rerun;
}

function openQuickRunModal(selectedPaths) {
  if (!quickRunModal) return;
  quickRunVideoPaths = selectedPaths.length ? selectedPaths : null;
  if (quickRunScopeText) {
    quickRunScopeText.textContent = quickRunVideoPaths
      ? `Will run on ${quickRunVideoPaths.length} selected video(s).`
      : "Will run on all videos in this project.";
  }
  if (quickRunModalStatus) quickRunModalStatus.textContent = "";
  applyQuickRunDefaultsToForm();
  if (
    quickRunOutputDir &&
    cachedProjectDetail &&
    cachedProjectDetail.name === selectedProject &&
    (cachedProjectDetail.quickrun_output || "").trim()
  ) {
    quickRunOutputDir.value = cachedProjectDetail.quickrun_output.trim();
  }
  quickRunModal.classList.remove("hidden");
}

function closeQuickRunModal() {
  if (quickRunModal) quickRunModal.classList.add("hidden");
}

function summarizeTrackingBatchItems(items) {
  const videos = items.filter((it) => it && it.type === "video").length;
  const subclips = items.filter((it) => it && it.type === "subclip").length;
  const parts = [];
  if (videos) parts.push(`${videos} video(s)`);
  if (subclips) parts.push(`${subclips} subclip(s)`);
  return parts.length ? parts.join(" and ") : "no targets";
}

function projectVideoEntryByPath(path) {
  const videos = cachedProjectDetail && Array.isArray(cachedProjectDetail.videos)
    ? cachedProjectDetail.videos
    : [];
  return videos.find((v) => {
    if (typeof v === "string") return v === path;
    return v && v.path === path;
  });
}

function trackingBatchItemPreview(item) {
  if (!item || !item.video_path) return "";
  const name = item.video_path.split(/[\\/]/).pop() || item.video_path;
  if (item.type === "subclip") {
    const bits = [`subclip ${item.clip_id}`, name];
    if (item.frame_start || item.frame_end) {
      bits.push(`frames ${item.frame_start || "?"}-${item.frame_end || "?"}`);
    }
    return bits.join(" · ");
  }
  const ent = projectVideoEntryByPath(item.video_path);
  if (ent && typeof ent === "object") {
    const fs = ent.frame_start != null && ent.frame_start !== "" ? ent.frame_start : "auto 1";
    const fe = ent.frame_end != null && ent.frame_end !== "" ? ent.frame_end : "auto none";
    return `video · ${name} · frames ${fs}-${fe}`;
  }
  return `video · ${name}`;
}

function updateTrackingBatchDerivedPreview() {
  if (!trackingBatchItems) return;
  if (trackingBatchSourcePreview) {
    trackingBatchSourcePreview.value = trackingBatchItems
      .map((it) => it.video_path)
      .join("\n");
  }
  if (trackingBatchDerivedPreview) {
    const lines = [
      "--source: one selected video path per job",
      "Optional flags section: when checked, adds --bh-count, --tar-track + --tar-tr-start, --head-bind (default off)",
      "--tar-tr-start override in Frames only applies when the optional flags section is enabled",
      "--frame-start: row/subclip start unless override is set",
      "--frame-end: row/subclip end when available unless override is set",
      "--device: if set to comma-separated GPU ids (e.g. 0,1,2), tracking jobs run in parallel (one active job per GPU)",
      "--name: auto-generated per target unless a single-target override is set",
      "--tracking-dir: <tracking output base>/<auto name> unless a single-target override is set",
      "--init-label-path: Snapshot label lookup unless an override is set",
      "",
      "Targets:",
      ...trackingBatchItems.map((it, idx) => `${idx + 1}. ${trackingBatchItemPreview(it)}`),
    ];
    trackingBatchDerivedPreview.value = lines.join("\n");
  }
}

function openTrackingBatchModal(items) {
  if (!trackingBatchModal) return;
  trackingBatchItems = items.slice();
  if (trackingBatchScopeText) {
    trackingBatchScopeText.textContent = `Will run tracking on ${summarizeTrackingBatchItems(trackingBatchItems)}.`;
  }
  if (trackingBatchModalStatus) trackingBatchModalStatus.textContent = "";
  applyTrackingBatchDefaultsToForm();
  if (
    trackingBatchOutputDir &&
    cachedProjectDetail &&
    cachedProjectDetail.name === selectedProject &&
    (cachedProjectDetail.tracking_output || "").trim()
  ) {
    trackingBatchOutputDir.value = cachedProjectDetail.tracking_output.trim();
  }
  updateTrackingBatchDerivedPreview();
  trackingBatchModal.classList.remove("hidden");
}

function closeTrackingBatchModal() {
  if (trackingBatchModal) trackingBatchModal.classList.add("hidden");
}

function setProjectMetaStatus(msg) {
  if (projectMetaModalStatus) projectMetaModalStatus.textContent = msg || "";
}

function openProjectMetaModal(project) {
  if (!projectMetaModal || !project || !projectCurrently || !projectAbstract || !projectQuickrunOutput) return;
  if (projectMetaModalTitle) projectMetaModalTitle.textContent = `Project meta: ${project.name}`;
  if (projectMetaNameLabel) projectMetaNameLabel.textContent = project.name;
  projectCurrently.value = project.currently || "";
  projectAbstract.value = project.abstract || "";
  projectQuickrunOutput.value = project.quickrun_output || "";
  if (projectSnapshotOutput) {
    projectSnapshotOutput.value = project.snapshot_output || "";
  }
  if (projectTrackingOutput) {
    projectTrackingOutput.value = project.tracking_output || "traking";
  }
  setProjectMetaStatus("");
  projectMetaModal.classList.remove("hidden");
}

function closeProjectMetaModal() {
  if (projectMetaModal) projectMetaModal.classList.add("hidden");
}

function collectQuickRunPayload() {
  const workers = parseInt(quickRunWorkers.value, 10);
  const window_overlap = parseInt(quickRunWindowOverlap.value, 10);
  const speed_window = parseInt(quickRunSpeedWindow.value, 10);
  const gam_splines = parseInt(quickRunGamSplines.value, 10);
  const gam_grid_points = parseInt(quickRunGamGridPoints.value, 10);
  const peak_prominence_factor = parseFloat(quickRunPeakPromFactor.value);
  const peak_min_distance_frames = parseInt(quickRunPeakMinDistance.value, 10);
  const auto_clip_duration_min = parseFloat(quickRunClipDurationMin.value);
  const auto_clip_fps = parseInt(quickRunClipFps.value, 10);
  const frame_skip = parseInt(quickRunFrameSkip.value, 10);
  const imgsz = parseInt(quickRunImgsz.value, 10);
  const limit = parseInt(quickRunLimit.value, 10);
  const conf_thres = parseFloat(quickRunConfThres.value);
  const iou_thres = parseFloat(quickRunIouThres.value);
  const fields = [
    ["workers", workers],
    ["window overlap", window_overlap],
    ["speed window", speed_window],
    ["GAM splines", gam_splines],
    ["GAM grid frames", gam_grid_points],
    ["peak prominence factor", peak_prominence_factor],
    ["peak min distance frames", peak_min_distance_frames],
    ["auto subclip duration", auto_clip_duration_min],
    ["auto subclip fps", auto_clip_fps],
    ["frame skip", frame_skip],
    ["imgsz", imgsz],
    ["limit", limit],
    ["confidence threshold", conf_thres],
    ["IoU threshold", iou_thres],
  ];
  for (const [label, n] of fields) {
    if (!Number.isFinite(n)) throw new Error(`${label} must be a valid number.`);
  }
  const body = {
    name: selectedProject,
    workers,
    window_overlap,
    speed_window,
    gam_splines,
    gam_grid_points,
    peak_prominence_factor,
    peak_min_distance_frames,
    auto_clip_duration_min,
    auto_clip_fps,
    output_dir: (quickRunOutputDir.value || "").trim(),
    weights: (quickRunWeights.value || "").trim(),
    frame_skip,
    conf_thres,
    iou_thres,
    imgsz,
    device: (quickRunDevice.value || "").trim(),
    limit,
    skip_detect: quickRunSkipDetect.checked,
    skip_track: quickRunSkipTrack.checked,
    skip_visualize: quickRunSkipViz.checked,
    rerun: quickRunRerun.checked,
  };
  if (quickRunVideoPaths && quickRunVideoPaths.length) {
    body.video_paths = quickRunVideoPaths;
  }
  return body;
}

function collectTrackingBatchPayload(allowMissingInitOverride) {
  if (!trackingBatchItems || !trackingBatchItems.length) {
    throw new Error("Select at least one video and/or subclip for tracking batch.");
  }
  const conf_thres = parseFloat(trackingBatchConfThres.value);
  const img_size = parseInt(trackingBatchImgSize.value, 10);
  const tar_tr_start_override = parseOptionalPositiveIntInput(trackingBatchTarTrStart, "--tar-tr-start override");
  const frame_start_override = parseOptionalPositiveIntInput(trackingBatchFrameStart, "--frame-start override");
  const frame_end_override = parseOptionalPositiveIntInput(trackingBatchFrameEnd, "--frame-end override");
  if (!Number.isFinite(conf_thres)) throw new Error("confidence threshold must be a valid number.");
  if (!Number.isFinite(img_size)) throw new Error("image size must be a valid integer.");
  return {
    name: selectedProject,
    items: trackingBatchItems,
    weights: (trackingBatchWeights.value || "").trim(),
    device: (trackingBatchDevice && trackingBatchDevice.value ? trackingBatchDevice.value : "").trim(),
    tracking_output: (trackingBatchOutputDir.value || "").trim(),
    conf_thres,
    img_size,
    run_name_override: (trackingBatchNameOverride.value || "").trim(),
    tracking_dir_override: (trackingBatchDirOverride.value || "").trim(),
    tar_tr_start_override,
    frame_start_override,
    frame_end_override,
    quiet: trackingBatchQuiet.checked,
    exist_ok: trackingBatchExistOk.checked,
    tracking_detect_flags: !!(trackingBatchDetectFlagsSection && trackingBatchDetectFlagsSection.checked),
    init_label_path_override: (trackingBatchInitLabelPath.value || "").trim(),
    use_snapshot_init: trackingBatchUseSnapshotInit.checked,
    allow_missing_init:
      allowMissingInitOverride != null
        ? !!allowMissingInitOverride
        : trackingBatchAllowMissingInit.checked,
    rerun: trackingBatchRerun.checked,
  };
}

function parseOptionalPositiveIntInput(el, label) {
  const raw = (el && el.value ? el.value : "").trim();
  if (!raw) return null;
  const n = parseInt(raw, 10);
  if (!Number.isFinite(n) || n < 1) {
    throw new Error(`${label} must be a positive integer, or blank for auto.`);
  }
  return n;
}

function setStatus(msg) {
  homeStatus.textContent = msg || "";
}

async function req(url, options) {
  const resp = await fetch(url, options);
  const data = await resp.json();
  if (!resp.ok || data.error) {
    throw new Error(data.error || `Request failed: ${resp.status}`);
  }
  return data;
}

function videoRecordPath(rec) {
  if (typeof rec === "string") return rec;
  if (rec && rec.absolute_path) return rec.absolute_path;
  return rec && rec.path ? rec.path : "";
}

function fmtVideoMetaCell(val) {
  if (val === null || val === undefined || val === "") return "—";
  return String(val);
}

function videoRowSplitText(rec) {
  const r = typeof rec === "object" && rec ? rec : {};
  const hasSplit =
    (r.split_x != null && r.split_x !== "") || (r.split_y != null && r.split_y !== "");
  if (!hasSplit) return "—";
  return `${fmtVideoMetaCell(r.split_x)}×${fmtVideoMetaCell(r.split_y)}`;
}

function videoMetaNumber(rec, field) {
  const r = typeof rec === "object" && rec ? rec : {};
  const v = r[field];
  if (v == null || v === "") return null;
  const n = Number(String(v).replace(",", "."));
  return Number.isFinite(n) ? n : null;
}

/**
 * @param {unknown} a
 * @param {unknown} b
 * @param {string} key
 * @param {"asc"|"desc"} dir
 */
function clipCountSortValue(videoPath) {
  const v = gClipCountsForSort[videoPath];
  if (typeof v === "number" && Number.isFinite(v)) return v;
  return Infinity;
}

function compareVideoRows(a, b, key, dir) {
  const mul = dir === "desc" ? -1 : 1;
  const ra = typeof a === "object" && a ? a : {};
  const rb = typeof b === "object" && b ? b : {};
  const pa = videoRecordPath(a);
  const pb = videoRecordPath(b);
  let cmp = 0;
  switch (key) {
    case "name": {
      const na = (pa.split(/[\\/]/).pop() || pa).toLowerCase();
      const nb = (pb.split(/[\\/]/).pop() || pb).toLowerCase();
      cmp = na.localeCompare(nb, undefined, { sensitivity: "base", numeric: true });
      break;
    }
    case "snap": {
      const ca = gSnapshotCountsForSort.counts[pa];
      const cb = gSnapshotCountsForSort.counts[pb];
      const ta = ca && ca.has_snapshot ? Number(ca.total) : null;
      const tb = cb && cb.has_snapshot ? Number(cb.total) : null;
      const sa = ta != null && Number.isFinite(ta) ? ta : -1;
      const sb = tb != null && Number.isFinite(tb) ? tb : -1;
      cmp = sa - sb;
      if (cmp === 0) {
        const c0a = ca && ca.has_snapshot ? Number(ca.class0) : -1;
        const c0b = cb && cb.has_snapshot ? Number(cb.class0) : -1;
        cmp = (Number.isFinite(c0a) ? c0a : -1) - (Number.isFinite(c0b) ? c0b : -1);
      }
      break;
    }
    case "disk_pixel":
    case "disk_radius_mm":
    case "frame_start":
    case "frame_end":
    case "fly_count": {
      const na = videoMetaNumber(ra, key);
      const nb = videoMetaNumber(rb, key);
      cmp = (na ?? -Infinity) - (nb ?? -Infinity);
      break;
    }
    case "split": {
      const sa = videoRowSplitText(ra);
      const sb = videoRowSplitText(rb);
      cmp = sa.localeCompare(sb, undefined, { sensitivity: "base", numeric: true });
      const xa = videoMetaNumber(ra, "split_x");
      const xb = videoMetaNumber(rb, "split_x");
      if (cmp === 0) cmp = (xa ?? -Infinity) - (xb ?? -Infinity);
      if (cmp === 0) {
        const ya = videoMetaNumber(ra, "split_y");
        const yb = videoMetaNumber(rb, "split_y");
        cmp = (ya ?? -Infinity) - (yb ?? -Infinity);
      }
      break;
    }
    case "clips": {
      cmp = clipCountSortValue(pa) - clipCountSortValue(pb);
      break;
    }
    default:
      cmp = 0;
  }
  if (cmp !== 0) return mul * cmp;
  return pa.localeCompare(pb);
}

function updateVideoSortHeaderIndicators() {
  const table = document.querySelector(".video-table");
  if (!table) return;
  for (const th of table.querySelectorAll("thead th[data-video-sort]")) {
    const dirEl = th.querySelector(".video-sort-dir");
    if (!dirEl) continue;
    const k = th.getAttribute("data-video-sort");
    if (videoTableSort.key && k === videoTableSort.key) {
      dirEl.textContent = videoTableSort.dir === "asc" ? " ▲" : " ▼";
    } else {
      dirEl.textContent = "";
    }
  }
}

const VIDEO_TABLE_COLSPAN = 11;

function mkVideoNumTd(val) {
  const td = document.createElement("td");
  td.className = "col-num-cell";
  td.textContent = fmtVideoMetaCell(val);
  return td;
}

/** Snapshot class 0/1 counts cell (filled after /api/project/snapshot_label_counts). */
/** Main video row: clip count (filled after /api/project/video_subclips). */
function mkVideoClipsTdMainPlaceholder() {
  const td = document.createElement("td");
  td.className = "col-num-cell col-clips-cell";
  td.textContent = "…";
  td.title = "Number of saved clips for this video";
  return td;
}

/** Subclip rows: count is only on the parent main row. */
function mkVideoClipsTdSubclipDash() {
  const td = document.createElement("td");
  td.className = "col-num-cell col-clips-cell video-clips-cell-sub";
  td.textContent = "—";
  return td;
}

function setMainVideoRowClipsCell(mainTr, text) {
  const td = mainTr && mainTr.querySelector("td.col-clips-cell");
  if (td) td.textContent = text;
  const path = mainTr && mainTr.dataset.videoPath;
  if (!path) return;
  if (text === "…") {
    delete gClipCountsForSort[path];
  } else if (text === "—") {
    gClipCountsForSort[path] = null;
  } else {
    const n = parseInt(String(text), 10);
    gClipCountsForSort[path] = Number.isFinite(n) ? n : null;
  }
}

function mkVideoSnapTdPlaceholder() {
  const td = document.createElement("td");
  td.className = "col-num-cell video-snap-cell";
  td.textContent = "…";
  td.title = "Loading snapshot label counts…";
  return td;
}

/** Green when sum of classes equals Flies, or both class counts equal Flies (e.g. 24/24 vs 24). */
function snapCountsMatchFlies(c, fly) {
  if (c == null || !c.has_snapshot) return false;
  if (fly == null || fly === "") return false;
  const f = Number(fly);
  if (!Number.isFinite(f)) return false;
  if (c.matches_flies === true) return true;
  const c0 = Number(c.class0);
  const c1 = Number(c.class1);
  const t = c.total != null ? Number(c.total) : NaN;
  if (Number.isFinite(t) && t === f) return true;
  if (Number.isFinite(c0) && Number.isFinite(c1) && c0 === f && c1 === f) return true;
  return false;
}

function applySnapCountsToCell(td, c, emptyTitle) {
  td.classList.remove("video-snap-match", "video-snap-mismatch");
  if (!c || !c.has_snapshot) {
    td.textContent = "—";
    td.title = emptyTitle;
    return;
  }
  td.textContent = `${c.class0}/${c.class1}`;
  const fly = c.fly_count;
  td.title = `Snapshot label: class 0 = ${c.class0}, class 1 = ${c.class1}, sum = ${c.total}. Flies (meta) = ${fly != null ? fly : "—"}. Match: sum equals Flies, or both classes equal Flies.`;
  if (fly != null && fly !== "") {
    if (snapCountsMatchFlies(c, fly)) td.classList.add("video-snap-match");
    else td.classList.add("video-snap-mismatch");
  }
}

async function loadSnapshotLabelCounts(projectName) {
  if (!projectName || !videoList) return;
  try {
    const r = await req(
      `/api/project/snapshot_label_counts?name=${encodeURIComponent(projectName)}`,
    );
    if (selectedProject !== projectName) return;
    const counts = r.counts || {};
    const clipCounts = r.clip_counts || {};
    gSnapshotCountsForSort = { counts, clip_counts: clipCounts };
    for (const row of videoList.querySelectorAll(
      "tr.video-row:not(.video-subclip-row)[data-video-path]",
    )) {
      const p = row.dataset.videoPath;
      const td = row.querySelector(".video-snap-cell");
      if (!td) continue;
      applySnapCountsToCell(
        td,
        counts[p],
        "No snapshot label indexed for this video (run Snapshot batch)",
      );
    }
    for (const row of videoList.querySelectorAll(
      "tr.video-subclip-row[data-video-path][data-clip-id]",
    )) {
      const p = row.dataset.videoPath;
      const cid = row.dataset.clipId;
      const td = row.querySelector(".video-snap-cell");
      if (!td) continue;
      const byClip = clipCounts[p] || {};
      const c = byClip[cid];
      applySnapCountsToCell(
        td,
        c,
        "No snapshot label for this subclip (run Snapshot with subclip selected)",
      );
    }
  } catch (_err) {
    for (const td of videoList.querySelectorAll(
      "tr[data-video-path] .video-snap-cell",
    )) {
      td.textContent = "—";
      td.title = "Could not load snapshot counts";
    }
  }
}

function fmtTrackingParam(v) {
  if (v == null || v === "") return "—";
  return String(v);
}

function setTrackingOpenDl(dlEl, rows) {
  if (!dlEl) return;
  dlEl.textContent = "";
  for (const { label, value } of rows) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = fmtTrackingParam(value);
    dlEl.appendChild(dt);
    dlEl.appendChild(dd);
  }
}

function closeTrackingOpenModal() {
  if (!trackingOpenModal) return;
  trackingOpenModal.classList.add("hidden");
  trackingOpenPendingUrl = "";
}

/**
 * Subclip “Tracking”: show all query/manifest parameters, then user opens Detect explorer.
 */
function openTrackingReviewModal({ trackingDir, videoPath, clip }) {
  if (!trackingOpenModal || !trackingOpenModalParams) return;
  const q2 = new URLSearchParams({
    tracking_dir: String(trackingDir),
    video_path: String(videoPath),
  });
  trackingOpenPendingUrl = `/detect_explore?${q2.toString()}`;
  const origin = typeof window !== "undefined" && window.location && window.location.origin
    ? window.location.origin
    : "";
  const absUrl = origin ? `${origin}${trackingOpenPendingUrl}` : trackingOpenPendingUrl;

  const clipLabel =
    clip && clip.name && String(clip.name).trim()
      ? String(clip.name).trim()
      : clip && clip.id != null
        ? `Clip ${clip.id}`
        : "—";

  setTrackingOpenDl(trackingOpenModalParams, [
    { label: "Project", value: selectedProject || "—" },
    { label: "Parent video path", value: videoPath },
    { label: "Subclip id", value: clip && clip.id != null ? String(clip.id) : "—" },
    { label: "Subclip label", value: clipLabel },
    {
      label: "Subclip frame range (plot window)",
      value:
        clip && (clip.start != null || clip.end != null)
          ? `${clip.start ?? "—"} … ${clip.end ?? "—"}`
          : "—",
    },
    {
      label: "Total-speed CSV (subclip)",
      value: clip && clip.source_csv ? String(clip.source_csv) : "—",
    },
    { label: "Tracking output directory (tracking_dir)", value: trackingDir },
    { label: "video_path (query)", value: videoPath },
    { label: "Full URL to open", value: absUrl },
  ]);

  if (trackingOpenModalManifest) {
    trackingOpenModalManifest.textContent = "Resolving paths from server (/api/tracking_explore/manifest)…";
  }

  trackingOpenModal.classList.remove("hidden");

  const qsp = new URLSearchParams({ dir: String(trackingDir) });
  qsp.set("video_path", String(videoPath));
  fetch(`/api/tracking_explore/manifest?${qsp.toString()}`)
    .then((resp) => resp.json().then((j) => ({ resp, j })))
    .then(({ resp, j }) => {
      if (!trackingOpenModal || trackingOpenModal.classList.contains("hidden")) return;
      if (!trackingOpenModalManifest) return;
      if (!resp.ok || j.error) {
        trackingOpenModalManifest.textContent = `Manifest: ${j.error || `request failed (${resp.status})`}`;
        return;
      }
      trackingOpenModalManifest.textContent = [
        "Resolved by server (same assets Detect explorer will load):",
        `  save_dir: ${j.save_dir || "—"}`,
        `  video_path: ${j.video_path || "—"}`,
        `  csv_abs_path: ${j.csv_abs_path || "—"}`,
        `  json_abs_path: ${j.json_abs_path || "—"}`,
        `  frame_min … frame_max: ${j.frame_min ?? "—"} … ${j.frame_max ?? "—"}`,
      ].join("\n");
    })
    .catch((err) => {
      if (!trackingOpenModalManifest || !trackingOpenModal || trackingOpenModal.classList.contains("hidden")) {
        return;
      }
      trackingOpenModalManifest.textContent = `Manifest request failed: ${err.message || err}`;
    });
}

function fillVideoActionsTd(tdAct, path, videoEntry, playbackOverride, resultsOpts) {
  tdAct.textContent = "";
  const playBtn = document.createElement("button");
  playBtn.className = "video-play-btn";
  playBtn.type = "button";
  playBtn.textContent = "Play";
  playBtn.onclick = () => {
    let fs = null;
    let fe = null;
    let labelSuffix;
    if (playbackOverride) {
      fs = playbackOverride.frameStart != null ? playbackOverride.frameStart : null;
      fe = playbackOverride.frameEnd != null ? playbackOverride.frameEnd : null;
      labelSuffix = playbackOverride.labelSuffix;
    } else if (typeof videoEntry === "object" && videoEntry) {
      fs = videoEntry.frame_start != null ? videoEntry.frame_start : null;
      fe = videoEntry.frame_end != null ? videoEntry.frame_end : null;
    }
    openVideoModal(path, { frameStart: fs, frameEnd: fe, labelSuffix }).catch(() => {});
  };
  const editBtn = document.createElement("button");
  editBtn.className = "video-edit-btn";
  editBtn.type = "button";
  editBtn.textContent = "Edit";
      editBtn.onclick = () => {
        const base = typeof videoEntry === "object" && videoEntry ? videoEntry : { path };
        openMetaModal(base);
      };
  const resultsBtn = document.createElement("button");
  resultsBtn.className = "video-results-btn";
  resultsBtn.type = "button";
  resultsBtn.textContent = "Results";
  resultsBtn.onclick = () => {
    const q = new URLSearchParams({ video_path: path });
    if (selectedProject) q.set("project", selectedProject);
    const cid =
      resultsOpts && resultsOpts.clipId != null && String(resultsOpts.clipId).trim() !== ""
        ? String(resultsOpts.clipId).trim()
        : "";
    if (cid) q.set("clip_id", cid);
    window.open(`/video-results?${q.toString()}`, "_blank", "noopener");
  };
  const trendingBtn = document.createElement("button");
  trendingBtn.className = "video-results-btn";
  trendingBtn.type = "button";
  trendingBtn.textContent = "Trending";
  trendingBtn.title = "Open total speed interactive plot";
  trendingBtn.onclick = () => {
    if (!selectedProject) {
      setStatus("Select a project first.");
      return;
    }
    const q = new URLSearchParams({ name: selectedProject, video_path: path });
    const cid =
      resultsOpts && resultsOpts.clipId != null && String(resultsOpts.clipId).trim() !== ""
        ? String(resultsOpts.clipId).trim()
        : "";
    if (cid) q.set("clip_id", cid);
    req(`/api/project/total_speed_plot_url?${q.toString()}`)
      .then((r) => {
        const u = new URLSearchParams({ path: r.path, video_path: r.video_path || path });
        u.set("project", selectedProject);
        if (r.clip_id != null && r.clip_id !== "") u.set("clip_id", String(r.clip_id));
        window.open(`/total-speed-plot?${u.toString()}`, "_blank", "noopener");
      })
      .catch((err) => setStatus(err.message || "No total speed plot for this row."));
  };
  tdAct.appendChild(playBtn);
  tdAct.appendChild(editBtn);
  tdAct.appendChild(resultsBtn);
  tdAct.appendChild(trendingBtn);
}

function buildVideoSubclipRow(parentPath, videoEntry, clip) {
  const recObj = typeof videoEntry === "object" && videoEntry ? videoEntry : {};
  const clipLabel =
    clip.name && String(clip.name).trim() ? String(clip.name).trim() : `Clip ${clip.id}`;
  const s0 = Number(clip.start);
  const s1 = Number(clip.end);

  const tr = document.createElement("tr");
  tr.className = "video-row video-subclip-row";
  tr.title = `${parentPath} · ${clipLabel}`;
  tr.dataset.videoPath = parentPath;
  tr.dataset.clipId = String(clip.id);

  const tdCb = document.createElement("td");
  tdCb.className = "col-cb";
  const subCb = document.createElement("input");
  subCb.type = "checkbox";
  subCb.className = "video-subclip-select-cb";
  subCb.dataset.parentPath = parentPath;
  subCb.dataset.clipId = String(clip.id);
  subCb.dataset.sourceCsv = clip.source_csv ? String(clip.source_csv) : "";
  if (Number.isFinite(s0)) subCb.dataset.frameStart = String(Math.round(s0));
  if (Number.isFinite(s1)) subCb.dataset.frameEnd = String(Math.round(s1));
  subCb.title = "Select subclip";
  subCb.addEventListener("change", updateSubclipSelectAllState);
  tdCb.appendChild(subCb);

  const tdName = document.createElement("td");
  tdName.className = "col-name-cell video-subclip-name-cell";
  const titleDiv = document.createElement("div");
  titleDiv.className = "video-subclip-title";
  titleDiv.textContent = clipLabel;
  tdName.appendChild(titleDiv);
  const metaDiv = document.createElement("div");
  metaDiv.className = "video-subclip-meta-line";
  const parentFile = parentPath.split(/[\\/]/).pop() || parentPath;
  metaDiv.appendChild(document.createTextNode(`↳ ${parentFile}`));
  if (clip.source_csv) {
    metaDiv.appendChild(document.createTextNode(" · "));
    const a = document.createElement("a");
    const q = new URLSearchParams({ path: clip.source_csv, video_path: parentPath });
    if (selectedProject) q.set("project", selectedProject);
    q.set("clip_id", String(clip.id));
    a.href = `/total-speed-plot?${q.toString()}`;
    a.textContent = "Plot";
    a.className = "video-subclip-plot-link";
    metaDiv.appendChild(a);
  }
  if (clip.has_tracking && clip.tracking_output_dir) {
    metaDiv.appendChild(document.createTextNode(" · "));
    const t = document.createElement("button");
    t.type = "button";
    t.textContent = "Tracking";
    t.className = "video-subclip-plot-link";
    t.title = "Review parameters, then open Detect explorer in a new tab";
    t.onclick = () => {
      openTrackingReviewModal({
        trackingDir: String(clip.tracking_output_dir),
        videoPath: String(parentPath),
        clip,
      });
    };
    metaDiv.appendChild(t);
  }
  tdName.appendChild(metaDiv);

  const tdSplit = document.createElement("td");
  tdSplit.className = "col-split-cell";
  tdSplit.textContent = videoRowSplitText(recObj);

  const tdAct = document.createElement("td");
  tdAct.className = "col-actions-cell";
  fillVideoActionsTd(
    tdAct,
    parentPath,
    videoEntry,
    {
      frameStart: Number.isFinite(s0) ? s0 : null,
      frameEnd: Number.isFinite(s1) ? s1 : null,
      labelSuffix: clipLabel,
    },
    { clipId: clip.id },
  );

  tr.appendChild(tdCb);
  tr.appendChild(tdName);
  tr.appendChild(mkVideoClipsTdSubclipDash());
  tr.appendChild(mkVideoSnapTdPlaceholder());
  tr.appendChild(mkVideoNumTd(recObj.disk_pixel));
  tr.appendChild(mkVideoNumTd(recObj.disk_radius_mm));
  tr.appendChild(mkVideoNumTd(Number.isFinite(s0) ? Math.round(s0) : null));
  tr.appendChild(mkVideoNumTd(Number.isFinite(s1) ? Math.round(s1) : null));
  tr.appendChild(mkVideoNumTd(recObj.fly_count));
  tr.appendChild(tdSplit);
  tr.appendChild(tdAct);
  return tr;
}

function appendSubclipColspanRow(afterEl, className, message, extraClass) {
  const tr = document.createElement("tr");
  tr.className = className;
  const td = document.createElement("td");
  td.colSpan = VIDEO_TABLE_COLSPAN;
  td.className = `video-subclip-status-cell${extraClass ? ` ${extraClass}` : ""}`;
  td.textContent = message;
  tr.appendChild(td);
  afterEl.after(tr);
  return tr;
}

/** Subclip / loading / error rows inserted immediately after a main video row. */
function getVideoRowsUnderMain(mainTr) {
  const out = [];
  let el = mainTr.nextElementSibling;
  while (
    el
    && el.tagName === "TR"
    && (el.classList.contains("video-subclip-row")
      || el.classList.contains("video-subclip-loading-row")
      || el.classList.contains("video-subclip-status-row"))
  ) {
    out.push(el);
    el = el.nextElementSibling;
  }
  return out;
}

function updateSubclipFoldIcon(mainTr) {
  const icon = mainTr.querySelector(".video-subclip-fold-icon");
  if (!icon) return;
  const collapsed = mainTr.classList.contains("video-subclips-collapsed");
  icon.textContent = collapsed ? "▶" : "▼";
}

function setSubclipsFolded(mainTr, folded) {
  for (const r of getVideoRowsUnderMain(mainTr)) {
    r.classList.toggle("video-subclip-fold-hidden", folded);
  }
  mainTr.classList.toggle("video-subclips-collapsed", folded);
  mainTr.setAttribute("aria-expanded", folded ? "false" : "true");
  updateSubclipFoldIcon(mainTr);
}

function toggleSubclipsFold(mainTr) {
  setSubclipsFolded(mainTr, !mainTr.classList.contains("video-subclips-collapsed"));
}

function getMainRowsWithSubclips() {
  return [...videoList.querySelectorAll("tr.video-row.video-has-subclips")];
}

function updateVideoColumnFoldHeaderState() {
  const foldBtn = videoFoldAllBtn;
  const headerTh = videoColHeader;
  if (!foldBtn || !headerTh) return;
  const mains = getMainRowsWithSubclips();
  if (!mains.length) {
    headerTh.classList.remove("is-clickable");
    foldBtn.classList.add("hidden");
    foldBtn.disabled = true;
    headerTh.title = "Click to sort by file name";
    return;
  }
  headerTh.classList.add("is-clickable");
  foldBtn.classList.remove("hidden");
  foldBtn.disabled = false;
  const anyExpanded = mains.some((tr) => !tr.classList.contains("video-subclips-collapsed"));
  foldBtn.textContent = anyExpanded ? "▼" : "▶";
  foldBtn.title = anyExpanded ? "Fold all sub-clips" : "Unfold all sub-clips";
  headerTh.title = "Click to sort by file name (use ▶/▼ to fold or unfold all sub-clips)";
}

function setAllSubclipsFolded(folded) {
  for (const tr of getMainRowsWithSubclips()) {
    setSubclipsFolded(tr, folded);
  }
  updateVideoColumnFoldHeaderState();
}

function toggleAllSubclipsFold() {
  const mains = getMainRowsWithSubclips();
  if (!mains.length) return;
  const anyExpanded = mains.some((tr) => !tr.classList.contains("video-subclips-collapsed"));
  setAllSubclipsFolded(anyExpanded);
}

function onMainVideoRowFoldClick(e) {
  const tr = e.currentTarget;
  if (!tr.classList.contains("video-has-subclips")) return;
  if (e.target.closest("input, button, a, label, .col-clips-cell")) return;
  toggleSubclipsFold(tr);
  updateVideoColumnFoldHeaderState();
}

/** Main rows with subclips: fold affordance, row click toggles visibility (starts collapsed). */
function enableVideoSubclipFolding(mainTr, parentPath) {
  mainTr.classList.add("video-has-subclips");
  mainTr.setAttribute("aria-expanded", "false");
  const hint = " · Click row (outside buttons/checkbox) to show or hide subclips.";
  mainTr.title = (mainTr.title || parentPath || "").replace(/\s*· Click row[^\n]*$/, "") + hint;

  const nameTd = mainTr.querySelector(".col-name-cell");
  if (nameTd && !nameTd.querySelector(".video-subclip-fold-icon")) {
    const icon = document.createElement("span");
    icon.className = "video-subclip-fold-icon";
    icon.setAttribute("aria-hidden", "true");
    let inner = nameTd.querySelector(".video-name-cell-inner");
    const nameText = nameTd.querySelector(".video-name-text");
    if (!inner && nameText) {
      inner = document.createElement("div");
      inner.className = "video-name-cell-inner";
      nameTd.insertBefore(inner, nameText);
      inner.appendChild(nameText);
    }
    if (inner && nameText) inner.insertBefore(icon, nameText);
    else if (nameText) nameTd.insertBefore(icon, nameText);
    else nameTd.insertBefore(icon, nameTd.firstChild);
  }

  if (!mainTr.dataset.subclipFoldBound) {
    mainTr.dataset.subclipFoldBound = "1";
    mainTr.addEventListener("click", onMainVideoRowFoldClick);
  }

  setSubclipsFolded(mainTr, true);
  updateVideoColumnFoldHeaderState();
}

async function loadVideoSubclipsAfterMainRow(mainTr, parentPath, videoEntry) {
  const proj = selectedProject;
  const loadingTr = document.createElement("tr");
  loadingTr.className = "video-subclip-loading-row";
  const loadingTd = document.createElement("td");
  loadingTd.colSpan = VIDEO_TABLE_COLSPAN;
  loadingTd.className = "video-subclip-status-cell";
  loadingTd.textContent = "Loading subclips…";
  loadingTr.appendChild(loadingTd);
  mainTr.after(loadingTr);

  if (!proj) {
    loadingTr.remove();
    setMainVideoRowClipsCell(mainTr, "—");
    return;
  }

  try {
    const resp = await fetch(
      `/api/project/video_subclips?name=${encodeURIComponent(proj)}&video_path=${encodeURIComponent(parentPath)}`,
    );
    const data = await resp.json();
    loadingTr.remove();

    if (!resp.ok || data.error) {
      setMainVideoRowClipsCell(mainTr, "—");
      appendSubclipColspanRow(
        mainTr,
        "video-subclip-status-row",
        data.error || "Could not load subclips.",
        "video-subclips-empty",
      );
      updateSubclipSelectAllState();
      return;
    }

    const clips = data.clips || [];
    setMainVideoRowClipsCell(mainTr, String(clips.length));
    if (!clips.length) {
      updateSubclipSelectAllState();
      return;
    }

    let insertAfter = mainTr;
    for (const c of clips) {
      const row = buildVideoSubclipRow(parentPath, videoEntry, c);
      insertAfter.after(row);
      insertAfter = row;
    }
    enableVideoSubclipFolding(mainTr, parentPath);
    updateSubclipSelectAllState();
  } catch (_e) {
    loadingTr.remove();
    setMainVideoRowClipsCell(mainTr, "—");
    appendSubclipColspanRow(mainTr, "video-subclip-status-row", "Failed to load subclips.", "video-subclips-empty");
    updateSubclipSelectAllState();
  }
}

function setMetaStatus(msg) {
  metaModalStatus.textContent = msg || "";
}

function intOrNullInput(el) {
  const t = (el.value || "").trim();
  if (t === "") return null;
  const n = parseInt(t, 10);
  return Number.isFinite(n) ? n : null;
}

function floatOrNullInput(el) {
  const t = (el.value || "").trim();
  if (t === "") return null;
  const n = parseFloat(t);
  return Number.isFinite(n) ? n : null;
}

function displayOpt(n) {
  if (n === null || n === undefined) return "";
  return String(n);
}

async function fetchVideoInfo(videoPath) {
  const resp = await fetch(`/api/video_info_by_path?video_path=${encodeURIComponent(videoPath)}`);
  const data = await resp.json();
  if (!resp.ok || data.error) {
    throw new Error(data.error || "Failed to read video");
  }
  const fps = typeof data.fps === "number" && Number.isFinite(data.fps) ? data.fps : 0;
  const frame_count =
    typeof data.frame_count === "number" && Number.isFinite(data.frame_count)
      ? data.frame_count
      : parseInt(data.frame_count, 10) || 0;
  let w = null;
  let h = null;
  if (data.width != null && data.width !== "") {
    const nw = Number(data.width);
    if (Number.isFinite(nw) && nw > 0) w = nw;
  }
  if (data.height != null && data.height !== "") {
    const nh = Number(data.height);
    if (Number.isFinite(nh) && nh > 0) h = nh;
  }
  return { frame_count, fps, width: w, height: h };
}

async function fetchVideoFrameCount(videoPath) {
  const info = await fetchVideoInfo(videoPath);
  return info.frame_count ?? 0;
}

function asPlaybackFrame(v) {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

/**
 * Map 1-based inclusive frame indices to [t0, t1] seconds (clip speed CSV uses same convention).
 * Returns null when both bounds are unspecified (play entire file).
 */
function computePlaybackTimes(duration, fpsApi, totalFrames, frameStart, frameEnd) {
  if (!Number.isFinite(duration) || duration <= 0) return null;
  const fs = asPlaybackFrame(frameStart);
  let fe = asPlaybackFrame(frameEnd);
  if (fe != null && fe < 1) fe = null;
  if (fs == null && fe == null) return null;

  let effFps = fpsApi > 1e-6 ? fpsApi : null;
  if (effFps == null && totalFrames > 0 && duration > 0) effFps = totalFrames / duration;
  if (effFps == null || effFps <= 1e-6) effFps = 30;

  const fStart = fs != null ? fs : 1;
  const t0 = Math.max(0, (fStart - 1) / effFps);
  let t1;
  if (fe != null) {
    t1 = fe / effFps;
  } else {
    t1 = duration;
  }
  t1 = Math.min(duration, Math.max(t0 + 1 / effFps * 0.05, t1));
  if (t1 - t0 < 1e-3) return null;
  const fullSpan = duration <= 1e-6 ? true : t0 <= 1e-3 && t1 >= duration - 1e-3;
  if (fullSpan) return null;
  return { t0, t1, effFps };
}

function clearVideoPlaybackGuards() {
  if (videoPlaybackTimeHandler) {
    videoModalPlayer.removeEventListener("timeupdate", videoPlaybackTimeHandler);
    videoPlaybackTimeHandler = null;
  }
}

function attachPlaybackWindow(t0, t1) {
  clearVideoPlaybackGuards();
  const dur = Number.isFinite(videoModalPlayer.duration) ? videoModalPlayer.duration : t1;
  const start = Math.min(Math.max(0, t0), dur);
  const end = Math.min(Math.max(start + 1e-4, t1), dur);
  videoModalPlayer.currentTime = start;
  videoPlaybackTimeHandler = () => {
    const ct = videoModalPlayer.currentTime;
    if (!Number.isFinite(ct)) return;
    if (ct >= end - 0.02) {
      videoModalPlayer.pause();
      try {
        videoModalPlayer.currentTime = Math.min(end, dur);
      } catch (_e) {
        /* ignore */
      }
    }
  };
  videoModalPlayer.addEventListener("timeupdate", videoPlaybackTimeHandler);
}

/** Pause at endSec when Media Fragments handle start but not stop (or no fragment). */
function attachPlaybackEndPause(endSec, fullDuration) {
  clearVideoPlaybackGuards();
  const dur = Number.isFinite(fullDuration) ? fullDuration : endSec;
  const end = Math.min(Math.max(endSec, 0), dur);
  videoPlaybackTimeHandler = () => {
    const ct = videoModalPlayer.currentTime;
    if (!Number.isFinite(ct)) return;
    if (ct >= end - 0.025) {
      videoModalPlayer.pause();
      try {
        videoModalPlayer.currentTime = Math.min(end, dur);
      } catch (_e) {
        /* ignore */
      }
    }
  };
  videoModalPlayer.addEventListener("timeupdate", videoPlaybackTimeHandler);
}

function fillMetaFormFromRecord(rec) {
  const p = videoRecordPath(rec);
  const abs = typeof rec === "object" && rec && rec.absolute_path ? rec.absolute_path : p;
  videoMetaModalTitle.textContent = p.split(/[\\/]/).pop() || "Video metadata";
  videoMetaPathLabel.textContent = abs;
  metaDiskPixel.value = displayOpt(rec.disk_pixel);
  metaDiskRadiusMm.value = displayOpt(rec.disk_radius_mm);
  metaFrameStart.value = displayOpt(rec.frame_start);
  metaFrameEnd.value = displayOpt(rec.frame_end);
  metaFlyCount.value = displayOpt(rec.fly_count);
  metaSplitX.value = displayOpt(rec.split_x);
  metaSplitY.value = displayOpt(rec.split_y);
  metaDetailedLocation.value = rec.detailed_location || "";
  metaTotalFrames.value = rec.total_frames != null ? String(rec.total_frames) : "";
  if (metaVideoWidth) metaVideoWidth.value = rec.video_width != null ? String(rec.video_width) : "";
  if (metaVideoHeight) metaVideoHeight.value = rec.video_height != null ? String(rec.video_height) : "";
}

function closeMetaModal() {
  editingVideoPath = "";
  videoMetaModal.classList.add("hidden");
  setMetaStatus("");
}

function openMetaModal(rec) {
  const p = videoRecordPath(rec);
  if (!p) return;
  editingVideoPath = p;
  const base = typeof rec === "object" && rec ? rec : { path: p };
  fillMetaFormFromRecord(base);
  videoMetaModal.classList.remove("hidden");
  setMetaStatus("");
  if (
    base.total_frames == null
    || base.video_width == null
    || base.video_height == null
  ) {
    void detectVideoStreamMeta();
  }
}

function updateVideoSelectAllState() {
  const boxes = videoList.querySelectorAll(".video-select-cb");
  const list = [...boxes];
  const n = list.length;
  const checked = list.filter((c) => c.checked).length;
  videoSelectAll.checked = n > 0 && checked === n;
  videoSelectAll.indeterminate = checked > 0 && checked < n;
}

function updateSubclipSelectAllState() {
  if (!videoSubclipSelectAll) return;
  const boxes = videoList.querySelectorAll(".video-subclip-select-cb");
  const list = [...boxes];
  const n = list.length;
  if (n === 0) {
    videoSubclipSelectAll.checked = false;
    videoSubclipSelectAll.indeterminate = false;
    return;
  }
  const checked = list.filter((c) => c.checked).length;
  videoSubclipSelectAll.checked = checked === n;
  videoSubclipSelectAll.indeterminate = checked > 0 && checked < n;
}

/** Targets for snapshot batch: main videos + subclips (with csv + clip id). */
function collectSnapshotBatchItems() {
  const items = [];
  for (const c of videoList.querySelectorAll(".video-select-cb:checked")) {
    const p = c.dataset.path;
    if (p) items.push({ type: "video", video_path: p });
  }
  for (const c of videoList.querySelectorAll(".video-subclip-select-cb:checked")) {
    const vp = c.dataset.parentPath;
    const cid = c.dataset.clipId;
    const src = (c.dataset.sourceCsv || "").trim();
    if (!vp || !cid) continue;
    if (!src) continue;
    const n = parseInt(cid, 10);
    if (!Number.isFinite(n)) continue;
    items.push({
      type: "subclip",
      video_path: vp,
      source_csv: src,
      clip_id: n,
      frame_start: c.dataset.frameStart ? parseInt(c.dataset.frameStart, 10) : null,
      frame_end: c.dataset.frameEnd ? parseInt(c.dataset.frameEnd, 10) : null,
    });
  }
  return items;
}

/** Checked main video rows only (for delete / operations that remove whole videos). */
function collectSelectedMainVideoPaths() {
  return [
    ...new Set(
      [...videoList.querySelectorAll(".video-select-cb:checked")]
        .map((c) => c.dataset.path)
        .filter(Boolean),
    ),
  ];
}

/** Parent video paths from checked main rows and checked subclip rows (deduped). */
function collectSelectedVideoPathsUnion() {
  const fromMain = collectSelectedMainVideoPaths();
  const fromSubclips = [...videoList.querySelectorAll(".video-subclip-select-cb:checked")]
    .map((c) => c.dataset.parentPath)
    .filter(Boolean);
  return [...new Set([...fromMain, ...fromSubclips])];
}

/** Checked subclip rows for delete (clip DB rows only; does not unregister parent videos). */
function collectSelectedSubclipsForDelete() {
  const out = [];
  for (const c of videoList.querySelectorAll(".video-subclip-select-cb:checked")) {
    const vp = c.dataset.parentPath;
    const cid = c.dataset.clipId;
    const src = (c.dataset.sourceCsv || "").trim();
    if (!vp || !cid || !src) continue;
    const n = parseInt(cid, 10);
    if (!Number.isFinite(n)) continue;
    out.push({ parentPath: vp, clipId: n, sourceCsv: src });
  }
  return out;
}

function resetBatchSelectHeaders() {
  videoSelectAll.checked = false;
  videoSelectAll.indeterminate = false;
  if (videoSubclipSelectAll) {
    videoSubclipSelectAll.checked = false;
    videoSubclipSelectAll.indeterminate = false;
  }
}

async function detectVideoStreamMeta() {
  if (!editingVideoPath) return;
  setMetaStatus("Detecting frames and resolution…");
  try {
    const info = await fetchVideoInfo(editingVideoPath);
    const fc = info.frame_count ?? 0;
    metaTotalFrames.value = fc > 0 ? String(fc) : "";
    if (metaVideoWidth) {
      metaVideoWidth.value = info.width != null ? String(info.width) : "";
    }
    if (metaVideoHeight) {
      metaVideoHeight.value = info.height != null ? String(info.height) : "";
    }
    const bits = [];
    if (fc > 0) bits.push(`${fc} frames`);
    if (info.width != null && info.height != null) bits.push(`${info.width}×${info.height} px`);
    setMetaStatus(bits.length ? `Detected: ${bits.join(" · ")}.` : "Could not read frame count or resolution.");
  } catch (err) {
    setMetaStatus(err.message || "Detection failed.");
  }
}

function clearProjectDragUi() {
  dragProjectName = null;
  if (!projectList) return;
  for (const el of projectList.querySelectorAll(".project-row.drag-over, .project-row.dragging")) {
    el.classList.remove("drag-over", "dragging");
  }
}

function reorderProjectsLocal(fromName, toName) {
  const names = projects.map((x) => x.name);
  const fromI = names.indexOf(fromName);
  const toI = names.indexOf(toName);
  if (fromI < 0 || toI < 0 || fromI === toI) return null;
  const next = [...projects];
  const [item] = next.splice(fromI, 1);
  next.splice(toI, 0, item);
  return next;
}

function renderProjectList() {
  projectList.innerHTML = "";
  clearProjectDragUi();
  if (!projects.length) {
    projectList.textContent = "No projects yet.";
    return;
  }
  for (const p of projects) {
    const row = document.createElement("div");
    row.className = "project-row" + (p.name === selectedProject ? " selected" : "");
    row.dataset.projectName = p.name;

    const dragHandle = document.createElement("span");
    dragHandle.className = "project-drag-handle";
    dragHandle.textContent = "⋮⋮";
    dragHandle.title = "Drag to reorder (top = first in list)";
    dragHandle.draggable = true;
    dragHandle.addEventListener("dragstart", (e) => {
      dragProjectName = p.name;
      row.classList.add("dragging");
      try {
        e.dataTransfer.setData("text/plain", p.name);
        e.dataTransfer.effectAllowed = "move";
      } catch (_err) {
        /* ignore */
      }
    });
    dragHandle.addEventListener("dragend", () => {
      clearProjectDragUi();
    });

    row.addEventListener("dragover", (e) => {
      if (!dragProjectName || dragProjectName === p.name) return;
      e.preventDefault();
      try {
        e.dataTransfer.dropEffect = "move";
      } catch (_err) {
        /* ignore */
      }
      for (const el of projectList.querySelectorAll(".project-row.drag-over")) {
        el.classList.remove("drag-over");
      }
      row.classList.add("drag-over");
    });

    row.addEventListener("dragleave", (e) => {
      if (row.contains(e.relatedTarget)) return;
      row.classList.remove("drag-over");
    });

    row.addEventListener("drop", async (e) => {
      e.preventDefault();
      const from =
        dragProjectName ||
        (() => {
          try {
            return (e.dataTransfer.getData("text/plain") || "").trim();
          } catch (_err) {
            return "";
          }
        })();
      row.classList.remove("drag-over");
      if (!from || from === p.name) {
        clearProjectDragUi();
        return;
      }
      const next = reorderProjectsLocal(from, p.name);
      if (!next) {
        clearProjectDragUi();
        return;
      }
      const order = next.map((x) => x.name);
      try {
        const r = await req("/api/projects/reorder", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ order }),
        });
        projects = r.projects || next;
        setStatus("Project order saved.");
      } catch (err) {
        setStatus(err.message || "Could not save project order.");
        await refreshProjects();
      } finally {
        clearProjectDragUi();
        renderProjectList();
      }
    });

    const nameBtn = document.createElement("button");
    nameBtn.className = "project-name-btn";
    nameBtn.textContent = `${p.name} (${p.video_count || 0} videos)`;
    nameBtn.onclick = () => loadProjectDetail(p.name);

    const delBtn = document.createElement("button");
    delBtn.className = "project-del-btn";
    delBtn.textContent = "Delete";
    delBtn.onclick = async () => {
      const ok = window.confirm(`Delete project "${p.name}"?`);
      if (!ok) return;
      try {
        await req("/api/projects", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: p.name }),
        });
        if (selectedProject === p.name) {
          selectedProject = "";
          renderProjectDetail(null);
        }
        await refreshProjects();
        setStatus(`Deleted project: ${p.name}`);
      } catch (err) {
        setStatus(err.message);
      }
    };

    const metaBtn = document.createElement("button");
    metaBtn.className = "project-meta-btn";
    metaBtn.type = "button";
    metaBtn.textContent = "Meta";
    metaBtn.onclick = async () => {
      try {
        const r = await req(`/api/project?name=${encodeURIComponent(p.name)}`);
        selectedProject = p.name;
        renderProjectList();
        renderProjectDetail(r.project);
        openProjectMetaModal(r.project);
      } catch (err) {
        setStatus(err.message);
      }
    };

    row.appendChild(dragHandle);
    row.appendChild(nameBtn);
    row.appendChild(metaBtn);
    row.appendChild(delBtn);
    projectList.appendChild(row);
  }
}

async function renderRegisteredVideoRows(project, { preserveSelection = false } = {}) {
  if (!project) return;
  const projName = project.name || "";
  if (gClipCountsForSortProject !== projName) {
    gClipCountsForSort = Object.create(null);
    gClipCountsForSortProject = projName;
  }
  const rawVideos = project.videos || [];
  const selectedMain = new Set();
  const selectedSubKeys = new Set();
  if (preserveSelection && videoList) {
    for (const c of videoList.querySelectorAll(".video-select-cb:checked")) {
      if (c.dataset.path) selectedMain.add(c.dataset.path);
    }
    for (const c of videoList.querySelectorAll(".video-subclip-select-cb:checked")) {
      const vp = c.dataset.parentPath;
      const cid = c.dataset.clipId;
      if (vp && cid) selectedSubKeys.add(`${vp}\t${cid}`);
    }
  }
  videoList.innerHTML = "";
  updateVideoColumnFoldHeaderState();
  updateVideoSortHeaderIndicators();
  const sorted = videoTableSort.key
    ? [...rawVideos].sort((a, b) =>
        compareVideoRows(a, b, videoTableSort.key, videoTableSort.dir),
      )
    : [...rawVideos];
  const subPromises = [];
  for (const v of sorted) {
    const path = videoRecordPath(v);
    const recObj = typeof v === "object" && v ? v : {};
    const tr = document.createElement("tr");
    tr.className = "video-row";
    tr.title = path;

    const tdCb = document.createElement("td");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.className = "video-select-cb";
    cb.dataset.path = path;
    cb.addEventListener("change", updateVideoSelectAllState);
    tdCb.appendChild(cb);

    const tdName = document.createElement("td");
    tdName.className = "col-name-cell";
    const nameInner = document.createElement("div");
    nameInner.className = "video-name-cell-inner";
    const nameSpan = document.createElement("span");
    nameSpan.className = "video-name-text";
    nameSpan.textContent = path.split(/[\\/]/).pop() || path;
    nameInner.appendChild(nameSpan);
    tdName.appendChild(nameInner);

    const tdClips = mkVideoClipsTdMainPlaceholder();

    const tdSnap = mkVideoSnapTdPlaceholder();

    const tdSplit = document.createElement("td");
    tdSplit.className = "col-split-cell";
    tdSplit.textContent = videoRowSplitText(recObj);

    const tdAct = document.createElement("td");
    tdAct.className = "col-actions-cell";
    fillVideoActionsTd(tdAct, path, v);

    tr.dataset.videoPath = path;
    tr.appendChild(tdCb);
    tr.appendChild(tdName);
    tr.appendChild(tdClips);
    tr.appendChild(tdSnap);
    tr.appendChild(mkVideoNumTd(recObj.disk_pixel));
    tr.appendChild(mkVideoNumTd(recObj.disk_radius_mm));
    tr.appendChild(mkVideoNumTd(recObj.frame_start));
    tr.appendChild(mkVideoNumTd(recObj.frame_end));
    tr.appendChild(mkVideoNumTd(recObj.fly_count));
    tr.appendChild(tdSplit);
    tr.appendChild(tdAct);
    videoList.appendChild(tr);

    subPromises.push(loadVideoSubclipsAfterMainRow(tr, path, v));
  }
  if (preserveSelection && selectedMain.size) {
    for (const c of videoList.querySelectorAll(".video-select-cb")) {
      if (c.dataset.path && selectedMain.has(c.dataset.path)) c.checked = true;
    }
  }
  await Promise.all(subPromises);
  if (preserveSelection && selectedSubKeys.size) {
    for (const c of videoList.querySelectorAll(".video-subclip-select-cb")) {
      const vp = c.dataset.parentPath;
      const cid = c.dataset.clipId;
      if (vp && cid && selectedSubKeys.has(`${vp}\t${cid}`)) c.checked = true;
    }
  }
  updateVideoSelectAllState();
  updateSubclipSelectAllState();
  updateVideoColumnFoldHeaderState();
  await loadSnapshotLabelCounts(project.name);
}

function renderProjectDetail(project) {
  cachedProjectDetail = project || null;
  if (!project) {
    detailTitle.textContent = "No project selected";
    detailLabInfo.value = "";
    detailLabInfo.disabled = true;
    saveLabInfoBtn.disabled = true;
    videoBatchBar.classList.add("hidden");
    videoTableWrap.classList.add("hidden");
    videoListEmpty.classList.remove("hidden");
    videoListEmpty.textContent = "No videos.";
    resetBatchSelectHeaders();
    videoList.innerHTML = "";
    updateVideoColumnFoldHeaderState();
    updateVideoSortHeaderIndicators();
    return;
  }
  detailTitle.textContent = `Project: ${project.name}`;
  detailLabInfo.value = project.lab_info || "";
  detailLabInfo.disabled = false;
  saveLabInfoBtn.disabled = false;
  const videos = project.videos || [];
  if (!videos.length) {
    videoBatchBar.classList.add("hidden");
    videoTableWrap.classList.add("hidden");
    videoListEmpty.classList.remove("hidden");
    videoListEmpty.textContent = "No videos.";
    resetBatchSelectHeaders();
    videoList.innerHTML = "";
    updateVideoColumnFoldHeaderState();
    updateVideoSortHeaderIndicators();
  } else {
    videoBatchBar.classList.remove("hidden");
    videoTableWrap.classList.remove("hidden");
    videoListEmpty.classList.add("hidden");
    resetBatchSelectHeaders();
    void renderRegisteredVideoRows(project);
  }
}

function videoModalIsFullscreen() {
  const fs =
    document.fullscreenElement
    || document.webkitFullscreenElement
    || document.mozFullScreenElement
    || document.msFullscreenElement;
  return fs === videoModalStage || fs === videoModalPlayer;
}

function syncVideoFullscreenButtonLabel() {
  if (!videoFullscreenBtn) return;
  videoFullscreenBtn.textContent = videoModalIsFullscreen() ? "Exit fullscreen" : "Fullscreen";
  resizeVideoMeasureCanvas();
}

async function toggleVideoModalFullscreen() {
  const stage = videoModalStage;
  const v = videoModalPlayer;
  const target = stage || v;
  try {
    if (videoModalIsFullscreen()) {
      if (document.exitFullscreen) await document.exitFullscreen();
      else if (document.webkitExitFullscreen) await document.webkitExitFullscreen();
      else if (document.mozCancelFullScreen) await document.mozCancelFullScreen();
      else if (document.msExitFullscreen) await document.msExitFullscreen();
    } else if (target.requestFullscreen) {
      await target.requestFullscreen();
    } else if (target.webkitRequestFullscreen) {
      target.webkitRequestFullscreen();
    } else if (target.mozRequestFullScreen) {
      target.mozRequestFullScreen();
    } else if (target.msRequestFullscreen) {
      target.msRequestFullscreen();
    } else if (v.webkitEnterFullscreen) {
      v.webkitEnterFullscreen();
    }
  } catch (_err) {
    /* ignore */
  }
  syncVideoFullscreenButtonLabel();
}

async function openVideoModal(videoPath, opts = {}) {
  clearVideoPlaybackGuards();
  resetVideoMeasure();
  const name = String(videoPath).split(/[\\/]/).pop() || "Video";
  const suffix = opts.labelSuffix ? ` · ${opts.labelSuffix}` : "";
  videoModalTitle.textContent = `${name}${suffix}`;
  videoModalPlayer.pause();
  const baseSrc = `/api/video_file_by_path?video_path=${encodeURIComponent(videoPath)}`;
  videoModalPlayer.src = baseSrc;
  videoModal.classList.remove("hidden");
  syncVideoFullscreenButtonLabel();

  const frameStart = opts.frameStart != null ? opts.frameStart : null;
  const frameEnd = opts.frameEnd != null ? opts.frameEnd : null;

  videoModalPlayer.onloadedmetadata = () => {
    void (async () => {
      resizeVideoMeasureCanvas();
      try {
        const info = await fetchVideoInfo(videoPath);
        const duration = videoModalPlayer.duration;
        const metaTf =
          opts.cachedTotalFrames != null && Number.isFinite(Number(opts.cachedTotalFrames))
            ? Number(opts.cachedTotalFrames)
            : null;
        const tf =
          info.frame_count > 0 ? info.frame_count : metaTf != null && metaTf > 0 ? metaTf : null;
        const win = computePlaybackTimes(duration, info.fps, tf, frameStart, frameEnd);
        const srcHasFrag = String(videoModalPlayer.src || "").includes("#t=");

        if (win && !srcHasFrag) {
          videoModalPlayer.pause();
          videoModalPlayer.src = `${baseSrc}#t=${win.t0.toFixed(4)},${win.t1.toFixed(4)}`;
          videoModalPlayer.load();
          return;
        }

        if (win && srcHasFrag) {
          clearVideoPlaybackGuards();
          requestAnimationFrame(() => {
            const ct = videoModalPlayer.currentTime;
            const startedInClip =
              Number.isFinite(ct) && ct >= win.t0 - 0.35 && ct <= win.t1 + 0.35;
            if (startedInClip) {
              attachPlaybackEndPause(win.t1, duration);
            } else {
              attachPlaybackWindow(win.t0, win.t1);
            }
          });
        } else if (!win) {
          clearVideoPlaybackGuards();
          videoModalPlayer.currentTime = 0;
        }
      } catch (_err) {
        clearVideoPlaybackGuards();
        videoModalPlayer.currentTime = 0;
      }
      resizeVideoMeasureCanvas();
      videoModalPlayer.play().catch(() => {});
    })();
  };

  requestAnimationFrame(() => resizeVideoMeasureCanvas());
}

function closeVideoModal() {
  if (videoModalIsFullscreen()) {
    if (document.exitFullscreen) document.exitFullscreen().catch(() => {});
    else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
    else if (document.mozCancelFullScreen) document.mozCancelFullScreen();
    else if (document.msExitFullscreen) document.msExitFullscreen();
  }
  clearVideoPlaybackGuards();
  resetVideoMeasure();
  videoModalPlayer.onloadedmetadata = null;
  videoModalPlayer.pause();
  videoModalPlayer.removeAttribute("src");
  videoModalPlayer.load();
  videoModal.classList.add("hidden");
  syncVideoFullscreenButtonLabel();
}

async function loadProjectDetail(name) {
  try {
    if (!cachedProjectDetail || cachedProjectDetail.name !== name) {
      videoTableSort = { key: null, dir: "asc" };
    }
    const r = await req(`/api/project?name=${encodeURIComponent(name)}`);
    selectedProject = name;
    renderProjectList();
    renderProjectDetail(r.project);
  } catch (err) {
    setStatus(err.message);
  }
}

async function refreshProjects() {
  const r = await req("/api/projects");
  projects = r.projects || [];
  if (!projects.length) {
    selectedProject = "";
    renderProjectList();
    renderProjectDetail(null);
    return;
  }
  const names = new Set(projects.map((p) => p.name));
  if (!selectedProject || !names.has(selectedProject)) {
    await loadProjectDetail(projects[0].name);
    return;
  }
  renderProjectList();
}

createProjectBtn.onclick = async () => {
  const name = (projectNameInput.value || "").trim();
  const lab_info = (labInfoInput.value || "").trim();
  if (!name) {
    setStatus("Project name is required.");
    return;
  }
  try {
    await req("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, lab_info }),
    });
    projectNameInput.value = "";
    labInfoInput.value = "";
    await refreshProjects();
    await loadProjectDetail(name);
    setStatus(`Created project: ${name}`);
  } catch (err) {
    setStatus(err.message);
  }
};

saveLabInfoBtn.onclick = async () => {
  if (!selectedProject) return;
  try {
    await req("/api/project/lab_info", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: selectedProject,
        lab_info: (detailLabInfo.value || "").trim(),
      }),
    });
    await refreshProjects();
    await loadProjectDetail(selectedProject);
    setStatus(`Saved lab info for: ${selectedProject}`);
  } catch (err) {
    setStatus(err.message);
  }
};

addDirBtn.onclick = async () => {
  if (!selectedProject) {
    setStatus("Select a project first.");
    return;
  }
  const directory = (videoDirInput.value || "").trim();
  if (!directory) {
    setStatus("Directory path is required.");
    return;
  }
  try {
    const r = await req("/api/project/videos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: selectedProject, mode: "directory", directory }),
    });
    await refreshProjects();
    renderProjectDetail(r.project);
    setStatus(`Added ${r.added} video(s) from directory.`);
  } catch (err) {
    setStatus(err.message);
  }
};

addPathsBtn.onclick = async () => {
  if (!selectedProject) {
    setStatus("Select a project first.");
    return;
  }
  const lines = (videoPathsInput.value || "")
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (!lines.length) {
    setStatus("Enter at least one video path.");
    return;
  }
  try {
    const r = await req("/api/project/videos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: selectedProject, mode: "paths", paths: lines }),
    });
    await refreshProjects();
    renderProjectDetail(r.project);
    setStatus(`Added ${r.added} video(s) by paths.`);
  } catch (err) {
    setStatus(err.message);
  }
};

refreshProjectsBtn.onclick = () => {
  refreshProjects().catch((err) => setStatus(err.message));
};

videoSelectAll.onchange = () => {
  const on = videoSelectAll.checked;
  videoList.querySelectorAll(".video-select-cb").forEach((c) => {
    c.checked = on;
  });
  videoSelectAll.indeterminate = false;
};

if (videoFoldAllBtn) {
  videoFoldAllBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleAllSubclipsFold();
  });
}
(function bindRegisteredVideoTableSort() {
  const thead = document.querySelector(".video-table thead");
  if (!thead) return;
  thead.addEventListener("click", (e) => {
    const th = e.target.closest("th[data-video-sort]");
    if (!th) return;
    if (e.target.closest("button, input, a, label")) return;
    const key = th.getAttribute("data-video-sort");
    if (!key) return;
    const proj = cachedProjectDetail;
    if (!proj || !(proj.videos || []).length) return;
    if (videoTableSort.key === key) {
      videoTableSort.dir = videoTableSort.dir === "asc" ? "desc" : "asc";
    } else {
      videoTableSort.key = key;
      videoTableSort.dir = "asc";
    }
    updateVideoSortHeaderIndicators();
    void renderRegisteredVideoRows(proj, { preserveSelection: true });
  });
})();
updateVideoColumnFoldHeaderState();

if (videoSubclipSelectAll) {
  videoSubclipSelectAll.onchange = () => {
    const boxes = videoList.querySelectorAll(".video-subclip-select-cb");
    if (!boxes.length) {
      videoSubclipSelectAll.checked = false;
      videoSubclipSelectAll.indeterminate = false;
      return;
    }
    const on = videoSubclipSelectAll.checked;
    boxes.forEach((c) => {
      c.checked = on;
    });
    videoSubclipSelectAll.indeterminate = false;
  };
}

batchDeleteVideosBtn.onclick = async () => {
  if (!selectedProject) return;
  const mainPaths = collectSelectedMainVideoPaths();
  const subclips = collectSelectedSubclipsForDelete();
  if (!mainPaths.length && !subclips.length) {
    setStatus("Select at least one video or subclip to delete.");
    return;
  }
  const parts = [];
  if (subclips.length) {
    parts.push(
      `delete ${subclips.length} saved subclip(s) (parent videos stay registered)`,
    );
  }
  if (mainPaths.length) {
    parts.push(`remove ${mainPaths.length} video(s) from the project`);
  }
  const ok = window.confirm(
    `${parts.join(" and ")}? Nothing is deleted from disk.`,
  );
  if (!ok) return;
  try {
    let removedClips = 0;
    for (const sc of subclips) {
      const url = `/api/total_speed_clips/${encodeURIComponent(sc.clipId)}?path=${encodeURIComponent(sc.sourceCsv)}`;
      const resp = await fetch(url, { method: "DELETE" });
      const d = await resp.json().catch(() => ({}));
      if (!resp.ok || d.error) {
        throw new Error(d.error || `Could not delete subclip ${sc.clipId}`);
      }
      removedClips += 1;
    }
    let r;
    if (mainPaths.length) {
      r = await req("/api/project/videos", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: selectedProject, video_paths: mainPaths }),
      });
    } else {
      r = { removed: 0, project: (await req(`/api/project?name=${encodeURIComponent(selectedProject)}`)).project };
    }
    await refreshProjects();
    renderProjectDetail(r.project);
    const bits = [];
    if (removedClips) bits.push(`deleted ${removedClips} subclip(s)`);
    if (mainPaths.length && r.removed != null) bits.push(`removed ${r.removed} video(s) from project`);
    setStatus(bits.length ? bits.join(" · ") : "Done.");
  } catch (err) {
    setStatus(err.message);
  }
};

quickRunFastviewBtn.onclick = () => {
  if (!selectedProject) {
    setStatus("Select a project first.");
    return;
  }
  if (!quickRunModal) {
    setStatus("QuickRun dialog is unavailable.");
    return;
  }
  const checked = collectSelectedVideoPathsUnion();
  openQuickRunModal(checked);
};

if (snapshotBatchBtn) {
  snapshotBatchBtn.onclick = async () => {
    if (!selectedProject) {
      setStatus("Select a project first.");
      return;
    }
    const items = collectSnapshotBatchItems();
    if (items === null) return;
    if (!items.length) {
      setStatus("Select at least one video and/or subclip for snapshot batch.");
      return;
    }
    setStatus("Starting snapshot batch (queued like QuickRun)…");
    try {
      const r = await req("/api/project/snapshot_batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: selectedProject, items }),
      });
      const dest = r.quickrun_url || `/quickrun?session=${encodeURIComponent(r.session_id)}`;
      window.location.href = dest;
    } catch (err) {
      setStatus(err.message || "Snapshot batch failed.");
    }
  };
}

if (trackingBatchBtn) {
  trackingBatchBtn.onclick = () => {
    if (!selectedProject) {
      setStatus("Select a project first.");
      return;
    }
    if (!trackingBatchModal) {
      setStatus("Tracking dialog is unavailable.");
      return;
    }
    const items = collectSnapshotBatchItems();
    if (items === null) return;
    if (!items.length) {
      setStatus("Select at least one video and/or subclip for tracking batch.");
      return;
    }
    openTrackingBatchModal(items);
  };
}

if (trackingBatchStartBtn) {
  trackingBatchStartBtn.onclick = async () => {
    if (!selectedProject) return;
    const runTrackingBatch = async (allowMissingInit) => {
      const resp = await fetch("/api/project/tracking_batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectTrackingBatchPayload(allowMissingInit)),
      });
      const data = await resp.json();
      return { resp, data };
    };

    if (trackingBatchModalStatus) {
      trackingBatchModalStatus.textContent = "Starting tracking batch…";
    } else {
      setStatus("Starting tracking batch…");
    }
    try {
      let { resp, data } = await runTrackingBatch(null);
      if (resp.status === 409 && data && data.missing_count) {
        const ok = window.confirm(
          `${data.missing_count} selected target(s) are missing snapshot labels for tracking start.\n\nContinue and fallback to model output for those targets?`,
        );
        if (!ok) {
          if (trackingBatchModalStatus) trackingBatchModalStatus.textContent = "Tracking batch canceled.";
          else setStatus("Tracking batch canceled.");
          return;
        }
        ({ resp, data } = await runTrackingBatch(true));
      }
      if (!resp.ok || data.error) {
        throw new Error(data.error || `Request failed: ${resp.status}`);
      }
      closeTrackingBatchModal();
      const dest = data.quickrun_url || `/quickrun?session=${encodeURIComponent(data.session_id)}`;
      window.location.href = dest;
    } catch (err) {
      if (trackingBatchModalStatus) trackingBatchModalStatus.textContent = err.message || "Tracking batch failed.";
      else setStatus(err.message || "Tracking batch failed.");
    }
  };
}

if (quickRunStartBtn) {
  quickRunStartBtn.onclick = async () => {
    if (!selectedProject) return;
    if (quickRunModalStatus) quickRunModalStatus.textContent = "Starting…";
    try {
      const body = collectQuickRunPayload();
      const r = await req("/api/quickrun/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      closeQuickRunModal();
      const dest = r.quickrun_url || `/quickrun?session=${encodeURIComponent(r.session_id)}`;
      window.location.href = dest;
    } catch (err) {
      if (quickRunModalStatus) quickRunModalStatus.textContent = err.message || "Failed.";
      else setStatus(err.message);
    }
  };
}

if (quickRunResetDefaultsBtn) {
  quickRunResetDefaultsBtn.onclick = () => applyQuickRunDefaultsToForm();
}

if (trackingBatchResetDefaultsBtn) {
  trackingBatchResetDefaultsBtn.onclick = () => applyTrackingBatchDefaultsToForm();
}

if (closeQuickRunModalBtn) {
  closeQuickRunModalBtn.onclick = closeQuickRunModal;
}

if (closeTrackingBatchModalBtn) {
  closeTrackingBatchModalBtn.onclick = closeTrackingBatchModal;
}

if (quickRunModal) {
  quickRunModal.onclick = (e) => {
    if (e.target === quickRunModal) closeQuickRunModal();
  };
}

if (trackingBatchModal) {
  trackingBatchModal.onclick = (e) => {
    if (e.target === trackingBatchModal) closeTrackingBatchModal();
  };
}

importVideoMetaTsvBtn.onclick = async () => {
  if (!selectedProject) {
    setStatus("Select a project first.");
    return;
  }
  const tsv_path = (videoMetaTsvPath.value || "").trim();
  if (!tsv_path) {
    setStatus("Enter the path to the TSV file.");
    return;
  }
  try {
    const r = await req("/api/project/import_video_meta_tsv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: selectedProject, tsv_path }),
    });
    await refreshProjects();
    renderProjectDetail(r.project);
    const unk = r.unknown_count ? `; ${r.unknown_count} row(s) had no matching video by filename` : "";
    setStatus(`TSV import: updated ${r.entries_updated} video entr(y/ies)${unk}.`);
  } catch (err) {
    setStatus(err.message);
  }
};

exportVideoMetaTsvBtn.onclick = async () => {
  if (!selectedProject) {
    setStatus("Select a project first.");
    return;
  }
  const tsv_path = (videoMetaTsvPath.value || "").trim();
  if (!tsv_path) {
    setStatus("Enter the path for the TSV file to write.");
    return;
  }
  try {
    const r = await req("/api/project/export_video_meta_tsv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: selectedProject, tsv_path }),
    });
    await refreshProjects();
    setStatus(`Saved ${r.row_count} row(s) to ${r.written}`);
  } catch (err) {
    setStatus(err.message);
  }
};

closeVideoModalBtn.onclick = closeVideoModal;
if (videoMeasureToggleBtn) {
  videoMeasureToggleBtn.onclick = () => setVideoMeasureMode(!videoMeasureMode);
}
if (videoMeasureClearBtn) {
  videoMeasureClearBtn.onclick = () => {
    videoMeasurePoints = [];
    drawVideoMeasureOverlay();
  };
}
if (videoMeasureCanvas) {
  videoMeasureCanvas.addEventListener("mousedown", (e) => {
    if (!videoMeasureMode) return;
    const hit = findMeasureHitIndex(e.clientX, e.clientY);
    if (hit >= 0) {
      videoMeasureDragIdx = hit;
      e.preventDefault();
      return;
    }
    const nat = clientToNativeMeasure(e.clientX, e.clientY);
    if (nat) {
      videoMeasurePoints.push(nat);
      e.preventDefault();
      drawVideoMeasureOverlay();
    }
  });
  videoMeasureCanvas.addEventListener("mousemove", (e) => {
    if (!videoMeasureMode || videoMeasureDragIdx < 0) return;
    const nat = clientToNativeMeasure(e.clientX, e.clientY);
    if (nat) {
      videoMeasurePoints[videoMeasureDragIdx] = nat;
      drawVideoMeasureOverlay();
    }
  });
}
window.addEventListener("mouseup", () => {
  videoMeasureDragIdx = -1;
});
if (videoModalStage && typeof ResizeObserver !== "undefined") {
  new ResizeObserver(() => resizeVideoMeasureCanvas()).observe(videoModalStage);
}
videoFullscreenBtn.onclick = () => {
  toggleVideoModalFullscreen();
};
document.addEventListener("fullscreenchange", syncVideoFullscreenButtonLabel);
document.addEventListener("webkitfullscreenchange", syncVideoFullscreenButtonLabel);
document.addEventListener("mozfullscreenchange", syncVideoFullscreenButtonLabel);
document.addEventListener("MSFullscreenChange", syncVideoFullscreenButtonLabel);
videoModal.onclick = (e) => {
  if (e.target === videoModal) closeVideoModal();
};
closeVideoMetaModalBtn.onclick = closeMetaModal;
videoMetaModal.onclick = (e) => {
  if (e.target === videoMetaModal) closeMetaModal();
};
if (closeProjectMetaModalBtn) closeProjectMetaModalBtn.onclick = closeProjectMetaModal;
if (projectMetaModal) {
  projectMetaModal.onclick = (e) => {
    if (e.target === projectMetaModal) closeProjectMetaModal();
  };
}
if (projectMetaSaveBtn) {
  projectMetaSaveBtn.onclick = async () => {
    if (!selectedProject) return;
    setProjectMetaStatus("Saving…");
    try {
      const r = await req("/api/project/meta", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: selectedProject,
          currently: (projectCurrently && projectCurrently.value) || "",
          abstract: (projectAbstract && projectAbstract.value) || "",
          quickrun_output: (projectQuickrunOutput && projectQuickrunOutput.value) || "",
          snapshot_output: (projectSnapshotOutput && projectSnapshotOutput.value) || "",
          tracking_output: (projectTrackingOutput && projectTrackingOutput.value) || "traking",
        }),
      });
      await refreshProjects();
      renderProjectDetail(r.project);
      setProjectMetaStatus("Saved.");
      setTimeout(() => closeProjectMetaModal(), 400);
    } catch (err) {
      setProjectMetaStatus(err.message || "Save failed.");
    }
  };
}
metaDetectFramesBtn.onclick = () => {
  detectVideoStreamMeta();
};
metaSaveBtn.onclick = async () => {
  if (!selectedProject || !editingVideoPath) return;
  setMetaStatus("Saving…");
  try {
    const meta = {
      disk_pixel: floatOrNullInput(metaDiskPixel),
      disk_radius_mm: floatOrNullInput(metaDiskRadiusMm),
      frame_start: intOrNullInput(metaFrameStart),
      frame_end: intOrNullInput(metaFrameEnd),
      fly_count: intOrNullInput(metaFlyCount),
      detailed_location: (metaDetailedLocation.value || "").trim(),
      split_x: intOrNullInput(metaSplitX),
      split_y: intOrNullInput(metaSplitY),
    };
    if ((metaTotalFrames.value || "").trim() !== "") {
      meta.total_frames = intOrNullInput(metaTotalFrames);
    }
    if (metaVideoWidth && (metaVideoWidth.value || "").trim() !== "") {
      meta.video_width = intOrNullInput(metaVideoWidth);
    }
    if (metaVideoHeight && (metaVideoHeight.value || "").trim() !== "") {
      meta.video_height = intOrNullInput(metaVideoHeight);
    }
    const r = await req("/api/project/video_meta", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: selectedProject,
        video_path: editingVideoPath,
        meta,
      }),
    });
    await refreshProjects();
    renderProjectDetail(r.project);
    setMetaStatus("Saved.");
    setTimeout(() => closeMetaModal(), 400);
  } catch (err) {
    setMetaStatus(err.message || "Save failed.");
  }
};
if (closeTrackingOpenModalBtn) {
  closeTrackingOpenModalBtn.onclick = closeTrackingOpenModal;
}
if (trackingOpenConfirmBtn) {
  trackingOpenConfirmBtn.onclick = () => {
    if (trackingOpenPendingUrl) {
      window.open(trackingOpenPendingUrl, "_blank", "noopener");
    }
    closeTrackingOpenModal();
  };
}
if (trackingOpenModal) {
  trackingOpenModal.onclick = (e) => {
    if (e.target === trackingOpenModal) closeTrackingOpenModal();
  };
}

window.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (trackingOpenModal && !trackingOpenModal.classList.contains("hidden")) {
    closeTrackingOpenModal();
    return;
  }
  if (trackingBatchModal && !trackingBatchModal.classList.contains("hidden")) {
    closeTrackingBatchModal();
    return;
  }
  if (projectMetaModal && !projectMetaModal.classList.contains("hidden")) {
    closeProjectMetaModal();
    return;
  }
  if (!videoMetaModal.classList.contains("hidden")) {
    closeMetaModal();
    return;
  }
  if (quickRunModal && !quickRunModal.classList.contains("hidden")) {
    closeQuickRunModal();
    return;
  }
  if (!videoModal.classList.contains("hidden")) {
    closeVideoModal();
  }
});

refreshProjects().catch((err) => setStatus(err.message));
