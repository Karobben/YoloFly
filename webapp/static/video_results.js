const qs = new URLSearchParams(window.location.search);
const videoPath = (qs.get("video_path") || "").trim();
/** Canonical path from `/api/quickrun/results_for_video` once loaded; used for deep links. */
let linkVideoPath = videoPath;
const projectFilter = (qs.get("project") || "").trim();
const clipIdFilter = (qs.get("clip_id") || "").trim();

const vrError = document.getElementById("vrError");
const vrActionMsg = document.getElementById("vrActionMsg");
const vrEmpty = document.getElementById("vrEmpty");
const vrIntro = document.getElementById("vrIntro");
const vrVideoPath = document.getElementById("vrVideoPath");
const vrProjectRow = document.getElementById("vrProjectRow");
const vrProjectName = document.getElementById("vrProjectName");
const vrClipRow = document.getElementById("vrClipRow");
const vrClipId = document.getElementById("vrClipId");
const vrRuns = document.getElementById("vrRuns");

function setVrActionMsg(text, isErr) {
  if (!vrActionMsg) return;
  if (!text) {
    vrActionMsg.textContent = "";
    vrActionMsg.classList.add("hidden");
    vrActionMsg.classList.remove("vr-action-msg-err");
    return;
  }
  vrActionMsg.textContent = text;
  vrActionMsg.classList.remove("hidden");
  vrActionMsg.classList.toggle("vr-action-msg-err", !!isErr);
}

function buildDeleteSectionPayload(scope, resultsType, section) {
  const body = {
    video_path: linkVideoPath || videoPath,
    scope,
  };
  if (projectFilter) body.project = projectFilter;
  if (clipIdFilter) body.clip_id = clipIdFilter;
  if (scope === "all") {
    return body;
  }
  body.results_type = resultsType;
  if (section) body.section = section;
  return body;
}

async function postDeleteResultsSection(payload) {
  const r = await fetch("/api/quickrun/results_delete_section", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
  return d;
}

async function deleteQuickrunSession(sid) {
  const r = await fetch(`/api/quickrun/session/${encodeURIComponent(sid)}`, {
    method: "DELETE",
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
  return d;
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function localFileUrl(absPath) {
  return `/api/local_file?path=${encodeURIComponent(absPath)}`;
}

function csvTablePageUrl(absPath) {
  const q = new URLSearchParams({ path: absPath });
  if (linkVideoPath) q.set("video_path", linkVideoPath);
  if (projectFilter) q.set("project", projectFilter);
  if (clipIdFilter) q.set("clip_id", clipIdFilter);
  return `/csv-table?${q.toString()}`;
}

function isTabularCsvPath(absPath) {
  if (!absPath) return false;
  return /\.(csv|tsv)$/i.test(absPath);
}

function isTotalSpeedCsvPath(absPath) {
  return /\.total_speed\.csv$/i.test(String(absPath || ""));
}

function totalSpeedPlotPageUrl(absPath) {
  const q = new URLSearchParams({ path: absPath });
  if (linkVideoPath) q.set("video_path", linkVideoPath);
  if (projectFilter) q.set("project", projectFilter);
  if (clipIdFilter) q.set("clip_id", clipIdFilter);
  return `/total-speed-plot?${q.toString()}`;
}

function fmtTime(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
  } catch {
    return iso;
  }
}

function renderPathOnly(label, absPath, note = "") {
  if (!absPath) return "";
  const noteHtml = note ? ` <span class="vr-muted">${esc(note)}</span>` : "";
  return `
    <div class="vr-file-block">
      <div><strong>${esc(label)}</strong>${noteHtml}</div>
      <div class="vr-path"><code>${esc(absPath)}</code></div>
    </div>
  `;
}

function renderFileRow(label, absPath, opts = {}) {
  if (!absPath) return "";
  const exists = opts.exists !== false;
  const ext = (absPath.split(".").pop() || "").toLowerCase();
  const isImg = ["png", "jpg", "jpeg", "webp", "gif", "svg"].includes(ext);
  const href = localFileUrl(absPath);
  const miss = exists ? "" : ' <span class="vr-missing">(missing on disk)</span>';
  let preview = "";
  if (isImg && exists) {
    preview = `<div class="vr-img-wrap"><img src="${href}" alt="" loading="lazy" /></div>`;
  }
  const viewTable =
    exists && isTabularCsvPath(absPath)
      ? `<a href="${csvTablePageUrl(absPath)}">View table</a>`
      : "";
  const speedPlot =
    exists && isTotalSpeedCsvPath(absPath)
      ? `<a href="${totalSpeedPlotPageUrl(absPath)}">Interactive plot</a>`
      : "";
  const actions = [`<a href="${href}" target="_blank" rel="noopener">Open</a>`];
  if (viewTable) actions.push(viewTable);
  if (speedPlot) actions.push(speedPlot);
  return `
    <div class="vr-file-block">
      <div><strong>${esc(label)}</strong>${miss}</div>
      <div class="vr-path"><code>${esc(absPath)}</code></div>
      <div class="vr-file-actions">
        ${actions.join(" · ")}
      </div>
      ${preview}
    </div>
  `;
}

function provenanceLine(a) {
  const parts = [];
  if (a.project) parts.push(`project <strong>${esc(a.project)}</strong>`);
  if (a.finished_at) parts.push(`finished ${esc(fmtTime(a.finished_at))}`);
  if (a.session_id)
    parts.push(
      `session <code>${esc(String(a.session_id).slice(0, 8))}…</code>`,
    );
  if (!parts.length) return "";
  return `<p class="vr-muted vr-prov">${parts.join(" · ")}</p>`;
}

function pathExtension(absPath) {
  const m = String(absPath || "").match(/\.([^.\\/]+)$/);
  return m ? m[1].toLowerCase() : "";
}

const IMAGE_EXTENSIONS = new Set([
  "png",
  "jpg",
  "jpeg",
  "webp",
  "gif",
  "svg",
  "bmp",
]);

/** Section key for grouping output files by type */
function artifactSectionKey(a) {
  if (a.kind === "output_directory") return "directories";
  const ext = pathExtension(a.path);
  if (
    a.kind === "detection_csv" ||
    a.kind === "total_speed_csv" ||
    ext === "csv" ||
    ext === "tsv"
  ) {
    return "csv";
  }
  if (ext === "json" || a.kind === "tracked_json") return "json";
  if (ext === "pdf") return "other";
  if (IMAGE_EXTENSIONS.has(ext)) return "images";
  if (a.kind === "plot") return "images";
  return "other";
}

const SECTION_ORDER = ["csv", "images", "json", "directories", "other"];
const SECTION_TITLE = {
  csv: "CSV & TSV",
  images: "Images",
  json: "JSON",
  directories: "Directories",
  other: "Other files",
};

function groupArtifactsBySection(artifacts) {
  const buckets = Object.fromEntries(SECTION_ORDER.map((k) => [k, []]));
  for (const a of artifacts) {
    const key = artifactSectionKey(a);
    if (buckets[key]) buckets[key].push(a);
    else buckets.other.push(a);
  }
  for (const k of SECTION_ORDER) {
    buckets[k].sort((x, y) => String(x.path).localeCompare(String(y.path)));
  }
  return SECTION_ORDER.filter((k) => buckets[k].length > 0).map((k) => ({
    key: k,
    title: SECTION_TITLE[k],
    items: buckets[k],
  }));
}

/** Top-level grouping: Quick Run (FastView) vs Snapshot, then unknown kinds last. */
const RESULT_TYPE_ORDER_FIXED = ["quick_run", "snapshot"];

function artifactResultsTypeKey(a) {
  const sk = String(
    (a.pipeline_params && a.pipeline_params.session_kind) || "fastview",
  ).toLowerCase();
  if (sk === "snapshot") return "snapshot";
  if (sk === "fastview") return "quick_run";
  return sk;
}

function resultsTypeTitle(key) {
  if (key === "quick_run") return "Quick Run";
  if (key === "snapshot") return "Snapshot";
  if (!key) return "Other";
  const pretty = String(key).replace(/_/g, " ");
  return pretty.charAt(0).toUpperCase() + pretty.slice(1);
}

function sortResultsTypeKeys(keys) {
  const set = new Set(keys);
  const out = [];
  for (const k of RESULT_TYPE_ORDER_FIXED) {
    if (set.has(k)) out.push(k);
  }
  const rest = [...keys]
    .filter((k) => !RESULT_TYPE_ORDER_FIXED.includes(k))
    .sort((a, b) => String(a).localeCompare(String(b)));
  out.push(...rest);
  return out;
}

function detectExploreSnapshotUrl(absDir) {
  const u = new URLSearchParams({ snapshot_dir: absDir });
  return `/detect_explore?${u.toString()}`;
}

function detectExploreTrackingUrl(absDir) {
  const u = new URLSearchParams({ tracking_dir: absDir });
  const vp = (linkVideoPath || "").trim();
  if (vp) u.set("video_path", vp);
  return `/detect_explore?${u.toString()}`;
}

function isSnapshotOutputDirectory(a) {
  return (
    a.kind === "output_directory" &&
    String(a.label || "").toLowerCase().includes("snapshot")
  );
}

function isTrackingOutputDirectory(a) {
  const label = String(a.label || "").toLowerCase();
  return (
    a.kind === "output_directory" &&
    (label.includes("tracking") || label.includes("post-track"))
  );
}

function groupArtifactsByResultsType(artifacts) {
  const byType = new Map();
  for (const a of artifacts) {
    const k = artifactResultsTypeKey(a);
    if (!byType.has(k)) byType.set(k, []);
    byType.get(k).push(a);
  }
  return sortResultsTypeKeys([...byType.keys()]).map((key) => ({
    key,
    title: resultsTypeTitle(key),
    items: byType.get(key),
  }));
}

function renderArtifactBlock(a) {
  let body = "";
  if (a.kind === "output_directory") {
    body = renderPathOnly(
      a.label || "Output directory",
      a.path,
      "directory on disk",
    );
    if (isSnapshotOutputDirectory(a) && a.path) {
      const href = detectExploreSnapshotUrl(a.path);
      body += `<div class="vr-detect-explore-actions"><a class="vr-btn-detect-explore" href="${esc(href)}" target="_blank" rel="noopener">Open raw frame + labels in detect_explore</a></div>`;
    } else if (isTrackingOutputDirectory(a) && a.path) {
      const href = detectExploreTrackingUrl(a.path);
      body += `<div class="vr-detect-explore-actions"><a class="vr-btn-detect-explore" href="${esc(href)}" target="_blank" rel="noopener">Open Video View in detect_explore</a></div>`;
    }
  } else {
    body = renderFileRow(a.label || a.kind, a.path, { exists: a.exists });
  }
  return `<div class="vr-artifact">${body}${provenanceLine(a)}</div>`;
}

function renderImageArtifactCard(a) {
  const exists = a.exists !== false;
  const href = localFileUrl(a.path);
  const miss = exists
    ? ""
    : ' <span class="vr-missing">(missing)</span>';
  const preview = exists
    ? `<div class="vr-img-cell-preview"><img src="${href}" alt="" loading="lazy" /></div>`
    : `<div class="vr-img-cell-preview vr-img-cell-missing" aria-hidden="true">No preview</div>`;
  const label = esc(a.label || "Image");
  return `
    <div class="vr-img-cell">
      ${preview}
      <div class="vr-img-cell-meta">
        <div class="vr-img-cell-label"><strong>${label}</strong>${miss}</div>
        <div class="vr-img-cell-path"><code>${esc(a.path)}</code></div>
        <div class="vr-img-cell-actions">
          <a href="${href}" target="_blank" rel="noopener">Open</a>
        </div>
      </div>
      ${provenanceLine(a)}
    </div>`;
}

function renderFoldableSection(g, resultsTypeKey) {
  const count = g.items.length;
  const isImages = g.key === "images";
  const inner = isImages
    ? `<div class="vr-artifacts vr-artifacts-grid">${g.items.map(renderImageArtifactCard).join("")}</div>`
    : `<div class="vr-artifacts">${g.items.map(renderArtifactBlock).join("")}</div>`;
  const delBtn = `<button type="button" class="qr-btn-danger qr-btn-small vr-section-delete-btn" data-vr-del-scope="section" data-vr-results-type="${esc(resultsTypeKey)}" data-vr-section="${esc(g.key)}" title="Delete these files on disk and remove them from the results index">Delete</button>`;
  return `
    <details class="vr-type-fold" data-section="${esc(g.key)}" open>
      <summary class="vr-type-summary">
        <span class="vr-type-summary-label">${esc(g.title)}</span>
        <span class="vr-summary-end">
          <span class="vr-type-count">${count}</span>
          ${delBtn}
        </span>
      </summary>
      <div class="vr-type-body">${inner}</div>
    </details>`;
}

function renderFileTypeSections(artifacts, resultsTypeKey) {
  const groups = groupArtifactsBySection(artifacts);
  return groups.map((g) => renderFoldableSection(g, resultsTypeKey)).join("");
}

function renderArtifactSections(artifacts) {
  const typeGroups = groupArtifactsByResultsType(artifacts);
  return typeGroups
    .map((tg) => {
      const n = tg.items.length;
      const delBtn = `<button type="button" class="qr-btn-danger qr-btn-small vr-section-delete-btn" data-vr-del-scope="results_type" data-vr-results-type="${esc(tg.key)}" title="Delete every output listed under ${esc(tg.title)} (all file types)">Delete all</button>`;
      return `
    <details class="vr-result-type-fold" data-results-type="${esc(tg.key)}" open>
      <summary class="vr-type-summary vr-result-type-summary">
        <span class="vr-type-summary-label">${esc(tg.title)}</span>
        <span class="vr-summary-end">
          <span class="vr-type-count">${n}</span>
          ${delBtn}
        </span>
      </summary>
      <div class="vr-result-type-body">
        ${renderFileTypeSections(tg.items, tg.key)}
      </div>
    </details>`;
    })
    .join("");
}

function renderFailedCard(run) {
  const sessionLink = `/quickrun?session=${encodeURIComponent(run.session_id)}`;
  const err =
    run.error_message != null && run.error_message !== ""
      ? `<p class="qr-warn">${esc(run.error_message)}</p>`
      : "";
  const log =
    run.log_path != null && run.log_path !== ""
      ? `<p class="qr-muted">Log: <code>${esc(run.log_path)}</code></p>`
      : "";
  return `
    <article class="vr-card vr-card-fail">
      <div class="vr-card-head">
        <span class="qr-badge qr-badge-fail">failed</span>
        <strong>${esc(run.project)}</strong>
        <span class="vr-muted">session <code>${esc(String(run.session_id).slice(0, 8))}…</code></span>
      </div>
      <div class="vr-card-meta">
        <span>Session: ${esc(fmtTime(run.session_created_at))}</span>
        <span>Job finished: ${esc(fmtTime(run.finished_at))}</span>
        <span>Exit: ${run.exit_code != null ? esc(run.exit_code) : "—"}</span>
        <a href="${sessionLink}">Open in Running progress →</a>
      </div>
      ${err}
      ${log}
    </article>
  `;
}

async function loadResults() {
  vrError.classList.add("hidden");
  vrError.textContent = "";
  vrEmpty.classList.add("hidden");
  setVrActionMsg("", false);

  const q = new URLSearchParams({ video_path: videoPath });
  if (projectFilter) q.set("project", projectFilter);
  if (clipIdFilter) q.set("clip_id", clipIdFilter);
  const r = await fetch(`/api/quickrun/results_for_video?${q}`);
  const data = await r.json();
  if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`);

  linkVideoPath = String(data.video_path || videoPath || "").trim();

  vrIntro.classList.remove("hidden");
  vrVideoPath.textContent = data.video_path || videoPath;
  if (data.project_filter) {
    vrProjectRow.classList.remove("hidden");
    vrProjectName.textContent = data.project_filter;
  }
  if (clipIdFilter && vrClipRow && vrClipId) {
    vrClipRow.classList.remove("hidden");
    vrClipId.textContent = clipIdFilter;
  }

  const artifacts = data.artifacts || [];
  const failedRuns = data.failed_runs || [];

  if (!artifacts.length && !failedRuns.length) {
    vrRuns.innerHTML = "";
    vrEmpty.classList.remove("hidden");
    return;
  }

  let html = "";
  if (artifacts.length) {
    const nArt = artifacts.length;
    const delAllOut = `<button type="button" class="qr-btn-danger qr-btn-small vr-section-delete-btn" data-vr-del-scope="all" title="Delete every listed output file and directory on disk, and clear the results index for this view">Delete all outputs</button>`;
    html += `<section class="vr-section vr-output-root">`;
    html += `<details class="vr-output-files-fold" open>`;
    html += `<summary class="vr-type-summary vr-output-files-fold-summary">`;
    html += `<span class="vr-type-summary-label">Output files</span>`;
    html += `<span class="vr-summary-end">`;
    html += `<span class="vr-type-count">${nArt}</span>${delAllOut}`;
    html += `</span></summary>`;
    html += `<div class="vr-output-files-fold-body">`;
    html += `<p class="qr-muted vr-section-note">Each path is listed once, grouped by <strong>result type</strong> (Quick Run vs Snapshot), then by <strong>file type</strong>. Use <strong>Delete</strong> on a section to remove those files or folders from disk and drop them from this index (completed job records are unchanged; a future sync may re-index outputs if the job folder still exists). <strong>Clear failed</strong> removes failed-session rows and each session’s <code>/tmp</code> log folder.</p>`;
    html += renderArtifactSections(artifacts);
    html += `</div></details></section>`;
  }
  if (failedRuns.length) {
    const sids = [...new Set(failedRuns.map((x) => x.session_id).filter(Boolean))];
    const sidAttr = esc(sids.join(","));
    const clearBtn = `<button type="button" class="qr-btn-danger qr-btn-small vr-section-delete-btn" data-vr-clear-failed="${sidAttr}" title="Remove failed sessions from the database and delete their /tmp session folders">Clear failed</button>`;
    html += `<section class="vr-section vr-failed-root"><details class="vr-type-fold" open>`;
    html += `<summary class="vr-type-summary"><span class="vr-type-summary-label">Failed attempts</span>`;
    html += `<span class="vr-summary-end"><span class="vr-type-count">${failedRuns.length}</span>${clearBtn}</span></summary>`;
    html += `<div class="vr-type-body"><p class="qr-muted vr-fold-note">Only jobs whose session still exists in the monitor history.</p>`;
    html += failedRuns.map(renderFailedCard).join("");
    html += `</div></details></section>`;
  }
  vrRuns.innerHTML = html;
}

async function main() {
  if (!videoPath) {
    vrError.textContent =
      "Missing video_path in URL. Open Results from a project video row.";
    vrError.classList.remove("hidden");
    return;
  }

  try {
    await loadResults();
  } catch (e) {
    vrError.textContent = e.message || String(e);
    vrError.classList.remove("hidden");
  }
}

vrRuns.addEventListener("click", async (e) => {
  const btn = e.target.closest(".vr-section-delete-btn");
  if (!btn) return;
  e.preventDefault();
  e.stopPropagation();

  const clearFailed = btn.getAttribute("data-vr-clear-failed");
  if (clearFailed != null && clearFailed !== "") {
    const sids = clearFailed.split(",").map((s) => s.trim()).filter((s) => /^[0-9a-f]{32}$/.test(s));
    if (!sids.length) return;
    if (
      !window.confirm(
        `Remove ${sids.length} failed session(s) from the database and delete each session’s folder under /tmp?`,
      )
    ) {
      return;
    }
    btn.disabled = true;
    try {
      for (const sid of sids) {
        await deleteQuickrunSession(sid);
      }
      setVrActionMsg(`Cleared ${sids.length} failed session(s).`, false);
      await loadResults();
    } catch (err) {
      setVrActionMsg(err.message || String(err), true);
    } finally {
      btn.disabled = false;
    }
    return;
  }

  const scope = btn.getAttribute("data-vr-del-scope");
  if (!scope) return;
  const resultsType = (btn.getAttribute("data-vr-results-type") || "").trim();
  const section = (btn.getAttribute("data-vr-section") || "").trim() || undefined;

  let confirmMsg = "";
  if (scope === "all") {
    confirmMsg =
      "Delete every output file and directory listed here from disk, and remove them from the results index?";
  } else if (scope === "results_type") {
    const label = resultsType.replace(/_/g, " ");
    confirmMsg = `Delete all outputs in “${label}” from disk and remove them from the index?`;
  } else if (scope === "section") {
    confirmMsg = `Delete all files in the “${section}” section from disk and remove them from the index?`;
  } else {
    return;
  }
  if (!window.confirm(confirmMsg)) return;

  btn.disabled = true;
  try {
    const payload = buildDeleteSectionPayload(scope, resultsType, section);
    const d = await postDeleteResultsSection(payload);
    setVrActionMsg(d.message || "Deleted.", false);
    await loadResults();
  } catch (err) {
    setVrActionMsg(err.message || String(err), true);
  } finally {
    btn.disabled = false;
  }
});

main();
