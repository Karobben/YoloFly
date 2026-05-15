const qs = new URLSearchParams(window.location.search);
const sessionId = (qs.get("session") || "").trim().toLowerCase();

const noSessionEl = document.getElementById("quickrunNoSession");
const errorEl = document.getElementById("quickrunError");
const bodyEl = document.getElementById("quickrunBody");
const qrProject = document.getElementById("qrProject");
const qrSessionId = document.getElementById("qrSessionId");
const qrWorkdir = document.getElementById("qrWorkdir");
const qrSessionStatus = document.getElementById("qrSessionStatus");
const qrCounts = document.getElementById("qrCounts");
const qrParams = document.getElementById("qrParams");
const qrJobs = document.getElementById("qrJobs");
const qrFatal = document.getElementById("qrFatal");
const qrHistoryTable = document.getElementById("qrHistoryTable");
const qrHistoryStatus = document.getElementById("qrHistoryStatus");
const qrDeleteCurrentBtn = document.getElementById("qrDeleteCurrentBtn");

let pollTimer = null;

const DELETE_CONFIRM =
  "Remove this run from history? This deletes the database record and the session’s TSV/log folder under /tmp. Pipeline outputs (CSVs, tracks, plots) on disk are not deleted.";

function badgeClass(status) {
  if (status === "done") return "qr-badge qr-badge-done";
  if (status === "failed") return "qr-badge qr-badge-fail";
  if (status === "processing") return "qr-badge qr-badge-run";
  return "qr-badge qr-badge-queue";
}

function sessionStatusClass(status) {
  if (status === "complete") return "qr-badge qr-badge-done";
  if (status === "failed") return "qr-badge qr-badge-fail";
  if (status === "running") return "qr-badge qr-badge-run";
  return "qr-badge qr-badge-queue";
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

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function detectExploreSnapshotHref(absDir) {
  const u = new URLSearchParams({ snapshot_dir: absDir });
  return `/detect_explore?${u.toString()}`;
}

function renderMetaDl(snapshot) {
  if (!snapshot || typeof snapshot !== "object") return "";
  const rows = [
    ["Video path", snapshot.path],
    ["Filename", snapshot.filename],
    ["Detailed location", snapshot.detailed_location],
    ["Disk px/mm", snapshot.disk_pixel],
    ["Disk radius (mm)", snapshot.disk_radius_mm],
    ["Frame range", [snapshot.frame_start, snapshot.frame_end].filter((x) => x != null).join(" – ") || "—"],
    ["Fly count", snapshot.fly_count],
    ["Splits", [snapshot.split_x, snapshot.split_y].every((x) => x == null || x === "") ? "—" : `${snapshot.split_x ?? "—"} × ${snapshot.split_y ?? "—"}`],
    ["Total frames", snapshot.total_frames],
  ];
  if (snapshot.job_kind === "snapshot") {
    rows.push(
      ["Job type", "detect_2 snapshot (single frame)"],
      ["Snapshot frame (detect_2)", snapshot.snapshot_frame],
    );
    if (snapshot.subclip_clip_id != null) {
      rows.push(["Subclip id", snapshot.subclip_clip_id]);
    }
  }
  if (snapshot.job_kind === "post_track") {
    rows.push(
      ["Job type", "Post_track (fly paths JSON)"],
      ["Detection CSV (input)", snapshot.post_track_input_csv],
      ["Post_track output CSV (intermediate)", snapshot.post_track_output_csv],
      ["Workers", snapshot.post_track_workers],
      ["JSON-only cleanup", snapshot.post_track_json_only ? "yes" : "no"],
    );
    if (snapshot.subclip_clip_id != null) {
      rows.push(["Subclip id", snapshot.subclip_clip_id]);
    }
  }
  const parts = rows
    .filter(([, v]) => v != null && v !== "")
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`);
  if (!parts.length) return "";
  return `<dl class="qr-meta-dl">${parts.join("")}</dl>`;
}

function renderOutputs(outputs) {
  if (!outputs || typeof outputs !== "object") return "<p class=\"qr-muted\">No output paths resolved yet.</p>";
  const lines = [];
  if (outputs.skipped_existing) {
    lines.push(
      "<p class=\"qr-muted\">Skipped running the pipeline: expected outputs were already on disk. Enable <strong>Rerun even if outputs exist</strong> (QuickRun) or pass <code>rerun</code> (snapshot batch) to force.</p>",
    );
  }
  if (outputs.resolve_error) lines.push(`<p class="qr-warn">${esc(outputs.resolve_error)}</p>`);
  if (outputs.job_kind === "post_track" && outputs.post_track_save_dir) {
    lines.push(`<div><strong>Tracking folder</strong> <code>${esc(outputs.post_track_save_dir)}</code></div>`);
  }
  if (outputs.video_file) lines.push(`<div><strong>Video</strong> <code>${esc(outputs.video_file)}</code></div>`);
  if (outputs.detection_csv) lines.push(`<div><strong>Detection CSV</strong> <code>${esc(outputs.detection_csv)}</code></div>`);
  if (outputs.track_stem) lines.push(`<div><strong>Track base</strong> <code>${esc(outputs.track_stem)}</code></div>`);
  if (outputs.tracked_json) lines.push(`<div><strong>Tracked JSON</strong> <code>${esc(outputs.tracked_json)}</code></div>`);
  if (outputs.output_directory) lines.push(`<div><strong>Plot directory</strong> <code>${esc(outputs.output_directory)}</code></div>`);
  if (outputs.snapshot_save_dir) {
    const deHref = detectExploreSnapshotHref(outputs.snapshot_save_dir);
    lines.push(
      `<div><strong>Snapshot output</strong> <code>${esc(outputs.snapshot_save_dir)}</code></div>` +
      `<div class="qr-detect-explore-actions"><a class="qr-btn-detect-explore" href="${esc(deHref)}" target="_blank" rel="noopener">Open raw frame + labels in detect_explore</a></div>`,
    );
  }
  if (outputs.weights_resolved) lines.push(`<div><strong>Weights</strong> <code>${esc(outputs.weights_resolved)}</code></div>`);
  if (Array.isArray(outputs.plots) && outputs.plots.length) {
    const plots = outputs.plots
      .map((p) => `<li><code>${esc(p.path)}</code> ${p.exists ? "✓" : "missing"} <span class="qr-muted">${esc(p.label || "")}</span></li>`)
      .join("");
    lines.push(`<div><strong>Plots</strong><ul class="qr-path-list">${plots}</ul></div>`);
  }
  if (!lines.length) return "<p class=\"qr-muted\">No output details.</p>";
  return `<div class="qr-outputs">${lines.join("")}</div>`;
}

function renderJobCard(job) {
  const st = job.status || "unknown";
  const metaHtml = renderMetaDl(job.entry_snapshot);
  const logBlock =
    job.log_tail && String(job.log_tail).trim()
      ? `<h4>Log (tail)</h4><pre class="quickrun-pre qr-log">${esc(job.log_tail)}</pre>`
      : "<p class=\"qr-muted\">No log output yet.</p>";
  const outHtml = st === "done" ? renderOutputs(job.outputs) : "";
  const runCmd = job.run_command && String(job.run_command).trim();
  const cmdBlock = runCmd
    ? `<details class="qr-details" open><summary>Command line</summary><pre class="quickrun-pre qr-run-cmd">${esc(runCmd)}</pre></details>`
    : "";

  return `
    <article class="qr-job-card">
      <div class="qr-job-head">
        <span class="${badgeClass(st)}">${esc(st)}</span>
        <strong class="qr-job-title">${esc(job.video_label || job.video_path || job.id)}</strong>
        <span class="qr-job-sub">${esc(job.video_path || "")}</span>
      </div>
      <div class="qr-job-meta-row">
        <span>PID: ${job.pid != null ? esc(job.pid) : "—"}</span>
        <span>Exit: ${job.exit_code != null ? esc(job.exit_code) : "—"}</span>
        <span>Started: ${esc(fmtTime(job.started_at))}</span>
        <span>Finished: ${esc(fmtTime(job.finished_at))}</span>
      </div>
      ${job.error_message ? `<p class="qr-warn"><strong>Error:</strong> ${esc(job.error_message)}</p>` : ""}
      <details class="qr-details">
        <summary>Video metadata &amp; location</summary>
        ${metaHtml || "<p class=\"qr-muted\">No metadata.</p>"}
      </details>
      <details class="qr-details" open>
        <summary>Processing log</summary>
        ${logBlock}
        <p class="qr-muted"><small>Full log file: <code>${esc(job.log_path || "")}</code></small></p>
      </details>
      ${
        st === "done"
          ? `<details class="qr-details" open><summary>Output files</summary>${outHtml}</details>`
          : ""
      }
      ${cmdBlock}
    </article>
  `;
}

function renderHistoryTable(sessions) {
  if (!sessions.length) {
    return '<p class="qr-muted">No saved runs yet. Start a QuickRun from the Project Manager.</p>';
  }
  const rows = sessions
    .map((s) => {
      const active = s.id === sessionId ? ' class="qr-history-current"' : "";
      const jobSummary = `${s.job_done}/${s.job_total} done, ${s.job_failed} failed`;
      const running =
        s.job_processing > 0 || s.job_queued > 0
          ? ` (${s.job_processing} running, ${s.job_queued} queued)`
          : "";
      return `
      <tr${active}>
        <td><code title="${esc(s.id)}">${esc(s.id.slice(0, 8))}…</code></td>
        <td>${esc(s.project)}</td>
        <td>${esc(fmtTime(s.created_at))}</td>
        <td><span class="${sessionStatusClass(s.session_status)}">${esc(s.session_status)}</span></td>
        <td>${esc(jobSummary)}${esc(running)}</td>
        <td class="qr-history-actions">
          <a href="/quickrun?session=${encodeURIComponent(s.id)}">Open</a>
          <button type="button" class="qr-btn-danger qr-btn-small" data-qr-delete="${esc(s.id)}">Delete</button>
        </td>
      </tr>`;
    })
    .join("");
  return `
    <table class="qr-history-table">
      <thead>
        <tr>
          <th>Session</th>
          <th>Project</th>
          <th>Started</th>
          <th>Status</th>
          <th>Jobs</th>
          <th></th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function loadHistory() {
  if (!qrHistoryTable) return;
  try {
    const r = await fetch("/api/quickrun/history?limit=80");
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`);
    qrHistoryTable.innerHTML = renderHistoryTable(data.sessions || []);
    if (qrHistoryStatus) qrHistoryStatus.textContent = "";
  } catch (e) {
    if (qrHistoryStatus) qrHistoryStatus.textContent = e.message || String(e);
  }
}

async function deleteSessionById(sid) {
  if (!sid || !/^[0-9a-f]{32}$/.test(sid)) return;
  if (!window.confirm(DELETE_CONFIRM)) return;
  if (qrHistoryStatus) qrHistoryStatus.textContent = "Deleting…";
  try {
    const r = await fetch(`/api/quickrun/session/${encodeURIComponent(sid)}`, { method: "DELETE" });
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`);
    if (qrHistoryStatus) qrHistoryStatus.textContent = "Removed from history.";
    if (sid === sessionId) {
      window.location.href = "/quickrun";
      return;
    }
    await loadHistory();
  } catch (e) {
    if (qrHistoryStatus) qrHistoryStatus.textContent = e.message || String(e);
  }
}

if (qrHistoryTable) {
  qrHistoryTable.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-qr-delete]");
    if (!btn) return;
    e.preventDefault();
    const sid = btn.getAttribute("data-qr-delete");
    if (sid) deleteSessionById(sid);
  });
}

if (qrDeleteCurrentBtn) {
  qrDeleteCurrentBtn.addEventListener("click", () => {
    if (/^[0-9a-f]{32}$/.test(sessionId)) deleteSessionById(sessionId);
  });
}

async function fetchSession() {
  const r = await fetch(`/api/quickrun/session/${encodeURIComponent(sessionId)}`);
  const data = await r.json();
  if (!r.ok || data.error) {
    throw new Error(data.error || `HTTP ${r.status}`);
  }
  return data.session;
}

function applySession(sess) {
  errorEl.classList.add("hidden");
  bodyEl.classList.remove("hidden");
  if (sess.fatal_error && qrFatal) {
    qrFatal.textContent = sess.fatal_error;
    qrFatal.classList.remove("hidden");
  } else if (qrFatal) {
    qrFatal.classList.add("hidden");
  }
  qrProject.textContent = sess.project || "—";
  qrSessionId.textContent = sess.id || sessionId;
  qrWorkdir.textContent = sess.workdir || "—";
  qrSessionStatus.textContent = sess.session_status || "—";
  const c = sess.counts || {};
  qrCounts.textContent = `${c.done ?? 0} done · ${c.failed ?? 0} failed · ${c.processing ?? 0} running · ${c.queued ?? 0} queued (${c.total ?? 0} total)`;
  qrParams.textContent = JSON.stringify(sess.pipeline_params || {}, null, 2);
  qrJobs.innerHTML = (sess.jobs || []).map(renderJobCard).join("");

  const complete =
    sess.session_status === "complete" ||
    sess.session_status === "failed" ||
    Boolean(sess.fatal_error);
  if (complete && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function pollLoop() {
  try {
    const sess = await fetchSession();
    applySession(sess);
    await loadHistory();
  } catch (e) {
    errorEl.textContent = e.message || String(e);
    errorEl.classList.remove("hidden");
    bodyEl.classList.add("hidden");
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    await loadHistory();
  }
}

async function main() {
  await loadHistory();
  if (!/^[0-9a-f]{32}$/.test(sessionId)) {
    noSessionEl.classList.remove("hidden");
    bodyEl.classList.add("hidden");
    return;
  }
  noSessionEl.classList.add("hidden");
  pollLoop();
  pollTimer = setInterval(pollLoop, 2000);
}

main();
