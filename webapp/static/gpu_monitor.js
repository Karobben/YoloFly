const statusEl = document.getElementById("gpuMonitorStatus");
const gpuTableWrap = document.getElementById("gpuTableWrap");
const unmappedWrap = document.getElementById("gpuUnmappedJobs");

let pollTimer = null;

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
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

function renderJobCell(job, pid) {
  if (!job) {
    return `
      <div><span class="qr-muted">external process</span></div>
      <div style="margin-top:6px">
        <button class="qr-btn-danger qr-btn-small" data-kill-pid="${esc(pid)}">Kill PID</button>
      </div>
    `;
  }
  const killBtn = job.killable
    ? `<button class="qr-btn-danger qr-btn-small" data-kill-session="${esc(job.session_id)}" data-kill-job="${esc(job.job_id)}">Kill</button>`
    : '<span class="qr-muted">not killable</span>';
  return `
    <div><strong>${esc(job.video_label || job.job_id)}</strong></div>
    <div class="qr-muted">${esc(job.project)} · ${esc(job.session_kind)} · started ${esc(fmtTime(job.started_at))}</div>
    <div class="qr-muted"><code>${esc(job.session_id)}</code> / <code>${esc(job.job_id)}</code></div>
    <div class="qr-muted">device=${esc(job.tracking_device || "auto")}</div>
    <div style="margin-top:6px">${killBtn}</div>
  `;
}

function renderGpuTable(payload) {
  const gpus = payload.gpus || [];
  if (!gpus.length) {
    gpuTableWrap.innerHTML = '<p class="qr-muted">No GPUs found or nvidia-smi unavailable.</p>';
    return;
  }
  const rows = gpus.map((g) => {
    const procs = (g.processes || []).length
      ? g.processes.map((p) => `
          <tr>
            <td><code>${esc(p.pid)}</code></td>
            <td>${esc(p.process_name || "—")}</td>
            <td>${esc(p.used_memory_mb || "—")} MB</td>
            <td>${renderJobCell(p.job, p.pid)}</td>
          </tr>
        `).join("")
      : '<tr><td colspan="4" class="qr-muted">No active compute processes.</td></tr>';
    return `
      <article class="qr-job-card">
        <div class="qr-job-head">
          <strong class="qr-job-title">GPU ${esc(g.index)}: ${esc(g.name)}</strong>
          <span class="qr-job-sub">UUID ${esc(g.uuid)}</span>
        </div>
        <div class="qr-job-meta-row">
          <span>Util: ${esc(g.util_gpu)}%</span>
          <span>Mem: ${esc(g.mem_used_mb)} / ${esc(g.mem_total_mb)} MB</span>
          <span>Temp: ${esc(g.temp_c)} C</span>
        </div>
        <table class="qr-history-table">
          <thead>
            <tr>
              <th>PID</th><th>Process</th><th>GPU Mem</th><th>QuickRun job</th>
            </tr>
          </thead>
          <tbody>${procs}</tbody>
        </table>
      </article>
    `;
  }).join("");
  gpuTableWrap.innerHTML = rows;
}

function renderUnmappedJobs(payload) {
  const jobs = payload.running_jobs_unmapped || [];
  if (!jobs.length) {
    unmappedWrap.innerHTML = '<p class="qr-muted">None.</p>';
    return;
  }
  const cards = jobs.map((j) => `
    <article class="qr-job-card">
      <div class="qr-job-head">
        <strong class="qr-job-title">${esc(j.video_label || j.job_id)}</strong>
        <span class="qr-job-sub">${esc(j.project)} · ${esc(j.session_kind)}</span>
      </div>
      <div class="qr-job-meta-row">
        <span>PID: <code>${esc(j.pid)}</code></span>
        <span>Device: ${esc(j.tracking_device || "auto")}</span>
        <span>Started: ${esc(fmtTime(j.started_at))}</span>
      </div>
      <div><button class="qr-btn-danger qr-btn-small" data-kill-session="${esc(j.session_id)}" data-kill-job="${esc(j.job_id)}">Kill</button></div>
    </article>
  `).join("");
  unmappedWrap.innerHTML = cards;
}

async function refresh() {
  try {
    const r = await fetch("/api/gpu/monitor");
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`);
    renderGpuTable(data);
    renderUnmappedJobs(data);
    const errs = [];
    if (data.nvidia_smi_error) errs.push(`gpu query: ${data.nvidia_smi_error}`);
    if (data.nvidia_process_error) errs.push(`process query: ${data.nvidia_process_error}`);
    statusEl.textContent = errs.length
      ? `Updated ${fmtTime(data.updated_at)} · ${errs.join(" | ")}`
      : `Updated ${fmtTime(data.updated_at)}`;
  } catch (e) {
    statusEl.textContent = e.message || String(e);
  }
}

async function killJob(sessionId, jobId) {
  if (!sessionId || !jobId) return;
  const ok = window.confirm(`Kill job ${jobId}?`);
  if (!ok) return;
  statusEl.textContent = "Killing job…";
  try {
    const r = await fetch("/api/quickrun/kill_job", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, job_id: jobId }),
    });
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`);
    await refresh();
  } catch (e) {
    statusEl.textContent = e.message || String(e);
  }
}

async function killPid(pid) {
  if (!pid) return;
  const ok = window.confirm(`Kill external GPU process PID ${pid}?`);
  if (!ok) return;
  statusEl.textContent = "Killing PID…";
  try {
    const r = await fetch("/api/gpu/kill_pid", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pid: Number(pid) }),
    });
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`);
    await refresh();
  } catch (e) {
    statusEl.textContent = e.message || String(e);
  }
}

document.body.addEventListener("click", (e) => {
  const pidBtn = e.target.closest("[data-kill-pid]");
  if (pidBtn) {
    const pid = pidBtn.getAttribute("data-kill-pid");
    killPid(pid);
    return;
  }
  const btn = e.target.closest("[data-kill-session][data-kill-job]");
  if (!btn) return;
  const sid = btn.getAttribute("data-kill-session");
  const jobId = btn.getAttribute("data-kill-job");
  killJob(sid, jobId);
});

refresh();
pollTimer = setInterval(refresh, 2000);

window.addEventListener("beforeunload", () => {
  if (pollTimer) clearInterval(pollTimer);
});
