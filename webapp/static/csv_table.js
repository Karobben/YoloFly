const qs = new URLSearchParams(window.location.search);
const filePath = (qs.get("path") || "").trim();
const returnVideoPath = (qs.get("video_path") || "").trim();
const returnProject = (qs.get("project") || "").trim();
const returnClipId = (qs.get("clip_id") || "").trim();

const csvTablePath = document.getElementById("csvTablePath");
const csvTableMeta = document.getElementById("csvTableMeta");
const csvTableError = document.getElementById("csvTableError");
const csvTableScroll = document.getElementById("csvTableScroll");
const csvThead = document.getElementById("csvThead");
const csvTbody = document.getElementById("csvTbody");
const csvTableEmpty = document.getElementById("csvTableEmpty");
const csvBackResults = document.getElementById("csvBackResults");
const csvPlotActions = document.getElementById("csvPlotActions");
const csvTotalSpeedPlotLink = document.getElementById("csvTotalSpeedPlotLink");

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function buildBackResultsHref() {
  if (!returnVideoPath) return "";
  const q = new URLSearchParams({ video_path: returnVideoPath });
  if (returnProject) q.set("project", returnProject);
  if (returnClipId) q.set("clip_id", returnClipId);
  return `/video-results?${q.toString()}`;
}

function isTotalSpeedCsvPath(p) {
  return /\.total_speed\.csv$/i.test(String(p || ""));
}

function totalSpeedPlotPageUrl(absPath) {
  const q = new URLSearchParams({ path: absPath });
  if (returnVideoPath) q.set("video_path", returnVideoPath);
  if (returnProject) q.set("project", returnProject);
  if (returnClipId) q.set("clip_id", returnClipId);
  return `/total-speed-plot?${q.toString()}`;
}

async function loadTable() {
  if (!filePath) {
    csvTableError.textContent = "Missing path query parameter.";
    csvTableError.classList.remove("hidden");
    return;
  }

  const back = buildBackResultsHref();
  if (back && csvBackResults) {
    csvBackResults.href = back;
    csvBackResults.classList.remove("hidden");
  }

  csvTablePath.innerHTML = `<code>${esc(filePath)}</code>`;

  try {
    const url = `/api/csv_table?path=${encodeURIComponent(filePath)}`;
    const r = await fetch(url);
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`);
    if (!data.ok) throw new Error("Unexpected response");

    const delimHint =
      data.delimiter === "\t" ? "tab" : data.delimiter === ";" ? "semicolon" : "comma";
    let meta = `Delimiter: ${delimHint} · ${data.row_count_returned} data row(s)`;
    if (data.headers && data.headers.length) {
      meta += ` · ${data.headers.length} column(s)`;
    }
    if (data.truncated) {
      meta += ` · truncated (showing first ${data.max_rows} rows only)`;
    }
    csvTableMeta.textContent = meta;

    if (isTotalSpeedCsvPath(filePath) && csvPlotActions && csvTotalSpeedPlotLink) {
      csvTotalSpeedPlotLink.href = totalSpeedPlotPageUrl(filePath);
      csvPlotActions.classList.remove("hidden");
    } else if (csvPlotActions) {
      csvPlotActions.classList.add("hidden");
    }

    const headers = data.headers || [];
    const rows = data.rows || [];

    csvThead.innerHTML = "";
    csvTbody.innerHTML = "";

    if (headers.length) {
      const tr = document.createElement("tr");
      for (const h of headers) {
        const th = document.createElement("th");
        th.textContent = h;
        tr.appendChild(th);
      }
      csvThead.appendChild(tr);
    }

    if (!rows.length) {
      csvTableScroll.classList.add("hidden");
      csvTableEmpty.classList.remove("hidden");
      return;
    }

    csvTableEmpty.classList.add("hidden");
    csvTableScroll.classList.remove("hidden");

    const frag = document.createDocumentFragment();
    for (const row of rows) {
      const tr = document.createElement("tr");
      for (let i = 0; i < Math.max(row.length, headers.length); i++) {
        const td = document.createElement("td");
        td.textContent = row[i] != null ? String(row[i]) : "";
        tr.appendChild(td);
      }
      frag.appendChild(tr);
    }
    csvTbody.appendChild(frag);
  } catch (e) {
    csvTableError.textContent = e.message || String(e);
    csvTableError.classList.remove("hidden");
    csvTableScroll.classList.add("hidden");
    csvTableEmpty.classList.add("hidden");
    if (csvPlotActions) csvPlotActions.classList.add("hidden");
  }
}

loadTable();
