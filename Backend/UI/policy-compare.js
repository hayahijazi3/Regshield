const API = window.BACKEND_URL || "http://127.0.0.1:5001";
const $ = (s) => document.querySelector(s);

function getToken() {
  return localStorage.getItem("token") || sessionStorage.getItem("token");
}
function show(el, on = true) { el.style.display = on ? "" : "none"; }
function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function renderMarkdown(md) {
  if (!md) return "";
  const lines = String(md).replace(/\r/g, "").split("\n");
  const out = [];
  let i = 0;
  const inline = (txt) =>
    esc(txt)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      .replace(/~~([^~]+)~~/g, "<s>$1</s>");

  while (i < lines.length) {
    let line = lines[i];

    if (/^```/.test(line)) {
      let code = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) {
        code.push(lines[i]);
        i++;
      }
      if (i < lines.length) i++;
      out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
      continue;
    }

    const isTableHeader =
      /\|/.test(line) &&
      i + 1 < lines.length &&
      /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[i + 1]);

    if (isTableHeader) {
      const headerRow = line;
      const rows = [];
      i += 2;
      while (i < lines.length && /\|/.test(lines[i]) && !/^\s*$/.test(lines[i])) {
        rows.push(lines[i]);
        i++;
      }
      const splitRow = (r) =>
        r.trim()
          .replace(/^\|/, "")
          .replace(/\|$/, "")
          .split("|")
          .map((c) => inline(c.trim()));
      const heads = splitRow(headerRow);
      const body = rows.map(splitRow);

      let tbl = `<table><thead><tr>${heads.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>`;
      for (const r of body) {
        const cells = r.length < heads.length ? r.concat(Array(heads.length - r.length).fill("")) : r;
        tbl += `<tr>${cells.map((c) => `<td>${c}</td>`).join("")}</tr>`;
      }
      tbl += `</tbody></table>`;
      out.push(tbl);
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const block = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        block.push(lines[i].replace(/^\s*>\s?/, ""));
        i++;
      }
      out.push(`<blockquote>${inline(block.join("\n"))}</blockquote>`);
      continue;
    }

    if (/^#{1,6}\s+/.test(line)) {
      const level = (line.match(/^#+/) || ["#"])[0].length;
      const text = line.replace(/^#{1,6}\s+/, "");
      i++;
      out.push(`<h${level}>${inline(text)}</h${level}>`);
      continue;
    }

    if (/^\s*[-*+]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*+]\s+/, ""));
        i++;
      }
      out.push(`<ul>${items.map((x) => `<li>${inline(x)}</li>`).join("")}</ul>`);
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ""));
        i++;
      }
      out.push(`<ol>${items.map((x) => `<li>${inline(x)}</li>`).join("")}</ol>`);
      continue;
    }

    if (!/^\s*$/.test(line)) {
      const para = [line];
      i++;
      while (i < lines.length && !/^\s*$/.test(lines[i])) {
        para.push(lines[i]);
        i++;
      }
      out.push(`<p>${inline(para.join(" "))}</p>`);
      continue;
    }

    i++;
  }
  return out.join("\n");
}

const form = $("#uploadForm");
const fileEl = $("#file");
const method = $("#method");
const sourcesSel = $("#sources");
const topk = $("#topk");
const statusBox = $("#status");
const loaderEl = $("#loader");          
const errBox = $("#error");
const results = $("#results");
const exportBtn = $("#exportBtn");

let LAST_COMPARE = null;
let LAST_COMPARE_META = null;

function setStatus(msg) { statusBox.textContent = msg || ""; }
function setError(msg) { errBox.textContent = msg || ""; show(errBox, !!msg); }
function showLoader(on) { if (loaderEl) show(loaderEl, on); }

function keepFirstTable(md) {
  if (!md) return "";
  const lines = String(md).replace(/\r/g, "").split("\n");
  let i = 0;
  while (i < lines.length) {
    const header = /^\s*\|.*\|\s*$/.test(lines[i]);
    const sep = i + 1 < lines.length &&
      /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[i + 1]);
    if (header && sep) break;
    i++;
  }
  if (i >= lines.length) return md;
  const out = [lines[i], lines[i + 1]];
  i += 2;
  while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
    out.push(lines[i]); i++;
  }
  return out.join("\n");
}

function renderResults(payload) {
  LAST_COMPARE = payload;
  const items = Array.isArray(payload.items) ? payload.items : [];
  results.innerHTML = "";

  if (!items.length) {
    results.innerHTML = `<div class="card">No sections detected in the uploaded policy.</div>`;
    return;
  }

  const frag = document.createDocumentFragment();
  for (const it of items) {
    const card = document.createElement("article");
    card.className = "card";

    const title = it.title ? esc(it.title) : `Section ${it.section_id}`;
    const policy = esc(it.policy_excerpt || "");

    const matches = (it.matches || []).map((m) => {
      const refBits = [
        esc(m.source || "-"),
        esc(m.reference || "-"),
        `p.${esc(String(m.page ?? ""))}`,
        esc(m.filename || "")
      ].join(" • ");
      const snippet = esc(String(m.text || "").slice(0, 1200)) + (m.text && m.text.length > 1200 ? "…" : "");
      return `
        <div>
          <div class="badge">${refBits}</div>
          <div class="text">${snippet}</div>
        </div>
      `;
    }).join("");

    const mdOnlyTable = keepFirstTable(it.ai_comparison_markdown || "");
    const mdHtml = renderMarkdown(mdOnlyTable || "_No AI comparison was generated._");

    card.innerHTML = `
      <h3>${title}</h3>
      <div class="meta">Top matches: ${it.matches?.length || 0}</div>

      <div class="grid">
        <div class="col">
          <h4>Policy Section</h4>
          <div class="text">${policy}</div>
        </div>
        <div class="col">
          <h4>Relevant Clauses</h4>
          <div class="matches">${matches || "<em>No matches</em>"}</div>
        </div>
      </div>

      <div class="markdown">${mdHtml}</div>
    `;
    frag.appendChild(card);
  }
  results.appendChild(frag);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  setError("");
  setStatus("");          
  showLoader(true);       
  results.innerHTML = "";

  const t0 = performance.now();   

  const f = fileEl.files?.[0];
  if (!f) {
    setError("Please choose a file (.pdf, .docx, .txt).");
    showLoader(false);
    return;
  }

  const fd = new FormData();
  fd.append("file", f, f.name);
  fd.append("method", method.value);
  fd.append("sources", sourcesSel?.value || "all");
  const topKValue = Math.max(1, Math.min(8, Number(topk.value) || 3));
  fd.append("top_k", String(topKValue));

  try {
    const res = await fetch(`${API}/compare/policy`, {
      method: "POST",
      headers: { Authorization: `Bearer ${getToken()}` },
      body: fd,
    });

    if (res.status === 401) {
      const loc = new URL(window.location.href);
      window.location.assign(`login.html?from=${encodeURIComponent(loc.pathname.replace(/^\//, ""))}`);
      return;
    }

    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.msg || data.error || `HTTP ${res.status}`);

    LAST_COMPARE_META = {
      method: method.value,
      sources: sourcesSel?.value || "all",
      topK: topKValue,
      when: new Date().toLocaleString(),
      filename: data.filename || f.name
    };

    renderResults(data);

    const elapsedMs = Math.round(performance.now() - t0);  // <-- NEW

    setStatus(`Compared ${data.sections_compared} section(s) using ${data.method} • Top_k=${data.top_k} • Response time: ${elapsedMs} ms`);
  } catch (err) {
    setError(err.message || "Comparison failed.");
  } finally {
    showLoader(false);  // hide spinner
    if (!errBox.textContent) setTimeout(() => setStatus(""), 600);
  }
});

fileEl.addEventListener("change", () => {
  setError("");
  const f = fileEl.files?.[0];
  if (!f) return;
  if (!/\.(pdf|docx|txt)$/i.test(f.name)) {
    setError("Unsupported file type. Allowed: .pdf, .docx, .txt");
  }
});

// -------- Export to PDF (Print) ----------
function buildPrintableHTML() {
  if (!LAST_COMPARE) return null;
  const meta = LAST_COMPARE_META || {};
  const items = LAST_COMPARE.items || [];

  const style = `
  <style>
    @page { size: A4; margin: 16mm; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; color:#111; }
    h1,h2,h3,h4 { margin: 0 0 .5rem; }
    h1 { font-size: 20px; }
    h2 { font-size: 16px; margin-top: 1.2rem; }
    .meta { font-size: 12px; color:#555; margin-bottom: 12px; }
    .card { border:1px solid #e5e5e5; border-radius:8px; padding:10px; margin-bottom:12px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .badge { display:inline-block; border:1px solid #ccc; border-radius:999px; padding:2px 8px; margin-right:6px; font-size:11px; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }
    th { background: #f6f8fa; }
    .small { font-size: 11px; color:#666; }
  </style>`;

  const header = `
    <h1>Policy vs Regulation — Export</h1>
    <div class="meta">
      <div><strong>File:</strong> ${esc(meta.filename || "-")}</div>
      <div><strong>Method:</strong> ${esc(meta.method || "-")}</div>
      <div><strong>Sources:</strong> ${esc(String(meta.sources || "all").toUpperCase())}</div>
      <div><strong>Top K:</strong> ${esc(String(meta.topK || "-"))}</div>
      <div class="small">Exported: ${esc(meta.when || new Date().toLocaleString())}</div>
    </div>
  `;

  const secHtml = items.map((it) => {
    const matches = (it.matches || []).map((m) => {
      const refBits = [
        esc(m.source || "-"),
        esc(m.reference || "-"),
        `p.${esc(String(m.page ?? ""))}`,
        esc(m.filename || "")
      ].join(" • ");
      const snippet = esc(String(m.text || ""));
      return `
        <div style="margin-bottom:6px;">
          <div class="badge">${refBits}</div>
          <div class="small">${snippet}</div>
        </div>
      `;
    }).join("");

    const mdHtml = renderMarkdown(it.ai_comparison_markdown || "");

    return `
      <div class="card">
        <h3>${esc(it.title || `Section ${it.section_id}`)}</h3>
        <div class="grid">
          <div>
            <h4>Policy Section</h4>
            <div class="small">${esc(it.policy_excerpt || "")}</div>
          </div>
          <div>
            <h4>Relevant Clauses</h4>
            ${matches || "<em>No matches</em>"}
          </div>
        </div>
        <h4 style="margin-top:10px;">AI Comparison</h4>
        <div>${mdHtml}</div>
      </div>
    `;
  }).join("");

  return `<!doctype html><html><head><meta charset="utf-8"><title>Policy Compare Export</title>${style}</head>
  <body>${header}${secHtml}</body></html>`;
}

function exportPDF() {
  if (!LAST_COMPARE) { alert("Run a comparison first."); return; }
  const html = buildPrintableHTML();
  const w = window.open("", "_blank");
  w.document.open();
  w.document.write(html);
  w.document.close();
  w.focus();
  setTimeout(() => w.print(), 400);
}
document.getElementById("exportBtn")?.addEventListener("click", exportPDF);
