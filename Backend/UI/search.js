const API_URL   = "http://127.0.0.1:5001";
const LOGIN_PAGE = "login.html";

const form = document.getElementById("searchForm");
const qInput = document.getElementById("q");
const topKInput = document.getElementById("topK");
const methodSel = document.getElementById("method");
const sourceSel = document.getElementById("source");

const statText = document.getElementById("statText");
const countPill = document.getElementById("countPill");
const errBox = document.getElementById("err");
const infoBox = document.getElementById("info");
const answerBox = document.getElementById("llmAnswer");
const resultsBox = document.getElementById("results");
const loader = document.getElementById("loader");   

function show(el, on = true){ if (el) el.style.display = on ? "" : "none"; }
function setText(el, t){ if (el) el.textContent = t; }
function safe(s){
  return String(s||"").replace(/[&<>"']/g, m => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'
  }[m]));
}
function highlight(text, q){
  if (!q || !text) return safe(text || "");
  const esc = q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp("(" + esc + ")", "ig");
  return safe(text).replace(re, "<mark>$1</mark>");
}
function copy(text){ navigator.clipboard?.writeText(text).catch(()=>{}); }

function renderMarkdown(md) {
  if (!md) return "";
  let html = safe(md);
  html = html
    .replace(/^###### (.*)$/gm, "<h6>$1</h6>")
    .replace(/^##### (.*)$/gm, "<h5>$1</h5>")
    .replace(/^#### (.*)$/gm, "<h4>$1</h4>")
    .replace(/^### (.*)$/gm, "<h3>$1</h3>")
    .replace(/^## (.*)$/gm, "<h2>$1</h2>")
    .replace(/^# (.*)$/gm, "<h1>$1</h1>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/^- (.*)$/gm, "<li>$1</li>")
    .replace(/(\n<li>.*<\/li>)+/g, m => `<ul>${m}</ul>`)
    .replace(/\[(.+?)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  html = html.replace(/\n{2,}/g, "<br><br>");
  return html;
}

function updateControls() {}
methodSel.addEventListener("change", updateControls);
updateControls();

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  show(errBox, false);
  show(infoBox, true);
  show(loader, true);                        
  show(countPill, false);
  resultsBox.innerHTML = "";
  show(answerBox, false);
  answerBox.innerHTML = "";

  const t0 = performance.now(); 

  const q = qInput.value.trim();
  const topK = Math.max(5, Math.min(100, Number(topKInput.value) || 20));
  const method = methodSel.value;
  const source = sourceSel.value; 
  try {
    const url = new URL(`${API_URL}/search`);
    url.searchParams.set("query", q);
    url.searchParams.set("top_k", String(topK));
    url.searchParams.set("method", method);
    url.searchParams.set("sources", source);

    const res = await fetch(url.toString(), { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const items = Array.isArray(data.results) ? data.results : [];
    const total = typeof data.total_matches === "number" ? data.total_matches : items.length;

    if (data.answer_markdown) {
      let ragMetricsHtml = "";

      if (method === "rag" && items.length) {
        const withBert = items.filter(r => typeof r.bert_f1 === "number");

        if (withBert.length) {
          const vals = withBert.map(r => Number(r.bert_f1));
          const avg = vals.reduce((a,b) => a + b, 0) / vals.length;
          const max = Math.max(...vals);

          ragMetricsHtml = `
            <div class="rag-metrics" style="margin-top:12px; font-size:0.9rem; border-top:1px solid #eee; padding-top:8px;">
              <div style="font-weight:600; margin-bottom:4px;">Quality metrics (BERT F1 vs query)</div>
              <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:4px;">
                <span>Avg: ${avg.toFixed(3)}</span>
                <span>Best: ${max.toFixed(3)}</span>
              </div>
              <details style="margin-top:2px;">
                <summary style="cursor:pointer;">Per clause</summary>
                <ul style="margin:6px 0 0 16px; padding:0;">
                  ${withBert.map(r => `
                    <li>[${safe(r.reference || "-")}] BERT F1 ${Number(r.bert_f1).toFixed(3)}</li>
                  `).join("")}
                </ul>
              </details>
            </div>
          `;
        }
      }

      answerBox.innerHTML = `
        <h3 style="margin-top:0">RAG Response</h3>
        <div class="markdown">${renderMarkdown(data.answer_markdown)}</div>
        ${ragMetricsHtml}
      `;
      show(answerBox, true);
    }

    
    if (!items.length) {
      const elapsedMs = Math.round(performance.now() - t0);
      resultsBox.innerHTML = `<div class="card">No results.</div>`;
      setText(statText, `No results. (time: ${elapsedMs} ms)`);
      show(infoBox, false);
      show(loader, false);                     
      return;
    }

    const frag = document.createDocumentFragment();
    items.slice(0, topK).forEach((c) => {
      const card = document.createElement("article");
      card.className = "card";

      const citation = `${c.source || ""} • ${c.reference || ""} • p.${c.page ?? ""}`.trim();
      const bertText = typeof c.bert_f1 === "number"
        ? ` • BERT F1 ${Number(c.bert_f1).toFixed(3)}`
        : "";

      card.innerHTML = `
        <div class="card__top">
          <div class="title">${safe(c.source || "Source")}</div>
          <div class="score">
            score ${Number(c.score ?? 0).toFixed(2)}${bertText}
          </div>
        </div>
        <div class="meta">
          <span class="badge">${safe(c.reference || "—")}</span>
          <span class="badge">p.${safe(String(c.page ?? ""))}</span>
          <span class="badge">${safe(c.filename || "")}</span>
        </div>
        <div class="text">${highlight(c.text || "", q)}</div>
        <div class="actions">
          <button class="btn" type="button">Copy citation
            <img src="copy.png" width="16" height="16" alt="copy">
          </button>
        </div>
      `;
      card.querySelector(".btn").addEventListener("click", () => copy(citation));
      frag.appendChild(card);
    });
    resultsBox.appendChild(frag);

    const methodLabel =
      method === "rag" ? "RAG (LLM)"
      : method === "hybrid" ? "Hybrid"
      : method;

    const elapsedMs = Math.round(performance.now() - t0);  // <-- NEW

    setText(
      statText,
      `Method: ${methodLabel} • Source: ${source.toUpperCase()} • time: ${elapsedMs} ms`
    );
    setText(countPill, `${Math.min(items.length, topK)} / ${total}`);
    show(countPill, true);
    show(infoBox, false);

  } catch (err) {
    setText(errBox, err.message || "Error");
    show(errBox, true);
    show(infoBox, false);
    setText(statText, "Error");
  } finally {
    show(loader, false);                       
  }
});

window.addEventListener("DOMContentLoaded", () => {
  form.dispatchEvent(new Event("submit"));
});
