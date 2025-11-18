import os, re, json, threading
from dataclasses import dataclass, asdict
from typing import List, Tuple
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np


# --- TF-IDF (lexical) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# --- Embeddings (semantic / hybrid / rag retrieval) ---
from sentence_transformers import SentenceTransformer

# --- PDF parsing ---
try:
    import PyPDF2
except Exception:
    PyPDF2 = None
try:
    import pdfplumber
except Exception:
    pdfplumber = None

# --- .env ---
from dotenv import load_dotenv, find_dotenv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = find_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(ENV_PATH, override=True)

def resolve_env_or_default(key: str, default_filename: str) -> str:
    val = os.environ.get(key)
    if val and os.path.isabs(val) and os.path.exists(val):
        return val
    if val:
        cand = os.path.normpath(os.path.join(BASE_DIR, val))
        if os.path.exists(cand):
            return cand
        cand2 = os.path.normpath(os.path.join(os.path.dirname(BASE_DIR), val))
        if os.path.exists(cand2):
            return cand2
    return os.path.join(BASE_DIR, default_filename)

PDPL_PATH = resolve_env_or_default("PDPL_PATH", "PDPL.pdf")
ECC_PATH  = resolve_env_or_default("ECC_PATH",  "ecc-en.pdf")

print(f"[startup] PDPL_PATH = {PDPL_PATH} (exists={os.path.exists(PDPL_PATH)})")
print(f"[startup] ECC_PATH  = {ECC_PATH} (exists={os.path.exists(ECC_PATH)})")

APP_HOST  = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT  = int(os.getenv("APP_PORT", "8000"))
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# PDFs we index
PDFS = [
    (PDPL_PATH, "PDPL (Implementing Regulation)"),
    (ECC_PATH,  "NCA Essential Cybersecurity Controls (ECC-1:2018)"),
]

# Source filters (UI 'sources' param)
SOURCE_LABELS = {
    "pdpl": "PDPL (Implementing Regulation)",
    "ecc":  "NCA Essential Cybersecurity Controls (ECC-1:2018)",
}
def parse_sources_param(s: str):
    if not s or s.lower().strip() in ("", "all", "*"):
        return None
    wanted = set()
    for part in re.split(r"[,;\s]+", s.strip()):
        if not part: continue
        key = part.lower()
        wanted.add(SOURCE_LABELS.get(key, part))
    return wanted

def source_allowed(clause_source: str, allowed_set):
    if allowed_set is None:
        return True
    return clause_source in allowed_set

# ---------- PDF reading / clause splitting ----------
def read_pdf_text(path: str) -> List[Tuple[int, str]]:
    """Return list[(page_no, text)] with non-empty text (raw page text)."""
    pages: List[Tuple[int, str]] = []
    if not os.path.exists(path):
        return pages
    if PyPDF2 is not None:
        try:
            with open(path, "rb") as f:
                r = PyPDF2.PdfReader(f)
                for i, page in enumerate(r.pages):
                    try:
                        t = page.extract_text() or ""
                    except Exception:
                        t = ""
                    if t.strip():
                        pages.append((i + 1, t))
        except Exception:
            pass
    if not pages and pdfplumber is not None:
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    t = page.extract_text() or ""
                    if t and t.strip():
                        pages.append((i + 1, t))
        except Exception:
            pass
    return pages

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def split_into_clauses_raw_norm(text: str) -> List[Tuple[str, str]]:
    """Return (raw, norm) chunks. raw keeps original line breaks; norm is whitespace-normalized."""
    t = text.replace("\r", "")
    t = re.sub(r"\n{3,}", "\n\n", t)
    anchors = re.split(
        r"(?=^Article\s+\d+)|(?=^\d-\d-(?:\d|-){1,6}\b)|(?=^[A-Z][A-Za-z \-/()]{5,}\s+\d-\d\b)",
        t, flags=re.MULTILINE
    )
    parts: List[Tuple[str, str]] = []
    for seg in anchors:
        seg = seg.strip("\n")
        if not seg:
            continue
        for p in re.split(r"\n{2,}", seg):
            raw = p.strip("\n ")
            norm = normalize_whitespace(raw)
            if len(norm) > 50:
                parts.append((raw, norm))
    return parts

def guess_reference(chunk: str, source_label: str) -> str:
    m = re.search(r"(Article\s+\d+)(?::?\s*([^\n]+)?)?", chunk, re.IGNORECASE)
    if m:
        title = (m.group(2) or "").strip()
        ref = m.group(1).title()
        return f"{ref}" + (f": {title}" if title else "")
    m2 = re.search(r"\b\d-\d-(?:\d|-){1,6}\b", chunk)
    if m2:
        return m2.group(0)
    words = chunk.split()
    return " ".join(words[:8]) + ("..." if len(words) > 8 else "")

# ---------- Data models ----------
@dataclass
class Clause:
    source: str
    filename: str
    page: int
    reference: str
    text_raw: str   # verbatim
    text_norm: str  # normalized

class SearchResponseItem(BaseModel):
    source: str
    filename: str
    page: int
    reference: str
    text: str
    score: float
    bert_f1: float | None = None   # quality metric in [0,1]

class SearchResponse(BaseModel):
    query: str
    total_matches: int
    returned: int
    results: List[SearchResponseItem]
    answer_markdown: str | None = None

# ---------- Index / cache ----------
INDEX: List[Clause] = []
INDEX_PATH = os.path.join(BASE_DIR, "index.json")

def load_index_from_disk() -> bool:
    global INDEX
    if os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not items or ("text_raw" not in items[0] or "text_norm" not in items[0]):
                print("[index] cached index schema is old; will rebuild.")
                return False
            INDEX = [Clause(**it) for it in items]
            print(f"[index] loaded cached index: {len(INDEX)} clauses")
            return True
        except Exception as e:
            print(f"[index] failed to load cache: {e}")
    return False

def save_index_to_disk():
    try:
        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in INDEX], f, ensure_ascii=False)
        print(f"[index] saved index: {len(INDEX)} clauses")
    except Exception as e:
        print(f"[index] failed to save cache: {e}")

def build_index() -> List[Clause]:
    if load_index_from_disk():
        return INDEX
    clauses: List[Clause] = []
    for path, label in PDFS:
        page_texts = read_pdf_text(path)
        print(f"[index] {label}: {len(page_texts)} pages with text (file={os.path.basename(path)})")
        for page_no, text in page_texts:
            if not text or len(text.strip()) < 20:
                continue
            for raw, norm in split_into_clauses_raw_norm(text):
                ref = guess_reference(norm, label)
                clauses.append(Clause(
                    source=label,
                    filename=os.path.basename(path),
                    page=page_no,
                    reference=ref,
                    text_raw=raw,
                    text_norm=norm
                ))
    print(f"[index] built fresh: {len(clauses)} clauses")
    return clauses

def ensure_index():
    global INDEX
    if not INDEX:
        INDEX = build_index()
        save_index_to_disk()

# ---------- TF-IDF (lexical) ----------
TFV: TfidfVectorizer | None = None
TFIDF_MTX = None
LEXICAL_READY = False

def build_tfidf():
    global TFV, TFIDF_MTX, LEXICAL_READY
    ensure_index()
    if not INDEX:
        TFV, TFIDF_MTX, LEXICAL_READY = None, None, False
        return
    TFV = TfidfVectorizer(ngram_range=(1,2), stop_words="english", max_df=0.98)
    X = TFV.fit_transform([c.text_norm for c in INDEX])
    TFIDF_MTX = normalize(X, norm="l2", axis=1)  # cosine
    LEXICAL_READY = True
    print(f"[tfidf] built TF-IDF: {TFIDF_MTX.shape}")

def ensure_tfidf():
    if not LEXICAL_READY or TFV is None or TFIDF_MTX is None:
        build_tfidf()

def search_lexical_tfidf(query: str, top_k: int, candidates: list[Clause]):
    ensure_tfidf()
    if TFV is None or TFIDF_MTX is None or not candidates:
        return []
    cand_idx = [INDEX.index(c) for c in candidates]
    sub = TFIDF_MTX[cand_idx]                                 # (M,D)
    qv = normalize(TFV.transform([query]), norm="l2", axis=1) # (1,D)
    sims = (sub @ qv.T).toarray().ravel()                     # (M,)
    order = np.argsort(-sims)[:top_k]
    return [(candidates[i], float(sims[i])) for i in order if sims[i] > 0]

# ---------- Embeddings (semantic) ----------
EMBEDDER: SentenceTransformer | None = None
CLAUSE_EMB = None
EMBED_DIM = 384
EMB_PATH = os.path.join(BASE_DIR, "embeddings.npy")
MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedder():
    global EMBEDDER
    if EMBEDDER is None:
        EMBEDDER = SentenceTransformer(MODEL_NAME)
    return EMBEDDER

def clause_repr(c: Clause) -> str:
    body = c.text_norm if len(c.text_norm) < 1200 else c.text_norm[:1200]
    return f"{c.source} | {c.reference} | p.{c.page}\n{body}"

def try_load_embeddings_from_disk() -> bool:
    global CLAUSE_EMB
    if os.path.exists(EMB_PATH):
        try:
            arr = np.load(EMB_PATH)
            if len(INDEX) and arr.shape[0] == len(INDEX) and arr.shape[1] == EMBED_DIM:
                CLAUSE_EMB = arr
                print(f"[emb] loaded cached: {CLAUSE_EMB.shape}")
                return True
        except Exception as e:
            print(f"[emb] failed to load cache: {e}")
    return False

def build_embeddings():
    global CLAUSE_EMB
    ensure_index()
    if try_load_embeddings_from_disk():
        return
    if not INDEX:
        CLAUSE_EMB = np.empty((0, EMBED_DIM), dtype=np.float32)
        print("[emb] no clauses; embeddings empty")
        return
    enc = get_embedder()
    texts = [clause_repr(c) for c in INDEX]
    vecs = enc.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    CLAUSE_EMB = np.asarray(vecs, dtype=np.float32)
    np.save(EMB_PATH, CLAUSE_EMB)
    print(f"[emb] built & saved: {CLAUSE_EMB.shape}")

def ensure_embeddings():
    global CLAUSE_EMB
    if CLAUSE_EMB is None or (len(INDEX) and CLAUSE_EMB.shape[0] != len(INDEX)):
        build_embeddings()

def search_semantic_subset(query: str, candidates: List[Clause], top_k: int):
    """Semantic scoring restricted to candidates; return scores mapped to [0,1]."""
    ensure_index()
    try:
        ensure_embeddings()
    except Exception as e:
        print(f"[emb] ensure_embeddings failed: {e}")
        return []
    if CLAUSE_EMB is None or CLAUSE_EMB.size == 0 or not candidates:
        return []
    enc = get_embedder()
    qv = enc.encode([query], normalize_embeddings=True)[0]
    idx_map = [INDEX.index(c) for c in candidates]
    sims = [(c, float(CLAUSE_EMB[i] @ qv)) for c, i in zip(candidates, idx_map)]
    sims.sort(key=lambda x: x[1], reverse=True)
    sims = sims[:top_k]
    return [(c, (s + 1.0) / 2.0) for c, s in sims]  # [-1,1] -> [0,1]

# ---------- BERTScore ----------
# Uses a persistent scorer; rescale with baseline to map roughly into [0,1]
# Uses raw BERTScore F1 in [0,1]
BERTSCORE_MODEL = os.getenv("BERTSCORE_MODEL", "roberta-large")
_BERTSCORER = None

def get_bertscorer():
    """Initialize once; use RAW F1 (0..1), no baseline rescaling."""
    global _BERTSCORER
    if _BERTSCORER is None:
        from bert_score import BERTScorer
        _BERTSCORER = BERTScorer(
            model_type=BERTSCORE_MODEL,
            lang="en",
            rescale_with_baseline=False   # <-- key fix
        )
    return _BERTSCORER


def _shorten(s: str, limit: int = 700) -> str:
    s = (s or "").strip()
    return s if len(s) <= limit else s[:limit]


def compute_bertscore_f1(query: str, texts: list[str]) -> list[float] | None:
    """Return list of BERTScore F1 in [0,1]."""
    if not texts:
        return None
    try:
        scorer = get_bertscorer()
        cands = [_shorten(t) for t in texts]
        refs  = [_shorten(query)] * len(texts)

        P, R, F1 = scorer.score(cands, refs)   # RAW scores, no baseline
        vals = []
        for f in F1:
            x = float(f.item())
            # Force to [0,1]
            if x < 0.0: x = 0.0
            if x > 1.0: x = 1.0
            vals.append(x)
        return vals
    except Exception as e:
        print(f"[bertscore] error: {e}")
        return None


# ---------- LLM (RAG generation) ----------
def gemini_answer_from_results(query: str, items: list[SearchResponseItem]) -> str:
    if not GEMINI_API_KEY:
        return "_LLM disabled: set GEMINI_API_KEY_"
    try:
        import google.generativeai as genai
    except Exception as e:
        return f"_LLM unavailable: {e}_"

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    top = items[:5]
    corpus = []
    for i, it in enumerate(top, 1):
        label = f"[{i}] {it.source} — {it.reference} (p.{it.page}, {it.filename})"
        snippet = it.text.strip()
        if len(snippet) > 1500:
            snippet = snippet[:1500] + "…"
        corpus.append(f"{label}\n{snippet}")

    prompt = f"""
You are a compliance assistant. Answer the user's query using ONLY the quoted clauses below.
- Be concise and cite sources inline like [1], [2] referring to the items.
- If something is not covered, say so briefly.
- Do NOT fabricate; only derive from the given clauses.

Query:
{query}

Clauses:
{chr(10).join(corpus)}

Return Markdown with:
- A short answer (2–6 sentences)
- A bullet list of cited clauses with their labels
"""
    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip() if hasattr(resp, "text") else str(resp).strip()
        return text or "_No answer generated._"
    except Exception as e:
        return f"_Gemini error: {e}_"

# ---------- API ----------
from enum import Enum
class Method(str, Enum):
    lexical = "lexical"   # TF-IDF
    semantic = "semantic" # MiniLM
    hybrid = "hybrid"     # alpha blend
    rag = "rag"           # semantic retrieve + LLM generate

# CORS
raw = ALLOW_ORIGINS or ""
origins = ["*"] if raw.strip() == "*" else [o.strip() for o in re.split(r"[,;\s]+", raw) if o.strip()]
print(f"[startup] ALLOW_ORIGINS parsed: {origins}")

app = FastAPI(title="Regulation Clause Search API", version="0.9.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type", "Authorization"],
)

@app.on_event("startup")
def _startup():
    ensure_index()
    def _bg():
        try:
            build_embeddings()
        except Exception as e:
            print(f"[emb] background build failed: {e}")
        try:
            build_tfidf()
        except Exception as e:
            print(f"[tfidf] background build failed: {e}")
        # Optional: warm-up BERTScore so first request isn't slow
        try:
            if os.getenv("WARMUP_BERTSCORE", "1") == "1":
                s = get_bertscorer()
                _ = s.score(["warm up"], ["warm up"])
                print("[bertscore] warm-up done")
        except Exception as e:
            print(f"[bertscore] warm-up failed: {e}")
    threading.Thread(target=_bg, daemon=True).start()

@app.get("/health")
def health():
    emb_ready = CLAUSE_EMB is not None and getattr(CLAUSE_EMB, "size", 0) > 0
    return {
        "status": "ok",
        "pdpl": os.path.exists(PDPL_PATH),
        "ecc": os.path.exists(ECC_PATH),
        "count": len(INDEX),
        "embeddings_ready": emb_ready,
        "tfidf_ready": LEXICAL_READY,
        "bertscore_available": True,  # will initialize on demand
    }

@app.get("/index")
def get_index_preview(limit: int = Query(200, ge=1, le=5000)):
    ensure_index()
    return {"count": len(INDEX), "preview": [asdict(c) for c in INDEX[:limit]]}

@app.get("/search", response_model=SearchResponse)
def search(
    query: str,
    top_k: int = Query(10, ge=1, le=100),
    method: Method = Method.lexical,
    alpha: float = Query(0.6, ge=0.0, le=1.0),
    sources: str = Query("all", description="Which documents to search: all | pdpl | ecc | comma-separated | full labels"),
    with_bertscore: bool = Query(True, description="If available, include BERTScore F1 for returned hits"),
):
    ensure_index()
    if not INDEX:
        return SearchResponse(query=query, total_matches=0, returned=0, results=[], answer_markdown=None)

    allowed = parse_sources_param(sources)
    candidates = [c for c in INDEX if source_allowed(c.source, allowed)]
    if not candidates:
        return SearchResponse(query=query, total_matches=0, returned=0, results=[], answer_markdown=None)

    if method == Method.lexical:
        scored = search_lexical_tfidf(query, top_k=top_k, candidates=candidates)

    elif method == Method.semantic:
        scored = search_semantic_subset(query, candidates, top_k=top_k)

    elif method == Method.hybrid:
        lex = search_lexical_tfidf(query, top_k=max(5*top_k, 100), candidates=candidates) or []
        sem = search_semantic_subset(query, candidates, top_k=max(5*top_k, 100)) or []
        L = {id(c): s for c, s in lex}  # 0..1
        S = {id(c): s for c, s in sem}  # 0..1
        keys = set(L) | set(S)
        id2 = {id(c): c for c in candidates}
        scored = []
        for k in keys:
            ls = L.get(k, 0.0); ss = S.get(k, 0.0)
            scored.append((id2[k], alpha*ss + (1.0-alpha)*ls))
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

    else:  # Method.rag
        scored = search_semantic_subset(query, candidates, top_k=top_k)

    total = len(scored)
    top = scored[:top_k]

    # Prepare results (verbatim text)
    results = [SearchResponseItem(
        source=c.source,
        filename=c.filename,
        page=c.page,
        reference=c.reference,
        text=c.text_raw,
        score=round(float(s), 3),
    ) for c, s in top]

    # --- BERTScore on returned hits (robust & clamped) ---
    if with_bertscore and results:
        try:
            texts = [r.text for r in results]
            f1s = compute_bertscore_f1(query, texts)
            if f1s:
                for r, f in zip(results, f1s):
                    r.bert_f1 = round(float(f), 3)
        except Exception as e:
            print(f"[bertscore] attach error: {e}")

    # --- RAG answer only for rag mode ---
    answer_md = None
    if method == Method.rag and results:
        answer_md = gemini_answer_from_results(query, results)

    return SearchResponse(
        query=query,
        total_matches=total,
        returned=len(results),
        results=results,
        answer_markdown=answer_md
    )

@app.post("/reindex")
def reindex():
    global INDEX, CLAUSE_EMB
    INDEX = build_index()
    save_index_to_disk()
    CLAUSE_EMB = None
    try:
        if os.path.exists(EMB_PATH):
            os.remove(EMB_PATH)
    except Exception:
        pass
    def _bg():
        try: build_embeddings()
        except Exception: pass
        try: build_tfidf()
        except Exception: pass
        # (optional) refresh BERTScore model in memory
        try:
            global _BERTSCORER
            _BERTSCORER = None
            _ = get_bertscorer()
        except Exception: 
            pass
    threading.Thread(target=_bg, daemon=True).start()
    return {"status": "ok", "count": len(INDEX)}

if __name__ == "__main__":
    uvicorn.run("server_app:app", host=APP_HOST, port=APP_PORT, reload=True)
