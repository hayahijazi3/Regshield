
# Python Backend (FastAPI) for Clause Search

## Install
```bash
python3 -m venv .venv #python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt #python -m pip install --upgrade pip
```

## Run
Place your PDFs and set env vars (or keep defaults):
```bash
export PDPL_PATH=/path/to/PDPL.pdf
export ECC_PATH=/path/to/ecc-en.pdf
export ALLOW_ORIGINS=http://localhost:5173
python server_app.py
```

## Endpoints
- GET /health
- GET /index?limit=200
- GET /search?query=...&top_k=20
- POST /reindex
