# RegShield: AI-Powered Compliance Assistant for Saudi Regulations

RegShield is an AI-based compliance assistant that helps organizations measure their alignment with the Saudi Personal Data Protection Law (PDPL) and the National Cybersecurity Authority Essential Cybersecurity Controls (NCA-ECC). The system provides clause-level retrieval and policy comparison features that support faster, clearer, and more systematic compliance assessments.

---

##Overview

RegShield analyzes regulatory texts and organizational policies to:

- Answer compliance questions at the clause level.
- Compare internal policies to PDPL and NCA-ECC.
- Detect missing or weak coverage of regulatory obligations.
- Provide AI-supported guidance based on retrieved evidence.

The system supports four retrieval modes: lexical, semantic, hybrid, and RAG. This combination allows both precise keyword matching and deeper semantic understanding of queries and policy text.

---

##Key Features

- Clause-level search over PDPL and NCA-ECC.
- Multiple retrieval modes:
  - Lexical (TF-IDF)
  - Semantic (MiniLM embeddings)
  - Hybrid (weighted combination)
  - RAG (retrieval augmented generation)
- Query-based compliance questions.
- Policy comparison to detect coverage gaps.
- Backend API built with FastAPI.
- Frontend built with React (Vite) for interactive use.

---

##System Architecture

High level:

- **Backend (Python + FastAPI)**  
  - Loads and preprocesses PDPL and NCA-ECC PDFs.  
  - Splits text into clauses and chunks.  
  - Builds lexical and semantic indices.  
  - Exposes search and indexing endpoints.

- **Retrieval Layer**  
  - Lexical search with TF-IDF.  
  - Semantic search with MiniLM embeddings.  
  - Hybrid scoring that combines both.  
  - RAG endpoint that uses retrieved clauses as context.

- **Frontend (React + Vite)**  
  - Interface for query search.  
  - Interface for uploading and comparing policies.  
  - Displays ranked clauses, scores, and gap analysis.

(You can add an architecture image here later, for example `![Architecture](./docs/architecture.png)`.)

---

##Tech Stack

- **Backend**: Python,   
- **Retrieval**: TF-IDF, MiniLM embeddings, RAG  
- **Frontend**: React, JavaScript 
- **Vector Store / Indexing**: local storage
- **Others**: dotenv for environment variables, requests, etc.

---



