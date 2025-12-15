# Agentic Design Patterns RAG

Retrieval-Augmented Generation system built over the *Agentic Design Patterns* book. The project walks through the full lifecycle: extracting content from the PDF with OCR, enriching and chunking the text, storing it in a pgvector/Postgres database, and serving an interactive Streamlit assistant that combines hybrid search (semantic + lexical) with semantic caching and interaction logging. Utility scripts generate evaluation QA data and benchmark the pipeline end-to-end.

## Highlights
- **Document pipeline** – high-resolution OCR (DeepSeek-OCR) ➜ markdown ➜ contextual chunking with GPT structured output ➜ pgvector ingestion with metadata.
- **Hybrid retrieval** – cosine similarity + full-text search fused with Reciprocal Rank Fusion, backed by pgvector extensions.
- **Semantic cache** – nearest-neighbor lookups short-circuit generation when a similar question has already been answered.
- **Evaluation tooling** – synthetic QA generation plus automated correctness scoring to track the health of the RAG stack.
- **Streamlit front-end** – single-page chat UI that surfaces the last Q&A, reruns automatically, and logs latency/cache hits for analysis.

## Tech Stack
- Python 3.11+, Streamlit, LangChain text splitters, Pydantic
- OpenAI Responses API (`gpt-4o-mini`, `gpt-4.1-*`, `gpt-5-nano`) for embeddings, enrichment, and answer generation
- DeepSeek-OCR via Hugging Face Transformers + PyTorch
- Postgres 16 with [`pgvector`](https://github.com/pgvector/pgvector) for vector storage, HNSW indexes, and text search
- Docker Compose for the database environment

## Repository Layout

```text
├── data/
│   ├── raw/                # PDFs and OCR-ready image batches
│   ├── ocr/                # Intermediate OCR output per page
│   ├── processed/          # Markdown, enriched chunks, QA pairs, embeddings
│   └── pgvector/           # Docker volume for Postgres
├── prompts/                # YAML system/user prompts for generation + evaluation
├── src/
│   ├── preprocessing/      # PDF ➜ image utilities
│   ├── ocr/                # DeepSeek OCR driver
│   ├── llm/                # OpenAI wrapper (embeddings + answers)
│   ├── database/           # Postgres setup + hybrid search helpers
│   └── ...
├── streamlit_app.py        # Front-end application
├── step_[1-5].py           # Reproducible pipeline stages (ingest + eval)
└── docker-compose.yml      # pgvector service definition
```

## Prerequisites
- Python 3.11 or later
- Docker + Docker Compose (for Postgres/pgvector)
- GPU with CUDA for OCR (DeepSeek-OCR); CPU mode is possible but very slow
- Poppler (`pdf2image` dependency) and Tesseract prerequisites installed locally
- `.env` file with your OpenAI key:

```text
OPENAI_API_KEY=sk-...
```

Install Python dependencies inside a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install streamlit psycopg[binary] pgvector openai langchain-text-splitters python-dotenv pyyaml pdf2image
```

> `requirements.txt` only captures the heavy OCR dependencies; install the additional libraries above for the full pipeline.

## Bring Up Postgres + pgvector

```bash
docker compose up -d
```

The service exposes Postgres on `localhost:5434` with credentials defined in `docker-compose.yml`. `src/database/postgres.py` will automatically create tables and indexes on first run:
- `document_chunks` – chunk text, metadata JSONB, and embedding vectors
- `semantic_cache` – cached question embeddings + answers for fast reuse
- `interaction_log` – request/response audit trail with cache hit + latency

## Data Preparation & Ingestion (Steps 1–3)

All steps expect the OpenAI key to be set and Postgres running.

1. **Extract PDF ➜ Markdown (`step_1.py`)**

   ```bash
   python step_1.py
   ```

   - Calls `preprocess_pdf_4_ocr` to convert the PDF into page-level PNGs (`data/raw/<pdf_name>/page_XXXX.png`).
   - Uses `DeepSeek-OCR` (`src/ocr/ocr.py`) to write merged markdown to `data/processed/Agentic_Design_Patterns.mmd`.

2. **Chunk + Enrich (`step_2.py`)**

   ```bash
   python step_2.py
   ```

   - Splits markdown into overlapping 1,750-token windows with LangChain’s `RecursiveCharacterTextSplitter`.
   - Calls `OpenAI Responses API` with a structured `ChunkMetadata` schema to attach semantic titles, context expansions, headers, and keywords.
   - Persists chunks + metadata to `data/processed/enriched_chunks.json` (auto-saves every 25 chunks to survive rate limits).

3. **Vector Ingestion (`step_3.py`)**

   ```bash
   python step_3.py
   ```

   - Runs `setup_database()` to ensure all extensions/tables/indexes exist.
   - Uses `OpenaiLLM.get_embedding()` (text-embedding-3-small) per chunk.
   - Stores concatenated chunk text + context and metadata JSONB into Postgres.

## QA Dataset Generation & Evaluation (Steps 4–5)

4. **Synthetic QA Creation (`step_4.py`)**

   ```bash
   python step_4.py
   ```

   - Splits the full book into four parts and calls multiple OpenAI models (`gpt-4o-mini`, `gpt-4.1-mini`, `gpt-4.1-nano`) across assorted personas.
   - Produces ~thousands of question/answer pairs saved to `data/processed/qa_pairs.json`, used for offline evaluation and possible fine-tuning.

5. **Automated RAG Evaluation (`step_5.py`)**

   ```bash
   python step_5.py
   ```

   - Samples the QA set, retrieves supporting documents with `hybrid_search()`, and generates answers using the same prompt template as production.
   - Scores each answer via OpenAI (`AnswerEvaluation` schema) and aggregates counts for `correct`, `partially_correct`, `incorrect`, and `hallucinated`.
   - Results print to the console; extend the loop or export metrics to persist dashboards.

## Running the Streamlit Assistant

1. Ensure Postgres is running (`docker compose up -d`) and the knowledge base is ingested.
2. Export `OPENAI_API_KEY` or load from `.env` automatically via `python-dotenv`.
3. Start the UI:

```bash
streamlit run streamlit_app.py
```

The interface displays the previous Q&A at the top and pins the input at the bottom. When submitting a question:
- `hybrid_search()` retrieves top-k passages through Reciprocal Rank Fusion of vector search and Postgres full-text search.
- `search_semantic_cache()` checks for near-duplicate questions; cache hits skip generation and return instantly.
- Otherwise `OpenaiLLM.generate_answer()` merges retrieved chunks via the `generating_answer.yml` template, stores the semantic cache entry, and logs the interaction (including cache hits and end-to-end latency) in Postgres.

## Prompts & Customization
- `prompts/generating_answer.yml` – governs both answer generation and evaluation prompts.
- `prompts/rag_validation_openai.yml` and `prompts/rag_validation_gemini.yml` – optional templates for QA generation using different providers.
- Adjust chunk size, overlap, ranking weights (`top_k`, `rrf_k`), or cache threshold directly in `src/database/postgres.py` and `src/llm/openai.py`.

## Next Steps
- Wire up CI to regularly regenerate QA sets and evaluation scores.
- Extend interaction logs with retrieval stats for better observability.
- Package ingestion steps as a CLI (e.g., `python -m pipelines.ingest --pdf ...`) and automate with workflow runners.

Happy building!
