# RAFT — Journal Scraping, Chunking & RAG QA Pipeline

End-to-end pipeline that scrapes agricultural-science journal PDFs (e.g.
*The Indian Journal of Agricultural Sciences*, *The Indian Journal of
Animal Sciences*, *Potato Journal*), extracts and chunks their text,
pushes embeddings into Qdrant, generates QA pairs from the chunks, and
evaluates retrieval / faithfulness quality.

> **Note**: large data files (PDFs, CSVs, FAISS indexes, raw scraping
> output) are **not** tracked in git — see `.gitignore`. Only the code
> and notebooks needed to reproduce the pipeline are committed.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                              # Python scripts
│   ├── clean_filenames.py            # normalize downloaded journal filenames
│   ├── count_pdf_stats.py            # count PDFs / pages per journal
│   ├── pdf_processor.py              # Azure Document Intelligence extractor
│   ├── chunking.py                   # semantic chunking (OpenAI embeddings)
│   ├── analyze_chunk_tokens.py       # token-count diagnostics for chunks
│   ├── embedding_push_notebook.py    # push chunk embeddings to Qdrant
│   ├── retrieve_similar_chunks.py    # similar-chunk retrieval from Qdrant
│   └── qa_quality_analysis.py        # multi-chunk QA quality analysis
│
├── notebooks/                        # Jupyter notebooks
│   ├── scraping_journals.ipynb       # download PDFs from journal sites
│   ├── combine_chunks.ipynb          # combine / merge chunk variants
│   ├── embedding_push.ipynb          # push embeddings to Qdrant
│   ├── Qa_pair_gen.ipynb             # generate QA pairs from chunks
│   └── rag_impl.ipynb                # RAG implementation over chunks
│
├── rag_test/                         # RAG retrieval testing
│   └── rag_test.ipynb
│
├── run_eval/                         # Category-wise evaluation
│   ├── eval.ipynb
│   └── fixed_evaluation.py
│
└── (gitignored data dirs)
    chunk_data/   downloads/   journal_downloads/   search_results/
```

### Pipeline stages

1. **Scrape journals** → `notebooks/scraping_journals.ipynb`, then
   `src/clean_filenames.py`, `src/count_pdf_stats.py`
2. **Extract & chunk** → `src/pdf_processor.py`, `src/chunking.py`,
   `src/analyze_chunk_tokens.py`, `notebooks/combine_chunks.ipynb`
3. **Embeddings & vector DB** → `notebooks/embedding_push.ipynb`,
   `src/embedding_push_notebook.py`, `src/retrieve_similar_chunks.py`
4. **QA generation & RAG** → `notebooks/Qa_pair_gen.ipynb`,
   `notebooks/rag_impl.ipynb`
5. **Evaluation** → `src/qa_quality_analysis.py`,
   `rag_test/rag_test.ipynb`, `run_eval/eval.ipynb`,
   `run_eval/fixed_evaluation.py`

---

## Setup

### 1. Clone & create a virtual environment

```bash
git clone <your-repo-url> RAFT_exp
cd RAFT_exp
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root with at least:

```bash
OPENAI_API_KEY=sk-...
# Azure Document Intelligence (used by pdf_processor.py)
AZURE_DOC_INTEL_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_DOC_INTEL_KEY=...
# Qdrant (used by embedding_push_notebook.py and retrieve_similar_chunks.py)
QDRANT_URL=http://localhost:6333
```

`.env` is intentionally `.gitignore`d. Do **not** commit secrets.

### 4. (Optional) Selenium driver

`scraping_journals.ipynb` uses Selenium for journal pages that require
JavaScript. Install a matching driver (e.g. `chromedriver`) and make sure
it is on your `PATH`.

---

## Typical workflow

Run all commands from the **project root** so relative paths like
`journal_downloads/...` resolve correctly.

1. **Scrape PDFs** with `notebooks/scraping_journals.ipynb` — they land
   under `journal_downloads/<journal_name>/`.
2. **Normalize filenames**: `python src/clean_filenames.py`
3. **Extract text** from PDFs with `src/pdf_processor.py` (Azure Document
   Intelligence). Output goes into per-journal `*_text/` directories.
4. **Chunk text** using `src/chunking.py` (semantic chunking via OpenAI
   embeddings). Results are stored in `*_chunks/` directories and
   aggregated into CSVs under `chunk_data/`.
5. **Push embeddings to Qdrant** with `notebooks/embedding_push.ipynb`
   or `python src/embedding_push_notebook.py`.
6. **Generate QA pairs** from the chunks with
   `notebooks/Qa_pair_gen.ipynb`.
7. **Run RAG retrieval** experiments in `notebooks/rag_impl.ipynb` and
   `rag_test/rag_test.ipynb`.
8. **Evaluate** results with `run_eval/eval.ipynb` /
   `python run_eval/fixed_evaluation.py` and analyze QA quality with
   `python src/qa_quality_analysis.py`.

---

## Data files (not committed)

The following are produced by the pipeline and live on disk but are not
pushed to GitHub:

- `journal_downloads/`, `downloads/` — source PDFs
- `chunk_data/` — chunked text CSVs
- `chunk_embeddings*.index` — FAISS indexes
- `search_results/` — Qdrant search outputs
- `*_similar_chunks_*.{csv,json}` — retrieval analysis outputs
- `scraped_data*.csv`, `enrichment_data*.csv`, `final_combination_results*.csv`

If you are setting up the project fresh, re-run the pipeline stages
above to regenerate them.

---

## Notes & caveats

- `notebooks/Qa_pair_gen.ipynb` contains a few cells that import the
  proprietary `fc_ai` package and Google Cloud Text-to-Speech. Those
  cells are optional — skip them if you do not have access to those
  services.
- The Qdrant URL hard-coded in `src/retrieve_similar_chunks.py` and
  `src/embedding_push_notebook.py` points at an internal dev instance —
  override it with `QDRANT_URL` (or edit in place) for your environment.
- Some scripts use absolute paths (e.g. to `chunk_data/...`). Update
  them to match your local checkout before running.
