# Legal Case RAG — Retrieval-Augmented Generation over U.S. Supreme Court Opinions

A Retrieval-Augmented Generation pipeline that answers questions about, and summarizes, U.S. Supreme
Court cases — built end to end on **Databricks Free Edition** and fronted by a **Streamlit app**
running as a Databricks App.

Supreme Court opinions run to tens of thousands of characters, far past what fits in a single LLM
prompt. This project indexes them as embedded passages, retrieves only what a question actually needs,
and grounds the model's answer in those passages. Generated summaries are then scored against the
Court's own syllabus using ROUGE.

![Databricks RAG Pipeline Architecture](Databricks%20RAG%20Pipeline%20Architecture.png)

## What's in here

| File | What it is |
|---|---|
| `RAG_LEGAL_SUMMARIZATION.ipynb` | The full pipeline — load, chunk, embed, index, retrieve, generate, evaluate |
| `app.py` | Streamlit front end: ask a question, see the retrieved passages and the grounded answer |
| `app.yaml` | Databricks App entry point |
| `requirements.txt` | Python dependencies for the app |
| `Databricks RAG Pipeline Architecture.png` | Architecture diagram |

## Stack

| Layer | Choice |
|---|---|
| Dataset | [`ChicagoHAI/CaseSumm`](https://huggingface.co/datasets/ChicagoHAI/CaseSumm) — Supreme Court opinions paired with their official syllabi |
| Storage | Delta tables in Unity Catalog (`workspace.legal`) |
| Chunking | LangChain `RecursiveCharacterTextSplitter`, 800 chars / 150 overlap |
| Embeddings | `databricks-bge-large-en` (Foundation Model API) |
| Vector store | FAISS via the LangChain wrapper |
| Generation | `databricks-meta-llama-3-1-8b-instruct` (Foundation Model API) |
| Evaluation | ROUGE-1 / ROUGE-2 / ROUGE-L, precision · recall · F1 |
| UI | Streamlit, deployed as a Databricks App |

## How it works

**1 · Ingest.** A sample of 150 cases is pulled from Hugging Face, checked for nulls and empty
opinions, and written to `workspace.legal.cases` with the opinion as `text` and the syllabus as
`summary`.

**2 · Chunk.** Each opinion is split into 800-character passages with 150 characters of overlap —
small enough for the embedding model's context window, overlapped so a sentence spanning a boundary
keeps its meaning. 150 cases produce **3,736 chunks**, persisted to `workspace.legal.case_chunks`
with `case_id` / `chunk_id` so every passage traces back to its source.

**3 · Embed and index.** Chunks are embedded in batches through the BGE endpoint and loaded into a
FAISS index, carrying their case and chunk ids as metadata.

**4 · Retrieve and generate.** A question is embedded and matched against the index; the top-k
passages become the context for a prompt that instructs the model to answer *only* from that context
and to say so when the context is insufficient — the main guard against invented case law.

**5 · Evaluate.** The pipeline summarizes a case and the result is scored against that case's
official syllabus. The syllabus is written by the Court's own attorneys, which makes it an unusually
clean ground truth for legal summarization.

## Results

ROUGE scores for the generated summary against the ground-truth syllabus:

| Metric | Precision | Recall | F1 |
|---|---:|---:|---:|
| ROUGE-1 | – | – | – |
| ROUGE-2 | – | – | – |
| ROUGE-L | – | – | – |

Reading these: **ROUGE-1** (single-word overlap) shows whether the summary picked up the right
subject matter and legal terminology. **ROUGE-2** (two-word sequences) is always much lower, because
the model paraphrases rather than copies. **ROUGE-L** (longest common subsequence) rewards presenting
information in a similar order without requiring contiguous matches.

Precision above recall is the expected shape here: the generated summary is short and accurate, while
the syllabus covers the whole opinion. Retrieving a handful of passages from a very long opinion caps
how much of the syllabus can be covered.

## Running it

### The notebook

1. Import `RAG_LEGAL_SUMMARIZATION.ipynb` into a Databricks workspace.
2. Attach to serverless compute and run top to bottom.
3. Parameters — sample size, chunk size and overlap, top-k, and both endpoint names — are set in the
   configuration cell near the top.

If your workspace doesn't expose the default endpoints, check **Serving → Endpoints** and substitute
what's available (for example `databricks-gte-large-en`,
`databricks-meta-llama-3-3-70b-instruct`). The embedding model used at query time must match the one
used to build the index, or query vectors won't line up with it.

### The Streamlit app

Deployed as a Databricks App, so authentication is handled by the platform — there are no tokens in
this repo.

```yaml
# app.yaml
command: ["streamlit", "run", "app.py"]
```

Deploy from the workspace UI (**Compute → Apps → Create app**) or with the CLI:

```bash
databricks apps deploy <app-name> --source-code-path /Workspace/Users/<you>/legal-rag
```

Run the notebook first — the app reads the tables and index it produces. The app's service principal
needs `CAN QUERY` on both serving endpoints and read access to the `workspace.legal` schema.

## Design notes

**Why FAISS rather than Databricks Vector Search.** Vector Search isn't available on Free Edition.
FAISS keeps the whole pipeline reproducible on the free tier, and the LangChain wrapper makes the
retrieval interface the same either way — swapping in Vector Search later is a change to one object,
not to the pipeline.

**Why 800 / 150.** Legal reasoning is dense and cross-referential, so passages need enough room to
carry a complete thought while staying well inside the embedding model's window. The overlap stops a
holding that straddles a boundary from being split across two passages that each look irrelevant.

**Why the syllabus is the ground truth.** It's an abstractive summary of the same opinion, written by
domain experts, published alongside it. That's rarer and more reliable than the machine-generated
references most summarization benchmarks rely on.

## Limitations

- Evaluation is on a **sample of cases**, not the full 27k-opinion dataset.
- **ROUGE measures word overlap, not legal correctness.** A summary that is wrong but reuses the
  syllabus's vocabulary can score well; a correct paraphrase can score badly.
- Retrieving a small number of passages from a very long opinion **necessarily omits material**.
- Answer quality is bounded by **retrieval quality** — if the right passage isn't retrieved, the model
  can't recover it.
- Similarity search has **no notion of legal weight**: it can't tell a holding from dicta or from
  procedural history.

## Possible extensions

- Hybrid retrieval (BM25 + dense) with a cross-encoder reranker.
- Map-reduce summarization across all of a case's chunks, so coverage isn't capped by top-k.
- Semantic evaluation — BERTScore, or an LLM-as-judge rubric for legal accuracy — alongside ROUGE.
- Citation checking: verify each generated sentence against a specific retrieved passage.
- Migration to Databricks Vector Search and Model Serving on a paid workspace.

## Credits

Dataset: [CaseSumm](https://huggingface.co/datasets/ChicagoHAI/CaseSumm) (ChicagoHAI). Please check the
dataset card for its licence and terms before reusing it.
