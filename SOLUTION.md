# Kuku FM listener feedback triage

**Assignment choice:** Problem 3 - a useful AI prototype for Kuku FM

## Problem, hypothesis, and scope

Product and content teams may need to find recurring complaints and requests across listener feedback. The prototype tests a narrow workflow: ask a natural-language question about reviews and receive relevant, citable examples plus an optional short synthesis. This is a product hypothesis; no Kuku FM employee interviews or workflow validation were conducted.

The checked-in CSV contains 185 non-empty, deduplicated English Google Play reviews for Kuku FM's India listing. The scraper requested 200 reviews. Reviewer names and profile identifiers were omitted. Reviews are a small, self-selected sample: they do not represent all listeners and do not verify reported billing or product events. The prototype indexes app reviews, not every Kuku FM show or episode.

## Implemented RAG pipeline

1. **Question planning.** In LLM summary mode, Groq (default model `openai/gpt-oss-20b`) rewrites the question into a few short search queries, requesting JSON. In evidence-only mode, transparent rule-based synonym groups expand the query. If LLM planning fails or returns unusable JSON, the pipeline falls back to those rules.
2. **Parent-child passages.** Each full review remains a parent evidence record. Reviews longer than 48 words are split into 48-word child passages with a 12-word overlap; shorter reviews produce one child. This corpus produces 208 children. Citations and displayed evidence always use the full parent review.
3. **Stored corpus vectors.** `sentence-transformers/all-MiniLM-L6-v2` creates normalized child vectors. The current 208-by-384 float32 matrix is checked in under `artifacts/` and keyed by a digest of model name and child text. `scripts/build_review_embeddings.py` regenerates it. This skips embedding all review passages during first use; it does not eliminate loading the encoder to embed a new query.
4. **Bi-encoder retrieval.** The query (or query variants) are embedded with the same bi-encoder. Cosine similarity is max-pooled across query variants for each child. The best child per parent is retained, and up to 30 distinct parent reviews are passed onward.
5. **Cross-encoder reranking.** `cross-encoder/ms-marco-MiniLM-L6-v2` scores the original question against each retrieved child. The app removes candidates more than five score points below the best and keeps up to eight. These relative logits are uncalibrated scores, not probabilities. This stage loads a second local model.
6. **Evidence and answer.** Evidence-only mode returns the retrieved reviews and metadata without calling an LLM. LLM summary mode sends the full parent reviews, stable IDs (`R001`, etc.), ratings, and dates to Groq for a concise answer grounded in the evidence. A third LLM call asks for a citation/support audit. If synthesis fails, the app shows evidence-only output; if the audit fails, it preserves the draft and labels the audit unavailable. The audit is an LLM check, not a formal proof that every citation is correct.

The Streamlit interface exposes **Evidence only** and **LLM summary**. The CLI supports evidence-only use with `--offline`; otherwise it uses the configured LLM. `.env.example` is a placeholder template. A real Groq key belongs in a local ignored `.env` or a hosting provider's secret settings, never in GitHub. For a public hosted app, every LLM summary request uses the service owner's configured key and quota.

## Evidence of behavior and known failures

There are 20 automated unit tests covering rule-based rewrites, LLM planning fallback, provider configuration, retrieval ordering/citations, cross-encoder filtering, offline behavior, and pipeline fallback paths. Model and API stages are mocked where appropriate. Tests check code behavior; they do not establish retrieval quality.

`raw_outputs/` preserves four unedited offline runs and one Groq payment/refund run. The captured Groq answer cites individual reports and avoids claiming prevalence. The offline playback and reliability examples include weak or generic matches. No labeled relevance benchmark or numerical Recall@k, nDCG, or citation-accuracy score has been produced. Relevant reviews can be missed, and similarity or reranker scores can be misleading. Inspect cited text in the CSV and keep a person responsible for decisions.

## Assumptions, scaling, and constraints

The prototype assumes public app-store reviews can help demonstrate a feedback-triage workflow, that English India reviews are sufficient for a first demo, and that a human will verify findings. It makes no claim about the true rate of an issue.

To scale to every show, first add a permitted feedback source that identifies the show or episode, then store each item with `show_id`, source, locale, timestamp, and a stable review ID. Ingest incrementally and deduplicate; retain parent-child links and metadata filters. For a larger corpus, replace the in-memory matrix with a vector index, batch new embeddings, and filter by show, language, and time before reranking. Evaluate a labeled query/review set per show (Recall@k, nDCG@k, citation support, abstention quality, latency, and cost). Add access controls, retention rules, and human review before operational use. These are proposed next steps, not implemented features.

The checked-in embedding matrix avoids corpus re-encoding but the query encoder and cross-encoder still load in the application process. Render Free reported the instance exceeded its 512 MB memory limit, so this neural configuration is not reliable on that tier. A host with sufficient RAM or externally hosted/lightweight model inference is needed. LLM summary mode can make up to three sequential Groq calls (planning, synthesis, audit), adding latency and consuming quota. Provider terms and data handling should be reviewed before sending review text.

## Run and inspect

The repository README has terminal and Streamlit run steps. The project, raw outputs, code, dataset, and stored vectors are available in the public GitHub repository:

- Project: https://github.com/MrRobot2049/kuku-fm-feedback/tree/parent-child-retrieval
- Raw outputs: https://github.com/MrRobot2049/kuku-fm-feedback/tree/parent-child-retrieval/raw_outputs

The source README documents the current branch's exact commands and environment setup.
