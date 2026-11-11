---
title: Kuku FM Review Triage
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---

# Kuku FM listener feedback triage

An evidence-first prototype for turning public Kuku FM Google Play reviews into a concise, cited feedback brief.

## Problem statement

**Hypothesis:** a Kuku FM product or content teammate has to scan many listener reviews to identify recurring requests and complaints, then decide what to investigate. This prototype lets the teammate ask a question in plain language and returns relevant review evidence with a short summary. The public review sample is a proxy for feedback, not a representative survey or a verified record of product events.

## Dataset

- Kuku FM Google Play package: `com.vlv.aravali`
- Collected November 10, 2026: newest English (`en`) reviews in the India (`in`) storefront; requested 200
- Saved after removing empty/duplicate text: 185 reviews
- Fields retained: text, star rating, timestamp, app version, helpful-vote count, developer reply
- Reviewer names, IDs, and profile photos omitted
- The sample is recent, self-selected, English-only, and limited to reviews returned by Google Play at collection time

The collection script is `scrape_reviews.py`; the saved dataset is `kuku_fm_reviews_en_in.csv`.

## How the assistant works

1. An LLM rewrites the question into a few search queries. If no LLM is configured, token-based rules add simple terminology variants.
2. Each review is kept as a parent evidence record. Reviews longer than 48 words are split into overlapping 48-word child passages (12-word overlap); shorter reviews remain a single child. `sentence-transformers/all-MiniLM-L6-v2` embeds child passages and query variants, with child embeddings cached locally by content hash.
3. Bi-encoder cosine similarity retrieves child passages across all queries, then keeps the best-matching child for each parent review.
4. `cross-encoder/ms-marco-MiniLM-L6-v2` scores each retrieved child passage against the original question, removes large low-score outliers relative to the best match, and reranks the parent reviews. Answers and citations use the complete parent review. Cross-encoder scores are uncalibrated ranking scores, not probabilities.
5. The LLM drafts an answer using retrieved reviews and IDs, then a second call checks citation support. If an LLM is unavailable, evidence-only mode still works.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the embedding and reranking models. In the app, **Evidence only** makes no API calls. To generate and audit an LLM answer, select **LLM summary** and choose a provider. A server-configured key is used when available; otherwise, you can enter your own key for that browser session. The CLI also reads a local `.env` file; use `.env.example` as a template and never commit `.env`.

Providers currently supported by the CLI: Gemini, Groq, OpenRouter, Cerebras, and OpenAI. Free-tier quotas, usage charges, model availability, and data policies vary by provider; check the provider console before sending requests. Do not submit private customer data.

## Run the CLI

```bash
python check_api.py
python agentic_rag.py "What are listeners saying about unexpected payments and refund support?" --debug
```

No-key evidence-only example:

```bash
python agentic_rag.py "What are listeners saying about playback problems?" --offline --debug
```

## Automated tests

The test suite uses `unittest` and mocks the embedding model, reranker, and API client where appropriate. It covers topic expansion, provider selection, citation IDs, retrieval ordering, reranker outlier handling, and full pipeline orchestration without making paid API calls.

```bash
python -m unittest discover -s tests -v
```

## Evaluation and limitations

- Review IDs in answers map to row order in the checked-in CSV, allowing a reviewer to inspect every cited source.
- Review stars and counts describe only this sample; they do not estimate the prevalence of a problem among all listeners.
- The cross-encoder cutoff is a heuristic. A relevant review can be filtered or ranked too low; evaluate labeled examples before production use.
- The answer verifier is another LLM call, not a formal guarantee. Manually inspect citations and keep a human in the loop before treating a theme as a product decision.
- A useful next evaluation is a small labeled set of query/review pairs and measured Recall@k, citation support, and human usefulness ratings.

## Raw runs

`raw_outputs/` contains unedited terminal output from the sample query and evidence-only runs on payment, refund, playback, catalog discovery, and reliability topics. `raw_outputs/README.md` indexes the queries and modes.

## Deployment

This repository is configured as a Docker-based Streamlit app on Render. The public app works in evidence-only mode. The Render service can use a provider key stored in its environment settings for LLM summaries; keep keys out of repository files. Anyone with access to the public service can use the configured key through the app's LLM mode, subject to the provider's quotas and billing terms.

## Sources and tools

- Public inputs: [Kuku FM Google Play listing](https://play.google.com/store/apps/details?id=com.vlv.aravali&gl=IN)
- Collection package: [google-play-scraper](https://github.com/JoMingyu/google-play-scraper)
- Embeddings: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Reranking: [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- LLM used in captured run: Groq API with `openai/gpt-oss-20b`
- Build assistance: OpenAI Codex

One early rule-based rewrite incorrectly treated the word “listeners” as a request about audio playback and appended audio terms. I changed matching to whole-word cues and added a regression test.
