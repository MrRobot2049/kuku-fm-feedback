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
2. Each review is kept as a parent evidence record. Reviews longer than 48 words are split into overlapping 48-word child passages (12-word overlap); shorter reviews remain a single child. `sentence-transformers/all-MiniLM-L6-v2` embeds child passages and query variants. The precomputed corpus child vectors are checked into `artifacts/`, so the review corpus does not need to be re-embedded on first use.
3. Bi-encoder cosine similarity retrieves child passages across all queries, then keeps the best-matching child for each parent review.
4. `cross-encoder/ms-marco-MiniLM-L6-v2` scores each retrieved child passage against the original question, removes large low-score outliers relative to the best match, and reranks the parent reviews. Answers and citations use the complete parent review. Cross-encoder scores are uncalibrated ranking scores, not probabilities.
5. The LLM drafts an answer using retrieved reviews and IDs, then a second call checks citation support. If an LLM is unavailable, evidence-only mode still works.

## Quick start: terminal

Requires Python 3.11 or newer and Git.

```bash
git clone --branch parent-child-retrieval https://github.com/MrRobot2049/kuku-fm-feedback.git
cd kuku-fm-feedback
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

#### Evidence-only search (no API key)

```bash
python agentic_rag.py "What do reviewers say about payment problems?" --offline --debug
```

#### LLM summary with Groq

Copy `.env.example` to `.env`, replace the placeholder with your own Groq API key, then run a question without `--offline`:

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY to your key
python agentic_rag.py "What do reviewers say about payment problems?" --debug
```

The CLI reads `.env` automatically. **Never commit `.env` or put a real key in `.env.example`.** The checked-in `.env.example` is a safe template. On Render, the hosted website uses the `GROQ_API_KEY` secret configured for the service; visitors do not need to enter a key. Anyone using the public LLM summary mode consumes the service owner's Groq quota, so share the URL accordingly.

#### Run the web UI locally

After activating the virtual environment and configuring `.env` if you want summaries:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal. Choose **Evidence only** for a no-key search or **LLM summary** to use the configured Groq key.

On first search, the bi-encoder and cross-encoder still need to load; the checked-in vectors only skip embedding all review passages. This app can exceed Render Free's 512 MB memory limit while loading its neural models. See [the free-tier deployment note](#deployment).

Groq usage, model availability, and data policies can change; check Groq's current terms and quotas before use. Do not submit private customer data.

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

The web UI is a Docker-based Streamlit service. Set `GROQ_API_KEY` as a secret in the hosting provider's environment settings; do not upload `.env` or commit a real key. The public app is intended as a demo. Render Free provides 512 MB RAM, and this app's local bi-encoder plus cross-encoder exceeded that limit during a live request. The precomputed corpus vectors remove the document-embedding step, but the query encoder and reranker still load into the service process. Use a host with more memory or move model inference to a separate service before relying on the live app.

## Sources and tools

- Public inputs: [Kuku FM Google Play listing](https://play.google.com/store/apps/details?id=com.vlv.aravali&gl=IN)
- Collection package: [google-play-scraper](https://github.com/JoMingyu/google-play-scraper)
- Embeddings: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Reranking: [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- LLM used in captured run: Groq API with `openai/gpt-oss-20b`
- Build assistance: OpenAI Codex

One early rule-based rewrite incorrectly treated the word “listeners” as a request about audio playback and appended audio terms. I changed matching to whole-word cues and added a regression test.
