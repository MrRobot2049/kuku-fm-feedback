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

## Run the project on your computer

Yes. To run it locally, first get a copy of the project files onto your computer. The simplest route is to clone the public GitHub repository. This downloads the code, review CSV, precomputed review vectors, and examples; it does **not** deploy or publish anything. You can also use GitHub's **Code → Download ZIP**, extract the ZIP, and open Terminal in the extracted `kuku-fm-feedback` folder. If you already have the project folder, skip this download step.

You need Python 3.11 or newer and Git (only needed for the clone option). The commands below are for macOS/Linux Terminal. Run them one at a time:

```bash
git clone --branch parent-child-retrieval https://github.com/MrRobot2049/kuku-fm-feedback.git
cd kuku-fm-feedback
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you downloaded the ZIP or already have the folder, open Terminal there and start with `python3 -m venv .venv`; then run the remaining install commands. Keep the terminal window open. The `.venv` is a local Python environment for this project.

### Choose how to run it

**Option A: Evidence-only search (no API key).** In the same Terminal window, with `.venv` activated:

```bash
python agentic_rag.py "What do reviewers say about payment problems?" --offline --debug
```

**Option B: LLM summary using your Groq API key.** Get your own key from Groq, then copy the safe example file and open `.env` in a text editor:

```bash
cp .env.example .env
open -e .env
```

Replace the placeholder value for `GROQ_API_KEY` with your key, save the file, and run:

```bash
python agentic_rag.py "What do reviewers say about payment problems?" --debug
```

The CLI reads `.env` automatically. `.env` is ignored by Git; never commit or share it. `.env.example` contains placeholders only and is safe to keep in GitHub. LLM summary mode sends the question and retrieved public review text to Groq and uses your key/quota. Evidence-only mode does not call Groq.

**Option C: Open the local web interface.** With `.venv` activated, run:

```bash
streamlit run app.py
```

Wait for Streamlit to print a **Local URL**, usually `http://localhost:8501`, then open that address in your browser. Choose **Evidence only** to search without an API key or **LLM summary** to use the Groq key in `.env`. Stop the local server with `Control-C` in Terminal.

The first run downloads the bi-encoder and cross-encoder model weights and loads both models into memory. The checked-in vectors skip embedding the review corpus again, but the query encoder and reranker still need their model files. Allow time and disk space for that first model download; this neural setup can exceed Render Free's 512 MB memory limit. See [the deployment note](#deployment).

Groq model availability, quotas, and data policies can change; check Groq's current terms before use. Do not submit private customer data.

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
