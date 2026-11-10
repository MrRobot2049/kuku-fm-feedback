# A review evidence assistant for Kuku FM

## Problem and proposal

**Problem 3:** build a useful AI prototype for Kuku FM. I chose review triage as a testable product hypothesis: product and content teams may benefit from asking a natural-language question over public listener reviews and getting relevant evidence with citations. The prototype demonstrates this workflow; I have not validated the underlying internal workflow with Kuku FM employees.

The input is 185 English Google Play reviews for Kuku FM's India listing, collected on November 10, 2026 (200 requested, empty and duplicate text removed). Reviews are self-selected, English-only, and not representative of all listeners. They report experiences; the system does not verify whether billing or product events occurred.

## How it works

1. Plan 2–4 search queries with a configured LLM, or use transparent rule-based expansions in offline mode.
2. Embed each review and query with `sentence-transformers/all-MiniLM-L6-v2`; retrieve candidates by cosine similarity.
3. Check candidate relevance and rerank with `cross-encoder/ms-marco-MiniLM-L6-v2`. The relative score is uncalibrated and is not a probability.
4. Ask the LLM to synthesize only from cited reviews, then make a second LLM call to audit support and citation IDs.
5. If the LLM or audit is unavailable, return evidence-only output or preserve the draft with a visible audit-unavailable status.

The Streamlit app runs evidence-only with no API key. Optional LLM mode accepts a provider key for that session. The checked-in API example uses Groq `openai/gpt-oss-20b`; keys are not stored in the repository.

## Observed behavior and quality

The captured Groq run on unexpected payments/refund support returned reviews R176, R095, R162, R129, R092, R183, and R140, and the answer explicitly described these as individual reports rather than evidence of prevalence. The raw output is preserved in `raw_outputs/`.

Twenty-five automated tests cover dataset fields and integrity, query rewrites across payment, support, playback, catalog, reliability, language, and downloads, malformed LLM planning output, provider settings and missing keys, retrieval ordering and citations, reranker filtering, offline behavior, citation-audit failure, input validation, and pipeline integration. Tests mock the API and model stages where possible; no paid API is called.

The output folder also preserves weaker offline runs. For example, a vague playback query returns some weak matches; the reliability query includes generic “bad app” reviews. This shows the current retrieval and relevance cutoff can return misleading material. Review the evidence and manually label examples before relying on it.

## Assumptions, scale, and limits

- Public store reviews are a reasonable prototype proxy for feedback, but they are biased toward people who choose to post.
- Review IDs map to the CSV row order. Names and profile identifiers are not included.
- The current sample is small enough for local embeddings and cross-encoder reranking. For a larger review history, use a vector index, incremental embedding updates, language-aware retrieval, and monitored scheduled ingestion.
- Before production, evaluate a labeled query/review set with Recall@k, citation support, ranking quality, and human usefulness ratings. Track false positives and abstention quality; require a human to verify any product decision.
- The answer audit is an LLM check, not a formal guarantee. Provider availability, free quotas, charges, and data handling vary.

## Tools and one observed failure

Implemented with Python, Streamlit, sentence-transformers, pandas, and an OpenAI-compatible API client. An early rule-based query expander matched “listen” as a substring of “listeners,” adding irrelevant audio terms to payment searches. I changed matching to whole-word cues and added a regression test. Remaining retrieval weaknesses are shown in the unedited raw runs.
