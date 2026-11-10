# Raw outputs

These are direct, unedited outputs from running the prototype against the checked-in review sample. The four `*-offline.txt` files are stdout/stderr captured together from `agentic_rag.py --offline --debug`; they do not make LLM API calls. The Groq file is the terminal text supplied by the project author after an actual Groq run and is preserved byte-for-byte.

| File | Input / mode |
| --- | --- |
| `groq-live-payment-refund-2026-11-11.txt` | Unexpected payments and refund support; Groq `openai/gpt-oss-20b`, LLM query rewrites, cited synthesis, and citation audit. User-supplied terminal capture. |
| `playback-offline.txt` | Playback, buffering, and listening; offline rule-based query expansion. |
| `catalog-discovery-offline.txt` | Finding stories and available content; offline rule-based query expansion. |
| `offline-downloads-offline.txt` | Episode downloads and offline listening; offline rule-based query expansion. |
| `login-reliability-offline.txt` | Reliability, crashes, and login issues; offline rule-based query expansion. |

To reproduce an offline sample from the project root:

```bash
python agentic_rag.py "What problems do reviewers report with playback, buffering, or listening?" --offline --debug
```

The retrieved examples are not hand-edited or curated. Some results are weak or irrelevant, especially for vague technical-reliability questions; this is intentional evidence of current failure modes, not a claim that the system is production accurate. Inspect citations against the CSV before drawing conclusions.
