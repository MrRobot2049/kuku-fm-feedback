"""Small, inspectable agentic RAG pipeline for Kuku FM listener reviews.

Stages: LLM query planning/rewriting -> dense retrieval -> relevance gate and
cross-encoder reranking -> cited synthesis -> citation/support verification.
The LLM is optional for retrieval; without a configured provider key, query expansion is
transparent and rule-based, and the CLI prints evidence without synthesis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
DATA_PATH = ROOT / "kuku_fm_reviews_en_in.csv"
CACHE_DIR = ROOT / ".cache"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
PROVIDER_SETTINGS = {
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.8-flash"),
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1", "openai/gpt-oss-20b"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "openrouter/free"),
    "cerebras": ("CEREBRAS_API_KEY", "https://api.cerebras.ai/v1", "gpt-oss-120b"),
    "openai": ("OPENAI_API_KEY", None, "gpt-4.1-mini"),
}


@dataclass
class Candidate:
    row_index: int
    citation: str
    text: str
    rating: int | None
    date: str | None
    embedding_score: float
    rerank_score: float | None = None


def resolve_llm_config(overrides: dict[str, str] | None = None) -> tuple[str, str, str, str | None]:
    """Return provider, model, key environment variable, and base URL."""
    overrides = overrides or {}
    provider = overrides.get("provider", LLM_PROVIDER)
    if provider == "auto":
        provider = next(
            (name for name, (env_key, _, _) in PROVIDER_SETTINGS.items() if os.getenv(env_key)),
            "",
        )
    if provider not in PROVIDER_SETTINGS:
        raise RuntimeError(
            "No API key found. Set LLM_PROVIDER and its key in .env; see .env.example."
        )
    env_key, base_url, default_model = PROVIDER_SETTINGS[provider]
    model = overrides.get("model") or os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", default_model))
    return provider, model, env_key, base_url


def call_llm(
    instructions: str, prompt: str, llm_config: dict[str, str] | None = None
) -> str:
    """Call a configured API provider through the OpenAI-compatible client."""
    provider, model, env_key, base_url = resolve_llm_config(llm_config)
    api_key = (llm_config or {}).get("api_key") or os.getenv(env_key)
    if not api_key:
        raise RuntimeError(f"{env_key} is required when LLM_PROVIDER={provider}")
    from openai import OpenAI

    client_args = {"api_key": api_key}
    if base_url:
        client_args["base_url"] = base_url
    # OpenAI-compatible SDK retries transient 5xx/429 responses with backoff.
    client = OpenAI(timeout=45.0, max_retries=3, **client_args)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()


def offline_query_rewrites(question: str) -> list[str]:
    """Simple fallback; useful for retrieval but not a substitute for an LLM."""
    q = question.strip()
    variants = [q]
    lower = q.lower()
    words = set(re.findall(r"[a-z]+", lower))
    synonym_groups = [
        (("payment", "charge", "deduct", "debit", "refund", "subscription", "autopay"),
         "payment charge deducted debited refund subscription auto renewal autopay"),
        (("support", "customer service", "help", "response"),
         "customer support service complaint response refund help"),
        (("playback", "audio", "listen", "download", "buffer", "stream", "pause"),
         "audio playback listening streaming download buffering sound"),
        (("content", "story", "book", "show", "series", "recommend"),
         "content stories audiobooks audio shows series discovery recommendations"),
        (("crash", "crashes", "crashing", "login", "log in", "freeze", "bug", "error", "open", "reliable"),
         "app crashes login sign in freeze bugs errors reliability technical issue"),
        (("language", "hindi", "tamil", "telugu", "voice", "narrator"),
         "language Hindi Tamil Telugu narration narrator voice quality"),
        (("offline", "download", "saved"),
         "offline listening downloaded episodes saved content download"),
    ]
    for cues, expansion in synonym_groups:
        if any(cue in words or (" " in cue and cue in lower) for cue in cues):
            variants.append(f"{q} {expansion}")
    if len(variants) == 1:
        variants.append(f"Kuku FM listener feedback about {q}")
    return list(dict.fromkeys(variants))


def plan_queries(
    question: str, llm_config: dict[str, str] | None = None
) -> tuple[list[str], str]:
    """Ask the LLM to rewrite for recall; fall back explicitly when unavailable."""
    instructions = (
        "You plan retrieval over public Kuku FM Google Play reviews. Rewrite the user's "
        "question into 2-4 short search queries that preserve intent and cover likely "
        "wording variants. Do not invent issues or assume a review proves a general trend. "
        'Return only JSON: {"queries": ["..."]}.'
    )
    try:
        raw = call_llm(instructions, question, llm_config) if llm_config else call_llm(instructions, question)
        parsed = json.loads(raw)
        rewrites = [str(q).strip() for q in parsed.get("queries", []) if str(q).strip()]
        if rewrites:
            all_queries = list(dict.fromkeys([question.strip(), *rewrites]))[:5]
            return all_queries, "llm"
    except Exception as exc:  # Keep the pipeline usable in a no-key/free local setup.
        print(f"[query planning fallback: {exc}]")
    return offline_query_rewrites(question), "offline-rule-based"


class DenseRetriever:
    """Dense retrieval with a local embedding cache for this small corpus."""

    def __init__(self, data_path: Path = DATA_PATH):
        from sentence_transformers import SentenceTransformer

        self.df = pd.read_csv(data_path).fillna("")
        self.texts = self.df["content"].astype(str).tolist()
        if not self.texts:
            raise ValueError(f"No review text in {data_path}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        CACHE_DIR.mkdir(exist_ok=True)
        cache_input = f"{EMBEDDING_MODEL}\n" + "\n".join(self.texts)
        digest = hashlib.sha256(cache_input.encode("utf-8")).hexdigest()[:12]
        cache_file = CACHE_DIR / f"review_embeddings_{digest}.npy"
        if cache_file.exists():
            self.embeddings = np.load(cache_file)
            if len(self.embeddings) != len(self.texts):
                cache_file.unlink(missing_ok=True)
                self.embeddings = None
        else:
            self.embeddings = None
        if self.embeddings is None:
            self.embeddings = self.model.encode(
                self.texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            np.save(cache_file, self.embeddings)

    def retrieve(self, queries: list[str], top_k: int = 30) -> list[Candidate]:
        vectors = self.model.encode(
            queries, normalize_embeddings=True, convert_to_numpy=True
        )
        # Max pooling keeps a review if any rewritten query finds it strongly.
        scores = np.max(vectors @ self.embeddings.T, axis=0)
        indices = np.argsort(scores)[::-1][:top_k]
        candidates = []
        for idx in indices:
            row = self.df.iloc[int(idx)]
            rating = row.get("score", "")
            date = str(row.get("at", ""))
            candidates.append(
                Candidate(
                    row_index=int(idx),
                    citation=f"R{int(idx) + 1:03d}",
                    text=str(row["content"]).strip(),
                    rating=int(rating) if rating != "" else None,
                    date=date[:10] if date else None,
                    embedding_score=float(scores[idx]),
                )
            )
        return candidates


class RelevanceReranker:
    """Cross-encoder relevance check and reranker over retrieved candidates."""

    def __init__(self):
        self.model = None

    def rerank(
        self,
        question: str,
        candidates: list[Candidate],
        keep: int = 8,
        relevance_margin: float = 5.0,
    ) -> list[Candidate]:
        if not candidates:
            return []
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(RERANKER_MODEL)
        pairs = [(question, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
        ranked = sorted(
            candidates,
            key=lambda item: item.rerank_score if item.rerank_score is not None else -1,
            reverse=True,
        )
        # Cross-encoder logits are uncalibrated. This relative outlier gate
        # drops candidates far below the strongest match; it is not a probability.
        best_score = ranked[0].rerank_score
        relevant = [
            item for item in ranked
            if item.rerank_score is not None
            and best_score is not None
            and item.rerank_score >= best_score - relevance_margin
        ]
        return relevant[:keep]


def evidence_text(candidates: list[Candidate]) -> str:
    return "\n".join(
        f"[{c.citation}] ({c.rating}/5 stars; {c.date}; rerank={c.rerank_score:.3f}) {c.text}"
        for c in candidates
    )


def make_answer(
    question: str,
    evidence: list[Candidate],
    llm_config: dict[str, str] | None = None,
) -> str:
    instructions = (
        "Answer only from the supplied review evidence. Distinguish individual reports "
        "from a verified general trend. Do not claim prevalence or causality from this "
        "small, self-selected English review sample. Cite each factual claim with the "
        "provided [R###] IDs. If evidence is insufficient, say so. Keep the answer concise."
    )
    prompt = f"Question: {question}\n\nReview evidence:\n{evidence_text(evidence)}"
    return call_llm(instructions, prompt, llm_config) if llm_config else call_llm(instructions, prompt)


def verify_answer(
    question: str,
    answer: str,
    evidence: list[Candidate],
    llm_config: dict[str, str] | None = None,
) -> str:
    instructions = (
        "Audit the draft against the evidence. Remove or correct unsupported claims, "
        "ensure each factual claim has citations using only supplied [R###] IDs, and "
        "avoid treating review counts as representative. Return only the corrected answer."
    )
    prompt = (
        f"Question: {question}\n\nDraft answer:\n{answer}\n\n"
        f"Allowed evidence and citations:\n{evidence_text(evidence)}"
    )
    return call_llm(instructions, prompt, llm_config) if llm_config else call_llm(instructions, prompt)


def run(
    question: str,
    top_k: int = 8,
    debug: bool = False,
    llm_config: dict[str, str] | None = None,
    use_llm: bool = True,
    retriever: DenseRetriever | None = None,
    reranker: RelevanceReranker | None = None,
) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if use_llm:
        queries, rewrite_method = plan_queries(question, llm_config) if llm_config else plan_queries(question)
    else:
        queries, rewrite_method = offline_query_rewrites(question), "offline-rule-based"
    retriever = retriever or DenseRetriever()
    candidates = retriever.retrieve(queries, top_k=30)
    ranked = (reranker or RelevanceReranker()).rerank(question, candidates, keep=top_k)

    if debug:
        print("\n[1] Planned search queries:")
        for query in queries:
            print(f" - {query}")
        print(f"[2] Query rewrite method: {rewrite_method}")
        print("[3] Retrieved candidates; cross-encoder checked relevance, gated outliers, and reranked survivors.")
        for item in ranked:
            print(
                f" {item.citation}: dense={item.embedding_score:.3f}, "
                f"rerank={item.rerank_score:.3f}, rating={item.rating}, text={item.text[:150]}"
            )

    if not use_llm:
        answer = "LLM synthesis unavailable; inspect the retrieved evidence below."
        answer_method = "offline evidence-only mode"
    else:
        try:
            draft = make_answer(question, ranked, llm_config) if llm_config else make_answer(question, ranked)
        except Exception as exc:
            draft = None
            answer = "LLM synthesis unavailable; inspect the retrieved evidence below."
            answer_method = f"evidence-only fallback ({exc})"
        if draft is not None:
            try:
                answer = verify_answer(question, draft, ranked, llm_config) if llm_config else verify_answer(question, draft, ranked)
                answer_method = "LLM synthesis with citation audit"
            except Exception as exc:
                answer = draft
                answer_method = f"LLM synthesis; citation audit unavailable ({exc})"

    return {
        "question": question,
        "rewritten_queries": queries,
        "rewrite_method": rewrite_method,
        "answer_method": answer_method,
        "answer": answer,
        "evidence": [asdict(item) for item in ranked],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG over Kuku FM reviews")
    parser.add_argument("question", help="Ask about listener feedback in the sample")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--debug", action="store_true", help="Show planning and ranking stages")
    parser.add_argument("--offline", action="store_true", help="Skip API calls and print evidence only")
    args = parser.parse_args()
    result = run(args.question, top_k=args.top_k, debug=args.debug, use_llm=not args.offline)
    print("\nAnswer\n------")
    print(result["answer"])
    print(f"\nMethod: {result['answer_method']}")
    print("\nEvidence\n--------")
    for item in result["evidence"]:
        print(
            f"[{item['citation']}] ({item['rating']}/5, {item['date']}, "
            f"embedding={item['embedding_score']:.3f}, rerank={item['rerank_score']:.3f}) "
            f"{item['text']}"
        )


if __name__ == "__main__":
    main()
