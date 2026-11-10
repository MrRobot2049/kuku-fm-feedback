"""Local retrieval primitives for the Kuku FM review-feedback assistant."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path(__file__).parent / "kuku_fm_reviews_en_in.csv"


@dataclass(frozen=True)
class ReviewEvidence:
    """One retrieved review and the metadata needed to cite it."""

    citation: str
    text: str
    rating: int | None
    date: str | None
    relevance: float


class ReviewRetriever:
    """TF-IDF retriever suitable for a small, inspectable review corpus."""

    def __init__(self, csv_path: Path = DATA_PATH):
        self.csv_path = Path(csv_path)
        self.reviews = pd.read_csv(self.csv_path).fillna("")
        if "content" not in self.reviews or self.reviews.empty:
            raise ValueError(f"No review text found in {self.csv_path}")

        self.texts = self.reviews["content"].astype(str).tolist()
        # Word and character features catch both meaningful terms and variants
        # such as spelling differences in short app-store reviews.
        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, top_k: int = 5) -> list[ReviewEvidence]:
        """Return the most relevant reviews, labelled [R001], [R002], etc."""
        query = query.strip()
        if not query:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = scores.argsort()[::-1]

        evidence: list[ReviewEvidence] = []
        for idx in ranked[: max(0, top_k)]:
            score = float(scores[idx])
            if score <= 0:
                continue
            row = self.reviews.iloc[idx]
            rating = row.get("score", "")
            date = str(row.get("at", ""))
            evidence.append(
                ReviewEvidence(
                    citation=f"R{idx + 1:03d}",
                    text=str(row["content"]).strip(),
                    rating=int(rating) if rating != "" else None,
                    date=date or None,
                    relevance=score,
                )
            )
        return evidence


def format_evidence(items: Iterable[ReviewEvidence]) -> str:
    """Format retrieved reviews as compact, auditable context."""
    lines = []
    for item in items:
        metadata = []
        if item.rating is not None:
            metadata.append(f"{item.rating}/5 stars")
        if item.date:
            metadata.append(item.date[:10])
        suffix = f" ({', '.join(metadata)})" if metadata else ""
        lines.append(f"[{item.citation}]{suffix} {item.text}")
    return "\n".join(lines)
