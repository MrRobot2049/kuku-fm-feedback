"""Build the checked-in bi-encoder vectors for the review child passages."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import agentic_rag as rag  # noqa: E402


def main() -> None:
    frame = pd.read_csv(rag.DATA_PATH).fillna("")
    parent_texts = frame["content"].astype(str).tolist()
    child_texts = [
        child
        for parent in parent_texts
        for child in rag.child_chunks(parent)
    ]
    digest = rag.embedding_index_digest(child_texts)
    output_dir = rag.INDEX_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"review_child_embeddings_{digest}.npy"

    model = SentenceTransformer(rag.EMBEDDING_MODEL)
    embeddings = model.encode(
        child_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)
    np.save(output_path, embeddings)
    print(f"Saved {len(child_texts)} child vectors to {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
