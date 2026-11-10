"""Search Kuku FM reviews and print the matching evidence."""

import argparse

from rag_core import ReviewRetriever, format_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the Kuku FM review dataset")
    parser.add_argument("query", help="What kind of listener feedback to find")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum reviews to show")
    args = parser.parse_args()

    retriever = ReviewRetriever()
    evidence = retriever.search(args.query, top_k=args.top_k)
    if not evidence:
        print("No matching reviews found in this sample.")
        return
    print(format_evidence(evidence))


if __name__ == "__main__":
    main()
