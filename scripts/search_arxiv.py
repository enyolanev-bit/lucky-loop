#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from luckyloop.literature import _search_arxiv, rank_and_filter_papers


def main() -> None:
    parser = argparse.ArgumentParser(description="Search arXiv through the official Atom API.")
    parser.add_argument("query")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    papers = _search_arxiv(args.query, max_results=args.max_results)
    included, excluded = rank_and_filter_papers(args.query, papers, min_score=0.0)
    if args.json:
        print(json.dumps({"sources": [paper.__dict__ for paper in included], "excluded": [paper.__dict__ for paper in excluded]}, indent=2))
        return
    for paper in included:
        version = f"{paper.arxiv_id}{paper.arxiv_version or ''}" if paper.arxiv_id else "n/a"
        print(f"[{paper.citation_id}] {paper.title}")
        print(f"  url: {paper.url}")
        print(f"  arxiv: {version}")
        print(f"  year: {paper.year or 'n.d.'}")
        print(f"  score: {paper.relevance_score:.1f}")
        print()


if __name__ == "__main__":
    main()
