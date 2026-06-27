from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path


CURATED_REFERENCES = [
    {
        "title": "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery",
        "authors": ["Sakana AI"],
        "year": 2024,
        "url": "https://arxiv.org/abs/2408.06292",
        "abstract": (
            "An autonomous scientific discovery system that generates ideas, writes code, "
            "runs experiments, and drafts papers. Useful as a baseline for end-to-end "
            "AI scientist workflows."
        ),
        "tags": ["autonomous research", "ai scientist", "report generation"],
    },
    {
        "title": "Agent Laboratory: Using LLM Agents as Research Assistants",
        "authors": ["Agent Laboratory authors"],
        "year": 2025,
        "url": "https://arxiv.org/abs/2501.04227",
        "abstract": (
            "A multi-agent research assistant workflow covering literature review, "
            "experiments, and report writing."
        ),
        "tags": ["research agents", "multi-agent", "literature review"],
    },
    {
        "title": "MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering",
        "authors": ["OpenAI"],
        "year": 2024,
        "url": "https://arxiv.org/abs/2410.07095",
        "abstract": (
            "A benchmark for ML engineering agents where agents search over real machine "
            "learning experiment spaces and are evaluated by task performance."
        ),
        "tags": ["ml agents", "benchmark", "experiment search"],
    },
    {
        "title": "AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench",
        "authors": ["MLE-bench research agents authors"],
        "year": 2025,
        "url": "https://arxiv.org/abs/2507.02554",
        "abstract": (
            "Studies search and exploration behavior of AI research agents for machine "
            "learning tasks."
        ),
        "tags": ["ml agents", "search policy", "exploration"],
    },
    {
        "title": "Qwen-AgentWorld: Language World Models for General Agents",
        "authors": ["Qwen Team"],
        "year": 2026,
        "url": "https://arxiv.org/abs/2606.24597",
        "abstract": (
            "Frames language world models as predictors of environment dynamics from "
            "state and action. Lucky Loop applies this idea to experiment outcomes."
        ),
        "tags": ["world models", "agents", "qwen"],
    },
    {
        "title": "AI Scientists Fail Without Strong Implementation Capability",
        "authors": ["AI scientist critique authors"],
        "year": 2025,
        "url": "https://arxiv.org/abs/2506.01372",
        "abstract": (
            "Argues that autonomous scientist systems can fail when implementation quality, "
            "experimental validity, and claim discipline are weak."
        ),
        "tags": ["verification", "implementation", "claim risk"],
    },
]


@dataclass
class Paper:
    title: str
    authors: list[str]
    year: int | None
    url: str
    abstract: str
    tags: list[str]
    source: str = "curated"
    relevance_score: float = 0.0
    used_for: list[str] | None = None


@dataclass
class ResearchContext:
    question: str
    queries: list[str]
    papers: list[Paper]
    known_gaps: list[str]
    recommended_metrics: list[str]
    recommended_baselines: list[str]
    recommended_experiment_plan: list[str]


def generate_queries(question: str) -> list[str]:
    base = [
        question,
        "autonomous research agents AI Scientist claim verification",
        "MLE-bench machine learning agents experiment search",
        "language world models for agents Qwen-AgentWorld",
        "AI scientist hallucinated findings reproducibility verification",
        "automated machine learning agents claim calibration",
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for query in base:
        key = query.lower().strip()
        if key and key not in seen:
            deduped.append(query)
            seen.add(key)
    return deduped


def _year_from_text(text: str) -> int | None:
    match = re.search(r"(20\d{2})", text)
    return int(match.group(1)) if match else None


def _paper_from_curated(row: dict) -> Paper:
    return Paper(
        title=row["title"],
        authors=list(row.get("authors") or []),
        year=row.get("year"),
        url=row["url"],
        abstract=row.get("abstract", ""),
        tags=list(row.get("tags") or []),
        source="curated",
    )


def _search_arxiv(query: str, max_results: int = 3, timeout: int = 15) -> list[Paper]:
    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    url = f"https://export.arxiv.org/api/query?{params}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = response.read()
    except Exception:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return []

    papers: list[Paper] = []
    for entry in root.findall("atom:entry", ns):
        title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split())
        abstract = " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split())
        published = entry.findtext("atom:published", default="", namespaces=ns) or ""
        authors = [
            (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
            for author in entry.findall("atom:author", ns)
        ]
        link = entry.findtext("atom:id", default="", namespaces=ns) or ""
        if not title or not link:
            continue
        papers.append(
            Paper(
                title=title,
                authors=[author for author in authors if author],
                year=_year_from_text(published),
                url=link,
                abstract=abstract,
                tags=["arxiv"],
                source="arxiv",
            )
        )
    return papers


def collect_papers(question: str, max_arxiv_per_query: int = 2) -> tuple[list[str], list[Paper]]:
    queries = generate_queries(question)
    papers = [_paper_from_curated(row) for row in CURATED_REFERENCES]
    seen_urls = {paper.url for paper in papers}
    for query in queries:
        for paper in _search_arxiv(query, max_results=max_arxiv_per_query):
            if paper.url in seen_urls:
                continue
            papers.append(paper)
            seen_urls.add(paper.url)
    return queries, papers


def _tokens(text: str) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "under",
        "which",
        "can",
        "into",
        "more",
        "than",
    }
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 2 and tok not in stop}


def rank_papers(question: str, papers: list[Paper]) -> list[Paper]:
    q = _tokens(question)
    priority = {
        "ai scientist",
        "agent laboratory",
        "mle-bench",
        "qwen-agentworld",
        "implementation capability",
    }
    ranked = []
    for paper in papers:
        haystack = f"{paper.title} {paper.abstract} {' '.join(paper.tags)}"
        overlap = len(q & _tokens(haystack))
        bonus = 3 if any(name in paper.title.lower() for name in priority) else 0
        paper.relevance_score = float(overlap + bonus)
        if "world" in haystack.lower() or "verification" in haystack.lower():
            paper.relevance_score += 1.0
        ranked.append(paper)
    return sorted(ranked, key=lambda p: (p.relevance_score, p.year or 0), reverse=True)


def synthesize_context(question: str, max_papers: int = 12) -> ResearchContext:
    queries, papers = collect_papers(question)
    ranked = rank_papers(question, papers)[:max_papers]
    for paper in ranked:
        title = paper.title.lower()
        used_for = []
        if "qwen" in title or "world model" in title:
            used_for.append("world_model_framing")
        if "mle" in title or "machine learning" in title:
            used_for.append("ml_agent_baseline")
        if "scientist" in title or "laboratory" in title:
            used_for.append("autonomous_research_baseline")
        if "fail" in title or "verification" in title or "implementation" in title:
            used_for.append("claim_risk")
        paper.used_for = used_for or ["background"]
    return ResearchContext(
        question=question,
        queries=queries,
        papers=ranked,
        known_gaps=[
            "Autonomous research agents are often evaluated by final score or report plausibility rather than claim calibration.",
            "Most ML-agent benchmarks do not measure whether the agent predicted experiment outcomes before spending compute.",
            "A strong single-run result can become an unsupported claim when seed variance or matched multi-seed checks are missing.",
            "World-model predictions are usually not logged as auditable prediction-vs-reality evidence in research-agent loops.",
        ],
        recommended_metrics=[
            "best_single_run_score",
            "best_verified_mean_score",
            "best_claimable_score",
            "unsupported_best_model_claims",
            "prediction_interval_coverage",
            "prediction_miss_count",
            "runs_to_first_verification",
            "compute_per_claimable_claim",
        ],
        recommended_baselines=[
            "classic_autoresearch",
            "classic_verified",
            "lucky_loop_full",
        ],
        recommended_experiment_plan=[
            "Run classic autoresearch to measure score-chasing behavior without prospective simulation.",
            "Run classic verified to isolate the deterministic verifier contribution.",
            "Run Lucky Loop full with Qwen-AgentWorld candidate predictions before compute.",
            "Compare unsupported claim rate, prediction-vs-reality traces, verification timing, and claimable evidence.",
        ],
    )


def write_context(context: ResearchContext, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(context)
    (out_dir / "research_context.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Related Work Context",
        "",
        f"Research question: {context.question}",
        "",
        "## Search Queries",
        "",
        *[f"- {query}" for query in context.queries],
        "",
        "## Sources",
        "",
    ]
    for paper in context.papers:
        authors = ", ".join(paper.authors[:4]) if paper.authors else "unknown authors"
        year = paper.year or "n.d."
        used_for = ", ".join(paper.used_for or [])
        lines += [
            f"### {paper.title}",
            "",
            f"- Authors: {authors}",
            f"- Year: {year}",
            f"- URL: {paper.url}",
            f"- Source: {paper.source}",
            f"- Used for: {used_for}",
            f"- Relevance score: {paper.relevance_score:.1f}",
            "",
            paper.abstract,
            "",
        ]
    lines += [
        "## Gaps Lucky Loop Tests",
        "",
        *[f"- {gap}" for gap in context.known_gaps],
        "",
        "## Metrics Suggested By Literature",
        "",
        *[f"- {metric}" for metric in context.recommended_metrics],
        "",
        "## Baselines",
        "",
        *[f"- {baseline}" for baseline in context.recommended_baselines],
    ]
    (out_dir / "related_work.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bib_lines = []
    for i, paper in enumerate(context.papers, start=1):
        key = re.sub(r"[^a-zA-Z0-9]+", "", paper.title.split(":")[0])[:32] or f"paper{i}"
        bib_lines += [
            f"@misc{{{key}{paper.year or ''},",
            f"  title = {{{paper.title}}},",
            f"  author = {{{' and '.join(paper.authors) if paper.authors else 'Unknown'}}},",
            f"  year = {{{paper.year or 'n.d.'}}},",
            f"  url = {{{paper.url}}}",
            "}",
            "",
        ]
    (out_dir / "sources.bib").write_text("\n".join(bib_lines), encoding="utf-8")
