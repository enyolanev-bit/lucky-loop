from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path


ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_RATE_LIMIT_SECONDS = 3.1


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
        "used_for": ["autonomous_research_baseline", "end_to_end_loop"],
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
        "used_for": ["autonomous_research_baseline", "literature_to_report_loop"],
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
        "used_for": ["ml_agent_baseline", "score_based_evaluation"],
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
        "used_for": ["ml_agent_baseline", "search_policy"],
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
        "used_for": ["world_model_framing", "state_action_prediction"],
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
        "used_for": ["claim_risk", "implementation_validity"],
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
    arxiv_id: str | None = None
    arxiv_version: str | None = None
    categories: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    citation_id: str = ""
    used_for: list[str] = field(default_factory=list)
    exclusion_reason: str | None = None


@dataclass
class GapFinding:
    gap_id: str
    claim: str
    source_ids: list[str]
    implication: str
    metric: str
    experiment: str


@dataclass
class ResearchContext:
    question: str
    queries: list[str]
    papers: list[Paper]
    excluded_papers: list[Paper]
    gap_findings: list[GapFinding]
    recommended_metrics: list[str]
    recommended_baselines: list[str]
    recommended_experiment_plan: list[str]

    @property
    def known_gaps(self) -> list[str]:
        return [gap.claim for gap in self.gap_findings]


def generate_queries(question: str) -> list[str]:
    base = [
        question,
        'ti:"AI Scientist" OR abs:"autonomous scientific discovery"',
        'all:"Agent Laboratory" "research assistants"',
        'all:"MLE-bench" "machine learning agents"',
        'all:"Qwen-AgentWorld" OR all:"language world models"',
        'all:"AI scientists fail" implementation capability',
        'all:"claim verification" "AI scientist" hallucinated findings',
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


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _extract_arxiv_id(url: str) -> tuple[str | None, str | None]:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#/]+)", url, flags=re.I)
    if not match:
        return None, None
    raw = match.group(1).removesuffix(".pdf")
    version_match = re.match(r"(.+?)(v\d+)$", raw)
    if version_match:
        return version_match.group(1), version_match.group(2)
    return raw, None


def _canonical_url(url: str) -> str:
    arxiv_id, version = _extract_arxiv_id(url)
    if arxiv_id:
        suffix = f"{arxiv_id}{version or ''}"
        return f"https://arxiv.org/abs/{suffix}"
    return url.split("?")[0].replace("http://", "https://").rstrip("/")


def _citation_key(title: str, year: int | None, arxiv_id: str | None) -> str:
    if arxiv_id:
        base = "arxiv_" + arxiv_id.replace(".", "_").replace("/", "_")
    else:
        words = re.findall(r"[a-zA-Z0-9]+", title)
        base = "_".join(words[:4]).lower() or "source"
    if year:
        base = f"{base}_{year}"
    return base


def _paper_from_curated(row: dict) -> Paper:
    url = _canonical_url(row["url"])
    arxiv_id, version = _extract_arxiv_id(url)
    paper = Paper(
        title=row["title"],
        authors=list(row.get("authors") or []),
        year=row.get("year"),
        url=url,
        abstract=row.get("abstract", ""),
        tags=list(row.get("tags") or []),
        source="curated",
        arxiv_id=arxiv_id,
        arxiv_version=version,
        used_for=list(row.get("used_for") or []),
    )
    paper.citation_id = _citation_key(paper.title, paper.year, paper.arxiv_id)
    return paper


def _arxiv_query(search_query: str | None = None, id_list: list[str] | None = None, max_results: int = 3) -> str:
    params = {
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    if id_list:
        params["id_list"] = ",".join(id_list)
    else:
        params["search_query"] = search_query or ""
    return f"{ARXIV_API}?{urllib.parse.urlencode(params)}"


def _parse_arxiv(data: bytes) -> list[Paper]:
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
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
        categories = [
            node.attrib.get("term", "")
            for node in entry.findall("atom:category", ns)
            if node.attrib.get("term")
        ]
        if not title or not link:
            continue
        url = _canonical_url(link)
        arxiv_id, version = _extract_arxiv_id(url)
        paper = Paper(
            title=title,
            authors=[author for author in authors if author],
            year=_year_from_text(published),
            url=url,
            abstract=abstract,
            tags=["arxiv"],
            source="arxiv",
            arxiv_id=arxiv_id,
            arxiv_version=version,
            categories=categories,
        )
        paper.citation_id = _citation_key(paper.title, paper.year, paper.arxiv_id)
        papers.append(paper)
    return papers


def _fetch_arxiv(url: str, timeout: int = 15) -> list[Paper]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return _parse_arxiv(response.read())
    except Exception:
        return []


def _search_arxiv(query: str, max_results: int = 3) -> list[Paper]:
    return _fetch_arxiv(_arxiv_query(search_query=query, max_results=max_results))


def _fetch_arxiv_ids(ids: list[str]) -> list[Paper]:
    if not ids:
        return []
    return _fetch_arxiv(_arxiv_query(id_list=ids, max_results=len(ids)))


def _merge_paper(existing: Paper, incoming: Paper) -> Paper:
    # Prefer arXiv metadata when it matches a curated entry, but keep curated tags/usage.
    if existing.source == "curated" and incoming.source == "arxiv":
        primary, secondary = incoming, existing
    else:
        primary, secondary = existing, incoming
    tags = sorted(set(primary.tags + secondary.tags))
    used_for = sorted(set(primary.used_for + secondary.used_for))
    authors = primary.authors or secondary.authors
    abstract = primary.abstract if len(primary.abstract) >= len(secondary.abstract) else secondary.abstract
    merged = Paper(
        title=primary.title or secondary.title,
        authors=authors,
        year=primary.year or secondary.year,
        url=_canonical_url(primary.url or secondary.url),
        abstract=abstract,
        tags=tags,
        source="arxiv+curated" if {primary.source, secondary.source} == {"arxiv", "curated"} else primary.source,
        arxiv_id=primary.arxiv_id or secondary.arxiv_id,
        arxiv_version=primary.arxiv_version or secondary.arxiv_version,
        categories=sorted(set(primary.categories + secondary.categories)),
        used_for=used_for,
    )
    merged.citation_id = _citation_key(merged.title, merged.year, merged.arxiv_id)
    return merged


def _dedupe_papers(papers: list[Paper]) -> list[Paper]:
    deduped: dict[str, Paper] = {}
    for paper in papers:
        key = paper.arxiv_id or _normalize_title(paper.title)
        if key in deduped:
            deduped[key] = _merge_paper(deduped[key], paper)
        else:
            deduped[key] = paper
    return list(deduped.values())


def collect_papers(question: str, max_arxiv_per_query: int = 2, polite_delay: float = ARXIV_RATE_LIMIT_SECONDS) -> tuple[list[str], list[Paper]]:
    queries = generate_queries(question)
    curated = [_paper_from_curated(row) for row in CURATED_REFERENCES]
    papers = list(curated)
    curated_ids = [paper.arxiv_id for paper in curated if paper.arxiv_id]
    papers.extend(_fetch_arxiv_ids(curated_ids))
    if curated_ids:
        time.sleep(polite_delay)
    for query in queries:
        papers.extend(_search_arxiv(query, max_results=max_arxiv_per_query))
        time.sleep(polite_delay)
    return queries, _dedupe_papers(papers)


def _tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "under", "which", "can",
        "into", "more", "than", "using", "towards", "toward", "based", "about", "agents",
        "agent", "research",
    }
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 2 and tok not in stop}


def _assign_usage(paper: Paper) -> list[str]:
    text = f"{paper.title} {paper.abstract} {' '.join(paper.tags)}".lower()
    usage = set(paper.used_for)
    if "qwen" in text or "world model" in text or "world models" in text:
        usage.add("world_model_framing")
    if "mle-bench" in text or "machine learning agent" in text or "kaggle" in text:
        usage.add("ml_agent_baseline")
    if "ai scientist" in text or "agent laboratory" in text or "scientific discovery" in text:
        usage.add("autonomous_research_baseline")
    if "fail" in text or "verification" in text or "hallucinat" in text or "implementation capability" in text:
        usage.add("claim_risk")
    if "literature review" in text or "report" in text:
        usage.add("end_to_end_loop")
    return sorted(usage)


def rank_and_filter_papers(question: str, papers: list[Paper], min_score: float = 2.0) -> tuple[list[Paper], list[Paper]]:
    q = _tokens(question)
    priority_terms = [
        "ai scientist",
        "agent laboratory",
        "mle-bench",
        "qwen-agentworld",
        "implementation capability",
        "autoresearch",
    ]
    ranked: list[Paper] = []
    excluded: list[Paper] = []
    for paper in papers:
        haystack = f"{paper.title} {paper.abstract} {' '.join(paper.tags)} {' '.join(paper.categories)}"
        tokens = _tokens(haystack)
        overlap = len(q & tokens)
        bonus = 3 if any(term in haystack.lower() for term in priority_terms) else 0
        usage = _assign_usage(paper)
        paper.used_for = usage
        paper.relevance_score = float(overlap + bonus + min(len(usage), 3))
        if "withdrawn" in paper.title.lower() or "withdrawn" in paper.abstract.lower():
            paper.exclusion_reason = "withdrawn"
            excluded.append(paper)
        elif paper.relevance_score < min_score or not usage:
            paper.exclusion_reason = "low relevance to autonomous research, ML agents, world models, or claim verification"
            excluded.append(paper)
        else:
            ranked.append(paper)
    ranked.sort(key=lambda p: (p.relevance_score, p.year or 0), reverse=True)
    excluded.sort(key=lambda p: (p.relevance_score, p.year or 0), reverse=True)
    return ranked, excluded


def _find_sources(papers: list[Paper], usage: str, fallback: list[str]) -> list[str]:
    ids = [paper.citation_id for paper in papers if usage in paper.used_for]
    return ids[:4] or fallback


def build_gap_findings(papers: list[Paper]) -> list[GapFinding]:
    # No cross-topic fallback: a gap cites only papers actually tagged for its topic, otherwise it
    # renders as [no_source]. Falling back to the autonomous-research list made gaps about, e.g.,
    # world models cite unrelated AI-Scientist papers — a citation that does not support the claim.
    autonomous = _find_sources(papers, "autonomous_research_baseline", [])
    ml_agents = _find_sources(papers, "ml_agent_baseline", [])
    world_models = _find_sources(papers, "world_model_framing", [])
    claim_risk = _find_sources(papers, "claim_risk", [])
    end_to_end = _find_sources(papers, "end_to_end_loop", [])
    return [
        GapFinding(
            gap_id="gap_claim_calibration",
            claim="Autonomous research agents can automate idea-to-report workflows, but final reports still need explicit claim calibration.",
            source_ids=sorted(set(autonomous + claim_risk))[:5],
            implication="Lucky Loop must separate observations from claims and force report claims through a verifier.",
            metric="unsupported_best_model_claims",
            experiment="Compare classic_autoresearch against lucky_loop_full claim ledger outcomes.",
        ),
        GapFinding(
            gap_id="gap_prediction_before_compute",
            claim="ML-agent benchmarks and search policies emphasize final performance more than prospective prediction before compute.",
            source_ids=sorted(set(ml_agents + world_models))[:5],
            implication="Lucky Loop should log state, candidate action, predicted observation, real observation, and comparison.",
            metric="prediction_interval_coverage",
            experiment="Measure Qwen-AgentWorld prediction-vs-reality for every Lucky Loop candidate decision.",
        ),
        GapFinding(
            gap_id="gap_single_run_overclaim",
            claim="Single-run leaderboard wins can overstate robustness when top models are close across seeds.",
            source_ids=sorted(set(ml_agents + claim_risk))[:5],
            implication="A robust best-model claim requires matched multi-seed top-model verification.",
            metric="best_claimable_score",
            experiment="Detect top observed models, rerun matched seeds, and compare effect size against seed noise.",
        ),
        GapFinding(
            gap_id="gap_end_to_end_auditability",
            claim="End-to-end autoresearch needs literature context, execution, analysis, and report generation tied to auditable evidence.",
            source_ids=sorted(set(end_to_end + autonomous))[:5],
            implication="The workspace must preserve sources, experiment plans, commands, traces, claim ledgers, and final report links.",
            metric="evidence_manifest_completeness",
            experiment="Generate reports/autoresearch/<slug>/ with sources.json, research_context.json, experiment_plan.json, and evidence_manifest.json.",
        ),
    ]


def synthesize_context(question: str, max_papers: int = 12) -> ResearchContext:
    queries, papers = collect_papers(question)
    ranked, excluded = rank_and_filter_papers(question, papers)
    selected = ranked[:max_papers]
    return ResearchContext(
        question=question,
        queries=queries,
        papers=selected,
        excluded_papers=excluded,
        gap_findings=build_gap_findings(selected),
        recommended_metrics=[
            "best_single_run_score",
            "best_verified_mean_score",
            "best_claimable_score",
            "unsupported_best_model_claims",
            "prediction_interval_coverage",
            "prediction_miss_count",
            "runs_to_first_verification",
            "compute_per_claimable_claim",
            "evidence_manifest_completeness",
        ],
        recommended_baselines=["classic_autoresearch", "classic_verified", "lucky_loop_full"],
        recommended_experiment_plan=[
            "Run classic autoresearch to measure score-chasing behavior without prospective simulation.",
            "Run classic verified to isolate the deterministic verifier contribution.",
            "Run Lucky Loop full with Qwen-AgentWorld candidate predictions before compute.",
            "Compare unsupported claim rate, prediction-vs-reality traces, verification timing, and claimable evidence.",
        ],
    )


def _source_payload(context: ResearchContext) -> dict:
    return {
        "schema_version": "1.0",
        "question": context.question,
        "sources": [asdict(paper) for paper in context.papers],
        "excluded_sources": [asdict(paper) for paper in context.excluded_papers],
        "gap_findings": [asdict(gap) for gap in context.gap_findings],
    }


def _fmt_citations(ids: list[str]) -> str:
    return ", ".join(f"[{source_id}]" for source_id in ids) if ids else "[no_source]"


def write_context(context: ResearchContext, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "research_context.json").write_text(json.dumps(asdict(context), indent=2), encoding="utf-8")
    (out_dir / "sources.json").write_text(json.dumps(_source_payload(context), indent=2), encoding="utf-8")

    lines = [
        "# Related Work Context",
        "",
        f"Research question: {context.question}",
        "",
        "## Search Protocol",
        "",
        "- API: official arXiv Atom API (`https://export.arxiv.org/api/query`).",
        f"- Rate limit: one request every {ARXIV_RATE_LIMIT_SECONDS:.1f}s.",
        "- Curated fallback: known core papers are included and deduplicated against arXiv metadata.",
        "- Citation stability: arXiv IDs and versions are preserved when available.",
        "",
        "## Search Queries",
        "",
        *[f"- {query}" for query in context.queries],
        "",
        "## Source -> Gap -> Metric -> Experiment",
        "",
        "| Gap | Sources | Metric | Experiment |",
        "|---|---|---|---|",
    ]
    for gap in context.gap_findings:
        lines.append(f"| {gap.claim} | {_fmt_citations(gap.source_ids)} | `{gap.metric}` | {gap.experiment} |")

    lines += ["", "## Included Sources", ""]
    for paper in context.papers:
        authors = ", ".join(paper.authors[:4]) if paper.authors else "unknown authors"
        year = paper.year or "n.d."
        used_for = ", ".join(paper.used_for or [])
        version = f"{paper.arxiv_id}{paper.arxiv_version or ''}" if paper.arxiv_id else "n/a"
        categories = ", ".join(paper.categories) if paper.categories else "n/a"
        lines += [
            f"### [{paper.citation_id}] {paper.title}",
            "",
            f"- Authors: {authors}",
            f"- Year: {year}",
            f"- URL: {paper.url}",
            f"- arXiv: {version}",
            f"- Categories: {categories}",
            f"- Source: {paper.source}",
            f"- Used for: {used_for}",
            f"- Relevance score: {paper.relevance_score:.1f}",
            "",
            paper.abstract,
            "",
        ]

    lines += ["## Excluded / Low-Relevance Sources", ""]
    if context.excluded_papers:
        for paper in context.excluded_papers[:20]:
            lines.append(f"- {paper.title} ({paper.url}): {paper.exclusion_reason}; score={paper.relevance_score:.1f}")
    else:
        lines.append("- None.")

    lines += [
        "",
        "## Metrics Suggested By Literature",
        "",
        *[f"- `{metric}`" for metric in context.recommended_metrics],
        "",
        "## Baselines",
        "",
        *[f"- `{baseline}`" for baseline in context.recommended_baselines],
    ]
    (out_dir / "related_work.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    bib_lines = []
    for paper in context.papers:
        bib_lines += [
            f"@misc{{{paper.citation_id},",
            f"  title = {{{paper.title}}},",
            f"  author = {{{' and '.join(paper.authors) if paper.authors else 'Unknown'}}},",
            f"  year = {{{paper.year or 'n.d.'}}},",
            f"  url = {{{paper.url}}}",
            "}",
            "",
        ]
    (out_dir / "sources.bib").write_text("\n".join(bib_lines), encoding="utf-8")
