"""
Literature Retrieval for AutoScientist.

Provides arXiv API integration and semantic search for prior art discovery.
"""

import json
import logging
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from computronium.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ArxivPaper:
    """Represents an arXiv paper entry."""

    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    categories: list[str]
    published: str
    updated: str
    pdf_url: str
    primary_category: str
    comment: str | None = None
    journal_ref: str | None = None
    doi: str | None = None


@dataclass(frozen=True, slots=True)
class LiteratureSearchResult:
    """Result of a literature search with relevance scoring."""

    paper: ArxivPaper
    relevance_score: float
    matched_terms: list[str]
    reason: str = ""


class ArxivClient:
    """Client for querying the arXiv API."""

    BASE_URL = "http://export.arxiv.org/api/query"
    RATE_LIMIT_DELAY = 3.0  # seconds between requests (arXiv policy)

    def __init__(self, rate_limit: float = RATE_LIMIT_DELAY):
        self.rate_limit = rate_limit
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def search(
        self,
        query: str,
        max_results: int = 50,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> list[ArxivPaper]:
        """
        Search arXiv for papers matching query.

        Args:
            query: Search query (arXiv query syntax supported)
            max_results: Maximum number of results
            sort_by: 'relevance', 'lastUpdatedDate', 'submittedDate'
            sort_order: 'ascending' or 'descending'

        Returns:
            List of ArxivPaper objects
        """
        self._rate_limit()

        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "computronium-autoscientist/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_content = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            logger.error("arXiv API error %s: %s", e.code, e.read().decode())
            raise
        except urllib.error.URLError as e:
            logger.error("Network error querying arXiv: %s", e)
            raise

        return self._parse_atom_feed(xml_content)

    def _parse_atom_feed(self, xml_content: str) -> list[ArxivPaper]:
        """Parse arXiv Atom feed XML into ArxivPaper objects."""
        import xml.etree.ElementTree as ET

        papers = []
        root = ET.fromstring(xml_content)

        # Namespace handling
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        for entry in root.findall("atom:entry", ns):
            arxiv_id = entry.find("atom:id", ns).text
            if arxiv_id:
                arxiv_id = arxiv_id.split("/abs/")[-1]

            title = entry.find("atom:title", ns).text or ""
            summary = entry.find("atom:summary", ns).text or ""
            published = entry.find("atom:published", ns).text or ""
            updated = entry.find("atom:updated", ns).text or ""

            authors = [
                author.find("atom:name", ns).text or ""
                for author in entry.findall("atom:author", ns)
            ]

            categories = [
                cat.get("term", "") for cat in entry.findall("atom:category", ns)
            ]
            primary_category = categories[0] if categories else ""

            # Links
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break

            # Optional fields
            comment = None
            comment_elem = entry.find("arxiv:comment", ns)
            if comment_elem is not None and comment_elem.text:
                comment = comment_elem.text

            journal_ref = None
            jr_elem = entry.find("arxiv:journal_ref", ns)
            if jr_elem is not None and jr_elem.text:
                journal_ref = jr_elem.text

            doi = None
            doi_elem = entry.find("arxiv:doi", ns)
            if doi_elem is not None and doi_elem.text:
                doi = doi_elem.text

            papers.append(
                ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=title.strip(),
                    authors=authors,
                    summary=summary.strip(),
                    categories=categories,
                    published=published,
                    updated=updated,
                    pdf_url=pdf_url,
                    primary_category=primary_category,
                    comment=comment,
                    journal_ref=journal_ref,
                    doi=doi,
                )
            )

        return papers

    def search_by_category(
        self,
        category: str,
        max_results: int = 50,
        days_back: int = 30,
    ) -> list[ArxivPaper]:
        """Search for recent papers in a specific arXiv category."""
        from datetime import datetime, timedelta

        date_cutoff = datetime.now() - timedelta(days=days_back)
        date_str = date_cutoff.strftime("%Y%m%d")
        query = f"cat:{category} AND submittedDate:[{date_str} TO *]"
        return self.search(query, max_results=max_results, sort_by="submittedDate")

    def get_paper(self, arxiv_id: str) -> ArxivPaper | None:
        """Fetch a single paper by arXiv ID."""
        papers = self.search(f"id:{arxiv_id}", max_results=1)
        return papers[0] if papers else None


class LiteratureRetriever:
    """
    High-level literature retrieval with semantic search.

    Combines arXiv API with local vector search for finding relevant prior art.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        cache_dir: str | Path = "cache/literature",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.kb = knowledge_base
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = ArxivClient()

        self._embedding_model = None
        self._embedding_model_name = embedding_model

    def _get_embedding_model(self):
        """Lazy-load the sentence transformer model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(self._embedding_model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed, semantic search disabled"
                )
        return self._embedding_model

    def search_prior_art(
        self,
        research_question: str,
        max_results: int = 20,
        categories: list[str] | None = None,
        use_semantic: bool = True,
    ) -> list[LiteratureSearchResult]:
        """
        Search for prior art relevant to a research question.

        Args:
            research_question: Natural language description of the research problem
            max_results: Maximum results to return
            categories: Optional arXiv categories to restrict search (e.g., ["cs.LG", "cs.AI"])
            use_semantic: Whether to use semantic re-ranking

        Returns:
            List of LiteratureSearchResult sorted by relevance
        """
        # Build arXiv query from research question
        query_terms = self._extract_query_terms(research_question)
        arxiv_query = " OR ".join(query_terms)

        if categories:
            cat_query = " OR ".join(f"cat:{c}" for c in categories)
            arxiv_query = f"({arxiv_query}) AND ({cat_query})"

        logger.info("Searching arXiv with query: %s", arxiv_query)
        papers = self.client.search(arxiv_query, max_results=max_results * 2)

        # Score and filter
        results = []
        for paper in papers:
            score, matched = self._score_relevance(
                paper, research_question, query_terms
            )
            if score > 0:
                results.append(
                    LiteratureSearchResult(
                        paper=paper,
                        relevance_score=score,
                        matched_terms=matched,
                        reason=f"Matched terms: {', '.join(matched)}",
                    )
                )

        # Semantic re-ranking if enabled
        if use_semantic and self._get_embedding_model():
            results = self._semantic_rerank(research_question, results)

        # Sort by relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:max_results]

    def _extract_query_terms(self, question: str) -> list[str]:
        """Extract key search terms from a research question."""
        # Simple term extraction - in production could use NER/keyword extraction
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "what",
            "how",
            "when",
            "where",
            "why",
            "who",
            "which",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "their",
            "our",
            "your",
            "my",
            "i",
            "we",
        }

        words = question.lower().split()
        terms = [w for w in words if w not in stop_words and len(w) > 2]

        # Add domain-specific expansions
        domain_expansions = {
            "equilibrium propagation": [
                "equilibrium propagation",
                "eqprop",
                "contrastive hebbian",
            ],
            "feedback alignment": ["feedback alignment", "fa", "direct feedback"],
            "predictive coding": ["predictive coding", "pc", "free energy"],
            "target propagation": ["target propagation", "tp", "inverse propagation"],
            "hebbian": ["hebbian", "contrastive hebbian", "stability-plasticity"],
            "spiking": ["spiking", "snn", "stdp", "surrogate gradient"],
            "local learning": [
                "local learning",
                "local credit assignment",
                "biological plausibility",
            ],
            "backprop": ["backpropagation", "backprop", "gradient descent"],
            "neuromorphic": [
                "neuromorphic",
                "neuromorphic computing",
                "analog computing",
            ],
        }

        for key, expansions in domain_expansions.items():
            if key in question.lower():
                terms.extend(expansions)

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                unique_terms.append(t)

        return unique_terms[:10]  # Limit to avoid overly broad queries

    def _score_relevance(
        self, paper: ArxivPaper, question: str, query_terms: list[str]
    ) -> tuple[float, list[str]]:
        """Score a paper's relevance to the research question."""
        text = f"{paper.title} {paper.summary} {' '.join(paper.categories)}".lower()
        matched = [term for term in query_terms if term.lower() in text]

        if not matched:
            return 0.0, []

        # Base score from term matches
        score = len(matched) / len(query_terms)

        # Boost for title matches
        title_lower = paper.title.lower()
        title_matches = sum(1 for term in matched if term.lower() in title_lower)
        score += title_matches * 0.2

        # Boost for recent papers (last 2 years)
        try:
            pub_year = int(paper.published[:4])
            if pub_year >= datetime.now().year - 2:
                score += 0.1
        except ValueError, IndexError:
            pass

        return min(score, 1.0), matched

    def _semantic_rerank(
        self, question: str, results: list[LiteratureSearchResult]
    ) -> list[LiteratureSearchResult]:
        """Re-rank results using semantic similarity."""
        model = self._get_embedding_model()
        if model is None:
            return results

        question_emb = model.encode(question, normalize_embeddings=True)

        for result in results:
            paper_text = f"{result.paper.title} {result.paper.summary}"
            paper_emb = model.encode(paper_text, normalize_embeddings=True)
            semantic_score = float(np.dot(question_emb, paper_emb))
            # Blend keyword and semantic scores
            result.relevance_score = 0.5 * result.relevance_score + 0.5 * semantic_score

        return results

    def search_similar_to_paper(
        self, arxiv_id: str, max_results: int = 10
    ) -> list[LiteratureSearchResult]:
        """Find papers similar to a given arXiv paper."""
        paper = self.client.get_paper(arxiv_id)
        if not paper:
            return []

        return self.search_prior_art(
            f"{paper.title} {paper.summary}",
            max_results=max_results,
            use_semantic=True,
        )

    def get_trending_papers(
        self, categories: list[str], days_back: int = 7, max_per_category: int = 10
    ) -> dict[str, list[ArxivPaper]]:
        """Get recently trending papers in specified categories."""
        trending = {}
        for cat in categories:
            papers = self.client.search_by_category(
                cat, max_results=max_per_category, days_back=days_back
            )
            trending[cat] = papers
        return trending

    def save_paper_to_kb(self, paper: ArxivPaper, topic: str = "literature") -> str:
        """Save a paper to the knowledge base."""
        if not self.kb:
            logger.warning("No knowledge base configured, skipping save")
            return ""

        entry = self.kb.add_entry(
            topic=topic,
            model_family="literature",
            finding=paper.title,
            details=f"arXiv:{paper.arxiv_id} | {paper.summary[:500]}...",
            confidence=0.9,
            tags=["literature", "arxiv"] + paper.categories,
            source="arxiv",
            extra={
                "arxiv_id": paper.arxiv_id,
                "authors": paper.authors,
                "published": paper.published,
                "pdf_url": paper.pdf_url,
                "primary_category": paper.primary_category,
            },
        )
        return entry


class LiteratureCache:
    """Persistent cache for arXiv search results."""

    def __init__(self, cache_dir: str | Path = "cache/literature"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, query: str) -> Path:
        """Generate cache file path from query."""
        import hashlib

        query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
        return self.cache_dir / f"{query_hash}.json"

    def get(self, query: str, max_age_hours: float = 24) -> list[ArxivPaper] | None:
        """Get cached results if fresh enough."""
        cache_file = self._cache_path(query)
        if not cache_file.exists():
            return None

        # Check age
        mtime = cache_file.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600
        if age_hours > max_age_hours:
            return None

        try:
            with cache_file.open() as f:
                data = json.load(f)
            return [ArxivPaper(**p) for p in data]
        except json.JSONDecodeError, TypeError:
            return None

    def set(self, query: str, papers: list[ArxivPaper]) -> None:
        """Cache search results."""
        cache_file = self._cache_path(query)
        data = [
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "authors": p.authors,
                "summary": p.summary,
                "categories": p.categories,
                "published": p.published,
                "updated": p.updated,
                "pdf_url": p.pdf_url,
                "primary_category": p.primary_category,
                "comment": p.comment,
                "journal_ref": p.journal_ref,
                "doi": p.doi,
            }
            for p in papers
        ]
        with cache_file.open("w") as f:
            json.dump(data, f)


__all__ = [
    "ArxivClient",
    "ArxivPaper",
    "LiteratureCache",
    "LiteratureRetriever",
    "LiteratureSearchResult",
]
