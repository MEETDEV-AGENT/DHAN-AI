"""
researcher.py — Researcher Sub-Agent

Performs web-based market research using Pinchtab for search.
Gathers financial news, sentiment, analyst opinions, and
macro-economic data relevant to the Indian stock market.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from dhan_ai.agents.base_agent import AgentRole, BaseAgent

logger = logging.getLogger(__name__)


class Sentiment(str, Enum):
    """Sentiment classification for a research item."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ResearchCategory(str, Enum):
    """Categories of research content."""

    NEWS = "news"
    ANALYST_REPORT = "analyst_report"
    EARNINGS = "earnings"
    MACRO = "macro"
    SECTOR = "sector"
    REGULATORY = "regulatory"


@dataclass
class ResearchItem:
    """A single piece of market research."""

    title: str
    source: str
    category: ResearchCategory
    sentiment: Sentiment
    relevance_score: float
    summary: str = ""
    symbols: List[str] = field(default_factory=list)
    url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class PinchtabSearchClient:
    """Client wrapper for the Pinchtab web-search API.

    Pinchtab provides real-time web search capabilities.
    This client handles authentication, request building, and
    response parsing for financial market queries.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url or "https://api.pinchtab.com/v1"
        self._session_active = False

    async def search(
        self,
        query: str,
        max_results: int = 10,
        market: str = "india",
        recency: str = "24h",
    ) -> List[Dict[str, Any]]:
        """Execute a web search via the Pinchtab API.

        Parameters
        ----------
        query:
            The search query string.
        max_results:
            Maximum number of results to return.
        market:
            Market region filter (default ``"india"``).
        recency:
            Recency filter — e.g. ``"1h"``, ``"24h"``, ``"7d"``.

        Returns
        -------
        list[dict]:
            Search results with ``title``, ``url``, ``snippet``, and
            ``published_at`` fields.
        """
        self._logger_info(
            "Pinchtab search: query=%r max_results=%d market=%s recency=%s",
            query,
            max_results,
            market,
            recency,
        )
        # Placeholder — actual HTTP call will be wired up with
        # httpx / aiohttp once Pinchtab credentials are configured.
        return []

    @staticmethod
    def _logger_info(msg: str, *args: Any) -> None:
        logging.getLogger("dhan_ai.agents.researcher.pinchtab").info(msg, *args)


class ResearcherAgent(BaseAgent):
    """Gathers market intelligence via web research (Pinchtab).

    Responsibilities:
      - Search financial news and analyst reports
      - Classify sentiment for each piece of research
      - Score relevance of findings to the current watchlist
      - Aggregate and summarise insights for the Manager
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(role=AgentRole.RESEARCHER, config=config)
        self.pinchtab = PinchtabSearchClient(
            api_key=self.config.get("pinchtab_api_key"),
            base_url=self.config.get("pinchtab_base_url"),
        )
        self.max_results_per_query: int = self.config.get("max_results_per_query", 10)
        self.recency: str = self.config.get("recency", "24h")
        self.relevance_threshold: float = self.config.get("relevance_threshold", 0.5)

    async def _execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct market research for the given symbols.

        Expected context keys:
          - ``watchlist``: list of stock symbols to research
          - ``topics``: optional list of additional search topics
          - ``market_data``: optional dict of current market data
        """
        watchlist: List[str] = context.get("watchlist", [])
        extra_topics: List[str] = context.get("topics", [])

        queries = self._build_queries(watchlist, extra_topics)
        raw_results = await self._run_searches(queries)
        items = self._process_results(raw_results, watchlist)
        sentiments = self._aggregate_sentiments(items, watchlist)
        summary = self._build_summary(items)

        return {
            "research_items": [self._item_to_dict(i) for i in items],
            "sentiments": sentiments,
            "summary": summary,
            "queries_executed": len(queries),
            "total_results": len(items),
        }

    # ------------------------------------------------------------------
    # Query building
    # ------------------------------------------------------------------

    def _build_queries(
        self,
        watchlist: List[str],
        extra_topics: List[str],
    ) -> List[str]:
        """Build search queries from the watchlist and user-specified topics."""
        queries: List[str] = []

        for symbol in watchlist:
            queries.append(f"{symbol} stock market news India")
            queries.append(f"{symbol} analyst report latest")

        queries.append("Indian stock market today")
        queries.append("NSE BSE market outlook")
        queries.append("RBI monetary policy latest")

        for topic in extra_topics:
            queries.append(topic)

        return queries

    async def _run_searches(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Run all queries through Pinchtab and collect results."""
        all_results: List[Dict[str, Any]] = []
        for query in queries:
            results = await self.pinchtab.search(
                query=query,
                max_results=self.max_results_per_query,
                recency=self.recency,
            )
            for r in results:
                r["_query"] = query
            all_results.extend(results)
        return all_results

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _process_results(
        self,
        raw: List[Dict[str, Any]],
        watchlist: List[str],
    ) -> List[ResearchItem]:
        """Convert raw search results into structured ResearchItems."""
        items: List[ResearchItem] = []
        seen_urls: set[str] = set()

        for result in raw:
            url = result.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            title = result.get("title", "")
            snippet = result.get("snippet", "")
            category = self._classify_category(title, snippet)
            sentiment = self._classify_sentiment(title, snippet)
            relevance = self._score_relevance(title, snippet, watchlist)

            if relevance < self.relevance_threshold:
                continue

            symbols = [s for s in watchlist if s.upper() in (title + snippet).upper()]

            items.append(
                ResearchItem(
                    title=title,
                    source=result.get("source", "web"),
                    category=category,
                    sentiment=sentiment,
                    relevance_score=relevance,
                    summary=snippet,
                    symbols=symbols,
                    url=url,
                )
            )

        items.sort(key=lambda x: x.relevance_score, reverse=True)
        return items

    def _aggregate_sentiments(
        self,
        items: List[ResearchItem],
        watchlist: List[str],
    ) -> Dict[str, str]:
        """Compute per-symbol sentiment from research items."""
        symbol_scores: Dict[str, List[int]] = {s: [] for s in watchlist}

        sentiment_map = {
            Sentiment.POSITIVE: 1,
            Sentiment.NEGATIVE: -1,
            Sentiment.NEUTRAL: 0,
        }

        for item in items:
            score = sentiment_map[item.sentiment]
            for sym in item.symbols:
                if sym in symbol_scores:
                    symbol_scores[sym].append(score)

        result: Dict[str, str] = {}
        for sym, scores in symbol_scores.items():
            if not scores:
                result[sym] = "neutral"
            else:
                avg = sum(scores) / len(scores)
                if avg > 0.2:
                    result[sym] = "positive"
                elif avg < -0.2:
                    result[sym] = "negative"
                else:
                    result[sym] = "neutral"
        return result

    # ------------------------------------------------------------------
    # Classification helpers (simple keyword-based — to be replaced
    # with LLM-based classification in production)
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_category(title: str, snippet: str) -> ResearchCategory:
        text = (title + " " + snippet).lower()
        if any(kw in text for kw in ("earnings", "quarterly", "profit", "revenue")):
            return ResearchCategory.EARNINGS
        if any(kw in text for kw in ("rbi", "gdp", "inflation", "monetary")):
            return ResearchCategory.MACRO
        if any(kw in text for kw in ("sector", "industry", "banking", "it sector")):
            return ResearchCategory.SECTOR
        if any(kw in text for kw in ("sebi", "regulation", "compliance")):
            return ResearchCategory.REGULATORY
        if any(kw in text for kw in ("target", "rating", "upgrade", "downgrade")):
            return ResearchCategory.ANALYST_REPORT
        return ResearchCategory.NEWS

    @staticmethod
    def _classify_sentiment(title: str, snippet: str) -> Sentiment:
        text = (title + " " + snippet).lower()
        pos = sum(1 for kw in ("surge", "rally", "gain", "up", "bullish", "growth", "positive") if kw in text)
        neg = sum(1 for kw in ("drop", "fall", "loss", "down", "bearish", "decline", "negative") if kw in text)
        if pos > neg:
            return Sentiment.POSITIVE
        if neg > pos:
            return Sentiment.NEGATIVE
        return Sentiment.NEUTRAL

    @staticmethod
    def _score_relevance(title: str, snippet: str, watchlist: List[str]) -> float:
        text = (title + " " + snippet).upper()
        matches = sum(1 for s in watchlist if s.upper() in text)
        if not watchlist:
            return 0.5
        return min(1.0, matches / max(1, len(watchlist)) + 0.3)

    @staticmethod
    def _item_to_dict(item: ResearchItem) -> Dict[str, Any]:
        return {
            "title": item.title,
            "source": item.source,
            "category": item.category.value,
            "sentiment": item.sentiment.value,
            "relevance_score": item.relevance_score,
            "summary": item.summary,
            "symbols": item.symbols,
            "url": item.url,
            "metadata": item.metadata,
        }

    def _build_summary(self, items: List[ResearchItem]) -> str:
        pos = sum(1 for i in items if i.sentiment == Sentiment.POSITIVE)
        neg = sum(1 for i in items if i.sentiment == Sentiment.NEGATIVE)
        return (
            f"Research complete: {len(items)} items "
            f"({pos} positive, {neg} negative, {len(items) - pos - neg} neutral)"
        )
