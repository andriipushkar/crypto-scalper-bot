"""
Sentiment Analysis for crypto markets.

Sources:
- Twitter/X API
- Reddit API
- News APIs
- Fear & Greed Index
- On-chain metrics
"""

import asyncio
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

import aiohttp
from loguru import logger


# =============================================================================
# Models
# =============================================================================

class SentimentLevel(Enum):
    """Sentiment classification levels."""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class SentimentScore:
    """Sentiment score from a source."""
    source: str
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    level: SentimentLevel
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, source: str, value: float, confidence: float = 1.0) -> "SentimentScore":
        """Create from a value (-1 to 1)."""
        # Clamp value
        value = max(-1.0, min(1.0, value))

        # Determine level
        if value < -0.6:
            level = SentimentLevel.EXTREME_FEAR
        elif value < -0.2:
            level = SentimentLevel.FEAR
        elif value < 0.2:
            level = SentimentLevel.NEUTRAL
        elif value < 0.6:
            level = SentimentLevel.GREED
        else:
            level = SentimentLevel.EXTREME_GREED

        return cls(
            source=source,
            score=value,
            confidence=confidence,
            level=level,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "score": self.score,
            "confidence": self.confidence,
            "level": self.level.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AggregateSentiment:
    """Aggregated sentiment from multiple sources."""
    overall_score: float  # Weighted average
    overall_level: SentimentLevel
    sources: List[SentimentScore]
    timestamp: datetime = field(default_factory=datetime.now)

    # Component scores
    social_score: Optional[float] = None
    news_score: Optional[float] = None
    market_score: Optional[float] = None

    # Trading signal
    signal_strength: float = 0.0  # 0 to 1
    suggested_bias: str = "neutral"  # bullish, bearish, neutral

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "overall_level": self.overall_level.value,
            "social_score": self.social_score,
            "news_score": self.news_score,
            "market_score": self.market_score,
            "signal_strength": self.signal_strength,
            "suggested_bias": self.suggested_bias,
            "sources": [s.to_dict() for s in self.sources],
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Sentiment Providers
# =============================================================================

class BaseSentimentProvider:
    """Base class for sentiment providers."""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_sentiment(self, symbol: str = "BTC") -> Optional[SentimentScore]:
        """Get sentiment score. Override in subclasses."""
        raise NotImplementedError


class FearGreedIndexProvider(BaseSentimentProvider):
    """
    Crypto Fear & Greed Index provider.

    Uses the alternative.me API (free, no key required).
    """

    API_URL = "https://api.alternative.me/fng/"

    async def get_sentiment(self, symbol: str = "BTC") -> Optional[SentimentScore]:
        """Get Fear & Greed Index sentiment."""
        try:
            session = await self.get_session()

            async with session.get(self.API_URL) as resp:
                if resp.status != 200:
                    return None

                data = await resp.json()
                fng_data = data.get("data", [{}])[0]

                # Fear & Greed is 0-100, convert to -1 to 1
                value = int(fng_data.get("value", 50))
                normalized = (value - 50) / 50  # -1 to 1

                return SentimentScore(
                    source="fear_greed_index",
                    score=normalized,
                    confidence=0.9,
                    level=self._value_to_level(value),
                    raw_data={
                        "value": value,
                        "classification": fng_data.get("value_classification"),
                    },
                )

        except Exception as e:
            logger.error(f"Fear & Greed fetch error: {e}")
            return None

    def _value_to_level(self, value: int) -> SentimentLevel:
        """Convert 0-100 value to sentiment level."""
        if value <= 25:
            return SentimentLevel.EXTREME_FEAR
        elif value <= 45:
            return SentimentLevel.FEAR
        elif value <= 55:
            return SentimentLevel.NEUTRAL
        elif value <= 75:
            return SentimentLevel.GREED
        else:
            return SentimentLevel.EXTREME_GREED


class LunarCrushProvider(BaseSentimentProvider):
    """
    LunarCrush social sentiment provider.

    Requires API key from lunarcrush.com
    """

    API_URL = "https://lunarcrush.com/api4/public/coins"

    def __init__(self, api_key: str = None):
        super().__init__()
        self.api_key = api_key or os.getenv("LUNARCRUSH_API_KEY", "")

    async def get_sentiment(self, symbol: str = "BTC") -> Optional[SentimentScore]:
        """Get LunarCrush social sentiment."""
        if not self.api_key:
            return None

        try:
            session = await self.get_session()
            headers = {"Authorization": f"Bearer {self.api_key}"}

            async with session.get(
                f"{self.API_URL}/{symbol}/time-series/v1",
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    return None

                data = await resp.json()

                # Galaxy Score is 0-100
                galaxy_score = data.get("data", {}).get("galaxy_score", 50)
                normalized = (galaxy_score - 50) / 50

                # Alt Rank (lower is better, 1-100)
                alt_rank = data.get("data", {}).get("alt_rank", 50)

                # Social volume change
                social_volume = data.get("data", {}).get("social_volume", 0)

                return SentimentScore.from_value(
                    source="lunarcrush",
                    value=normalized,
                    confidence=0.8,
                )

        except Exception as e:
            logger.error(f"LunarCrush fetch error: {e}")
            return None


class CryptoNewsProvider(BaseSentimentProvider):
    """
    Crypto news sentiment provider.

    Uses CryptoPanic API (free tier available).
    """

    API_URL = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, api_key: str = None):
        super().__init__()
        self.api_key = api_key or os.getenv("CRYPTOPANIC_API_KEY", "")

    # Simple sentiment keywords
    POSITIVE_WORDS = {
        "bullish", "surge", "soar", "rally", "gain", "rise", "pump",
        "breakout", "moon", "buy", "accumulate", "support", "recovery",
        "adoption", "partnership", "upgrade", "milestone", "record",
    }

    NEGATIVE_WORDS = {
        "bearish", "crash", "dump", "plunge", "fall", "drop", "sell",
        "breakdown", "fear", "panic", "hack", "scam", "ban", "regulation",
        "lawsuit", "investigation", "loss", "decline", "warning",
    }

    async def get_sentiment(self, symbol: str = "BTC") -> Optional[SentimentScore]:
        """Get news sentiment."""
        if not self.api_key:
            return None

        try:
            session = await self.get_session()

            params = {
                "auth_token": self.api_key,
                "currencies": symbol,
                "filter": "important",
                "public": "true",
            }

            async with session.get(self.API_URL, params=params) as resp:
                if resp.status != 200:
                    return None

                data = await resp.json()
                posts = data.get("results", [])

                if not posts:
                    return SentimentScore.from_value("news", 0.0, 0.3)

                # Analyze headlines
                positive_count = 0
                negative_count = 0

                for post in posts[:20]:  # Last 20 articles
                    title = post.get("title", "").lower()
                    words = set(re.findall(r'\w+', title))

                    if words & self.POSITIVE_WORDS:
                        positive_count += 1
                    if words & self.NEGATIVE_WORDS:
                        negative_count += 1

                total = positive_count + negative_count
                if total == 0:
                    score = 0.0
                else:
                    score = (positive_count - negative_count) / total

                return SentimentScore.from_value(
                    source="news",
                    value=score,
                    confidence=0.6,
                )

        except Exception as e:
            logger.error(f"News fetch error: {e}")
            return None


class OnChainSentimentProvider(BaseSentimentProvider):
    """
    On-chain sentiment indicators.

    Uses Glassnode-like metrics (requires API key for real data).
    """

    def __init__(self, api_key: str = None):
        super().__init__()
        self.api_key = api_key or os.getenv("GLASSNODE_API_KEY", "")

    async def get_sentiment(self, symbol: str = "BTC") -> Optional[SentimentScore]:
        """Get on-chain sentiment indicators."""
        # Without API, return neutral
        if not self.api_key:
            return None

        # Placeholder for on-chain analysis
        # Would fetch:
        # - Exchange inflows/outflows
        # - Whale movements
        # - MVRV ratio
        # - NUPL (Net Unrealized Profit/Loss)
        # - Funding rates

        return None


# =============================================================================
# Sentiment Aggregator
# =============================================================================

class SentimentAnalyzer:
    """
    Aggregate sentiment from multiple sources.

    Usage:
        analyzer = SentimentAnalyzer()

        # Add providers
        analyzer.add_provider(FearGreedIndexProvider())
        analyzer.add_provider(CryptoNewsProvider(api_key="..."))

        # Get aggregated sentiment
        sentiment = await analyzer.analyze("BTC")
        print(f"Overall: {sentiment.overall_level.value}")
        print(f"Score: {sentiment.overall_score:.2f}")
    """

    def __init__(self):
        self._providers: List[Tuple[BaseSentimentProvider, float]] = []
        self._cache: Dict[str, Tuple[AggregateSentiment, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    def add_provider(
        self,
        provider: BaseSentimentProvider,
        weight: float = 1.0,
    ) -> None:
        """Add a sentiment provider with optional weight."""
        self._providers.append((provider, weight))

    async def analyze(
        self,
        symbol: str = "BTC",
        use_cache: bool = True,
    ) -> AggregateSentiment:
        """
        Analyze sentiment from all providers.

        Args:
            symbol: Crypto symbol
            use_cache: Whether to use cached results

        Returns:
            Aggregated sentiment analysis
        """
        # Check cache
        if use_cache and symbol in self._cache:
            cached, cached_time = self._cache[symbol]
            if datetime.now() - cached_time < self._cache_ttl:
                return cached

        # Fetch from all providers
        scores: List[SentimentScore] = []
        tasks = []

        for provider, weight in self._providers:
            tasks.append(self._fetch_with_weight(provider, symbol, weight))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, SentimentScore):
                scores.append(result)

        # Calculate aggregate
        aggregate = self._aggregate_scores(scores)

        # Cache result
        self._cache[symbol] = (aggregate, datetime.now())

        return aggregate

    async def _fetch_with_weight(
        self,
        provider: BaseSentimentProvider,
        symbol: str,
        weight: float,
    ) -> Optional[SentimentScore]:
        """Fetch sentiment and apply weight."""
        try:
            score = await provider.get_sentiment(symbol)
            if score:
                # Apply weight to confidence
                score.confidence *= weight
            return score
        except Exception as e:
            logger.error(f"Provider error: {e}")
            return None

    def _aggregate_scores(
        self,
        scores: List[SentimentScore],
    ) -> AggregateSentiment:
        """Aggregate multiple sentiment scores."""
        if not scores:
            return AggregateSentiment(
                overall_score=0.0,
                overall_level=SentimentLevel.NEUTRAL,
                sources=[],
            )

        # Weighted average by confidence
        total_weight = sum(s.confidence for s in scores)
        if total_weight == 0:
            weighted_score = 0.0
        else:
            weighted_score = sum(s.score * s.confidence for s in scores) / total_weight

        # Determine level
        level = SentimentScore.from_value("aggregate", weighted_score).level

        # Component scores
        social_scores = [s for s in scores if s.source in ["lunarcrush", "twitter", "reddit"]]
        news_scores = [s for s in scores if s.source == "news"]
        market_scores = [s for s in scores if s.source in ["fear_greed_index", "onchain"]]

        social_score = (
            sum(s.score * s.confidence for s in social_scores) /
            sum(s.confidence for s in social_scores)
            if social_scores else None
        )

        news_score = (
            sum(s.score * s.confidence for s in news_scores) /
            sum(s.confidence for s in news_scores)
            if news_scores else None
        )

        market_score = (
            sum(s.score * s.confidence for s in market_scores) /
            sum(s.confidence for s in market_scores)
            if market_scores else None
        )

        # Trading signal
        signal_strength = abs(weighted_score) * (total_weight / len(scores))
        signal_strength = min(1.0, signal_strength)

        if weighted_score > 0.3:
            bias = "bullish"
        elif weighted_score < -0.3:
            bias = "bearish"
        else:
            bias = "neutral"

        return AggregateSentiment(
            overall_score=weighted_score,
            overall_level=level,
            sources=scores,
            social_score=social_score,
            news_score=news_score,
            market_score=market_score,
            signal_strength=signal_strength,
            suggested_bias=bias,
        )

    async def get_fear_greed(self) -> Optional[SentimentScore]:
        """Get just the Fear & Greed Index."""
        provider = FearGreedIndexProvider()
        try:
            return await provider.get_sentiment()
        finally:
            await provider.close()

    async def close(self) -> None:
        """Close all providers."""
        for provider, _ in self._providers:
            await provider.close()


# =============================================================================
# Convenience Functions
# =============================================================================

async def get_market_sentiment(symbol: str = "BTC") -> AggregateSentiment:
    """
    Quick sentiment check with default providers.

    Uses only free APIs that don't require keys.
    """
    analyzer = SentimentAnalyzer()

    # Add free providers
    analyzer.add_provider(FearGreedIndexProvider(), weight=1.0)

    # Add optional providers if keys are available
    if os.getenv("LUNARCRUSH_API_KEY"):
        analyzer.add_provider(LunarCrushProvider(), weight=0.8)

    if os.getenv("CRYPTOPANIC_API_KEY"):
        analyzer.add_provider(CryptoNewsProvider(), weight=0.6)

    try:
        return await analyzer.analyze(symbol)
    finally:
        await analyzer.close()


async def get_fear_greed_index() -> Tuple[int, str]:
    """
    Get current Fear & Greed Index.

    Returns:
        (value 0-100, classification string)
    """
    provider = FearGreedIndexProvider()
    try:
        score = await provider.get_sentiment()
        if score:
            return (
                score.raw_data.get("value", 50),
                score.raw_data.get("classification", "Neutral"),
            )
        return 50, "Neutral"
    finally:
        await provider.close()


def sentiment_to_signal_modifier(sentiment: AggregateSentiment) -> float:
    """
    Convert sentiment to a signal strength modifier.

    Returns a value between 0.5 and 1.5 to multiply signal strength.

    - Extreme fear: Trade more cautiously (0.7)
    - Fear: Slightly cautious (0.85)
    - Neutral: No modification (1.0)
    - Greed: Trade with confidence (1.1)
    - Extreme greed: Be careful of reversal (0.9)
    """
    modifiers = {
        SentimentLevel.EXTREME_FEAR: 0.7,
        SentimentLevel.FEAR: 0.85,
        SentimentLevel.NEUTRAL: 1.0,
        SentimentLevel.GREED: 1.1,
        SentimentLevel.EXTREME_GREED: 0.9,  # Caution at extremes
    }

    return modifiers.get(sentiment.overall_level, 1.0)
