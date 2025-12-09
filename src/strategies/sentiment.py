"""
Sentiment Analysis

Social media and news sentiment analysis for trading signals.
"""
import asyncio
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from loguru import logger


class SentimentLevel(Enum):
    """Sentiment levels."""
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2


class DataSource(Enum):
    """Data sources for sentiment."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    FEAR_GREED = "fear_greed"


@dataclass
class SentimentData:
    """Single sentiment data point."""
    source: DataSource
    symbol: str
    text: str
    sentiment_score: float  # -1 to 1
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sentiment_level(self) -> SentimentLevel:
        """Convert score to level."""
        if self.sentiment_score <= -0.6:
            return SentimentLevel.VERY_BEARISH
        elif self.sentiment_score <= -0.2:
            return SentimentLevel.BEARISH
        elif self.sentiment_score <= 0.2:
            return SentimentLevel.NEUTRAL
        elif self.sentiment_score <= 0.6:
            return SentimentLevel.BULLISH
        else:
            return SentimentLevel.VERY_BULLISH


@dataclass
class AggregateSentiment:
    """Aggregated sentiment for a symbol."""
    symbol: str
    overall_score: float
    overall_level: SentimentLevel
    source_scores: Dict[str, float]
    sample_size: int
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    trend: str = "stable"  # improving, declining, stable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "score": self.overall_score,
            "level": self.overall_level.name,
            "sources": self.source_scores,
            "sample_size": self.sample_size,
            "confidence": self.confidence,
            "trend": self.trend,
            "timestamp": self.timestamp.isoformat(),
        }


class TextSentimentAnalyzer:
    """Analyze text sentiment."""

    def __init__(self, use_transformer: bool = False):
        self.use_transformer = use_transformer and HAS_TRANSFORMERS

        if self.use_transformer:
            self.analyzer = pipeline(
                "sentiment-analysis",
                model="finiteautomata/bertweet-base-sentiment-analysis"
            )
        elif not HAS_TEXTBLOB:
            logger.warning("TextBlob not installed, using basic analysis")

    def analyze(self, text: str) -> Tuple[float, float]:
        """Analyze text sentiment.

        Returns:
            Tuple of (sentiment_score, confidence)
        """
        # Clean text
        text = self._clean_text(text)

        if not text:
            return 0.0, 0.0

        if self.use_transformer:
            return self._analyze_transformer(text)
        elif HAS_TEXTBLOB:
            return self._analyze_textblob(text)
        else:
            return self._analyze_basic(text)

    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags (keep the word)
        text = re.sub(r'#(\w+)', r'\1', text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.lower()

    def _analyze_transformer(self, text: str) -> Tuple[float, float]:
        """Analyze using transformer model."""
        try:
            result = self.analyzer(text[:512])[0]  # Limit length
            label = result['label'].lower()
            score = result['score']

            if label == 'positive':
                sentiment = score
            elif label == 'negative':
                sentiment = -score
            else:
                sentiment = 0

            return sentiment, score
        except Exception as e:
            logger.warning(f"Transformer analysis failed: {e}")
            return 0.0, 0.0

    def _analyze_textblob(self, text: str) -> Tuple[float, float]:
        """Analyze using TextBlob."""
        blob = TextBlob(text)
        # Polarity is -1 to 1
        sentiment = blob.sentiment.polarity
        # Subjectivity as confidence proxy
        confidence = 0.5 + blob.sentiment.subjectivity * 0.5
        return sentiment, confidence

    def _analyze_basic(self, text: str) -> Tuple[float, float]:
        """Basic keyword-based analysis."""
        bullish_words = {
            'moon', 'pump', 'bull', 'buy', 'long', 'bullish', 'up', 'green',
            'profit', 'gain', 'surge', 'rocket', 'breakout', 'ath', 'hodl',
            'lambo', 'gem', 'undervalued', 'cheap', 'opportunity'
        }
        bearish_words = {
            'dump', 'bear', 'sell', 'short', 'bearish', 'down', 'red',
            'loss', 'crash', 'drop', 'fall', 'rekt', 'scam', 'overvalued',
            'expensive', 'bubble', 'dead', 'rug', 'ponzi'
        }

        words = text.split()
        bull_count = sum(1 for w in words if w in bullish_words)
        bear_count = sum(1 for w in words if w in bearish_words)
        total = bull_count + bear_count

        if total == 0:
            return 0.0, 0.3

        sentiment = (bull_count - bear_count) / total
        confidence = min(1.0, total / 5)  # More keywords = more confidence

        return sentiment, confidence


class TwitterSentimentFetcher:
    """Fetch sentiment from Twitter/X."""

    def __init__(self, bearer_token: Optional[str] = None):
        self.bearer_token = bearer_token
        self.analyzer = TextSentimentAnalyzer()

    async def fetch_tweets(
        self,
        query: str,
        max_results: int = 100,
    ) -> List[SentimentData]:
        """Fetch and analyze tweets."""
        if not self.bearer_token:
            logger.warning("Twitter API token not configured")
            return []

        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required")

        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        params = {
            "query": f"{query} -is:retweet lang:en",
            "max_results": min(max_results, 100),
            "tweet.fields": "created_at,public_metrics",
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    logger.error(f"Twitter API error: {response.status}")
                    return []

                data = await response.json()

        results = []
        for tweet in data.get("data", []):
            text = tweet.get("text", "")
            sentiment, confidence = self.analyzer.analyze(text)

            results.append(SentimentData(
                source=DataSource.TWITTER,
                symbol=query,
                text=text,
                sentiment_score=sentiment,
                confidence=confidence,
                metadata={
                    "likes": tweet.get("public_metrics", {}).get("like_count", 0),
                    "retweets": tweet.get("public_metrics", {}).get("retweet_count", 0),
                }
            ))

        return results


class RedditSentimentFetcher:
    """Fetch sentiment from Reddit."""

    def __init__(self):
        self.analyzer = TextSentimentAnalyzer()
        self.subreddits = ["cryptocurrency", "bitcoin", "ethtrader", "altcoin"]

    async def fetch_posts(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[SentimentData]:
        """Fetch and analyze Reddit posts."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required")

        results = []

        for subreddit in self.subreddits:
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                "q": symbol,
                "sort": "new",
                "limit": min(limit // len(self.subreddits), 25),
                "restrict_sr": "true",
                "t": "day",
            }
            headers = {"User-Agent": "TradingBot/1.0"}

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, headers=headers) as response:
                        if response.status != 200:
                            continue

                        data = await response.json()

                for post in data.get("data", {}).get("children", []):
                    post_data = post.get("data", {})
                    text = f"{post_data.get('title', '')} {post_data.get('selftext', '')}"

                    sentiment, confidence = self.analyzer.analyze(text)

                    results.append(SentimentData(
                        source=DataSource.REDDIT,
                        symbol=symbol,
                        text=text[:500],
                        sentiment_score=sentiment,
                        confidence=confidence,
                        metadata={
                            "score": post_data.get("score", 0),
                            "num_comments": post_data.get("num_comments", 0),
                            "subreddit": subreddit,
                        }
                    ))

            except Exception as e:
                logger.warning(f"Reddit fetch failed for r/{subreddit}: {e}")

        return results


class FearGreedIndexFetcher:
    """Fetch Crypto Fear & Greed Index."""

    async def fetch(self) -> Optional[SentimentData]:
        """Fetch current Fear & Greed Index."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required")

        url = "https://api.alternative.me/fng/"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return None

                    data = await response.json()

            fng_data = data.get("data", [{}])[0]
            value = int(fng_data.get("value", 50))
            classification = fng_data.get("value_classification", "Neutral")

            # Convert 0-100 to -1 to 1
            sentiment_score = (value - 50) / 50

            return SentimentData(
                source=DataSource.FEAR_GREED,
                symbol="MARKET",
                text=classification,
                sentiment_score=sentiment_score,
                confidence=0.8,
                metadata={
                    "value": value,
                    "classification": classification,
                }
            )

        except Exception as e:
            logger.error(f"Fear & Greed fetch failed: {e}")
            return None


class SentimentAggregator:
    """Aggregate sentiment from multiple sources."""

    def __init__(
        self,
        twitter_fetcher: Optional[TwitterSentimentFetcher] = None,
        reddit_fetcher: Optional[RedditSentimentFetcher] = None,
        history_size: int = 1000,
    ):
        self.twitter = twitter_fetcher
        self.reddit = reddit_fetcher or RedditSentimentFetcher()
        self.fear_greed = FearGreedIndexFetcher()

        self.history: Dict[str, deque] = {}
        self.history_size = history_size

        # Source weights
        self.weights = {
            DataSource.TWITTER: 1.0,
            DataSource.REDDIT: 0.8,
            DataSource.NEWS: 1.2,
            DataSource.FEAR_GREED: 0.5,
        }

        # Callbacks
        self.on_sentiment_update: Optional[Callable] = None

    def add_data(self, data: SentimentData):
        """Add sentiment data point."""
        symbol = data.symbol

        if symbol not in self.history:
            self.history[symbol] = deque(maxlen=self.history_size)

        self.history[symbol].append(data)

    async def fetch_all(self, symbol: str) -> List[SentimentData]:
        """Fetch sentiment from all sources."""
        all_data = []

        # Twitter
        if self.twitter:
            try:
                tweets = await self.twitter.fetch_tweets(symbol)
                all_data.extend(tweets)
            except Exception as e:
                logger.warning(f"Twitter fetch failed: {e}")

        # Reddit
        try:
            posts = await self.reddit.fetch_posts(symbol)
            all_data.extend(posts)
        except Exception as e:
            logger.warning(f"Reddit fetch failed: {e}")

        # Fear & Greed
        try:
            fng = await self.fear_greed.fetch()
            if fng:
                all_data.append(fng)
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")

        # Store in history
        for data in all_data:
            self.add_data(data)

        return all_data

    def get_aggregate(
        self,
        symbol: str,
        time_window: timedelta = timedelta(hours=24),
    ) -> AggregateSentiment:
        """Get aggregated sentiment for symbol."""
        now = datetime.now()
        cutoff = now - time_window

        # Get recent data
        data_points = [
            d for d in self.history.get(symbol, [])
            if d.timestamp >= cutoff
        ]

        if not data_points:
            return AggregateSentiment(
                symbol=symbol,
                overall_score=0,
                overall_level=SentimentLevel.NEUTRAL,
                source_scores={},
                sample_size=0,
                confidence=0,
            )

        # Calculate weighted average by source
        source_scores: Dict[str, List[float]] = {}
        for data in data_points:
            source = data.source.value
            if source not in source_scores:
                source_scores[source] = []
            source_scores[source].append(data.sentiment_score * data.confidence)

        # Average per source
        source_averages = {}
        for source, scores in source_scores.items():
            source_averages[source] = sum(scores) / len(scores) if scores else 0

        # Weighted overall score
        total_weight = 0
        weighted_sum = 0
        for source, avg in source_averages.items():
            weight = self.weights.get(DataSource(source), 1.0)
            weighted_sum += avg * weight
            total_weight += weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0

        # Determine level
        if overall_score <= -0.6:
            level = SentimentLevel.VERY_BEARISH
        elif overall_score <= -0.2:
            level = SentimentLevel.BEARISH
        elif overall_score <= 0.2:
            level = SentimentLevel.NEUTRAL
        elif overall_score <= 0.6:
            level = SentimentLevel.BULLISH
        else:
            level = SentimentLevel.VERY_BULLISH

        # Calculate trend
        trend = self._calculate_trend(symbol, time_window)

        # Confidence based on sample size
        confidence = min(1.0, len(data_points) / 50)

        return AggregateSentiment(
            symbol=symbol,
            overall_score=overall_score,
            overall_level=level,
            source_scores=source_averages,
            sample_size=len(data_points),
            confidence=confidence,
            trend=trend,
        )

    def _calculate_trend(self, symbol: str, window: timedelta) -> str:
        """Calculate sentiment trend."""
        now = datetime.now()
        half_window = window / 2

        recent = [
            d.sentiment_score for d in self.history.get(symbol, [])
            if d.timestamp >= now - half_window
        ]
        older = [
            d.sentiment_score for d in self.history.get(symbol, [])
            if now - window <= d.timestamp < now - half_window
        ]

        if not recent or not older:
            return "stable"

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        diff = recent_avg - older_avg

        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        return "stable"


class SentimentSignalGenerator:
    """Generate trading signals from sentiment."""

    def __init__(
        self,
        aggregator: SentimentAggregator,
        bullish_threshold: float = 0.3,
        bearish_threshold: float = -0.3,
    ):
        self.aggregator = aggregator
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold

        # Signal history
        self.signals: List[Dict[str, Any]] = []

        # Callbacks
        self.on_signal: Optional[Callable] = None

    async def generate_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Generate trading signal from sentiment."""
        # Fetch latest data
        await self.aggregator.fetch_all(symbol)

        # Get aggregate
        sentiment = self.aggregator.get_aggregate(symbol)

        if sentiment.sample_size < 10:
            return None  # Not enough data

        signal = None

        if sentiment.overall_score >= self.bullish_threshold:
            if sentiment.trend == "improving":
                signal = {
                    "symbol": symbol,
                    "action": "buy",
                    "strength": "strong",
                    "sentiment": sentiment.to_dict(),
                }
            elif sentiment.trend != "declining":
                signal = {
                    "symbol": symbol,
                    "action": "buy",
                    "strength": "moderate",
                    "sentiment": sentiment.to_dict(),
                }

        elif sentiment.overall_score <= self.bearish_threshold:
            if sentiment.trend == "declining":
                signal = {
                    "symbol": symbol,
                    "action": "sell",
                    "strength": "strong",
                    "sentiment": sentiment.to_dict(),
                }
            elif sentiment.trend != "improving":
                signal = {
                    "symbol": symbol,
                    "action": "sell",
                    "strength": "moderate",
                    "sentiment": sentiment.to_dict(),
                }

        if signal:
            signal["timestamp"] = datetime.now().isoformat()
            self.signals.append(signal)

            if self.on_signal:
                self.on_signal(signal)

            logger.info(
                f"Sentiment signal: {signal['action']} {symbol} "
                f"({signal['strength']}) score={sentiment.overall_score:.2f}"
            )

        return signal

    async def run_loop(
        self,
        symbols: List[str],
        interval_minutes: int = 15,
    ):
        """Run signal generation loop."""
        while True:
            for symbol in symbols:
                try:
                    await self.generate_signal(symbol)
                except Exception as e:
                    logger.error(f"Signal generation failed for {symbol}: {e}")

            await asyncio.sleep(interval_minutes * 60)
