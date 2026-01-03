"""
Divergence Detection module for the Crypto Narrative Pulse Tracker.

Detects sentiment-price divergences that may indicate market turning points:
- Bearish divergence: sentiment > 0.5 AND price_delta < -2%
- Bullish divergence: sentiment < -0.5 AND price_delta > 2%

Requirements: 4.4, 4.5
"""

from enum import Enum
from typing import Union


class DivergenceType(Enum):
    """Types of sentiment-price divergence."""

    ALIGNED = "aligned"
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"


# =============================================================================
# DIVERGENCE THRESHOLDS
# =============================================================================

# Sentiment thresholds for divergence detection
SENTIMENT_HIGH_THRESHOLD = 0.5  # High sentiment (bullish)
SENTIMENT_LOW_THRESHOLD = -0.5  # Low sentiment (bearish)

# Price delta thresholds for divergence detection (percentage)
PRICE_FALLING_THRESHOLD = -2.0  # Price falling significantly
PRICE_RISING_THRESHOLD = 2.0  # Price rising significantly


# =============================================================================
# DIVERGENCE DETECTION FUNCTIONS
# =============================================================================


def detect_divergence(
    sentiment: float,
    price_delta_pct: float,
    sentiment_high: float = SENTIMENT_HIGH_THRESHOLD,
    sentiment_low: float = SENTIMENT_LOW_THRESHOLD,
    price_falling: float = PRICE_FALLING_THRESHOLD,
    price_rising: float = PRICE_RISING_THRESHOLD,
) -> str:
    """
    Detect sentiment-price divergence.

    Divergence occurs when sentiment and price move in opposite directions:
    - Bearish divergence: High sentiment (>0.5) but price falling (<-2%)
      This suggests the market may be overheated and due for a correction.
    - Bullish divergence: Low sentiment (<-0.5) but price rising (>2%)
      This suggests the market may be oversold and due for a recovery.

    Args:
        sentiment: Sentiment velocity/score (-1 to 1)
        price_delta_pct: Price change percentage over the window
        sentiment_high: Threshold for high sentiment (default: 0.5)
        sentiment_low: Threshold for low sentiment (default: -0.5)
        price_falling: Threshold for falling price (default: -2.0%)
        price_rising: Threshold for rising price (default: 2.0%)

    Returns:
        Divergence type string:
        - "bearish_divergence" if sentiment > 0.5 AND price_delta < -2%
        - "bullish_divergence" if sentiment < -0.5 AND price_delta > 2%
        - "aligned" otherwise

    Requirements: 4.4, 4.5

    Example:
        >>> detect_divergence(0.7, -3.0)
        'bearish_divergence'  # High sentiment but price falling
        >>> detect_divergence(-0.6, 3.0)
        'bullish_divergence'  # Low sentiment but price rising
        >>> detect_divergence(0.7, 3.0)
        'aligned'  # Both sentiment and price are positive
    """
    # Bearish divergence: sentiment is high but price is falling
    # This is a warning sign - people are bullish but price is dropping
    # Requirement 4.4
    if sentiment > sentiment_high and price_delta_pct < price_falling:
        return DivergenceType.BEARISH_DIVERGENCE.value

    # Bullish divergence: sentiment is low but price is rising
    # This could be a buying opportunity - people are bearish but price is rising
    # Requirement 4.5
    if sentiment < sentiment_low and price_delta_pct > price_rising:
        return DivergenceType.BULLISH_DIVERGENCE.value

    # No divergence - sentiment and price are aligned
    return DivergenceType.ALIGNED.value


def detect_divergence_enum(
    sentiment: float,
    price_delta_pct: float,
) -> DivergenceType:
    """
    Detect sentiment-price divergence and return as enum.

    Same as detect_divergence but returns DivergenceType enum instead of string.

    Args:
        sentiment: Sentiment velocity/score (-1 to 1)
        price_delta_pct: Price change percentage over the window

    Returns:
        DivergenceType enum value

    Requirements: 4.4, 4.5
    """
    result = detect_divergence(sentiment, price_delta_pct)
    return DivergenceType(result)


def is_bearish_divergence(sentiment: float, price_delta_pct: float) -> bool:
    """
    Check if there is a bearish divergence.

    Bearish divergence: sentiment > 0.5 AND price_delta < -2%

    Args:
        sentiment: Sentiment velocity/score (-1 to 1)
        price_delta_pct: Price change percentage over the window

    Returns:
        True if bearish divergence detected

    Requirements: 4.4
    """
    return (
        detect_divergence(sentiment, price_delta_pct)
        == DivergenceType.BEARISH_DIVERGENCE.value
    )


def is_bullish_divergence(sentiment: float, price_delta_pct: float) -> bool:
    """
    Check if there is a bullish divergence.

    Bullish divergence: sentiment < -0.5 AND price_delta > 2%

    Args:
        sentiment: Sentiment velocity/score (-1 to 1)
        price_delta_pct: Price change percentage over the window

    Returns:
        True if bullish divergence detected

    Requirements: 4.5
    """
    return (
        detect_divergence(sentiment, price_delta_pct)
        == DivergenceType.BULLISH_DIVERGENCE.value
    )


def is_aligned(sentiment: float, price_delta_pct: float) -> bool:
    """
    Check if sentiment and price are aligned (no divergence).

    Args:
        sentiment: Sentiment velocity/score (-1 to 1)
        price_delta_pct: Price change percentage over the window

    Returns:
        True if no divergence detected
    """
    return detect_divergence(sentiment, price_delta_pct) == DivergenceType.ALIGNED.value


def get_divergence_description(divergence_type: Union[str, DivergenceType]) -> str:
    """
    Get a human-readable description of the divergence type.

    Args:
        divergence_type: Divergence type string or enum

    Returns:
        Human-readable description
    """
    if isinstance(divergence_type, DivergenceType):
        divergence_str = divergence_type.value
    else:
        divergence_str = str(divergence_type).lower()

    descriptions = {
        "bearish_divergence": "⚠️ Bearish Divergence: High sentiment but price falling - potential correction ahead",
        "bullish_divergence": "📈 Bullish Divergence: Low sentiment but price rising - potential buying opportunity",
        "aligned": "✅ Aligned: Sentiment and price moving in same direction",
    }

    return descriptions.get(divergence_str, "Unknown divergence type")


def get_divergence_emoji(divergence_type: Union[str, DivergenceType]) -> str:
    """
    Get an emoji indicator for the divergence type.

    Args:
        divergence_type: Divergence type string or enum

    Returns:
        Emoji string
    """
    if isinstance(divergence_type, DivergenceType):
        divergence_str = divergence_type.value
    else:
        divergence_str = str(divergence_type).lower()

    emojis = {
        "bearish_divergence": "⚠️",
        "bullish_divergence": "📈",
        "aligned": "✅",
    }

    return emojis.get(divergence_str, "❓")


# =============================================================================
# DIVERGENCE ANALYSIS
# =============================================================================


class DivergenceAnalyzer:
    """
    Analyzer for detecting and tracking sentiment-price divergences.

    Provides methods for detecting divergences and generating alerts.

    Requirements: 4.4, 4.5

    Example:
        >>> analyzer = DivergenceAnalyzer()
        >>> result = analyzer.analyze(sentiment=0.7, price_delta_pct=-3.0)
        >>> print(result['divergence_type'])  # 'bearish_divergence'
        >>> print(result['description'])  # Human-readable description
    """

    def __init__(
        self,
        sentiment_high: float = SENTIMENT_HIGH_THRESHOLD,
        sentiment_low: float = SENTIMENT_LOW_THRESHOLD,
        price_falling: float = PRICE_FALLING_THRESHOLD,
        price_rising: float = PRICE_RISING_THRESHOLD,
    ):
        """
        Initialize the divergence analyzer with custom thresholds.

        Args:
            sentiment_high: Threshold for high sentiment (default: 0.5)
            sentiment_low: Threshold for low sentiment (default: -0.5)
            price_falling: Threshold for falling price (default: -2.0%)
            price_rising: Threshold for rising price (default: 2.0%)
        """
        self.sentiment_high = sentiment_high
        self.sentiment_low = sentiment_low
        self.price_falling = price_falling
        self.price_rising = price_rising

    def analyze(
        self,
        sentiment: float,
        price_delta_pct: float,
    ) -> dict:
        """
        Analyze sentiment and price for divergence.

        Args:
            sentiment: Sentiment velocity/score (-1 to 1)
            price_delta_pct: Price change percentage over the window

        Returns:
            Dictionary with:
            - divergence_type: String type of divergence
            - is_divergent: Boolean indicating if divergence detected
            - description: Human-readable description
            - emoji: Emoji indicator
            - sentiment: Input sentiment value
            - price_delta_pct: Input price delta value

        Requirements: 4.4, 4.5
        """
        divergence_type = detect_divergence(
            sentiment,
            price_delta_pct,
            self.sentiment_high,
            self.sentiment_low,
            self.price_falling,
            self.price_rising,
        )

        return {
            "divergence_type": divergence_type,
            "is_divergent": divergence_type != DivergenceType.ALIGNED.value,
            "description": get_divergence_description(divergence_type),
            "emoji": get_divergence_emoji(divergence_type),
            "sentiment": sentiment,
            "price_delta_pct": price_delta_pct,
        }

    def should_alert(self, sentiment: float, price_delta_pct: float) -> bool:
        """
        Check if a divergence alert should be sent.

        Args:
            sentiment: Sentiment velocity/score (-1 to 1)
            price_delta_pct: Price change percentage over the window

        Returns:
            True if divergence detected and alert should be sent
        """
        divergence_type = detect_divergence(
            sentiment,
            price_delta_pct,
            self.sentiment_high,
            self.sentiment_low,
            self.price_falling,
            self.price_rising,
        )
        return divergence_type != DivergenceType.ALIGNED.value


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_default_analyzer: DivergenceAnalyzer | None = None


def get_divergence_analyzer() -> DivergenceAnalyzer:
    """
    Get the default divergence analyzer instance (singleton).

    Returns:
        DivergenceAnalyzer instance
    """
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = DivergenceAnalyzer()
    return _default_analyzer


def analyze_divergence(sentiment: float, price_delta_pct: float) -> dict:
    """
    Convenience function to analyze divergence using the default analyzer.

    Args:
        sentiment: Sentiment velocity/score (-1 to 1)
        price_delta_pct: Price change percentage over the window

    Returns:
        Divergence analysis dictionary
    """
    return get_divergence_analyzer().analyze(sentiment, price_delta_pct)
