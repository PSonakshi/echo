"""
Pulse Score Calculator module for the Crypto Narrative Pulse Tracker.

Combines sentiment velocity, phrase frequency, influencer signals, and
price correlation into a single momentum score (1-10) for traders.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9
"""

from enum import Enum
from typing import Union


class DivergenceType(Enum):
    """Types of sentiment-price divergence."""

    ALIGNED = "aligned"
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"


class PulseScoreCalculator:
    """
    Combines all signals into a 1-10 momentum score.

    The Pulse Score is calculated by combining:
    - Sentiment velocity: 0-4 points
    - Phrase frequency spike: 0-3 points
    - Influencer bullish ratio: 0-3 points
    - Divergence modifier: -1 to 0 points

    Final score is clamped to [1, 10] range.

    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9

    Example:
        >>> calculator = PulseScoreCalculator()
        >>> score = calculator.calculate(
        ...     sentiment_velocity=0.8,
        ...     phrase_frequency=25,
        ...     influencer_ratio=0.75,
        ...     divergence_type="aligned"
        ... )
        >>> print(score)  # 10.0 (4 + 3 + 3 = 10, clamped)
    """

    # Sentiment velocity thresholds and points (Requirements 7.2, 7.3)
    SENTIMENT_HIGH_THRESHOLD = 0.7
    SENTIMENT_MID_THRESHOLD = 0.4
    SENTIMENT_HIGH_POINTS = 4.0
    SENTIMENT_MID_POINTS = 2.0

    # Phrase frequency thresholds and points (Requirements 7.4, 7.5)
    PHRASE_HIGH_THRESHOLD = 20
    PHRASE_MID_THRESHOLD = 10
    PHRASE_HIGH_POINTS = 3.0
    PHRASE_MID_POINTS = 1.5

    # Influencer ratio thresholds and points (Requirements 7.6, 7.7)
    INFLUENCER_HIGH_THRESHOLD = 0.7
    INFLUENCER_MID_THRESHOLD = 0.5
    INFLUENCER_HIGH_POINTS = 3.0
    INFLUENCER_MID_POINTS = 1.5

    # Divergence penalty (Requirement 7.8)
    BEARISH_DIVERGENCE_PENALTY = 1.0

    # Score bounds (Requirement 7.9)
    MIN_SCORE = 1.0
    MAX_SCORE = 10.0

    def calculate(
        self,
        sentiment_velocity: float,
        phrase_frequency: int,
        influencer_ratio: float,
        divergence_type: Union[str, DivergenceType],
    ) -> float:
        """
        Calculate the Pulse Score from all input signals.

        Scoring formula (Requirement 7.1):
        - Sentiment velocity: 0-4 points
        - Phrase frequency spike: 0-3 points
        - Influencer bullish ratio: 0-3 points
        - Divergence modifier: -1 to 0 points

        Args:
            sentiment_velocity: Average sentiment over time window (-1 to 1)
            phrase_frequency: Count of trending phrase occurrences
            influencer_ratio: Ratio of bullish influencers (0 to 1)
            divergence_type: Type of sentiment-price divergence

        Returns:
            Pulse score clamped to [1, 10] range

        Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9
        """
        score = 0.0

        # Sentiment component (0-4 points) - Requirements 7.2, 7.3
        score += self._calculate_sentiment_points(sentiment_velocity)

        # Phrase frequency component (0-3 points) - Requirements 7.4, 7.5
        score += self._calculate_phrase_points(phrase_frequency)

        # Influencer component (0-3 points) - Requirements 7.6, 7.7
        score += self._calculate_influencer_points(influencer_ratio)

        # Divergence modifier (-1 to 0 points) - Requirement 7.8
        score += self._calculate_divergence_modifier(divergence_type)

        # Clamp to [1, 10] range - Requirement 7.9
        return max(self.MIN_SCORE, min(self.MAX_SCORE, score))

    def _calculate_sentiment_points(self, sentiment_velocity: float) -> float:
        """
        Calculate points from sentiment velocity.

        - velocity > 0.7: 4 points (Requirement 7.2)
        - velocity > 0.4: 2 points (Requirement 7.3)
        - otherwise: 0 points

        Args:
            sentiment_velocity: Average sentiment over time window

        Returns:
            Points contribution (0, 2, or 4)
        """
        if sentiment_velocity > self.SENTIMENT_HIGH_THRESHOLD:
            return self.SENTIMENT_HIGH_POINTS
        elif sentiment_velocity > self.SENTIMENT_MID_THRESHOLD:
            return self.SENTIMENT_MID_POINTS
        return 0.0

    def _calculate_phrase_points(self, phrase_frequency: int) -> float:
        """
        Calculate points from trending phrase frequency.

        - frequency > 20: 3 points (Requirement 7.4)
        - frequency > 10: 1.5 points (Requirement 7.5)
        - otherwise: 0 points

        Args:
            phrase_frequency: Count of trending phrase occurrences

        Returns:
            Points contribution (0, 1.5, or 3)
        """
        if phrase_frequency > self.PHRASE_HIGH_THRESHOLD:
            return self.PHRASE_HIGH_POINTS
        elif phrase_frequency > self.PHRASE_MID_THRESHOLD:
            return self.PHRASE_MID_POINTS
        return 0.0

    def _calculate_influencer_points(self, influencer_ratio: float) -> float:
        """
        Calculate points from influencer bullish ratio.

        - ratio > 0.7: 3 points (Requirement 7.6)
        - ratio > 0.5: 1.5 points (Requirement 7.7)
        - otherwise: 0 points

        Args:
            influencer_ratio: Ratio of bullish influencers (0 to 1)

        Returns:
            Points contribution (0, 1.5, or 3)
        """
        if influencer_ratio > self.INFLUENCER_HIGH_THRESHOLD:
            return self.INFLUENCER_HIGH_POINTS
        elif influencer_ratio > self.INFLUENCER_MID_THRESHOLD:
            return self.INFLUENCER_MID_POINTS
        return 0.0

    def _calculate_divergence_modifier(
        self, divergence_type: Union[str, DivergenceType]
    ) -> float:
        """
        Calculate divergence penalty.

        - bearish_divergence: -1 point (Requirement 7.8)
        - otherwise: 0 points

        Args:
            divergence_type: Type of sentiment-price divergence

        Returns:
            Points modifier (-1 or 0)
        """
        # Normalize to string for comparison
        if isinstance(divergence_type, DivergenceType):
            divergence_str = divergence_type.value
        else:
            divergence_str = str(divergence_type).lower()

        if divergence_str == "bearish_divergence":
            return -self.BEARISH_DIVERGENCE_PENALTY
        return 0.0

    def get_signal_type(self, score: float) -> str:
        """
        Get the signal type based on pulse score.

        Args:
            score: Pulse score (1-10)

        Returns:
            Signal type string:
            - "strong_buy" if score >= 7
            - "cooling_off" if score <= 3
            - "neutral" otherwise
        """
        if score >= 7:
            return "strong_buy"
        elif score <= 3:
            return "cooling_off"
        return "neutral"


# Module-level singleton for convenience
_default_calculator: PulseScoreCalculator | None = None


def get_pulse_calculator() -> PulseScoreCalculator:
    """
    Get the default pulse score calculator instance (singleton).

    Returns:
        PulseScoreCalculator instance
    """
    global _default_calculator
    if _default_calculator is None:
        _default_calculator = PulseScoreCalculator()
    return _default_calculator


def calculate_pulse_score(
    sentiment_velocity: float,
    phrase_frequency: int,
    influencer_ratio: float,
    divergence_type: Union[str, DivergenceType] = "aligned",
) -> float:
    """
    Convenience function to calculate pulse score using the default calculator.

    Args:
        sentiment_velocity: Average sentiment over time window (-1 to 1)
        phrase_frequency: Count of trending phrase occurrences
        influencer_ratio: Ratio of bullish influencers (0 to 1)
        divergence_type: Type of sentiment-price divergence

    Returns:
        Pulse score clamped to [1, 10] range
    """
    return get_pulse_calculator().calculate(
        sentiment_velocity, phrase_frequency, influencer_ratio, divergence_type
    )
