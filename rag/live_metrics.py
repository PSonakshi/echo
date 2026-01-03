"""
Live Metrics State Management for the Crypto Narrative Pulse Tracker.

Provides a shared state dictionary for real-time metrics that are used
to enrich RAG prompts with live context:
- pulse_score: Current momentum score (1-10)
- trending_phrases: Top trending phrases from recent messages
- influencer_consensus: Bullish/bearish/neutral consensus from influencers
- divergence_status: Current sentiment-price divergence status

Uses pw.io.subscribe() to update state from pipeline outputs.

Requirements: 8.4
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    import pathway as pw

logger = logging.getLogger(__name__)


# =============================================================================
# LIVE METRICS DATA CLASS
# =============================================================================


@dataclass
class LiveMetrics:
    """
    Shared state container for live pipeline metrics.

    Thread-safe container that holds the current state of all metrics
    used for RAG prompt enrichment.

    Attributes:
        pulse_score: Current momentum score (1-10)
        trending_phrases: List of top trending phrases
        influencer_consensus: Consensus classification string
        divergence_status: Current divergence type
        last_updated: Timestamp of last update
        message_count: Total messages processed in current window

    Requirements: 8.4
    """

    pulse_score: float = 5.0
    trending_phrases: list[str] = field(default_factory=list)
    influencer_consensus: str = "neutral"
    divergence_status: str = "aligned"
    last_updated: Optional[datetime] = None
    message_count: int = 0
    sentiment_velocity: float = 0.0
    influencer_bullish_count: int = 0
    influencer_bearish_count: int = 0
    current_price: Optional[float] = None
    price_delta_pct: float = 0.0

    # Thread lock for safe concurrent access
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs) -> None:
        """
        Thread-safe update of metrics.

        Args:
            **kwargs: Metric fields to update
        """
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key) and not key.startswith("_"):
                    setattr(self, key, value)
            self.last_updated = datetime.utcnow()

    def get_snapshot(self) -> dict[str, Any]:
        """
        Get a thread-safe snapshot of current metrics.

        Returns:
            Dictionary with all current metric values
        """
        with self._lock:
            return {
                "pulse_score": self.pulse_score,
                "trending_phrases": list(self.trending_phrases),
                "influencer_consensus": self.influencer_consensus,
                "divergence_status": self.divergence_status,
                "last_updated": self.last_updated.isoformat()
                if self.last_updated
                else None,
                "message_count": self.message_count,
                "sentiment_velocity": self.sentiment_velocity,
                "influencer_bullish_count": self.influencer_bullish_count,
                "influencer_bearish_count": self.influencer_bearish_count,
                "current_price": self.current_price,
                "price_delta_pct": self.price_delta_pct,
            }

    def get_rag_context(self) -> dict[str, str]:
        """
        Get metrics formatted for RAG prompt enrichment.

        Returns:
            Dictionary with string-formatted metrics for prompt template

        Requirements: 8.4
        """
        with self._lock:
            phrases_str = (
                ", ".join(self.trending_phrases[:3])
                if self.trending_phrases
                else "None detected"
            )
            return {
                "pulse_score": f"{self.pulse_score:.1f}",
                "trending_phrases": phrases_str,
                "influencer_consensus": self.influencer_consensus,
                "divergence_status": self._format_divergence_status(),
            }

    def _format_divergence_status(self) -> str:
        """Format divergence status for display."""
        status_map = {
            "aligned": "✅ Aligned",
            "bearish_divergence": "⚠️ Bearish Divergence",
            "bullish_divergence": "📈 Bullish Divergence",
        }
        return status_map.get(self.divergence_status, self.divergence_status)


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_live_metrics: Optional[LiveMetrics] = None
_metrics_lock = threading.Lock()


def get_live_metrics() -> LiveMetrics:
    """
    Get the singleton LiveMetrics instance.

    Returns:
        LiveMetrics instance (creates one if not exists)
    """
    global _live_metrics
    with _metrics_lock:
        if _live_metrics is None:
            _live_metrics = LiveMetrics()
        return _live_metrics


def reset_live_metrics() -> None:
    """Reset the live metrics to default values (useful for testing)."""
    global _live_metrics
    with _metrics_lock:
        _live_metrics = LiveMetrics()


# =============================================================================
# UPDATE FUNCTIONS
# =============================================================================


def update_pulse_score(score: float) -> None:
    """
    Update the current pulse score.

    Args:
        score: New pulse score (1-10)
    """
    metrics = get_live_metrics()
    metrics.update(pulse_score=max(1.0, min(10.0, score)))
    logger.debug(f"Updated pulse_score to {score}")


def update_trending_phrases(phrases: list[str]) -> None:
    """
    Update the trending phrases list.

    Args:
        phrases: List of trending phrases (will keep top 10)
    """
    metrics = get_live_metrics()
    metrics.update(trending_phrases=phrases[:10])
    logger.debug(f"Updated trending_phrases: {phrases[:3]}")


def update_influencer_consensus(
    bullish_count: int,
    bearish_count: int,
) -> None:
    """
    Update influencer consensus based on bullish/bearish counts.

    Args:
        bullish_count: Number of bullish influencer signals
        bearish_count: Number of bearish influencer signals
    """
    metrics = get_live_metrics()
    total = bullish_count + bearish_count

    if total == 0:
        consensus = "neutral"
    else:
        ratio = bullish_count / total
        if ratio > 0.7:
            consensus = "strongly bullish"
        elif ratio > 0.5:
            consensus = "moderately bullish"
        elif ratio < 0.3:
            consensus = "strongly bearish"
        elif ratio < 0.5:
            consensus = "moderately bearish"
        else:
            consensus = "neutral"

    metrics.update(
        influencer_consensus=consensus,
        influencer_bullish_count=bullish_count,
        influencer_bearish_count=bearish_count,
    )
    logger.debug(f"Updated influencer_consensus to {consensus}")


def update_divergence_status(divergence_type: str) -> None:
    """
    Update the divergence status.

    Args:
        divergence_type: One of "aligned", "bearish_divergence", "bullish_divergence"
    """
    metrics = get_live_metrics()
    metrics.update(divergence_status=divergence_type)
    logger.debug(f"Updated divergence_status to {divergence_type}")


def update_sentiment_velocity(velocity: float, message_count: int = 0) -> None:
    """
    Update sentiment velocity metrics.

    Args:
        velocity: Current sentiment velocity (-1 to 1)
        message_count: Number of messages in the window
    """
    metrics = get_live_metrics()
    metrics.update(
        sentiment_velocity=velocity,
        message_count=message_count,
    )
    logger.debug(f"Updated sentiment_velocity to {velocity}")


def update_price_data(price: float, delta_pct: float) -> None:
    """
    Update price-related metrics.

    Args:
        price: Current price in USD
        delta_pct: Price change percentage
    """
    metrics = get_live_metrics()
    metrics.update(
        current_price=price,
        price_delta_pct=delta_pct,
    )
    logger.debug(f"Updated price to {price}, delta {delta_pct}%")


# =============================================================================
# PATHWAY SUBSCRIPTION HANDLERS
# =============================================================================


def _create_pulse_score_handler() -> Callable:
    """Create handler for pulse score updates."""

    def handler(key, row, time, is_addition):
        if is_addition and row is not None:
            try:
                score = row.get("score", row.get("pulse_score", 5.0))
                update_pulse_score(float(score))
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Error updating pulse score: {e}")

    return handler


def _create_phrases_handler() -> Callable:
    """Create handler for trending phrases updates."""

    def handler(key, row, time, is_addition):
        if is_addition and row is not None:
            try:
                phrases = row.get("phrases", row.get("trending_phrases", []))
                if isinstance(phrases, str):
                    phrases = [phrases]
                update_trending_phrases(list(phrases))
            except (KeyError, TypeError) as e:
                logger.warning(f"Error updating trending phrases: {e}")

    return handler


def _create_influencer_handler() -> Callable:
    """Create handler for influencer consensus updates."""

    def handler(key, row, time, is_addition):
        if is_addition and row is not None:
            try:
                bullish = int(row.get("bullish_count", 0))
                bearish = int(row.get("bearish_count", 0))
                update_influencer_consensus(bullish, bearish)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Error updating influencer consensus: {e}")

    return handler


def _create_divergence_handler() -> Callable:
    """Create handler for divergence status updates."""

    def handler(key, row, time, is_addition):
        if is_addition and row is not None:
            try:
                divergence = row.get(
                    "divergence_type", row.get("divergence", "aligned")
                )
                update_divergence_status(str(divergence))
            except (KeyError, TypeError) as e:
                logger.warning(f"Error updating divergence status: {e}")

    return handler


def _create_velocity_handler() -> Callable:
    """Create handler for sentiment velocity updates."""

    def handler(key, row, time, is_addition):
        if is_addition and row is not None:
            try:
                velocity = float(row.get("velocity", 0.0))
                count = int(row.get("message_count", 0))
                update_sentiment_velocity(velocity, count)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Error updating sentiment velocity: {e}")

    return handler


# =============================================================================
# PIPELINE SUBSCRIPTION
# =============================================================================


def subscribe_to_pipeline_outputs(
    pulse_scores: Optional["pw.Table"] = None,
    trending_phrases: Optional["pw.Table"] = None,
    influencer_signals: Optional["pw.Table"] = None,
    price_correlation: Optional["pw.Table"] = None,
    sentiment_velocity: Optional["pw.Table"] = None,
) -> None:
    """
    Subscribe to pipeline output tables to update live metrics.

    Uses pw.io.subscribe() to receive updates from Pathway tables
    and update the shared LiveMetrics state.

    Args:
        pulse_scores: Table with pulse score calculations
        trending_phrases: Table with trending phrase data
        influencer_signals: Table with influencer consensus data
        price_correlation: Table with divergence detection results
        sentiment_velocity: Table with sentiment velocity data

    Requirements: 8.4

    Example:
        >>> subscribe_to_pipeline_outputs(
        ...     pulse_scores=pulse_table,
        ...     trending_phrases=phrases_table,
        ...     influencer_signals=influencer_table,
        ...     price_correlation=correlation_table,
        ... )
    """
    import pathway as pw

    if pulse_scores is not None:
        pw.io.subscribe(pulse_scores, on_change=_create_pulse_score_handler())
        logger.info("Subscribed to pulse_scores updates")

    if trending_phrases is not None:
        pw.io.subscribe(trending_phrases, on_change=_create_phrases_handler())
        logger.info("Subscribed to trending_phrases updates")

    if influencer_signals is not None:
        pw.io.subscribe(influencer_signals, on_change=_create_influencer_handler())
        logger.info("Subscribed to influencer_signals updates")

    if price_correlation is not None:
        pw.io.subscribe(price_correlation, on_change=_create_divergence_handler())
        logger.info("Subscribed to price_correlation updates")

    if sentiment_velocity is not None:
        pw.io.subscribe(sentiment_velocity, on_change=_create_velocity_handler())
        logger.info("Subscribed to sentiment_velocity updates")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_metrics_for_alert() -> dict[str, Any]:
    """
    Get metrics formatted for alert messages.

    Returns:
        Dictionary with metrics suitable for Telegram alerts
    """
    metrics = get_live_metrics()
    snapshot = metrics.get_snapshot()

    return {
        "score": snapshot["pulse_score"],
        "phrases": snapshot["trending_phrases"][:3],
        "consensus": snapshot["influencer_consensus"],
        "divergence": snapshot["divergence_status"],
        "velocity": snapshot["sentiment_velocity"],
    }


def get_metrics_for_dashboard() -> dict[str, Any]:
    """
    Get metrics formatted for dashboard display.

    Returns:
        Dictionary with all metrics for Streamlit dashboard
    """
    return get_live_metrics().get_snapshot()
