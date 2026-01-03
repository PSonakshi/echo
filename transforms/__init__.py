"""
Transforms module for the Crypto Narrative Pulse Tracker.

Contains transformation components for the Pathway streaming pipeline:
- SentimentAnalyzer: VADER-based sentiment analysis with crypto lexicon
- Sentiment Pipeline: Pathway integration for sentiment velocity
- Price Pipeline: Price delta calculation over sliding windows
- Divergence Detection: Sentiment-price divergence detection
- PulseScoreCalculator: Combined momentum scoring (1-10)
- (Future) PhraseClusterer: N-gram extraction and trending detection
- (Future) InfluenceCalculator: Author influence scoring
"""

from transforms.divergence import (
    DivergenceAnalyzer,
    analyze_divergence,
    detect_divergence,
    detect_divergence_enum,
    get_divergence_analyzer,
    get_divergence_description,
    get_divergence_emoji,
    is_aligned,
    is_bearish_divergence,
    is_bullish_divergence,
)
from transforms.divergence import (
    DivergenceType as DivergenceTypeEnum,
)
from transforms.pulse_score import (
    DivergenceType,
    PulseScoreCalculator,
    calculate_pulse_score,
    get_pulse_calculator,
)
from transforms.sentiment import (
    SentimentAnalyzer,
    analyze_sentiment,
    get_sentiment_analyzer,
)

__all__ = [
    # Sentiment Analyzer
    "SentimentAnalyzer",
    "analyze_sentiment",
    "get_sentiment_analyzer",
    # Pulse Score Calculator
    "DivergenceType",
    "PulseScoreCalculator",
    "calculate_pulse_score",
    "get_pulse_calculator",
    # Divergence Detection
    "DivergenceAnalyzer",
    "DivergenceTypeEnum",
    "analyze_divergence",
    "detect_divergence",
    "detect_divergence_enum",
    "get_divergence_analyzer",
    "get_divergence_description",
    "get_divergence_emoji",
    "is_aligned",
    "is_bearish_divergence",
    "is_bullish_divergence",
]

# Pathway-dependent imports (only available when pathway is installed)
try:
    from transforms.price_pipeline import (
        add_price_classification,
        calculate_price_delta,
        calculate_price_delta_pct,
        classify_price_movement,
        create_price_pipeline,
        filter_significant_price_moves,
        get_latest_price_delta,
    )
    from transforms.sentiment_pipeline import (
        STANDARD_WINDOW_DURATION,
        STANDARD_WINDOW_HOP,
        add_sentiment_scores,
        calculate_sentiment_velocity,
        classify_momentum,
        create_sentiment_pipeline,
        filter_by_momentum,
        get_latest_velocity,
    )

    __all__.extend(
        [
            # Sentiment Pipeline
            "STANDARD_WINDOW_DURATION",
            "STANDARD_WINDOW_HOP",
            "add_sentiment_scores",
            "calculate_sentiment_velocity",
            "classify_momentum",
            "create_sentiment_pipeline",
            "filter_by_momentum",
            "get_latest_velocity",
            # Price Pipeline
            "calculate_price_delta",
            "calculate_price_delta_pct",
            "create_price_pipeline",
            "get_latest_price_delta",
            "filter_significant_price_moves",
            "classify_price_movement",
            "add_price_classification",
        ]
    )
except ImportError:
    # Pathway not installed - pipeline functions not available
    pass
