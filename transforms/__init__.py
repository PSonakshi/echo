"""
Transforms module for the Crypto Narrative Pulse Tracker.

Contains transformation components for the Pathway streaming pipeline:
- SentimentAnalyzer: VADER-based sentiment analysis with crypto lexicon
- Sentiment Pipeline: Pathway integration for sentiment velocity
- PulseScoreCalculator: Combined momentum scoring (1-10)
- (Future) PhraseClusterer: N-gram extraction and trending detection
- (Future) InfluenceCalculator: Author influence scoring
"""

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
]

# Pathway-dependent imports (only available when pathway is installed)
try:
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
        ]
    )
except ImportError:
    # Pathway not installed - pipeline functions not available
    pass
