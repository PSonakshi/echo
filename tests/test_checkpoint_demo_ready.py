"""
Checkpoint 11: Demo-Ready Verification Tests

This module verifies the demo-ready features:
1. End-to-end flow: simulator → pipeline → alerts
2. Telegram bot alerts and queries
3. RAG system integration
4. Live metrics state management

Requirements: Checkpoint 11 verification
"""

import sys
from unittest.mock import MagicMock, patch

# Mock pathway module before importing modules that depend on it
mock_pw = MagicMock()
mock_pw.Schema = type("Schema", (), {})
mock_pw.Duration = MagicMock()
mock_pw.DateTimeUtc = MagicMock()
sys.modules["pathway"] = mock_pw

# Mock telegram module before importing bot modules
mock_telegram = MagicMock()
mock_telegram.Bot = MagicMock()
mock_telegram.Update = MagicMock()
sys.modules["telegram"] = mock_telegram
sys.modules["telegram.ext"] = MagicMock()

import pytest

from schemas import validate_message_payload
from simulator.hype_simulator import (
    PHASE_ORDER,
    generate_single_message,
    generate_single_price,
)
from transforms.divergence import detect_divergence
from transforms.pulse_score import PulseScoreCalculator, calculate_pulse_score
from transforms.sentiment import SentimentAnalyzer

# =============================================================================
# TEST 1: End-to-End Flow - Simulator → Sentiment → Pulse Score
# =============================================================================


class TestEndToEndFlow:
    """Verify the complete data flow from simulator to pulse score."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.pulse_calculator = PulseScoreCalculator()

    def test_simulator_to_sentiment_flow(self):
        """Messages from simulator should flow through sentiment analysis."""
        for phase in PHASE_ORDER:
            # Step 1: Generate message from simulator
            message = generate_single_message(coin_symbol="MEME", phase=phase)

            # Step 2: Validate message schema
            payload = {k: v for k, v in message.items() if not k.startswith("_")}
            result = validate_message_payload(payload)
            assert result.is_valid, f"Message validation failed: {result.errors}"

            # Step 3: Analyze sentiment
            sentiment = self.sentiment_analyzer.analyze(message["text"])
            assert -1.0 <= sentiment <= 1.0, f"Invalid sentiment: {sentiment}"

    def test_sentiment_to_pulse_score_flow(self):
        """Sentiment should flow into pulse score calculation."""
        # Simulate aggregated metrics
        test_cases = [
            # (velocity, phrase_freq, influencer_ratio, divergence, expected_range)
            (0.8, 25, 0.8, "aligned", (7, 10)),  # High activity
            (0.5, 15, 0.6, "aligned", (4, 7)),  # Moderate activity
            (0.2, 5, 0.3, "aligned", (1, 3)),  # Low activity
            (0.8, 25, 0.8, "bearish_divergence", (6, 9)),  # High with penalty
        ]

        for velocity, freq, ratio, divergence, (min_score, max_score) in test_cases:
            score = self.pulse_calculator.calculate(velocity, freq, ratio, divergence)
            assert min_score <= score <= max_score, (
                f"Score {score} not in range [{min_score}, {max_score}]"
            )

    def test_full_pipeline_simulation(self):
        """Simulate a full pipeline run with multiple messages."""
        messages = []
        sentiments = []

        # Generate messages across all phases
        for phase in PHASE_ORDER:
            for _ in range(5):
                msg = generate_single_message(coin_symbol="MEME", phase=phase)
                messages.append(msg)
                sentiment = self.sentiment_analyzer.analyze(msg["text"])
                sentiments.append(sentiment)

        # Verify we processed all messages
        assert len(messages) == 20
        assert len(sentiments) == 20

        # Calculate average sentiment (simulating velocity)
        avg_sentiment = sum(sentiments) / len(sentiments)
        assert -1.0 <= avg_sentiment <= 1.0

        # Calculate pulse score with simulated metrics
        score = calculate_pulse_score(
            sentiment_velocity=avg_sentiment,
            phrase_frequency=15,  # Simulated
            influencer_ratio=0.5,  # Simulated
            divergence_type="aligned",
        )
        assert 1.0 <= score <= 10.0


# =============================================================================
# TEST 2: Telegram Bot Alert Logic
# =============================================================================


class TestTelegramBotAlerts:
    """Verify Telegram bot alert logic without actual API calls."""

    def test_strong_buy_threshold(self):
        """Score >= 7 should trigger strong buy alert."""
        from bot.telegram_bot import TelegramAlertBot

        # Test threshold logic
        assert TelegramAlertBot.STRONG_BUY_THRESHOLD == 7.0

        # Scores that should trigger alert
        assert 7.0 >= TelegramAlertBot.STRONG_BUY_THRESHOLD
        assert 8.5 >= TelegramAlertBot.STRONG_BUY_THRESHOLD
        assert 10.0 >= TelegramAlertBot.STRONG_BUY_THRESHOLD

        # Scores that should not trigger
        assert 6.9 < TelegramAlertBot.STRONG_BUY_THRESHOLD
        assert 5.0 < TelegramAlertBot.STRONG_BUY_THRESHOLD

    def test_cooling_off_threshold(self):
        """Score <= 3 should trigger cooling off alert."""
        from bot.telegram_bot import TelegramAlertBot

        # Test threshold logic
        assert TelegramAlertBot.COOLING_OFF_THRESHOLD == 3.0

        # Scores that should trigger alert
        assert 3.0 <= TelegramAlertBot.COOLING_OFF_THRESHOLD
        assert 2.0 <= TelegramAlertBot.COOLING_OFF_THRESHOLD
        assert 1.0 <= TelegramAlertBot.COOLING_OFF_THRESHOLD

        # Scores that should not trigger
        assert 3.1 > TelegramAlertBot.COOLING_OFF_THRESHOLD
        assert 5.0 > TelegramAlertBot.COOLING_OFF_THRESHOLD

    def test_alert_message_templates(self):
        """Alert message templates should contain required elements."""
        from bot.telegram_bot import (
            COOLING_OFF_TEMPLATE,
            DIVERGENCE_WARNING_TEMPLATE,
            STRONG_BUY_TEMPLATE,
        )

        # Strong buy template
        assert "🚀" in STRONG_BUY_TEMPLATE
        assert "Strong Buy Signal" in STRONG_BUY_TEMPLATE
        assert "{score" in STRONG_BUY_TEMPLATE
        assert "{phrases}" in STRONG_BUY_TEMPLATE

        # Cooling off template
        assert "❄️" in COOLING_OFF_TEMPLATE
        assert "Cooling Off" in COOLING_OFF_TEMPLATE
        assert "{score" in COOLING_OFF_TEMPLATE

        # Divergence warning template
        assert "⚠️" in DIVERGENCE_WARNING_TEMPLATE
        assert "Divergence" in DIVERGENCE_WARNING_TEMPLATE

    @patch("bot.telegram_bot.Bot")
    def test_bot_initialization(self, mock_bot):
        """Bot should initialize with correct parameters."""
        from bot.telegram_bot import TelegramAlertBot

        bot = TelegramAlertBot(
            token="test_token",
            channel_id="test_channel",
            coin_symbol="MEME",
        )

        assert bot.token == "test_token"
        assert bot.channel_id == "test_channel"
        assert bot.coin == "MEME"

    @patch("bot.telegram_bot.Bot")
    def test_divergence_formatting(self, mock_bot):
        """Divergence status should be formatted correctly."""
        from bot.telegram_bot import TelegramAlertBot

        bot = TelegramAlertBot(
            token="test_token",
            channel_id="test_channel",
        )

        assert "✅" in bot._format_divergence("aligned")
        assert "⚠️" in bot._format_divergence("bearish_divergence")
        assert "📈" in bot._format_divergence("bullish_divergence")


# =============================================================================
# TEST 3: RAG System Integration
# =============================================================================


class TestRAGIntegration:
    """Verify RAG system components work correctly."""

    def test_live_metrics_initialization(self):
        """Live metrics should initialize with default values."""
        from rag.live_metrics import LiveMetrics, reset_live_metrics

        reset_live_metrics()
        metrics = LiveMetrics()

        assert metrics.pulse_score == 5.0
        assert metrics.trending_phrases == []
        assert metrics.influencer_consensus == "neutral"
        assert metrics.divergence_status == "aligned"

    def test_live_metrics_update(self):
        """Live metrics should update correctly."""
        from rag.live_metrics import LiveMetrics

        metrics = LiveMetrics()
        metrics.update(
            pulse_score=8.5,
            trending_phrases=["to the moon", "bullish"],
            influencer_consensus="strongly bullish",
            divergence_status="aligned",
        )

        assert metrics.pulse_score == 8.5
        assert "to the moon" in metrics.trending_phrases
        assert metrics.influencer_consensus == "strongly bullish"

    def test_live_metrics_snapshot(self):
        """Snapshot should return all metrics."""
        from rag.live_metrics import LiveMetrics

        metrics = LiveMetrics()
        metrics.update(pulse_score=7.5)

        snapshot = metrics.get_snapshot()

        assert "pulse_score" in snapshot
        assert "trending_phrases" in snapshot
        assert "influencer_consensus" in snapshot
        assert "divergence_status" in snapshot
        assert snapshot["pulse_score"] == 7.5

    def test_live_metrics_rag_context(self):
        """RAG context should be formatted for prompts."""
        from rag.live_metrics import LiveMetrics

        metrics = LiveMetrics()
        metrics.update(
            pulse_score=8.0,
            trending_phrases=["moon", "pump", "lfg"],
        )

        context = metrics.get_rag_context()

        assert context["pulse_score"] == "8.0"
        assert "moon" in context["trending_phrases"]

    def test_crypto_rag_initialization(self):
        """CryptoRAG should initialize correctly."""
        from rag.crypto_rag import CryptoRAG

        rag = CryptoRAG()

        assert rag.top_k == 15  # Default value
        assert rag.llm_client is None
        assert rag.retriever is None

    def test_crypto_rag_prompt_building(self):
        """CryptoRAG should build context-enriched prompts."""
        from rag.crypto_rag import CryptoRAG
        from rag.live_metrics import reset_live_metrics

        reset_live_metrics()
        rag = CryptoRAG()

        # Update metrics
        rag.live_metrics.update(
            pulse_score=8.0,
            trending_phrases=["moon", "pump"],
        )

        # Build prompt
        prompt = rag.build_prompt(
            query="What's the sentiment?",
            retrieved_messages=[
                {"text": "To the moon!", "author_id": "user1", "sentiment": 0.8}
            ],
        )

        assert "8.0" in prompt
        assert "moon" in prompt
        assert "What's the sentiment?" in prompt

    def test_crypto_rag_fallback_response(self):
        """CryptoRAG should generate fallback when LLM unavailable."""
        from rag.crypto_rag import CryptoRAG
        from rag.live_metrics import reset_live_metrics

        reset_live_metrics()
        rag = CryptoRAG()
        rag.live_metrics.update(pulse_score=7.5)

        response = rag._generate_fallback_response()

        assert "7.5" in response
        assert "Pulse Score" in response


# =============================================================================
# TEST 4: Divergence Detection Integration
# =============================================================================


class TestDivergenceIntegration:
    """Verify divergence detection integrates with alerts."""

    def test_divergence_triggers_warning(self):
        """Divergence should be detected and formatted for alerts."""
        # Test bearish divergence
        divergence = detect_divergence(sentiment=0.7, price_delta_pct=-3.0)
        assert divergence == "bearish_divergence"

        # Test bullish divergence
        divergence = detect_divergence(sentiment=-0.7, price_delta_pct=3.0)
        assert divergence == "bullish_divergence"

        # Test aligned
        divergence = detect_divergence(sentiment=0.7, price_delta_pct=3.0)
        assert divergence == "aligned"

    def test_divergence_affects_pulse_score(self):
        """Bearish divergence should reduce pulse score."""
        calculator = PulseScoreCalculator()

        score_aligned = calculator.calculate(0.8, 25, 0.8, "aligned")
        score_divergent = calculator.calculate(0.8, 25, 0.8, "bearish_divergence")

        assert score_aligned > score_divergent
        assert score_aligned - score_divergent == 1.0  # Exactly 1 point penalty


# =============================================================================
# TEST 5: Price Data Integration
# =============================================================================


class TestPriceIntegration:
    """Verify price data flows through the system."""

    def test_price_generation(self):
        """Simulator should generate valid price data."""
        for phase in PHASE_ORDER:
            price = generate_single_price(coin_symbol="MEME", phase=phase)

            assert "coin_symbol" in price
            assert "price_usd" in price
            assert "timestamp" in price
            assert "volume_24h" in price
            assert price["price_usd"] > 0

    def test_price_correlation_with_phase(self):
        """Price should correlate with hype cycle phase."""
        from simulator.hype_simulator import PRICE_MULTIPLIERS

        # Peak should have highest multiplier
        assert PRICE_MULTIPLIERS["peak"] > PRICE_MULTIPLIERS["seed"]
        assert PRICE_MULTIPLIERS["peak"] > PRICE_MULTIPLIERS["growth"]
        assert PRICE_MULTIPLIERS["peak"] > PRICE_MULTIPLIERS["decline"]


# =============================================================================
# TEST 6: Alert Subscription Logic
# =============================================================================


class TestAlertSubscription:
    """Verify alert subscription logic works correctly."""

    def test_metrics_for_alert_format(self):
        """Metrics should be formatted correctly for alerts."""
        from rag.live_metrics import get_metrics_for_alert, reset_live_metrics

        reset_live_metrics()

        # Update some metrics
        from rag.live_metrics import (
            update_divergence_status,
            update_pulse_score,
            update_trending_phrases,
        )

        update_pulse_score(8.5)
        update_trending_phrases(["moon", "pump", "lfg"])
        update_divergence_status("aligned")

        metrics = get_metrics_for_alert()

        assert metrics["score"] == 8.5
        assert "moon" in metrics["phrases"]
        assert metrics["divergence"] == "aligned"

    def test_influencer_consensus_calculation(self):
        """Influencer consensus should be calculated correctly."""
        from rag.live_metrics import (
            get_live_metrics,
            reset_live_metrics,
            update_influencer_consensus,
        )

        reset_live_metrics()

        # Strongly bullish (>70% bullish)
        update_influencer_consensus(bullish_count=8, bearish_count=2)
        assert get_live_metrics().influencer_consensus == "strongly bullish"

        # Moderately bullish (50-70% bullish)
        reset_live_metrics()
        update_influencer_consensus(bullish_count=6, bearish_count=4)
        assert get_live_metrics().influencer_consensus == "moderately bullish"

        # Strongly bearish (<30% bullish)
        reset_live_metrics()
        update_influencer_consensus(bullish_count=2, bearish_count=8)
        assert get_live_metrics().influencer_consensus == "strongly bearish"


# =============================================================================
# TEST 7: Query Response Format
# =============================================================================


class TestQueryResponse:
    """Verify query responses are formatted correctly."""

    def test_rag_response_structure(self):
        """RAG response should have all required fields."""
        from rag.crypto_rag import RAGResponse

        response = RAGResponse(
            answer="Test answer",
            pulse_score=7.5,
            trending_phrases=["moon", "pump"],
            sources=[{"text": "source1"}],
            relevance_scores=[0.9],
            influencer_consensus="bullish",
            divergence_status="aligned",
        )

        assert response.answer == "Test answer"
        assert response.pulse_score == 7.5
        assert len(response.trending_phrases) == 2
        assert len(response.sources) == 1
        assert len(response.relevance_scores) == 1

    def test_rag_response_to_dict(self):
        """RAG response should serialize to dict."""
        from rag.crypto_rag import RAGResponse

        response = RAGResponse(
            answer="Test",
            pulse_score=5.0,
            trending_phrases=[],
            sources=[],
            relevance_scores=[],
        )

        data = response.to_dict()

        assert isinstance(data, dict)
        assert "answer" in data
        assert "pulse_score" in data
        assert "trending_phrases" in data

    def test_query_response_template(self):
        """Query response template should contain required elements."""
        from bot.telegram_bot import QUERY_RESPONSE_TEMPLATE

        assert "{score" in QUERY_RESPONSE_TEMPLATE
        assert "{phrases}" in QUERY_RESPONSE_TEMPLATE
        assert "{consensus}" in QUERY_RESPONSE_TEMPLATE
        assert "{divergence}" in QUERY_RESPONSE_TEMPLATE
        assert "{answer}" in QUERY_RESPONSE_TEMPLATE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
