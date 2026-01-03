"""
Tests for Price Data Integration (Task 8).

Tests the following components:
1. PriceFetcher with caching and rate limiting
2. Price delta calculation
3. Divergence detection

Requirements: 4.1, 4.2, 4.4, 4.5, 4.7
"""

import sys
from unittest.mock import MagicMock

# Mock pathway module before importing modules that depend on it
mock_pw = MagicMock()
mock_pw.Schema = type("Schema", (), {})
mock_pw.Duration = MagicMock()
mock_pw.DateTimeUtc = MagicMock()
sys.modules["pathway"] = mock_pw

import time

import pytest

from connectors.price_fetcher import (
    PriceCache,
    PriceFetcher,
    RateLimiter,
    fetch_price,
)
from schemas import validate_price_payload
from transforms.divergence import (
    DivergenceAnalyzer,
    analyze_divergence,
    detect_divergence,
    get_divergence_description,
    is_aligned,
    is_bearish_divergence,
    is_bullish_divergence,
)
from transforms.price_pipeline import (
    calculate_price_delta_pct,
    classify_price_movement,
)

# =============================================================================
# TEST 1: Rate Limiter
# =============================================================================


class TestRateLimiter:
    """Test the rate limiter for API calls."""

    def test_acquire_within_limit(self):
        """Should allow calls within the rate limit."""
        limiter = RateLimiter(max_calls=5, period=60)

        # Should allow 5 calls
        for _ in range(5):
            assert limiter.acquire() is True

    def test_acquire_exceeds_limit(self):
        """Should block calls that exceed the rate limit."""
        limiter = RateLimiter(max_calls=3, period=60)

        # Use up the limit
        for _ in range(3):
            limiter.acquire()

        # Next call should be blocked
        assert limiter.acquire() is False

    def test_get_remaining_calls(self):
        """Should correctly report remaining calls."""
        limiter = RateLimiter(max_calls=5, period=60)

        assert limiter.get_remaining_calls() == 5
        limiter.acquire()
        assert limiter.get_remaining_calls() == 4
        limiter.acquire()
        assert limiter.get_remaining_calls() == 3


# =============================================================================
# TEST 2: Price Cache
# =============================================================================


class TestPriceCache:
    """Test the price cache with TTL."""

    def test_cache_set_and_get(self):
        """Should store and retrieve cached values."""
        cache = PriceCache(default_ttl=60)

        data = {"price_usd": 100.0, "coin_symbol": "TEST"}
        cache.set("test_key", data)

        result = cache.get("test_key")
        assert result == data

    def test_cache_miss(self):
        """Should return None for missing keys."""
        cache = PriceCache(default_ttl=60)

        result = cache.get("nonexistent_key")
        assert result is None

    def test_cache_expiry(self):
        """Should return None for expired entries."""
        cache = PriceCache(default_ttl=1)  # 1 second TTL

        data = {"price_usd": 100.0}
        cache.set("test_key", data, ttl=1)

        # Should be available immediately
        assert cache.get("test_key") == data

        # Wait for expiry
        time.sleep(1.1)

        # Should be expired
        assert cache.get("test_key") is None

    def test_cache_clear(self):
        """Should clear all cached entries."""
        cache = PriceCache(default_ttl=60)

        cache.set("key1", {"data": 1})
        cache.set("key2", {"data": 2})

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None


# =============================================================================
# TEST 3: Price Fetcher
# =============================================================================


class TestPriceFetcher:
    """Test the price fetcher with simulation mode."""

    def test_simulated_price_has_required_fields(self):
        """Simulated price should have all required fields."""
        fetcher = PriceFetcher(
            coin_id="memecoin",
            use_simulation=True,
            simulation_base_price=0.001,
        )

        price = fetcher.get_price()

        assert "coin_symbol" in price
        assert "price_usd" in price
        assert "timestamp" in price
        assert "volume_24h" in price

    def test_simulated_price_validates(self):
        """Simulated price should pass schema validation."""
        fetcher = PriceFetcher(
            coin_id="memecoin",
            use_simulation=True,
        )

        price = fetcher.get_price()
        result = validate_price_payload(price)

        assert result.is_valid, f"Validation failed: {result.errors}"

    def test_simulated_price_positive(self):
        """Simulated price should be positive."""
        fetcher = PriceFetcher(
            coin_id="memecoin",
            use_simulation=True,
            simulation_base_price=0.001,
        )

        price = fetcher.get_price()

        assert price["price_usd"] > 0

    def test_from_symbol_creates_fetcher(self):
        """Should create fetcher from coin symbol."""
        fetcher = PriceFetcher.from_symbol("SOL", use_simulation=True)

        assert fetcher.coin_id == "solana"

    def test_caching_returns_same_value(self):
        """Should return cached value on subsequent calls."""
        fetcher = PriceFetcher(
            coin_id="memecoin",
            use_simulation=True,
            cache_ttl=60,
        )

        price1 = fetcher.get_price()
        price2 = fetcher.get_price()

        # Should be the same cached value
        assert price1 == price2

    def test_fetch_price_convenience_function(self):
        """The convenience function should work."""
        # This will use simulation mode by default in test environment
        price = fetch_price("memecoin")

        assert "price_usd" in price
        assert "coin_symbol" in price


# =============================================================================
# TEST 4: Price Delta Calculation
# =============================================================================


class TestPriceDeltaCalculation:
    """Test price delta percentage calculation."""

    def test_price_increase(self):
        """Should calculate positive delta for price increase."""
        delta = calculate_price_delta_pct(100.0, 105.0)
        assert delta == 5.0

    def test_price_decrease(self):
        """Should calculate negative delta for price decrease."""
        delta = calculate_price_delta_pct(100.0, 95.0)
        assert delta == -5.0

    def test_no_change(self):
        """Should return 0 for no price change."""
        delta = calculate_price_delta_pct(100.0, 100.0)
        assert delta == 0.0

    def test_zero_start_price(self):
        """Should return 0 for zero start price (avoid division by zero)."""
        delta = calculate_price_delta_pct(0.0, 100.0)
        assert delta == 0.0

    def test_negative_start_price(self):
        """Should return 0 for negative start price."""
        delta = calculate_price_delta_pct(-100.0, 100.0)
        assert delta == 0.0

    def test_large_increase(self):
        """Should handle large percentage increases."""
        delta = calculate_price_delta_pct(100.0, 200.0)
        assert delta == 100.0  # 100% increase

    def test_large_decrease(self):
        """Should handle large percentage decreases."""
        delta = calculate_price_delta_pct(100.0, 50.0)
        assert delta == -50.0  # 50% decrease


# =============================================================================
# TEST 5: Price Movement Classification
# =============================================================================


class TestPriceMovementClassification:
    """Test price movement classification."""

    def test_strong_up(self):
        """Should classify >5% as strong_up."""
        assert classify_price_movement(6.0) == "strong_up"
        assert classify_price_movement(10.0) == "strong_up"

    def test_up(self):
        """Should classify 2-5% as up."""
        assert classify_price_movement(3.0) == "up"
        assert classify_price_movement(5.0) == "up"

    def test_strong_down(self):
        """Should classify <-5% as strong_down."""
        assert classify_price_movement(-6.0) == "strong_down"
        assert classify_price_movement(-10.0) == "strong_down"

    def test_down(self):
        """Should classify -2 to -5% as down."""
        assert classify_price_movement(-3.0) == "down"
        assert classify_price_movement(-5.0) == "down"

    def test_stable(self):
        """Should classify -2 to 2% as stable."""
        assert classify_price_movement(0.0) == "stable"
        assert classify_price_movement(1.5) == "stable"
        assert classify_price_movement(-1.5) == "stable"


# =============================================================================
# TEST 6: Divergence Detection
# =============================================================================


class TestDivergenceDetection:
    """Test sentiment-price divergence detection."""

    def test_bearish_divergence(self):
        """Should detect bearish divergence: high sentiment + falling price."""
        # Requirement 4.4: sentiment > 0.5 AND price_delta < -2%
        result = detect_divergence(0.7, -3.0)
        assert result == "bearish_divergence"

    def test_bullish_divergence(self):
        """Should detect bullish divergence: low sentiment + rising price."""
        # Requirement 4.5: sentiment < -0.5 AND price_delta > 2%
        result = detect_divergence(-0.7, 3.0)
        assert result == "bullish_divergence"

    def test_aligned_positive(self):
        """Should detect aligned when both sentiment and price are positive."""
        result = detect_divergence(0.7, 3.0)
        assert result == "aligned"

    def test_aligned_negative(self):
        """Should detect aligned when both sentiment and price are negative."""
        result = detect_divergence(-0.7, -3.0)
        assert result == "aligned"

    def test_aligned_neutral(self):
        """Should detect aligned for neutral values."""
        result = detect_divergence(0.3, 1.0)
        assert result == "aligned"

    def test_boundary_bearish_divergence(self):
        """Should detect bearish divergence at exact boundaries."""
        # Just above sentiment threshold, just below price threshold
        result = detect_divergence(0.51, -2.01)
        assert result == "bearish_divergence"

    def test_boundary_not_bearish(self):
        """Should not detect bearish divergence at boundary."""
        # At sentiment threshold, at price threshold (not exceeding)
        result = detect_divergence(0.5, -2.0)
        assert result == "aligned"

    def test_boundary_bullish_divergence(self):
        """Should detect bullish divergence at exact boundaries."""
        # Just below sentiment threshold, just above price threshold
        result = detect_divergence(-0.51, 2.01)
        assert result == "bullish_divergence"


# =============================================================================
# TEST 7: Divergence Helper Functions
# =============================================================================


class TestDivergenceHelpers:
    """Test divergence helper functions."""

    def test_is_bearish_divergence(self):
        """Should correctly identify bearish divergence."""
        assert is_bearish_divergence(0.7, -3.0) is True
        assert is_bearish_divergence(0.3, -3.0) is False
        assert is_bearish_divergence(0.7, 3.0) is False

    def test_is_bullish_divergence(self):
        """Should correctly identify bullish divergence."""
        assert is_bullish_divergence(-0.7, 3.0) is True
        assert is_bullish_divergence(-0.3, 3.0) is False
        assert is_bullish_divergence(-0.7, -3.0) is False

    def test_is_aligned(self):
        """Should correctly identify aligned state."""
        assert is_aligned(0.7, 3.0) is True
        assert is_aligned(-0.7, -3.0) is True
        assert is_aligned(0.7, -3.0) is False

    def test_get_divergence_description(self):
        """Should return human-readable descriptions."""
        desc = get_divergence_description("bearish_divergence")
        assert "Bearish" in desc
        assert "warning" in desc.lower() or "⚠️" in desc

        desc = get_divergence_description("bullish_divergence")
        assert "Bullish" in desc

        desc = get_divergence_description("aligned")
        assert "Aligned" in desc


# =============================================================================
# TEST 8: Divergence Analyzer
# =============================================================================


class TestDivergenceAnalyzer:
    """Test the DivergenceAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DivergenceAnalyzer()

    def test_analyze_returns_dict(self):
        """Should return a dictionary with all expected keys."""
        result = self.analyzer.analyze(0.7, -3.0)

        assert "divergence_type" in result
        assert "is_divergent" in result
        assert "description" in result
        assert "emoji" in result
        assert "sentiment" in result
        assert "price_delta_pct" in result

    def test_analyze_bearish_divergence(self):
        """Should correctly analyze bearish divergence."""
        result = self.analyzer.analyze(0.7, -3.0)

        assert result["divergence_type"] == "bearish_divergence"
        assert result["is_divergent"] is True
        assert result["emoji"] == "⚠️"

    def test_analyze_bullish_divergence(self):
        """Should correctly analyze bullish divergence."""
        result = self.analyzer.analyze(-0.7, 3.0)

        assert result["divergence_type"] == "bullish_divergence"
        assert result["is_divergent"] is True
        assert result["emoji"] == "📈"

    def test_analyze_aligned(self):
        """Should correctly analyze aligned state."""
        result = self.analyzer.analyze(0.7, 3.0)

        assert result["divergence_type"] == "aligned"
        assert result["is_divergent"] is False
        assert result["emoji"] == "✅"

    def test_should_alert_on_divergence(self):
        """Should return True for divergence alerts."""
        assert self.analyzer.should_alert(0.7, -3.0) is True
        assert self.analyzer.should_alert(-0.7, 3.0) is True

    def test_should_not_alert_when_aligned(self):
        """Should return False when aligned."""
        assert self.analyzer.should_alert(0.7, 3.0) is False
        assert self.analyzer.should_alert(-0.7, -3.0) is False

    def test_custom_thresholds(self):
        """Should respect custom thresholds."""
        # Create analyzer with stricter thresholds
        strict_analyzer = DivergenceAnalyzer(
            sentiment_high=0.8,
            sentiment_low=-0.8,
            price_falling=-5.0,
            price_rising=5.0,
        )

        # This would be bearish divergence with default thresholds
        # but not with stricter thresholds
        result = strict_analyzer.analyze(0.7, -3.0)
        assert result["divergence_type"] == "aligned"

    def test_analyze_divergence_convenience_function(self):
        """The convenience function should work."""
        result = analyze_divergence(0.7, -3.0)

        assert result["divergence_type"] == "bearish_divergence"


# =============================================================================
# TEST 9: Integration - Price + Divergence
# =============================================================================


class TestPriceDivergenceIntegration:
    """Test integration between price calculation and divergence detection."""

    def test_price_delta_to_divergence(self):
        """Should correctly flow from price delta to divergence detection."""
        # Simulate price data
        start_price = 100.0
        end_price = 97.0  # 3% drop

        # Calculate delta
        delta = calculate_price_delta_pct(start_price, end_price)
        assert delta == -3.0

        # Detect divergence with high sentiment
        sentiment = 0.7
        divergence = detect_divergence(sentiment, delta)

        assert divergence == "bearish_divergence"

    def test_full_analysis_flow(self):
        """Should correctly analyze full price + sentiment flow."""
        analyzer = DivergenceAnalyzer()

        # Scenario: Price rising but sentiment negative (bullish divergence)
        start_price = 100.0
        end_price = 105.0  # 5% rise
        sentiment = -0.6

        delta = calculate_price_delta_pct(start_price, end_price)
        result = analyzer.analyze(sentiment, delta)

        assert result["divergence_type"] == "bullish_divergence"
        assert result["is_divergent"] is True
        assert result["price_delta_pct"] == 5.0
        assert result["sentiment"] == -0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
