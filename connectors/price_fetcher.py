"""
Price Data Fetcher for the Crypto Narrative Pulse Tracker.

Provides price data fetching from CoinGecko API with caching and rate limiting,
or uses simulated data from the Hype Simulator for demos.

Requirements: 4.1, 4.7
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any

import requests

from schemas import create_price_dict

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# CoinGecko API configuration
COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_PRO_API_BASE = "https://pro-api.coingecko.com/api/v3"

# Rate limiting: 50 calls/min for free tier (Requirement 4.7)
DEFAULT_RATE_LIMIT_CALLS = 50
DEFAULT_RATE_LIMIT_PERIOD = 60  # seconds

# Default cache TTL: 60 seconds
DEFAULT_CACHE_TTL = 60


# =============================================================================
# RATE LIMITER
# =============================================================================


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Implements rate limiting to comply with CoinGecko's API limits
    (50 calls/min for free tier).

    Requirements: 4.7
    """

    def __init__(
        self,
        max_calls: int = DEFAULT_RATE_LIMIT_CALLS,
        period: int = DEFAULT_RATE_LIMIT_PERIOD,
    ):
        """
        Initialize the rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: list[float] = []
        self._lock = Lock()

    def acquire(self) -> bool:
        """
        Attempt to acquire a rate limit token.

        Returns:
            True if the call is allowed, False if rate limited
        """
        with self._lock:
            now = time.time()
            # Remove calls outside the current window
            self.calls = [t for t in self.calls if now - t < self.period]

            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            return False

    def wait_and_acquire(self, timeout: float = 10.0) -> bool:
        """
        Wait until a rate limit token is available, then acquire it.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if acquired within timeout, False otherwise
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.acquire():
                return True
            time.sleep(0.1)
        return False

    def get_remaining_calls(self) -> int:
        """Get the number of remaining calls in the current window."""
        with self._lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.period]
            return self.max_calls - len(self.calls)


# =============================================================================
# CACHE
# =============================================================================


@dataclass
class CacheEntry:
    """A cached price entry with TTL."""

    data: dict[str, Any]
    timestamp: float
    ttl: int


class PriceCache:
    """
    Simple TTL-based cache for price data.

    Caches price data to reduce API calls and handle rate limits.

    Requirements: 4.7
    """

    def __init__(self, default_ttl: int = DEFAULT_CACHE_TTL):
        """
        Initialize the cache.

        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.default_ttl = default_ttl
        self._cache: dict[str, CacheEntry] = {}
        self._lock = Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        """
        Get a cached value if it exists and hasn't expired.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            if time.time() - entry.timestamp > entry.ttl:
                # Entry expired
                del self._cache[key]
                return None

            return entry.data

    def set(self, key: str, data: dict[str, Any], ttl: int | None = None) -> None:
        """
        Set a cached value.

        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        with self._lock:
            self._cache[key] = CacheEntry(
                data=data,
                timestamp=time.time(),
                ttl=ttl if ttl is not None else self.default_ttl,
            )

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def remove(self, key: str) -> None:
        """Remove a specific cached entry."""
        with self._lock:
            self._cache.pop(key, None)


# =============================================================================
# PRICE FETCHER
# =============================================================================


class PriceFetcher:
    """
    Fetches cryptocurrency price data from CoinGecko API with caching.

    Supports both live API data and simulated data for demos.
    Implements caching with TTL for rate limit handling.

    Requirements: 4.1, 4.7

    Example:
        >>> fetcher = PriceFetcher(coin_id="solana", use_simulation=False)
        >>> price = fetcher.get_price()
        >>> print(f"SOL: ${price['price_usd']}")
    """

    # Mapping of common symbols to CoinGecko IDs
    SYMBOL_TO_ID = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "MEME": "memecoin",
        "DOGE": "dogecoin",
        "SHIB": "shiba-inu",
        "PEPE": "pepe",
        "WIF": "dogwifcoin",
        "BONK": "bonk",
    }

    def __init__(
        self,
        coin_id: str = "memecoin",
        api_key: str | None = None,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        use_simulation: bool = False,
        simulation_base_price: float = 0.001,
    ):
        """
        Initialize the price fetcher.

        Args:
            coin_id: CoinGecko coin ID (e.g., "solana", "bitcoin")
            api_key: Optional CoinGecko Pro API key
            cache_ttl: Cache time-to-live in seconds (default: 60)
            use_simulation: If True, use simulated data instead of API
            simulation_base_price: Base price for simulation mode
        """
        self.coin_id = coin_id.lower()
        self.api_key = api_key or os.environ.get("COINGECKO_API_KEY")
        self.use_simulation = use_simulation
        self.simulation_base_price = simulation_base_price

        # Initialize cache and rate limiter
        self.cache = PriceCache(default_ttl=cache_ttl)
        self.rate_limiter = RateLimiter()

        # Determine API base URL
        self.api_base = COINGECKO_PRO_API_BASE if self.api_key else COINGECKO_API_BASE

        logger.info(
            f"PriceFetcher initialized: coin_id={self.coin_id}, "
            f"simulation={self.use_simulation}, cache_ttl={cache_ttl}s"
        )

    @classmethod
    def from_symbol(
        cls,
        symbol: str,
        api_key: str | None = None,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        use_simulation: bool = False,
    ) -> "PriceFetcher":
        """
        Create a PriceFetcher from a coin symbol.

        Args:
            symbol: Coin symbol (e.g., "SOL", "BTC", "MEME")
            api_key: Optional CoinGecko Pro API key
            cache_ttl: Cache time-to-live in seconds
            use_simulation: If True, use simulated data

        Returns:
            PriceFetcher instance
        """
        coin_id = cls.SYMBOL_TO_ID.get(symbol.upper(), symbol.lower())
        return cls(
            coin_id=coin_id,
            api_key=api_key,
            cache_ttl=cache_ttl,
            use_simulation=use_simulation,
        )

    def get_price(self) -> dict[str, Any]:
        """
        Get the current price for the configured coin.

        Returns cached data if available, otherwise fetches from API
        or generates simulated data.

        Returns:
            Price dictionary conforming to PriceSchema

        Requirements: 4.1, 4.7
        """
        cache_key = f"price_{self.coin_id}"

        # Check cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {self.coin_id}")
            return cached

        # Fetch new data
        if self.use_simulation:
            price_data = self._get_simulated_price()
        else:
            price_data = self._fetch_from_api()

        # Cache the result
        if price_data:
            self.cache.set(cache_key, price_data)

        return price_data

    def _fetch_from_api(self) -> dict[str, Any]:
        """
        Fetch price data from CoinGecko API.

        Implements rate limiting and error handling.

        Returns:
            Price dictionary or fallback to cached/simulated data on error
        """
        # Check rate limit
        if not self.rate_limiter.wait_and_acquire(timeout=5.0):
            logger.warning(
                f"Rate limited, using cached or simulated data for {self.coin_id}"
            )
            # Try to return stale cache or simulation
            return self._get_simulated_price()

        try:
            url = f"{self.api_base}/simple/price"
            params = {
                "ids": self.coin_id,
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
            }
            headers = {}
            if self.api_key:
                headers["x-cg-pro-api-key"] = self.api_key

            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            if self.coin_id not in data:
                logger.warning(f"Coin {self.coin_id} not found in API response")
                return self._get_simulated_price()

            coin_data = data[self.coin_id]

            # Get symbol from coin_id (uppercase first part or full id)
            symbol = self.coin_id.upper()
            for sym, cid in self.SYMBOL_TO_ID.items():
                if cid == self.coin_id:
                    symbol = sym
                    break

            price_dict = create_price_dict(
                coin_symbol=symbol,
                price_usd=coin_data.get("usd", 0.0),
                volume_24h=coin_data.get("usd_24h_vol", 0.0),
            )

            logger.debug(
                f"Fetched price for {self.coin_id}: ${price_dict['price_usd']}"
            )
            return price_dict

        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed for {self.coin_id}: {e}")
            return self._get_simulated_price()
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to parse API response for {self.coin_id}: {e}")
            return self._get_simulated_price()

    def _get_simulated_price(self) -> dict[str, Any]:
        """
        Generate simulated price data.

        Used for demos or as fallback when API is unavailable.

        Returns:
            Simulated price dictionary
        """
        import random

        # Add some random variation (±5%)
        variation = random.uniform(-0.05, 0.05)
        price = self.simulation_base_price * (1 + variation)

        # Simulated volume
        volume = random.uniform(100000, 1000000)

        # Get symbol
        symbol = self.coin_id.upper()
        for sym, cid in self.SYMBOL_TO_ID.items():
            if cid == self.coin_id:
                symbol = sym
                break

        return create_price_dict(
            coin_symbol=symbol,
            price_usd=round(price, 8),
            volume_24h=round(volume, 2),
        )

    def get_price_history(self, days: int = 1) -> list[dict[str, Any]]:
        """
        Get historical price data (API only, not cached).

        Args:
            days: Number of days of history (1, 7, 14, 30, 90, 180, 365, max)

        Returns:
            List of price dictionaries with timestamps
        """
        if self.use_simulation:
            return self._get_simulated_history(days)

        if not self.rate_limiter.wait_and_acquire(timeout=5.0):
            logger.warning("Rate limited for history request")
            return self._get_simulated_history(days)

        try:
            url = f"{self.api_base}/coins/{self.coin_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": str(days),
            }
            headers = {}
            if self.api_key:
                headers["x-cg-pro-api-key"] = self.api_key

            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])

            # Get symbol
            symbol = self.coin_id.upper()
            for sym, cid in self.SYMBOL_TO_ID.items():
                if cid == self.coin_id:
                    symbol = sym
                    break

            result = []
            for i, (timestamp_ms, price) in enumerate(prices):
                volume = volumes[i][1] if i < len(volumes) else 0.0
                timestamp = (
                    datetime.utcfromtimestamp(timestamp_ms / 1000).isoformat() + "Z"
                )

                result.append(
                    create_price_dict(
                        coin_symbol=symbol,
                        price_usd=price,
                        timestamp=timestamp,
                        volume_24h=volume,
                    )
                )

            return result

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch history for {self.coin_id}: {e}")
            return self._get_simulated_history(days)

    def _get_simulated_history(self, days: int) -> list[dict[str, Any]]:
        """Generate simulated price history."""
        import random
        from datetime import timedelta

        # Get symbol
        symbol = self.coin_id.upper()
        for sym, cid in self.SYMBOL_TO_ID.items():
            if cid == self.coin_id:
                symbol = sym
                break

        result = []
        now = datetime.utcnow()
        points = min(days * 24, 168)  # Max 168 points (hourly for 7 days)

        price = self.simulation_base_price
        for i in range(points):
            # Random walk
            price *= 1 + random.uniform(-0.02, 0.02)
            timestamp = (now - timedelta(hours=points - i)).isoformat() + "Z"

            result.append(
                create_price_dict(
                    coin_symbol=symbol,
                    price_usd=round(price, 8),
                    timestamp=timestamp,
                    volume_24h=random.uniform(100000, 1000000),
                )
            )

        return result


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================

_default_fetcher: PriceFetcher | None = None


def get_price_fetcher(
    coin_id: str = "memecoin",
    use_simulation: bool | None = None,
) -> PriceFetcher:
    """
    Get or create a price fetcher instance.

    Args:
        coin_id: CoinGecko coin ID
        use_simulation: Override simulation mode (uses env var if None)

    Returns:
        PriceFetcher instance
    """
    global _default_fetcher

    if use_simulation is None:
        use_simulation = (
            os.environ.get("USE_PRICE_SIMULATION", "false").lower() == "true"
        )

    if _default_fetcher is None or _default_fetcher.coin_id != coin_id:
        _default_fetcher = PriceFetcher(
            coin_id=coin_id,
            use_simulation=use_simulation,
        )

    return _default_fetcher


def fetch_price(coin_id: str = "memecoin") -> dict[str, Any]:
    """
    Convenience function to fetch price for a coin.

    Args:
        coin_id: CoinGecko coin ID

    Returns:
        Price dictionary conforming to PriceSchema
    """
    fetcher = get_price_fetcher(coin_id)
    return fetcher.get_price()
