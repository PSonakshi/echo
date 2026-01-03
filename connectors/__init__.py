"""
Connectors module for the Crypto Narrative Pulse Tracker.

Provides Pathway connectors for ingesting data from various sources:
- HTTP webhooks (messages from simulator and external sources)
- Price data fetchers (CoinGecko API)
- Discord connector (stretch goal)
"""

from connectors.price_fetcher import (
    PriceCache,
    PriceFetcher,
    RateLimiter,
    fetch_price,
    get_price_fetcher,
)
from connectors.webhook import (
    create_filtered_webhook_connector,
    create_tag_filter,
    create_webhook_connector,
    filter_by_tags,
    filter_by_tags_any,
    matches_tag_pattern,
)

__all__ = [
    # Webhook connector
    "create_webhook_connector",
    "create_filtered_webhook_connector",
    "filter_by_tags",
    "filter_by_tags_any",
    "matches_tag_pattern",
    "create_tag_filter",
    # Price fetcher
    "PriceFetcher",
    "PriceCache",
    "RateLimiter",
    "fetch_price",
    "get_price_fetcher",
]
