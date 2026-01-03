"""
Core data schemas for the Crypto Narrative Pulse Tracker.

Defines Pathway schemas for message and price data ingestion,
along with validation utilities for ensuring data integrity.

Requirements: 1.1, 1.3, 4.1
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import pathway as pw

# =============================================================================
# ENUMS
# =============================================================================


class SourcePlatform(Enum):
    """Supported data source platforms."""

    SIMULATOR = "simulator"
    TWITTER = "twitter"
    DISCORD = "discord"
    WEBHOOK = "webhook"


class DivergenceType(Enum):
    """Price-sentiment divergence types."""

    ALIGNED = "aligned"
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"


# =============================================================================
# PATHWAY SCHEMAS
# =============================================================================


class MessageSchema(pw.Schema):
    """
    Unified schema for social media messages from all sources.

    Used by the Pipeline to parse and validate incoming webhook payloads.
    All data sources (simulator, Discord, Twitter) normalize to this schema.

    Requirements: 1.1
    """

    message_id: str  # Unique identifier for the message
    text: str  # Message content
    author_id: str  # Author identifier
    author_followers: int  # Follower/member count
    timestamp: str  # ISO format timestamp (e.g., "2024-01-15T10:30:00Z")
    tags: list  # Hashtags, cashtags, channel tags (e.g., ["#Solana", "$MEME"])
    engagement_count: int  # Likes, reactions, retweets
    source_platform: str  # Platform identifier: "simulator", "twitter", "discord"


class PriceSchema(pw.Schema):
    """
    Schema for cryptocurrency price data.

    Used by the Pipeline to ingest price data from CoinGecko API
    or the Hype Simulator for correlation analysis.

    Requirements: 4.1
    """

    coin_symbol: str  # Cryptocurrency symbol (e.g., "MEME", "SOL")
    price_usd: float  # Current price in USD
    timestamp: str  # ISO format timestamp
    volume_24h: float  # 24-hour trading volume in USD


# =============================================================================
# VALIDATION
# =============================================================================

# Required fields for MessageSchema validation
MESSAGE_REQUIRED_FIELDS = ["message_id", "text", "author_id", "timestamp"]

# Required fields for PriceSchema validation
PRICE_REQUIRED_FIELDS = ["coin_symbol", "price_usd", "timestamp"]


@dataclass
class ValidationError:
    """Structured error for validation failures."""

    field: str
    error_type: str
    message: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "field": self.field,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass
class ValidationResult:
    """Result of schema validation."""

    is_valid: bool
    errors: list[ValidationError]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {"is_valid": self.is_valid, "errors": [e.to_dict() for e in self.errors]}


def validate_message_payload(payload: dict[str, Any]) -> ValidationResult:
    """
    Validate a message payload against MessageSchema requirements.

    Checks for:
    - Required fields presence
    - Correct field types
    - Valid values (non-empty strings, non-negative integers)

    Args:
        payload: Dictionary containing message data

    Returns:
        ValidationResult with is_valid flag and list of errors

    Requirements: 1.3
    """
    errors: list[ValidationError] = []

    # Check required fields
    for field in MESSAGE_REQUIRED_FIELDS:
        if field not in payload:
            errors.append(
                ValidationError(
                    field=field,
                    error_type="missing_field",
                    message=f"Required field '{field}' is missing",
                )
            )

    # If required fields are missing, return early
    if errors:
        return ValidationResult(is_valid=False, errors=errors)

    # Type validation for string fields
    string_fields = ["message_id", "text", "author_id", "timestamp", "source_platform"]
    for field in string_fields:
        if field in payload and payload[field] is not None:
            if not isinstance(payload[field], str):
                errors.append(
                    ValidationError(
                        field=field,
                        error_type="invalid_type",
                        message=f"Field '{field}' must be a string, got {type(payload[field]).__name__}",
                    )
                )
            elif field in MESSAGE_REQUIRED_FIELDS and not payload[field].strip():
                errors.append(
                    ValidationError(
                        field=field,
                        error_type="empty_value",
                        message=f"Required field '{field}' cannot be empty",
                    )
                )

    # Type validation for integer fields
    int_fields = ["author_followers", "engagement_count"]
    for field in int_fields:
        if field in payload and payload[field] is not None:
            if not isinstance(payload[field], int):
                errors.append(
                    ValidationError(
                        field=field,
                        error_type="invalid_type",
                        message=f"Field '{field}' must be an integer, got {type(payload[field]).__name__}",
                    )
                )
            elif payload[field] < 0:
                errors.append(
                    ValidationError(
                        field=field,
                        error_type="invalid_value",
                        message=f"Field '{field}' must be non-negative",
                    )
                )

    # Type validation for list field
    if "tags" in payload and payload["tags"] is not None:
        if not isinstance(payload["tags"], list):
            errors.append(
                ValidationError(
                    field="tags",
                    error_type="invalid_type",
                    message=f"Field 'tags' must be a list, got {type(payload['tags']).__name__}",
                )
            )

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def validate_price_payload(payload: dict[str, Any]) -> ValidationResult:
    """
    Validate a price payload against PriceSchema requirements.

    Checks for:
    - Required fields presence
    - Correct field types
    - Valid values (non-empty strings, positive numbers)

    Args:
        payload: Dictionary containing price data

    Returns:
        ValidationResult with is_valid flag and list of errors

    Requirements: 1.3
    """
    errors: list[ValidationError] = []

    # Check required fields
    for field in PRICE_REQUIRED_FIELDS:
        if field not in payload:
            errors.append(
                ValidationError(
                    field=field,
                    error_type="missing_field",
                    message=f"Required field '{field}' is missing",
                )
            )

    # If required fields are missing, return early
    if errors:
        return ValidationResult(is_valid=False, errors=errors)

    # Type validation for string fields
    string_fields = ["coin_symbol", "timestamp"]
    for field in string_fields:
        if field in payload and payload[field] is not None:
            if not isinstance(payload[field], str):
                errors.append(
                    ValidationError(
                        field=field,
                        error_type="invalid_type",
                        message=f"Field '{field}' must be a string, got {type(payload[field]).__name__}",
                    )
                )
            elif not payload[field].strip():
                errors.append(
                    ValidationError(
                        field=field,
                        error_type="empty_value",
                        message=f"Required field '{field}' cannot be empty",
                    )
                )

    # Type validation for numeric fields
    float_fields = ["price_usd", "volume_24h"]
    for field in float_fields:
        if field in payload and payload[field] is not None:
            if not isinstance(payload[field], (int, float)):
                errors.append(
                    ValidationError(
                        field=field,
                        error_type="invalid_type",
                        message=f"Field '{field}' must be a number, got {type(payload[field]).__name__}",
                    )
                )
            elif field == "price_usd" and payload[field] <= 0:
                errors.append(
                    ValidationError(
                        field=field,
                        error_type="invalid_value",
                        message=f"Field '{field}' must be positive",
                    )
                )
            elif field == "volume_24h" and payload[field] < 0:
                errors.append(
                    ValidationError(
                        field=field,
                        error_type="invalid_value",
                        message=f"Field '{field}' must be non-negative",
                    )
                )

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_message_dict(
    message_id: str,
    text: str,
    author_id: str,
    author_followers: int = 0,
    timestamp: str | None = None,
    tags: list[str] | None = None,
    engagement_count: int = 0,
    source_platform: str = "webhook",
) -> dict[str, Any]:
    """
    Create a message dictionary conforming to MessageSchema.

    Useful for creating test data or transforming external data.

    Args:
        message_id: Unique identifier
        text: Message content
        author_id: Author identifier
        author_followers: Follower count (default: 0)
        timestamp: ISO timestamp (default: current time)
        tags: List of tags (default: empty list)
        engagement_count: Engagement count (default: 0)
        source_platform: Source platform (default: "webhook")

    Returns:
        Dictionary conforming to MessageSchema
    """
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat() + "Z"
    if tags is None:
        tags = []

    return {
        "message_id": message_id,
        "text": text,
        "author_id": author_id,
        "author_followers": author_followers,
        "timestamp": timestamp,
        "tags": tags,
        "engagement_count": engagement_count,
        "source_platform": source_platform,
    }


def create_price_dict(
    coin_symbol: str,
    price_usd: float,
    timestamp: str | None = None,
    volume_24h: float = 0.0,
) -> dict[str, Any]:
    """
    Create a price dictionary conforming to PriceSchema.

    Useful for creating test data or transforming external data.

    Args:
        coin_symbol: Cryptocurrency symbol
        price_usd: Price in USD
        timestamp: ISO timestamp (default: current time)
        volume_24h: 24h trading volume (default: 0.0)

    Returns:
        Dictionary conforming to PriceSchema
    """
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat() + "Z"

    return {
        "coin_symbol": coin_symbol,
        "price_usd": price_usd,
        "timestamp": timestamp,
        "volume_24h": volume_24h,
    }
