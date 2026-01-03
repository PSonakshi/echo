"""
Price Pipeline Integration for the Crypto Narrative Pulse Tracker.

Integrates price data processing into the Pathway streaming pipeline using:
- pw.temporal.sliding() for price delta calculation over 5-min windows
- Percentage change calculation: (end - start) / start * 100

Requirements: 4.2
"""

import pathway as pw

# =============================================================================
# WINDOW CONFIGURATION
# =============================================================================
# Standard window configuration for consistency across all components
# 5-minute sliding window with 1-minute hop (as per design doc)

STANDARD_WINDOW_DURATION = pw.Duration(minutes=5)
STANDARD_WINDOW_HOP = pw.Duration(minutes=1)


# =============================================================================
# PRICE SCHEMAS
# =============================================================================


class PriceDeltaSchema(pw.Schema):
    """Schema for price delta calculations."""

    coin_symbol: str
    start_price: float  # Price at start of window
    end_price: float  # Price at end of window
    price_delta_pct: float  # Percentage change: (end - start) / start * 100
    window_end: pw.DateTimeUtc  # End timestamp of the window


class PriceWithTimestampSchema(pw.Schema):
    """Schema for price data with parsed timestamp."""

    coin_symbol: str
    price_usd: float
    timestamp: pw.DateTimeUtc
    volume_24h: float


# =============================================================================
# PRICE DELTA CALCULATION
# =============================================================================


def calculate_price_delta_pct(start_price: float, end_price: float) -> float:
    """
    Calculate price delta as a percentage.

    Formula: (end_price - start_price) / start_price * 100

    Args:
        start_price: Price at start of window
        end_price: Price at end of window

    Returns:
        Percentage change (positive = price increase, negative = decrease)
        Returns 0.0 if start_price is 0 or invalid

    Requirements: 4.2

    Example:
        >>> calculate_price_delta_pct(100.0, 105.0)
        5.0  # 5% increase
        >>> calculate_price_delta_pct(100.0, 95.0)
        -5.0  # 5% decrease
    """
    if start_price <= 0:
        return 0.0
    return ((end_price - start_price) / start_price) * 100


def calculate_price_delta(
    price_stream: pw.Table,
    window_duration: pw.Duration = STANDARD_WINDOW_DURATION,
    window_hop: pw.Duration = STANDARD_WINDOW_HOP,
) -> pw.Table:
    """
    Calculate price delta percentage over sliding windows.

    Uses pw.temporal.sliding() with standard 5-min window to calculate
    the percentage change in price from the start to end of each window.

    Args:
        price_stream: Pathway table with PriceSchema columns
        window_duration: Duration of sliding window (default: 5 minutes)
        window_hop: Hop interval between windows (default: 1 minute)

    Returns:
        Pathway table with price delta information:
        - coin_symbol: The cryptocurrency symbol
        - start_price: Price at window start
        - end_price: Price at window end
        - price_delta_pct: Percentage change
        - window_end: Window end timestamp

    Requirements: 4.2

    Example:
        >>> prices = pw.io.http.rest_connector(...)
        >>> price_delta = calculate_price_delta(prices)
    """
    # Apply sliding window aggregation to get earliest and latest prices
    windowed = price_stream.windowby(
        pw.this.timestamp,
        window=pw.temporal.sliding(
            hop=window_hop,
            duration=window_duration,
        ),
    ).reduce(
        coin_symbol=pw.reducers.latest(pw.this.coin_symbol),
        start_price=pw.reducers.earliest(pw.this.price_usd),
        end_price=pw.reducers.latest(pw.this.price_usd),
        window_end=pw.this._pw_window_end,
    )

    # Calculate percentage change
    return windowed.select(
        coin_symbol=pw.this.coin_symbol,
        start_price=pw.this.start_price,
        end_price=pw.this.end_price,
        price_delta_pct=pw.apply(
            calculate_price_delta_pct,
            pw.this.start_price,
            pw.this.end_price,
        ),
        window_end=pw.this.window_end,
    )


def get_latest_price_delta(price_delta_table: pw.Table) -> pw.Table:
    """
    Get the most recent price delta reading.

    Useful for dashboard displays and divergence detection.

    Args:
        price_delta_table: Table with price delta columns

    Returns:
        Table with single row containing latest price delta
    """
    return price_delta_table.reduce(
        coin_symbol=pw.reducers.latest(pw.this.coin_symbol),
        start_price=pw.reducers.latest(pw.this.start_price),
        end_price=pw.reducers.latest(pw.this.end_price),
        price_delta_pct=pw.reducers.latest(pw.this.price_delta_pct),
        window_end=pw.reducers.max(pw.this.window_end),
    )


def filter_significant_price_moves(
    price_delta_table: pw.Table,
    threshold_pct: float = 2.0,
) -> pw.Table:
    """
    Filter price delta table to only significant price movements.

    Useful for triggering alerts on large price swings.

    Args:
        price_delta_table: Table with price_delta_pct column
        threshold_pct: Minimum absolute percentage change to include

    Returns:
        Filtered table containing only significant price moves

    Example:
        >>> significant = filter_significant_price_moves(price_delta, threshold_pct=2.0)
        >>> # Only includes moves >= 2% or <= -2%
    """
    return price_delta_table.filter(
        pw.apply(
            lambda delta: abs(delta) >= threshold_pct,
            pw.this.price_delta_pct,
        )
    )


def classify_price_movement(price_delta_pct: float) -> str:
    """
    Classify price movement based on percentage change.

    Args:
        price_delta_pct: Percentage change in price

    Returns:
        Classification string:
        - "strong_up" if delta > 5%
        - "up" if delta > 2%
        - "strong_down" if delta < -5%
        - "down" if delta < -2%
        - "stable" otherwise
    """
    if price_delta_pct > 5.0:
        return "strong_up"
    elif price_delta_pct > 2.0:
        return "up"
    elif price_delta_pct < -5.0:
        return "strong_down"
    elif price_delta_pct < -2.0:
        return "down"
    else:
        return "stable"


def add_price_classification(price_delta_table: pw.Table) -> pw.Table:
    """
    Add price movement classification to price delta table.

    Args:
        price_delta_table: Table with price_delta_pct column

    Returns:
        Table with additional price_movement column
    """
    return price_delta_table.select(
        coin_symbol=pw.this.coin_symbol,
        start_price=pw.this.start_price,
        end_price=pw.this.end_price,
        price_delta_pct=pw.this.price_delta_pct,
        window_end=pw.this.window_end,
        price_movement=pw.apply(classify_price_movement, pw.this.price_delta_pct),
    )


# =============================================================================
# PIPELINE CREATION
# =============================================================================


def create_price_pipeline(price_stream: pw.Table) -> pw.Table:
    """
    Create the complete price analysis pipeline.

    Calculates price delta and adds movement classification.

    Args:
        price_stream: Raw price stream from connectors

    Returns:
        Price delta table with classification

    Requirements: 4.2

    Example:
        >>> prices = pw.io.http.rest_connector(...)
        >>> price_analysis = create_price_pipeline(prices)
    """
    # Calculate price delta over sliding windows
    price_delta = calculate_price_delta(price_stream)

    # Add movement classification
    return add_price_classification(price_delta)
