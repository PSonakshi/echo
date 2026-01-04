#!/usr/bin/env python
"""
Crypto Narrative Pulse Tracker - Main Application

This is the main entry point for the backend server providing:
1. REST API endpoints for frontend integration
2. WebSocket support for real-time updates
3. Demo simulation mode with Telegram alerts
4. Real-time metrics serving

Usage:
  python main.py                    # Start API server (default port 8000)
  python main.py --demo             # Run demo simulation with Telegram
  python main.py --demo --no-telegram  # Run demo without Telegram
  python main.py --port 5000        # Start API server on custom port

API Endpoints:
  GET  /api/metrics          - Current pulse score and metrics
  GET  /api/metrics/history  - Historical data for charts
  POST /api/config           - Update tracked coin configuration
  POST /api/query            - RAG query endpoint
  GET  /api/performance      - Performance metrics (latency/throughput)
  GET  /health               - Health check
  WS   /ws                   - WebSocket for real-time updates

Requirements: 11.1, 11.2, 11.4
"""

import asyncio
import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Callable, Optional

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, ".")

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Mock pathway module for Windows compatibility
from unittest.mock import MagicMock

mock_pw = MagicMock()
mock_pw.Schema = type("Schema", (), {})
mock_pw.Duration = MagicMock()
mock_pw.DateTimeUtc = MagicMock()
sys.modules["pathway"] = mock_pw

from simulator.hype_simulator import (
    PHASE_ORDER,
    PHASES,
    generate_single_message,
)
from transforms.divergence import detect_divergence
from transforms.pulse_score import PulseScoreCalculator
from transforms.sentiment import SentimentAnalyzer

# =============================================================================
# METRICS HISTORY STORAGE
# =============================================================================


class MetricsHistory:
    """
    Store historical metrics for charting.

    Supports both in-memory storage (default) and SQLite persistence.
    Retains last 24-48 hours of data points.

    Requirements: 11.1
    """

    def __init__(
        self,
        max_hours: int = 48,
        use_sqlite: bool = False,
        db_path: str = "data/metrics_history.db",
    ):
        self.max_points = max_hours * 60  # One point per minute
        self.use_sqlite = use_sqlite
        self.db_path = db_path
        self._lock = threading.Lock()

        if use_sqlite:
            self._init_sqlite()
        else:
            self.history = deque(maxlen=self.max_points)

    def _init_sqlite(self):
        """Initialize SQLite database for persistent storage."""
        import os
        import sqlite3

        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pulse_score REAL NOT NULL,
                sentiment REAL NOT NULL,
                divergence TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON metrics_history(timestamp)"
        )
        conn.commit()
        conn.close()
        logger.info(f"SQLite metrics history initialized at {self.db_path}")

    def add(self, pulse_score: float, sentiment: float, divergence: str):
        """Add a new data point."""
        timestamp = datetime.utcnow().isoformat() + "Z"

        with self._lock:
            if self.use_sqlite:
                import sqlite3

                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO metrics_history (timestamp, pulse_score, sentiment, divergence) VALUES (?, ?, ?, ?)",
                    (timestamp, pulse_score, sentiment, divergence),
                )
                # Clean up old entries
                cutoff = (datetime.utcnow() - timedelta(hours=48)).isoformat()
                cursor.execute(
                    "DELETE FROM metrics_history WHERE timestamp < ?", (cutoff,)
                )
                conn.commit()
                conn.close()
            else:
                self.history.append(
                    {
                        "timestamp": timestamp,
                        "pulse_score": pulse_score,
                        "sentiment": sentiment,
                        "divergence": divergence,
                    }
                )

    def get_history(self, hours: int = 24) -> list:
        """Get history for the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        with self._lock:
            if self.use_sqlite:
                import sqlite3

                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT timestamp, pulse_score, sentiment, divergence FROM metrics_history WHERE timestamp > ? ORDER BY timestamp ASC",
                    (cutoff.isoformat(),),
                )
                rows = cursor.fetchall()
                conn.close()
                return [
                    {
                        "timestamp": row[0],
                        "pulse_score": row[1],
                        "sentiment": row[2],
                        "divergence": row[3],
                    }
                    for row in rows
                ]
            else:
                return [
                    point
                    for point in self.history
                    if datetime.fromisoformat(point["timestamp"].replace("Z", ""))
                    > cutoff
                ]

    def get_latest(self, count: int = 100) -> list:
        """Get the latest N data points."""
        with self._lock:
            if self.use_sqlite:
                import sqlite3

                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT timestamp, pulse_score, sentiment, divergence FROM metrics_history ORDER BY timestamp DESC LIMIT ?",
                    (count,),
                )
                rows = cursor.fetchall()
                conn.close()
                return [
                    {
                        "timestamp": row[0],
                        "pulse_score": row[1],
                        "sentiment": row[2],
                        "divergence": row[3],
                    }
                    for row in reversed(rows)
                ]
            else:
                return list(self.history)[-count:]


# Global metrics history - use SQLite if DATA_PERSISTENCE env var is set
USE_SQLITE = os.getenv("DATA_PERSISTENCE", "memory").lower() == "sqlite"
metrics_history = MetricsHistory(use_sqlite=USE_SQLITE)


# =============================================================================
# INFLUENCER LEADERBOARD STORAGE (Task 18.2)
# =============================================================================


class InfluencerTracker:
    """
    Track influencer activity and sentiment for leaderboard.

    Requirements: 11.3
    """

    def __init__(self, max_influencers: int = 100):
        self.max_influencers = max_influencers
        self._lock = threading.Lock()
        self._influencers: dict[str, dict] = {}

    def update_influencer(
        self,
        author_id: str,
        followers: int,
        engagement: int,
        sentiment: float,
        message_count: int = 1,
    ):
        """Update or add an influencer's data."""
        from transforms.influence import calculate_influence_score

        influence_score = calculate_influence_score(followers, engagement)

        with self._lock:
            if author_id in self._influencers:
                # Update existing influencer with rolling average sentiment
                existing = self._influencers[author_id]
                total_messages = existing.get("message_count", 1) + message_count
                avg_sentiment = (
                    existing.get("sentiment", 0) * existing.get("message_count", 1)
                    + sentiment * message_count
                ) / total_messages

                self._influencers[author_id] = {
                    "author_id": author_id,
                    "followers": followers,
                    "engagement": engagement,
                    "influence_score": influence_score,
                    "sentiment": avg_sentiment,
                    "message_count": total_messages,
                    "last_updated": datetime.utcnow().isoformat() + "Z",
                }
            else:
                self._influencers[author_id] = {
                    "author_id": author_id,
                    "followers": followers,
                    "engagement": engagement,
                    "influence_score": influence_score,
                    "sentiment": sentiment,
                    "message_count": message_count,
                    "last_updated": datetime.utcnow().isoformat() + "Z",
                }

            # Prune to max size if needed
            if len(self._influencers) > self.max_influencers:
                # Keep top influencers by score
                sorted_influencers = sorted(
                    self._influencers.items(),
                    key=lambda x: x[1]["influence_score"],
                    reverse=True,
                )
                self._influencers = dict(sorted_influencers[: self.max_influencers])

    def get_leaderboard(self, limit: int = 10) -> list[dict]:
        """Get top influencers sorted by influence score."""
        with self._lock:
            sorted_influencers = sorted(
                self._influencers.values(),
                key=lambda x: x["influence_score"],
                reverse=True,
            )
            return sorted_influencers[:limit]


# Global influencer tracker
influencer_tracker = InfluencerTracker()


def _get_influencer_leaderboard(limit: int = 10) -> list[dict]:
    """Get the influencer leaderboard from the tracker."""
    leaderboard = influencer_tracker.get_leaderboard(limit)
    if leaderboard:
        return leaderboard
    # Return simulated data if no real data
    return _get_simulated_influencers(limit)


def _get_simulated_influencers(limit: int = 10) -> list[dict]:
    """Generate simulated influencer data for demo purposes."""
    import random

    # Simulated influencer accounts
    influencers = [
        {"author_id": "crypto_whale_1", "followers": 500000, "base_engagement": 15000},
        {"author_id": "degen_trader_2", "followers": 250000, "base_engagement": 8000},
        {"author_id": "nft_guru_3", "followers": 150000, "base_engagement": 5000},
        {"author_id": "defi_master_4", "followers": 120000, "base_engagement": 4000},
        {"author_id": "moon_hunter_5", "followers": 100000, "base_engagement": 3500},
        {"author_id": "alpha_seeker_6", "followers": 80000, "base_engagement": 2800},
        {"author_id": "chart_wizard_7", "followers": 75000, "base_engagement": 2500},
        {"author_id": "token_analyst_8", "followers": 60000, "base_engagement": 2000},
        {"author_id": "yield_farmer_9", "followers": 50000, "base_engagement": 1800},
        {"author_id": "gem_finder_10", "followers": 45000, "base_engagement": 1500},
    ]

    result = []
    for inf in influencers[:limit]:
        engagement = inf["base_engagement"] + random.randint(-500, 500)
        influence_score = (inf["followers"] * 0.6) + (engagement * 0.4)
        sentiment = random.uniform(-0.5, 0.8)

        result.append(
            {
                "author_id": inf["author_id"],
                "followers": inf["followers"],
                "engagement": engagement,
                "influence_score": influence_score,
                "sentiment": sentiment,
                "message_count": random.randint(5, 50),
                "last_updated": datetime.utcnow().isoformat() + "Z",
            }
        )

    return result


# =============================================================================
# CURRENT METRICS STATE
# =============================================================================


class CurrentMetrics:
    """
    Current metrics state for API responses.

    Thread-safe container for real-time metrics.

    Requirements: 11.1, 11.2
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.pulse_score = 5.0
        self.trending_phrases = []
        self.influencer_consensus = "neutral"
        self.divergence_status = "aligned"
        self.sentiment_velocity = 0.0
        self.tracked_coin = os.getenv("TRACKED_COIN", "MEME")
        self.last_updated = datetime.utcnow()
        self.message_count = 0
        self.influencer_bullish_count = 0
        self.influencer_bearish_count = 0
        self.current_price = None
        self.price_delta_pct = 0.0

        # WebSocket subscribers for real-time updates
        self._subscribers: list[Callable[[dict], None]] = []

    def update(self, **kwargs):
        """Update metrics (thread-safe)."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key) and not key.startswith("_"):
                    setattr(self, key, value)
            self.last_updated = datetime.utcnow()

        # Notify WebSocket subscribers
        self._notify_subscribers()

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        with self._lock:
            return {
                "pulse_score": self.pulse_score,
                "trending_phrases": self.trending_phrases,
                "influencer_consensus": self.influencer_consensus,
                "divergence_status": self.divergence_status,
                "sentiment_velocity": self.sentiment_velocity,
                "tracked_coin": self.tracked_coin,
                "timestamp": self.last_updated.isoformat() + "Z",
                "message_count": self.message_count,
                "influencer_bullish_count": self.influencer_bullish_count,
                "influencer_bearish_count": self.influencer_bearish_count,
                "current_price": self.current_price,
                "price_delta_pct": self.price_delta_pct,
            }

    def subscribe(self, callback: Callable[[dict], None]):
        """Subscribe to metrics updates for WebSocket push."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[dict], None]):
        """Unsubscribe from metrics updates."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _notify_subscribers(self):
        """Notify all WebSocket subscribers of metrics update."""
        data = self.to_dict()
        for callback in self._subscribers[
            :
        ]:  # Copy list to avoid modification during iteration
            try:
                callback(data)
            except Exception as e:
                logger.warning(f"Error notifying subscriber: {e}")


# Global current metrics
current_metrics = CurrentMetrics()


# =============================================================================
# FLASK API SERVER
# =============================================================================


def create_api_app():
    """
    Create Flask application with API endpoints.

    Endpoints:
    - GET /api/metrics - Return current pulse score, phrases, divergence, consensus
    - GET /api/metrics/history - Return historical pulse scores for charting
    - POST /api/config - Update tracked coin configuration
    - POST /api/query - Handle RAG queries from frontend
    - GET /api/performance - Performance metrics (latency/throughput)
    - GET /health - Health check

    Requirements: 11.1, 11.2, 11.4
    """
    from functools import wraps

    from flask import Flask, jsonify, request
    from flask_cors import CORS

    app = Flask(__name__)

    # ==========================================================================
    # CORS Configuration (Task 12.2)
    # ==========================================================================
    cors_origins = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")

    # Add production domain if configured
    production_domain = os.getenv("PRODUCTION_DOMAIN")
    if production_domain:
        cors_origins.append(production_domain)

    CORS(app, origins=cors_origins, supports_credentials=True)
    logger.info(f"CORS configured for origins: {cors_origins}")

    # ==========================================================================
    # API Key Authentication Middleware (Task 12.2)
    # ==========================================================================
    API_KEY = os.getenv("API_KEY")
    # Note: Protected endpoints use the @require_api_key decorator

    def require_api_key(f):
        """Decorator to require API key for protected endpoints."""

        @wraps(f)
        def decorated(*args, **kwargs):
            if not API_KEY:
                # No API key configured, allow all requests
                return f(*args, **kwargs)

            # Check for API key in header or query param
            provided_key = request.headers.get("X-API-Key") or request.args.get(
                "api_key"
            )

            if provided_key != API_KEY:
                return jsonify(
                    {"error": "Unauthorized", "message": "Invalid or missing API key"}
                ), 401

            return f(*args, **kwargs)

        return decorated

    # ==========================================================================
    # Health Check Endpoint
    # ==========================================================================
    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "version": "1.0.0",
                "tracked_coin": current_metrics.tracked_coin,
            }
        )

    # ==========================================================================
    # GET /api/metrics - Current Metrics (Task 12.1)
    # ==========================================================================
    @app.route("/api/metrics", methods=["GET"])
    def get_metrics():
        """
        Get current pulse tracker metrics.

        Returns:
            JSON with pulse_score, trending_phrases, divergence, consensus, etc.

        Requirements: 11.1, 11.2
        """
        return jsonify(current_metrics.to_dict())

    # ==========================================================================
    # GET /api/metrics/history - Historical Metrics (Task 12.1, 12.3)
    # ==========================================================================
    @app.route("/api/metrics/history", methods=["GET"])
    def get_metrics_history():
        """
        Get historical metrics for charting.

        Query params:
            hours (int): Number of hours of history (default: 24, max: 48)
            limit (int): Maximum number of data points (default: 1000)

        Returns:
            JSON with history array and metadata

        Requirements: 11.1
        """
        hours = request.args.get("hours", 24, type=int)
        hours = min(hours, 48)  # Cap at 48 hours
        limit = request.args.get("limit", 1000, type=int)

        history = metrics_history.get_history(hours)

        # Apply limit if needed
        if len(history) > limit:
            # Sample evenly across the time range
            step = len(history) // limit
            history = history[::step][:limit]

        return jsonify(
            {
                "history": history,
                "hours": hours,
                "count": len(history),
                "tracked_coin": current_metrics.tracked_coin,
            }
        )

    # ==========================================================================
    # GET /api/performance - Performance Metrics (Task 12.1)
    # ==========================================================================
    @app.route("/api/performance", methods=["GET"])
    def get_performance():
        """
        Get current performance metrics (latency and throughput).

        Returns:
            JSON with latency and throughput statistics

        Requirements: 12.1, 12.2, 12.3
        """
        try:
            from transforms.performance import get_performance_metrics

            metrics = get_performance_metrics()
            return jsonify(metrics.to_dict())
        except ImportError:
            return jsonify(
                {
                    "error": "Performance monitoring not available",
                    "latency": {"avg_ms": 0, "warnings_count": 0},
                    "throughput": {"current_mps": 0, "total_messages": 0},
                }
            )

    # ==========================================================================
    # GET/POST /api/config - Configuration (Task 12.1)
    # ==========================================================================
    @app.route("/api/config", methods=["GET", "POST"])
    @require_api_key
    def config():
        """
        Get or update configuration.

        GET: Returns current configuration
        POST: Updates tracked coin (requires API key if configured)

        POST body:
            {"coin": "BTC"}

        Returns:
            JSON with configuration

        Requirements: 11.1
        """
        if request.method == "POST":
            data = request.get_json() or {}
            coin = data.get("coin")
            if coin:
                old_coin = current_metrics.tracked_coin
                current_metrics.tracked_coin = coin.upper()
                os.environ["TRACKED_COIN"] = coin.upper()
                logger.info(f"Tracked coin changed from {old_coin} to {coin.upper()}")
                return jsonify(
                    {
                        "success": True,
                        "coin": current_metrics.tracked_coin,
                        "previous_coin": old_coin,
                    }
                )
            return jsonify({"error": "Missing 'coin' field"}), 400

        return jsonify(
            {
                "coin": current_metrics.tracked_coin,
                "alert_threshold_high": 7.0,
                "alert_threshold_low": 3.0,
                "cors_origins": cors_origins,
                "api_key_required": bool(API_KEY),
            }
        )

    # ==========================================================================
    # POST /api/query - RAG Query (Task 12.1)
    # ==========================================================================
    @app.route("/api/query", methods=["POST"])
    def query():
        """
        Handle RAG queries from frontend.

        POST body:
            {"question": "What's the sentiment on $MEME?"}

        Returns:
            JSON with answer, pulse_score, trending_phrases, sources

        Requirements: 11.4
        """
        data = request.get_json() or {}
        question = data.get("question", "")

        if not question:
            return jsonify({"error": "Missing 'question' field"}), 400

        try:
            from rag.crypto_rag import CryptoRAG

            rag = CryptoRAG()
            response = rag.answer(question)
            return jsonify(response.to_dict())
        except Exception as e:
            logger.warning(f"RAG query failed: {e}")
            # Fallback response without RAG
            return jsonify(
                {
                    "answer": f"RAG system unavailable. Current pulse score is {current_metrics.pulse_score:.1f}/10 for ${current_metrics.tracked_coin}. Trending phrases: {', '.join(current_metrics.trending_phrases[:3]) if current_metrics.trending_phrases else 'None'}.",
                    "pulse_score": current_metrics.pulse_score,
                    "trending_phrases": current_metrics.trending_phrases,
                    "influencer_consensus": current_metrics.influencer_consensus,
                    "divergence_status": current_metrics.divergence_status,
                    "sources": [],
                    "relevance_scores": [],
                    "error": str(e),
                }
            )

    # ==========================================================================
    # GET /api/influencers - Influencer Leaderboard (Task 18.2)
    # ==========================================================================
    @app.route("/api/influencers", methods=["GET"])
    def get_influencers():
        """
        Get influencer leaderboard showing top contributors and their sentiment.

        Query params:
            limit (int): Maximum number of influencers to return (default: 10)

        Returns:
            JSON with influencers array containing author_id, influence_score,
            sentiment, followers, engagement_count

        Requirements: 11.3
        """
        limit = request.args.get("limit", 10, type=int)
        limit = min(limit, 50)  # Cap at 50

        # Get influencer data from the influencer tracker
        try:
            # Get influencer data from current state
            influencers = _get_influencer_leaderboard(limit)

            return jsonify(
                {
                    "influencers": influencers,
                    "count": len(influencers),
                    "tracked_coin": current_metrics.tracked_coin,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            )
        except Exception as e:
            logger.warning(f"Error getting influencer data: {e}")
            # Return simulated data for demo purposes
            return jsonify(
                {
                    "influencers": _get_simulated_influencers(limit),
                    "count": limit,
                    "tracked_coin": current_metrics.tracked_coin,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "simulated": True,
                }
            )

    # ==========================================================================
    # GET /api/rag/stats - RAG Relevance Statistics (Task 18.3)
    # ==========================================================================
    @app.route("/api/rag/stats", methods=["GET"])
    def get_rag_stats():
        """
        Get RAG retrieval relevance score statistics.

        Returns:
            JSON with total_queries, avg_relevance, avg_latency_ms, low_relevance_count

        Requirements: 12.5
        """
        try:
            from rag.crypto_rag import get_rag_logger

            rag_logger = get_rag_logger()
            stats = rag_logger.get_statistics()
            return jsonify(
                {
                    **stats,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            )
        except Exception as e:
            logger.warning(f"Error getting RAG stats: {e}")
            return jsonify(
                {
                    "total_queries": 0,
                    "avg_relevance": 0.0,
                    "avg_latency_ms": 0.0,
                    "low_relevance_count": 0,
                    "error": str(e),
                }
            )

    # ==========================================================================
    # GET /api/rag/logs - RAG Query Logs (Task 18.3)
    # ==========================================================================
    @app.route("/api/rag/logs", methods=["GET"])
    def get_rag_logs():
        """
        Get recent RAG query logs with relevance scores.

        Query params:
            limit (int): Maximum number of logs to return (default: 100)

        Returns:
            JSON with logs array containing query, relevance_scores, latency_ms

        Requirements: 12.5
        """
        limit = request.args.get("limit", 100, type=int)
        limit = min(limit, 500)  # Cap at 500

        try:
            from rag.crypto_rag import get_rag_logger

            rag_logger = get_rag_logger()
            logs = rag_logger.get_recent_logs(limit)
            return jsonify(
                {
                    "logs": logs,
                    "count": len(logs),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            )
        except Exception as e:
            logger.warning(f"Error getting RAG logs: {e}")
            return jsonify(
                {
                    "logs": [],
                    "count": 0,
                    "error": str(e),
                }
            )

    # ==========================================================================
    # POST /api/simulate - Simulation Step (for testing)
    # ==========================================================================
    @app.route("/api/simulate", methods=["POST"])
    def simulate():
        """
        Trigger a simulation step (for testing).

        POST body:
            {"phase": "growth"}  # Optional, defaults to "growth"

        Returns:
            JSON with message, sentiment, pulse_score, divergence
        """
        import time as time_module

        data = request.get_json() or {}
        phase = data.get("phase", "growth")

        # Record ingestion time for latency tracking
        ingestion_time = time_module.time()

        # Run one simulation step
        sentiment_analyzer = SentimentAnalyzer()
        pulse_calculator = PulseScoreCalculator()

        message = generate_single_message(
            coin_symbol=current_metrics.tracked_coin, phase=phase
        )

        # Track performance
        try:
            from transforms.performance import (
                record_ingestion,
                record_message_processed,
            )

            record_ingestion(message["message_id"], ingestion_time)
            record_message_processed()
        except ImportError:
            pass

        sentiment = sentiment_analyzer.analyze(message["text"])

        # Update metrics
        phrase_freq = {"seed": 5, "growth": 15, "peak": 25, "decline": 8}.get(phase, 10)
        divergence = detect_divergence(sentiment, 5.0)  # Simulated price delta

        pulse_score = pulse_calculator.calculate(
            sentiment_velocity=sentiment,
            phrase_frequency=phrase_freq,
            influencer_ratio=0.5,
            divergence_type=divergence,
        )

        current_metrics.update(
            pulse_score=pulse_score,
            sentiment_velocity=sentiment,
            divergence_status=divergence,
            trending_phrases=PHASES.get(phase, {}).get("phrases", [])[:5],
            message_count=current_metrics.message_count + 1,
        )

        metrics_history.add(pulse_score, sentiment, divergence)

        return jsonify(
            {
                "message": message,
                "sentiment": sentiment,
                "pulse_score": pulse_score,
                "divergence": divergence,
                "phase": phase,
            }
        )

    # ==========================================================================
    # Error Handlers
    # ==========================================================================
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found", "message": str(e)}), 404

    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"Internal server error: {e}")
        return jsonify({"error": "Internal server error", "message": str(e)}), 500

    return app


# =============================================================================
# TELEGRAM BOT WRAPPER (for demo mode)
# =============================================================================


class DemoTelegramBot:
    """Simplified Telegram bot for demo alerts."""

    def __init__(self, token: str, channel_id: str, coin_symbol: str = "MEME"):
        self.token = token
        self.channel_id = channel_id
        self.coin = coin_symbol
        self.bot = None
        self.enabled = True
        self._last_alert_score: Optional[float] = None

        try:
            from telegram import Bot

            self.bot = Bot(token=token)
            print(f"  ✅ Telegram bot initialized for channel: {channel_id}")
        except Exception as e:
            print(f"  ⚠️ Telegram bot initialization failed: {e}")
            self.enabled = False

    async def send_message(self, text: str) -> bool:
        """Send a message to the Telegram channel."""
        if not self.enabled or not self.bot:
            return False

        try:
            await self.bot.send_message(
                chat_id=self.channel_id,
                text=text,
                parse_mode="Markdown",
            )
            return True
        except Exception as e:
            print(f"  ⚠️ Failed to send Telegram message: {e}")
            return False

    async def send_alert(
        self, score: float, phase: str, phrases: list[str], divergence: str = "aligned"
    ) -> bool:
        """Send a momentum alert based on pulse score."""
        if not self.enabled:
            return False

        should_alert = False
        if score >= 7.0:
            if self._last_alert_score is None or self._last_alert_score < 7.0:
                should_alert = True
                emoji = "🚀"
                signal_type = "High Momentum"
                signal_desc = "Strong positive momentum detected! Social sentiment is very bullish and community engagement is elevated. This is typically when traders consider entry points."
        elif score <= 3.0:
            if self._last_alert_score is None or self._last_alert_score > 3.0:
                should_alert = True
                emoji = "❄️"
                signal_type = "Low Momentum"
                signal_desc = "Momentum is cooling off. Social sentiment has turned negative or neutral and community engagement is decreasing. This is typically when traders exercise caution."
        else:
            self._last_alert_score = score
            return False

        if not should_alert:
            return False

        divergence_warning = ""
        if divergence != "aligned":
            div_text = divergence.replace("_", " ").title()
            if divergence == "bearish_divergence":
                divergence_warning = f"\n\n⚠️ *Divergence Warning:* {div_text}\nSentiment is high but price is falling - this mismatch sometimes precedes reversals."
            else:
                divergence_warning = f"\n\n⚠️ *Divergence Warning:* {div_text}\nSentiment is low but price is rising - this mismatch sometimes precedes reversals."

        phrases_str = ", ".join(phrases[:3]) if phrases else "None detected"

        # Score interpretation
        if score >= 8:
            score_interp = "Very High - Extreme bullish sentiment"
        elif score >= 7:
            score_interp = "High - Strong positive momentum"
        elif score >= 5:
            score_interp = "Moderate - Mixed sentiment"
        elif score >= 3:
            score_interp = "Low - Declining interest"
        else:
            score_interp = "Very Low - Minimal buzz"

        message = f"""{emoji} *{signal_type} Alert: ${self.coin}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 *Pulse Score:* {score:.1f}/10
_{score_interp}_

🔄 *Hype Cycle Phase:* {phase.title()}

*What this means:*
{signal_desc}

🔥 *What people are saying:* {phrases_str}{divergence_warning}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ _This is sentiment analysis only, not financial advice. Always DYOR!_
_Demo simulation in progress..._"""

        result = await self.send_message(message)
        if result:
            self._last_alert_score = score
        return result

    async def send_summary(self, results: list[dict], avg_score: float) -> bool:
        """Send final simulation summary."""
        if not self.enabled:
            return False

        # Build phase summary
        phase_lines = []
        for r in results:
            signal = (
                "🚀" if r["pulse_score"] >= 7 else "❄️" if r["pulse_score"] <= 3 else "➡️"
            )
            phase_lines.append(
                f"• {r['phase'].title()}: {r['pulse_score']:.1f}/10 {signal}"
            )

        phases_summary = "\n".join(phase_lines)

        # Simple interpretation based on average score
        if avg_score >= 7:
            meaning = "Strong bullish momentum detected! High social buzz usually indicates growing interest - but remember, hype can fade quickly."
        elif avg_score <= 3:
            meaning = "Low momentum detected. This usually means declining interest or fear in the market - could be a dip or continued downtrend."
        else:
            meaning = "Mixed signals detected. The market sentiment is undecided - typically means sideways movement until a clear trend forms."

        message = f"""📈 *Simulation Complete: ${self.coin}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*Phase Results:*
{phases_summary}

📊 *Average Score:* {avg_score:.1f}/10

*What this means:*
{meaning}

⏩ _This demo simulated weeks of market activity in seconds to show the tracker across various conditions._
_In production, this runs 24/7 with real social media data._
⚠️ _Not financial advice - always DYOR!_"""

        return await self.send_message(message)


# =============================================================================
# DEMO SIMULATION
# =============================================================================


def print_header(text: str):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_subheader(text: str):
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print(f"{'─' * 70}")


def get_phase_price_trend(phase: str) -> tuple:
    trends = {
        "seed": (0.001, 5.0),
        "growth": (0.00105, 15.0),
        "peak": (0.00121, 8.0),
        "decline": (0.00130, -12.0),
    }
    return trends.get(phase, (0.001, 0.0))


async def run_demo(use_telegram: bool = True):
    """Run the demo simulation."""
    print_header("🚀 CRYPTO NARRATIVE PULSE TRACKER - DEMO")

    telegram_bot: Optional[DemoTelegramBot] = None

    if use_telegram:
        token = os.getenv("TELEGRAM_TOKEN")
        channel_id = os.getenv("TELEGRAM_CHANNEL_ID")
        coin_symbol = os.getenv("TRACKED_COIN", "MEME")

        if token and channel_id:
            print("\n  📱 Initializing Telegram bot...")
            telegram_bot = DemoTelegramBot(token, channel_id, coin_symbol)

            if telegram_bot.enabled:
                await telegram_bot.send_message(
                    f"🚀 *Crypto Pulse Demo Starting*\n\n"
                    f"Tracking: *${coin_symbol}*\n\n"
                    f"⏩ *What you're about to see:*\n"
                    f"We're simulating weeks/months of market activity in just a few seconds! "
                    f"This rapid simulation shows how the tracker behaves across different market conditions.\n\n"
                    f"_Watch for alerts as momentum shifts!_"
                )
                print("  📤 Sent start notification to Telegram")
        else:
            print("\n  ⚠️ Telegram credentials not found in .env")
    else:
        print("\n  📵 Telegram alerts disabled")

    sentiment_analyzer = SentimentAnalyzer()
    pulse_calculator = PulseScoreCalculator()
    coin_symbol = os.getenv("TRACKED_COIN", "MEME")
    messages_per_phase = 8
    all_results = []

    # Demo-specific phase configs to ensure full range of pulse scores
    # These override the simulator defaults to show the system in various conditions
    demo_phase_configs = {
        "seed": {
            "sentiment_boost": -0.2,  # Start low/cautious
            "phrase_freq": 3,
            "influencer_ratio": 0.2,
            "price_delta": -5.0,
        },
        "growth": {
            "sentiment_boost": 0.3,  # Building excitement
            "phrase_freq": 18,
            "influencer_ratio": 0.6,
            "price_delta": 25.0,
        },
        "peak": {
            "sentiment_boost": 0.5,  # Maximum hype
            "phrase_freq": 30,
            "influencer_ratio": 0.85,
            "price_delta": 15.0,
        },
        "decline": {
            "sentiment_boost": -0.4,  # Fear/capitulation
            "phrase_freq": 5,
            "influencer_ratio": 0.15,
            "price_delta": -20.0,
        },
    }

    print_header(f"📊 HYPE CYCLE SIMULATION: ${coin_symbol}")
    print("\n  ⏩ Simulating weeks of market activity at rapid speed...")
    print("     This shows how the tracker responds to different conditions.\n")

    for phase_idx, phase_name in enumerate(PHASE_ORDER):
        phase_config = PHASES[phase_name]
        demo_config = demo_phase_configs[phase_name]
        price_delta = demo_config["price_delta"]

        phase_emoji = {"seed": "🌱", "growth": "📈", "peak": "🔥", "decline": "📉"}
        print_subheader(
            f"{phase_emoji.get(phase_name, '📊')} PHASE {phase_idx + 1}/4: {phase_name.upper()}"
        )

        print(f"\n  Expected Volume: {phase_config['volume_pct'] * 100:.0f}%")
        print(f"  Sentiment Range: {phase_config['sentiment_range']}")
        print(f"  Key Phrases: {', '.join(phase_config['phrases'][:3])}")
        print(f"  Price Trend: {price_delta:+.1f}%")

        sentiments = []
        influencer_count = 0

        print(f"\n  📨 Processing {messages_per_phase} messages...\n")

        for i in range(messages_per_phase):
            message = generate_single_message(coin_symbol=coin_symbol, phase=phase_name)
            sentiment = sentiment_analyzer.analyze(message["text"])
            # Apply demo boost to ensure wider range
            sentiment = max(-1.0, min(1.0, sentiment + demo_config["sentiment_boost"]))
            sentiments.append(sentiment)

            is_influencer = message["author_followers"] > 10000
            if is_influencer:
                influencer_count += 1

            emoji = "🟢" if sentiment > 0.3 else "🔴" if sentiment < -0.3 else "⚪"
            inf_marker = "⭐" if is_influencer else " "
            print(
                f'    {emoji}{inf_marker} "{message["text"][:50]}..." → {sentiment:.2f}'
            )
            time.sleep(0.1)

        avg_sentiment = sum(sentiments) / len(sentiments)
        phrase_freq = demo_config["phrase_freq"]
        influencer_ratio = demo_config["influencer_ratio"]

        divergence = detect_divergence(avg_sentiment, price_delta)
        pulse_score = pulse_calculator.calculate(
            sentiment_velocity=avg_sentiment,
            phrase_frequency=phrase_freq,
            influencer_ratio=influencer_ratio,
            divergence_type=divergence,
        )

        # Update global metrics
        current_metrics.update(
            pulse_score=pulse_score,
            sentiment_velocity=avg_sentiment,
            divergence_status=divergence,
            trending_phrases=phase_config["phrases"][:5],
        )
        metrics_history.add(pulse_score, avg_sentiment, divergence)

        result = {
            "phase": phase_name,
            "avg_sentiment": avg_sentiment,
            "price_delta": price_delta,
            "phrase_freq": phrase_freq,
            "influencer_ratio": influencer_ratio,
            "divergence": divergence,
            "pulse_score": pulse_score,
        }
        all_results.append(result)

        print(f"\n  {'─' * 50}")
        print("  📊 PHASE METRICS:")
        print(f"     Avg Sentiment:    {avg_sentiment:>7.3f}")
        print(f"     Price Delta:      {price_delta:>7.1f}%")
        print(f"     Phrase Frequency: {phrase_freq:>7}")
        print(f"     Divergence:       {divergence}")
        print(f"  {'─' * 50}")

        bar_length = int(pulse_score * 3)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        score_color = "🟢" if pulse_score >= 7 else "🔴" if pulse_score <= 3 else "🟡"
        print(f"\n  {score_color} PULSE SCORE: {pulse_score:.1f}/10")
        print(f"     [{bar}]")

        if pulse_score >= 7:
            print("\n  🚀🚀🚀 ALERT: STRONG BUY SIGNAL! 🚀🚀🚀")
        elif pulse_score <= 3:
            print("\n  ❄️❄️❄️ ALERT: COOLING OFF ❄️❄️❄️")

        if telegram_bot and telegram_bot.enabled:
            alert_sent = await telegram_bot.send_alert(
                score=pulse_score,
                phase=phase_name,
                phrases=phase_config["phrases"][:3],
                divergence=divergence,
            )
            if alert_sent:
                print("  📤 Telegram alert sent!")

        if phase_idx < len(PHASE_ORDER) - 1:
            print("\n  ⏳ Moving to next phase...")
            time.sleep(1.5)

    print_header("📈 SIMULATION COMPLETE")

    avg_score = sum(r["pulse_score"] for r in all_results) / len(all_results)
    print(f"\n  📈 OVERALL AVERAGE PULSE SCORE: {avg_score:.1f}/10")

    if telegram_bot and telegram_bot.enabled:
        await telegram_bot.send_summary(all_results, avg_score)
        print("\n  📤 Final summary sent to Telegram!")

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70 + "\n")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
# WEBSOCKET SUPPORT (Task 12.4)
# =============================================================================

# Global SocketIO instance
_socketio = None


def create_socketio_app(app):
    """
    Create Flask-SocketIO application for real-time WebSocket updates.

    Provides real-time metrics push to frontend clients as an alternative
    to polling for live updates.

    Requirements: 11.1
    """
    global _socketio

    try:
        from flask_socketio import SocketIO, emit

        cors_origins = os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")

        _socketio = SocketIO(
            app,
            cors_allowed_origins=cors_origins,
            async_mode="threading",
            logger=False,
            engineio_logger=False,
        )

        @_socketio.on("connect")
        def handle_connect():
            """Handle client connection."""
            logger.info("WebSocket client connected")
            # Send current metrics immediately on connect
            emit("metrics_update", current_metrics.to_dict())

        @_socketio.on("disconnect")
        def handle_disconnect():
            """Handle client disconnection."""
            logger.info("WebSocket client disconnected")

        @_socketio.on("subscribe_metrics")
        def handle_subscribe():
            """Handle metrics subscription request."""
            emit("metrics_update", current_metrics.to_dict())

        @_socketio.on("request_history")
        def handle_history_request(data):
            """Handle history request from client."""
            hours = data.get("hours", 24) if data else 24
            hours = min(hours, 48)
            history = metrics_history.get_history(hours)
            emit(
                "history_update",
                {
                    "history": history,
                    "hours": hours,
                    "count": len(history),
                },
            )

        # Subscribe to metrics updates to broadcast to all clients
        def broadcast_metrics(data: dict):
            """Broadcast metrics update to all connected clients."""
            if _socketio:
                _socketio.emit("metrics_update", data)

        current_metrics.subscribe(broadcast_metrics)

        logger.info("WebSocket support enabled with Flask-SocketIO")
        return _socketio

    except ImportError:
        logger.warning("Flask-SocketIO not installed. WebSocket support disabled.")
        logger.warning("Install with: pip install flask-socketio")
        return None


def broadcast_metrics_update():
    """Manually broadcast current metrics to all WebSocket clients."""
    global _socketio
    if _socketio:
        _socketio.emit("metrics_update", current_metrics.to_dict())


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Main entry point."""
    import argparse

    # Get port from environment variable (Railway sets this)
    default_port = int(os.environ.get("PORT", 8000))

    parser = argparse.ArgumentParser(description="Crypto Narrative Pulse Tracker")
    parser.add_argument("--demo", action="store_true", help="Run demo simulation")
    parser.add_argument(
        "--no-telegram", action="store_true", help="Disable Telegram in demo mode"
    )
    parser.add_argument(
        "--port", type=int, default=default_port, help="API server port"
    )
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument(
        "--no-websocket", action="store_true", help="Disable WebSocket support"
    )

    args = parser.parse_args()

    if args.demo:
        # Run demo simulation
        asyncio.run(run_demo(use_telegram=not args.no_telegram))
    else:
        # Start API server
        print_header("🚀 CRYPTO NARRATIVE PULSE TRACKER - API SERVER")
        print(f"\n  Starting server on http://{args.host}:{args.port}")
        print(f"  Tracked coin: ${current_metrics.tracked_coin}")
        print("\n  REST Endpoints:")
        print("    GET  /health              - Health check")
        print("    GET  /api/metrics         - Current metrics")
        print("    GET  /api/metrics/history - Historical data")
        print("    GET  /api/performance     - Performance metrics")
        print("    GET  /api/influencers     - Influencer leaderboard")
        print("    GET  /api/rag/stats       - RAG relevance statistics")
        print("    GET  /api/rag/logs        - RAG query logs")
        print("    POST /api/config          - Update config")
        print("    POST /api/query           - RAG query")
        print("    POST /api/simulate        - Trigger simulation step")

        try:
            app = create_api_app()

            # Try to enable WebSocket support
            socketio = None
            if not args.no_websocket:
                socketio = create_socketio_app(app)
                if socketio:
                    print("\n  WebSocket Events:")
                    print("    connect              - Client connected")
                    print("    metrics_update       - Real-time metrics push")
                    print("    subscribe_metrics    - Subscribe to updates")
                    print("    request_history      - Request historical data")

            print("\n  Press Ctrl+C to stop\n")

            if socketio:
                # Run with WebSocket support
                socketio.run(app, host=args.host, port=args.port, debug=False)
            else:
                # Run without WebSocket support
                app.run(host=args.host, port=args.port, debug=False)

        except ImportError as e:
            print(f"  ⚠️ Missing dependency: {e}")
            print("  Install with: pip install flask flask-cors flask-socketio")
            sys.exit(1)


if __name__ == "__main__":
    main()
