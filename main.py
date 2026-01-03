#!/usr/bin/env python
"""
Crypto Narrative Pulse Tracker - Main Application

This is the main entry point for the backend server providing:
1. REST API endpoints for frontend integration
2. Demo simulation mode with Telegram alerts
3. Real-time metrics serving

Usage:
  python main.py                    # Start API server (default port 8000)
  python main.py --demo             # Run demo simulation with Telegram
  python main.py --demo --no-telegram  # Run demo without Telegram
  python main.py --port 5000        # Start API server on custom port

API Endpoints:
  GET  /api/metrics          - Current pulse score and metrics
  GET  /api/metrics/history  - Historical data for charts
  POST /api/config           - Update tracked coin
  POST /api/query            - RAG query endpoint
  GET  /health               - Health check
"""

import asyncio
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

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
    """Store historical metrics for charting."""

    def __init__(self, max_hours: int = 48):
        self.max_points = max_hours * 60  # One point per minute
        self.history = deque(maxlen=self.max_points)

    def add(self, pulse_score: float, sentiment: float, divergence: str):
        """Add a new data point."""
        self.history.append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "pulse_score": pulse_score,
                "sentiment": sentiment,
                "divergence": divergence,
            }
        )

    def get_history(self, hours: int = 24) -> list:
        """Get history for the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            point
            for point in self.history
            if datetime.fromisoformat(point["timestamp"].replace("Z", "")) > cutoff
        ]


# Global metrics history
metrics_history = MetricsHistory()


# =============================================================================
# CURRENT METRICS STATE
# =============================================================================


class CurrentMetrics:
    """Current metrics state for API responses."""

    def __init__(self):
        self.pulse_score = 5.0
        self.trending_phrases = []
        self.influencer_consensus = "neutral"
        self.divergence_status = "aligned"
        self.sentiment_velocity = 0.0
        self.tracked_coin = os.getenv("TRACKED_COIN", "MEME")
        self.last_updated = datetime.utcnow()

    def update(self, **kwargs):
        """Update metrics."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = datetime.utcnow()

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "pulse_score": self.pulse_score,
            "trending_phrases": self.trending_phrases,
            "influencer_consensus": self.influencer_consensus,
            "divergence_status": self.divergence_status,
            "sentiment_velocity": self.sentiment_velocity,
            "tracked_coin": self.tracked_coin,
            "timestamp": self.last_updated.isoformat() + "Z",
        }


# Global current metrics
current_metrics = CurrentMetrics()


# =============================================================================
# FLASK API SERVER
# =============================================================================


def create_api_app():
    """Create Flask application with API endpoints."""
    from flask import Flask, jsonify, request
    from flask_cors import CORS

    app = Flask(__name__)

    # Configure CORS
    cors_origins = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    CORS(app, origins=cors_origins)

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify(
            {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        )

    @app.route("/api/metrics", methods=["GET"])
    def get_metrics():
        """Get current pulse tracker metrics."""
        return jsonify(current_metrics.to_dict())

    @app.route("/api/metrics/history", methods=["GET"])
    def get_metrics_history():
        """Get historical metrics for charting."""
        hours = request.args.get("hours", 24, type=int)
        hours = min(hours, 48)  # Cap at 48 hours
        return jsonify(
            {
                "history": metrics_history.get_history(hours),
                "hours": hours,
            }
        )

    @app.route("/api/config", methods=["GET", "POST"])
    def config():
        """Get or update configuration."""
        if request.method == "POST":
            data = request.get_json() or {}
            coin = data.get("coin")
            if coin:
                current_metrics.tracked_coin = coin.upper()
                os.environ["TRACKED_COIN"] = coin.upper()
                return jsonify({"success": True, "coin": current_metrics.tracked_coin})
            return jsonify({"error": "Missing 'coin' field"}), 400

        return jsonify(
            {
                "coin": current_metrics.tracked_coin,
                "alert_threshold_high": 7.0,
                "alert_threshold_low": 3.0,
            }
        )

    @app.route("/api/query", methods=["POST"])
    def query():
        """Handle RAG queries."""
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
            # Fallback response without RAG
            return jsonify(
                {
                    "answer": f"RAG system unavailable. Current pulse score is {current_metrics.pulse_score:.1f}/10 for ${current_metrics.tracked_coin}.",
                    "pulse_score": current_metrics.pulse_score,
                    "trending_phrases": current_metrics.trending_phrases,
                    "error": str(e),
                }
            )

    @app.route("/api/simulate", methods=["POST"])
    def simulate():
        """Trigger a simulation step (for testing)."""
        data = request.get_json() or {}
        phase = data.get("phase", "growth")

        # Run one simulation step
        sentiment_analyzer = SentimentAnalyzer()
        pulse_calculator = PulseScoreCalculator()

        message = generate_single_message(
            coin_symbol=current_metrics.tracked_coin, phase=phase
        )
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
        )

        metrics_history.add(pulse_score, sentiment, divergence)

        return jsonify(
            {
                "message": message,
                "sentiment": sentiment,
                "pulse_score": pulse_score,
                "divergence": divergence,
            }
        )

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

_Demo complete! In production, this runs 24/7 with real social media data._
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
                    f"Tracking: *${coin_symbol}*\n"
                    f"Simulating hype cycle phases...\n\n"
                    f"_Watch for alerts!_"
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

    print_header(f"📊 HYPE CYCLE SIMULATION: ${coin_symbol}")

    for phase_idx, phase_name in enumerate(PHASE_ORDER):
        phase_config = PHASES[phase_name]
        base_price, price_delta = get_phase_price_trend(phase_name)

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
        phrase_freq = {"seed": 5, "growth": 15, "peak": 25, "decline": 8}[phase_name]
        influencer_ratio = 0.3 + (influencer_count / messages_per_phase) * 0.5
        if phase_name == "peak":
            influencer_ratio = min(0.8, influencer_ratio + 0.2)

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


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Crypto Narrative Pulse Tracker")
    parser.add_argument("--demo", action="store_true", help="Run demo simulation")
    parser.add_argument(
        "--no-telegram", action="store_true", help="Disable Telegram in demo mode"
    )
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--host", default="0.0.0.0", help="API server host")

    args = parser.parse_args()

    if args.demo:
        # Run demo simulation
        asyncio.run(run_demo(use_telegram=not args.no_telegram))
    else:
        # Start API server
        print_header("🚀 CRYPTO NARRATIVE PULSE TRACKER - API SERVER")
        print(f"\n  Starting server on http://{args.host}:{args.port}")
        print(f"  Tracked coin: ${current_metrics.tracked_coin}")
        print("\n  Endpoints:")
        print("    GET  /health              - Health check")
        print("    GET  /api/metrics         - Current metrics")
        print("    GET  /api/metrics/history - Historical data")
        print("    POST /api/config          - Update config")
        print("    POST /api/query           - RAG query")
        print("    POST /api/simulate        - Trigger simulation step")
        print("\n  Press Ctrl+C to stop\n")

        try:
            app = create_api_app()
            app.run(host=args.host, port=args.port, debug=False)
        except ImportError:
            print("  ⚠️ Flask not installed. Install with: pip install flask flask-cors")
            sys.exit(1)


if __name__ == "__main__":
    main()
