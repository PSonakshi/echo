#!/usr/bin/env python
"""
Crypto Narrative Pulse Tracker - Demo with Telegram Bot Integration

This demo showcases the full hype cycle with REAL Telegram alerts:
1. Sentiment analysis with crypto-specific lexicon
2. Pulse score calculation (1-10 momentum indicator)
3. Price-sentiment divergence detection
4. LIVE Telegram alerts for buy/sell signals

Run with: python run_demo_telegram.py

Requirements:
- Set TELEGRAM_TOKEN and TELEGRAM_CHANNEL_ID in .env file
- Install: pip install python-telegram-bot python-dotenv

Usage:
  python run_demo_telegram.py              # Run with Telegram alerts
  python run_demo_telegram.py --no-telegram  # Run without Telegram (console only)
"""

import asyncio
import os
import sys
import time
from typing import Optional

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
# TELEGRAM BOT WRAPPER
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
        self,
        score: float,
        phase: str,
        phrases: list[str],
        divergence: str = "aligned",
    ) -> bool:
        """Send a momentum alert based on pulse score."""
        if not self.enabled:
            return False

        # Only alert on threshold crossings
        should_alert = False

        if score >= 7.0:
            if self._last_alert_score is None or self._last_alert_score < 7.0:
                should_alert = True
                emoji = "🚀"
                signal = "Strong Buy Signal"
        elif score <= 3.0:
            if self._last_alert_score is None or self._last_alert_score > 3.0:
                should_alert = True
                emoji = "❄️"
                signal = "Cooling Off"
        else:
            self._last_alert_score = score
            return False

        if not should_alert:
            return False

        # Format divergence warning
        divergence_warning = ""
        if divergence != "aligned":
            div_text = divergence.replace("_", " ").title()
            divergence_warning = f"\n⚠️ *Warning:* {div_text}"

        # Format phrases
        phrases_str = ", ".join(phrases[:3]) if phrases else "None detected"

        message = f"""{emoji} *Momentum Alert: ${self.coin}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 *Pulse Score:* {score:.1f}/10
📈 *Signal:* {signal}
🔄 *Phase:* {phase.title()}
🔥 *Trending:* {phrases_str}{divergence_warning}

_Demo simulation - not financial advice_"""

        result = await self.send_message(message)
        if result:
            self._last_alert_score = score
        return result

    async def send_divergence_warning(
        self,
        score: float,
        divergence_type: str,
        phrases: list[str],
    ) -> bool:
        """Send a divergence warning alert."""
        if not self.enabled or divergence_type == "aligned":
            return False

        if divergence_type == "bearish_divergence":
            explanation = "Sentiment is high but price is falling. Potential top!"
        else:
            explanation = "Sentiment is low but price is rising. Potential bottom!"

        phrases_str = ", ".join(phrases[:3]) if phrases else "None detected"
        div_text = divergence_type.replace("_", " ").title()

        message = f"""⚠️ *Divergence Warning: ${self.coin}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 *Pulse Score:* {score:.1f}/10
⚠️ *Warning:* {div_text}
📈 *Analysis:* {explanation}
🔥 *Trending:* {phrases_str}

_This may indicate a potential reversal._"""

        return await self.send_message(message)

    async def send_phase_update(
        self, phase: str, score: float, sentiment: float
    ) -> bool:
        """Send a phase transition update."""
        if not self.enabled:
            return False

        phase_emoji = {"seed": "🌱", "growth": "📈", "peak": "🔥", "decline": "📉"}
        emoji = phase_emoji.get(phase, "📊")

        message = f"""{emoji} *Phase Update: ${self.coin}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 *Current Phase:* {phase.title()}
📊 *Pulse Score:* {score:.1f}/10
💭 *Avg Sentiment:* {sentiment:.2f}

_Hype cycle simulation in progress..._"""

        return await self.send_message(message)

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
# DEMO FUNCTIONS
# =============================================================================


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_subheader(text: str):
    """Print a formatted subheader."""
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print(f"{'─' * 70}")


def get_phase_price_trend(phase: str) -> tuple[float, float]:
    """Get realistic price trend for each phase."""
    trends = {
        "seed": (0.001, 5.0),
        "growth": (0.00105, 15.0),
        "peak": (0.00121, 8.0),
        "decline": (0.00130, -12.0),
    }
    return trends.get(phase, (0.001, 0.0))


async def run_demo_with_telegram(use_telegram: bool = True):
    """Run the demo simulation with optional Telegram integration."""

    print_header("🚀 CRYPTO NARRATIVE PULSE TRACKER - TELEGRAM DEMO")

    # Initialize Telegram bot
    telegram_bot: Optional[DemoTelegramBot] = None

    if use_telegram:
        token = os.getenv("TELEGRAM_TOKEN")
        channel_id = os.getenv("TELEGRAM_CHANNEL_ID")
        coin_symbol = os.getenv("TRACKED_COIN", "MEME")

        if token and channel_id:
            print("\n  📱 Initializing Telegram bot...")
            telegram_bot = DemoTelegramBot(token, channel_id, coin_symbol)

            if telegram_bot.enabled:
                # Send start message
                await telegram_bot.send_message(
                    f"🚀 *Crypto Pulse Demo Starting*\n\n"
                    f"Tracking: *${coin_symbol}*\n"
                    f"Simulating hype cycle phases...\n\n"
                    f"_Watch for alerts!_"
                )
                print("  📤 Sent start notification to Telegram")
        else:
            print("\n  ⚠️ Telegram credentials not found in .env")
            print("     Set TELEGRAM_TOKEN and TELEGRAM_CHANNEL_ID to enable alerts")
    else:
        print("\n  📵 Telegram alerts disabled (--no-telegram flag)")

    telegram_status = (
        "LIVE Telegram alerts"
        if (telegram_bot and telegram_bot.enabled)
        else "Console alerts"
    )
    print(f"""
This demo simulates a complete crypto hype cycle showing:
• Real-time sentiment analysis with crypto-specific lexicon
• Pulse score calculation (1-10 momentum indicator)
• Price-sentiment divergence detection
• {telegram_status} for trading signals
    """)

    # Initialize components
    sentiment_analyzer = SentimentAnalyzer()
    pulse_calculator = PulseScoreCalculator()

    # Configuration
    coin_symbol = os.getenv("TRACKED_COIN", "MEME")
    messages_per_phase = 8

    # Track metrics
    all_results = []

    # Get trending phrases for the simulation
    trending_phrases = [
        "to the moon",
        "bullish af",
        "lfg",
        "100x potential",
        "hidden gem",
    ]

    print_header(f"📊 HYPE CYCLE SIMULATION: ${coin_symbol}")

    for phase_idx, phase_name in enumerate(PHASE_ORDER):
        phase_config = PHASES[phase_name]
        base_price, price_delta = get_phase_price_trend(phase_name)

        # Phase header with emoji
        phase_emoji = {"seed": "🌱", "growth": "📈", "peak": "🔥", "decline": "📉"}
        print_subheader(
            f"{phase_emoji.get(phase_name, '📊')} PHASE {phase_idx + 1}/4: {phase_name.upper()}"
        )

        print(f"\n  Expected Volume: {phase_config['volume_pct'] * 100:.0f}%")
        print(f"  Sentiment Range: {phase_config['sentiment_range']}")
        print(f"  Key Phrases: {', '.join(phase_config['phrases'][:3])}")
        print(f"  Price Trend: {price_delta:+.1f}%")

        # Generate and analyze messages
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

            # Show progress with sentiment indicator
            emoji = "🟢" if sentiment > 0.3 else "🔴" if sentiment < -0.3 else "⚪"
            inf_marker = "⭐" if is_influencer else " "
            print(
                f'    {emoji}{inf_marker} "{message["text"][:50]}..." → {sentiment:.2f}'
            )

            time.sleep(0.1)

        # Calculate phase metrics
        avg_sentiment = sum(sentiments) / len(sentiments)
        max_sentiment = max(sentiments)
        min_sentiment = min(sentiments)

        # Simulate phrase frequency based on phase
        phrase_freq = {"seed": 5, "growth": 15, "peak": 25, "decline": 8}[phase_name]

        # Calculate influencer ratio
        influencer_ratio = 0.3 + (influencer_count / messages_per_phase) * 0.5
        if phase_name == "peak":
            influencer_ratio = min(0.8, influencer_ratio + 0.2)

        # Detect divergence
        divergence = detect_divergence(avg_sentiment, price_delta)

        # Calculate pulse score
        pulse_score = pulse_calculator.calculate(
            sentiment_velocity=avg_sentiment,
            phrase_frequency=phrase_freq,
            influencer_ratio=influencer_ratio,
            divergence_type=divergence,
        )

        # Store results
        result = {
            "phase": phase_name,
            "avg_sentiment": avg_sentiment,
            "max_sentiment": max_sentiment,
            "min_sentiment": min_sentiment,
            "price_delta": price_delta,
            "phrase_freq": phrase_freq,
            "influencer_ratio": influencer_ratio,
            "divergence": divergence,
            "pulse_score": pulse_score,
        }
        all_results.append(result)

        # Phase summary
        print(f"\n  {'─' * 50}")
        print("  📊 PHASE METRICS:")
        print(
            f"     Avg Sentiment:    {avg_sentiment:>7.3f}  (range: {min_sentiment:.2f} to {max_sentiment:.2f})"
        )
        print(f"     Price Delta:      {price_delta:>7.1f}%")
        print(f"     Phrase Frequency: {phrase_freq:>7}")
        print(f"     Influencer Ratio: {influencer_ratio:>7.2f}")
        print(f"     Divergence:       {divergence}")
        print(f"  {'─' * 50}")

        # Pulse score with visual bar
        bar_length = int(pulse_score * 3)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        score_color = "🟢" if pulse_score >= 7 else "🔴" if pulse_score <= 3 else "🟡"

        print(f"\n  {score_color} PULSE SCORE: {pulse_score:.1f}/10")
        print(f"     [{bar}]")

        # Console alerts
        if pulse_score >= 7:
            print("\n  🚀🚀🚀 ALERT: STRONG BUY SIGNAL! 🚀🚀🚀")
            print("       Momentum is building - consider entry!")
        elif pulse_score <= 3:
            print("\n  ❄️❄️❄️ ALERT: COOLING OFF ❄️❄️❄️")
            print("       Momentum is fading - exercise caution!")

        if divergence != "aligned":
            print(f"\n  ⚠️ DIVERGENCE WARNING: {divergence.replace('_', ' ').title()}")
            if divergence == "bearish_divergence":
                print("       High sentiment but price falling - potential top!")
            else:
                print("       Low sentiment but price rising - potential bottom!")

        # Send Telegram alerts
        if telegram_bot and telegram_bot.enabled:
            # Send pulse score alert if threshold crossed
            alert_sent = await telegram_bot.send_alert(
                score=pulse_score,
                phase=phase_name,
                phrases=phase_config["phrases"][:3],
                divergence=divergence,
            )
            if alert_sent:
                print("  📤 Telegram alert sent!")

            # Send divergence warning
            if divergence != "aligned":
                div_sent = await telegram_bot.send_divergence_warning(
                    score=pulse_score,
                    divergence_type=divergence,
                    phrases=phase_config["phrases"][:3],
                )
                if div_sent:
                    print("  📤 Divergence warning sent to Telegram!")

        # Pause between phases
        if phase_idx < len(PHASE_ORDER) - 1:
            print("\n  ⏳ Moving to next phase...")
            time.sleep(1.5)

    # Final summary
    print_header("📈 SIMULATION COMPLETE - FINAL ANALYSIS")

    print("\n  📊 PHASE-BY-PHASE BREAKDOWN:")
    print(f"  {'─' * 66}")
    print(
        f"  {'Phase':<10} {'Sentiment':>10} {'Price Δ':>10} {'Phrases':>10} {'Score':>8} {'Signal':>12}"
    )
    print(f"  {'─' * 66}")

    for r in all_results:
        signal = (
            "🚀 BUY"
            if r["pulse_score"] >= 7
            else "❄️ COOL"
            if r["pulse_score"] <= 3
            else "➡️ HOLD"
        )
        print(
            f"  {r['phase']:<10} {r['avg_sentiment']:>10.3f} {r['price_delta']:>9.1f}% {r['phrase_freq']:>10} {r['pulse_score']:>8.1f} {signal:>12}"
        )

    print(f"  {'─' * 66}")

    # Key insights
    print("\n  🔑 KEY INSIGHTS:")

    peak_result = next(r for r in all_results if r["phase"] == "peak")
    if peak_result["pulse_score"] >= 7:
        print("     ✅ Peak phase showed strong buy signals - hype cycle detected!")

    decline_result = next(r for r in all_results if r["phase"] == "decline")
    if decline_result["pulse_score"] <= 3:
        print("     ✅ Decline phase correctly identified cooling momentum")

    divergences = [r for r in all_results if r["divergence"] != "aligned"]
    if divergences:
        print(
            f"     ⚠️ {len(divergences)} divergence(s) detected - important warning signals!"
        )

    # Overall assessment
    avg_score = sum(r["pulse_score"] for r in all_results) / len(all_results)
    print(f"\n  📈 OVERALL AVERAGE PULSE SCORE: {avg_score:.1f}/10")

    # Send final summary to Telegram
    if telegram_bot and telegram_bot.enabled:
        await telegram_bot.send_summary(all_results, avg_score)
        print("\n  📤 Final summary sent to Telegram!")

    print("\n" + "=" * 70)
    print("  Demo complete! In production, this runs continuously with:")
    print("  • Real-time social media data via webhooks")
    print("  • Live price feeds from CoinGecko")
    print("  • Telegram alerts for trading signals")
    print("  • RAG-powered Q&A for detailed analysis")
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    # Check for --no-telegram flag
    use_telegram = "--no-telegram" not in sys.argv

    # Run the async demo
    asyncio.run(run_demo_with_telegram(use_telegram))


if __name__ == "__main__":
    main()
