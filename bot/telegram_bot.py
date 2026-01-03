"""
Telegram Bot for the Crypto Narrative Pulse Tracker.

Provides:
- Alert notifications for pulse score thresholds (>= 7 or <= 3)
- Divergence warning alerts
- /query command for RAG-based Q&A

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import asyncio
import logging
import os
from typing import Any, Optional

from telegram import Bot, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ALERT MESSAGE TEMPLATES
# =============================================================================

STRONG_BUY_TEMPLATE = """🚀 *High Momentum Alert: ${coin}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 *Pulse Score:* {score:.1f} out of 10

*What does this mean?*
The Pulse Score measures social media buzz and sentiment. A score of {score:.1f} is considered HIGH - this usually indicates strong bullish momentum and increased community interest.

📈 *Current Signal:* Strong Positive Momentum
• Social sentiment is very positive
• Community engagement is elevated
• This is typically when traders consider entry points

🔥 *What people are saying:* {phrases}
{divergence_warning}
⚠️ *Remember:* High momentum can reverse quickly. This is not financial advice - always do your own research (DYOR) and never invest more than you can afford to lose.

_Type /query followed by your question for detailed AI analysis_
_Type /help to see all available commands_"""

COOLING_OFF_TEMPLATE = """❄️ *Low Momentum Alert: ${coin}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 *Pulse Score:* {score:.1f} out of 10

*What does this mean?*
The Pulse Score measures social media buzz and sentiment. A score of {score:.1f} is considered LOW - this usually indicates declining interest or bearish sentiment in the community.

📉 *Current Signal:* Cooling Off / Declining Momentum
• Social sentiment has turned negative or neutral
• Community engagement is decreasing
• This is typically when traders exercise caution

🔥 *What people are saying:* {phrases}
{divergence_warning}
💡 *Tip:* Low momentum periods can be accumulation opportunities for long-term believers, but they can also signal further decline. This is not financial advice - always DYOR.

_Type /query followed by your question for detailed AI analysis_
_Type /help to see all available commands_"""

DIVERGENCE_WARNING_TEMPLATE = """⚠️ *Divergence Alert: ${coin}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 *Pulse Score:* {score:.1f} out of 10
⚠️ *Alert Type:* {divergence_type}

*What is a Divergence?*
A divergence happens when price movement and social sentiment move in opposite directions. This is often considered an important warning signal by traders.

📈 *What's happening:*
{explanation}

🔥 *What people are saying:* {phrases}

*Why this matters:*
Divergences can sometimes predict trend reversals. When the crowd feels one way but price moves the opposite direction, it may indicate the trend is about to change.

⚠️ *Important:* Divergences are not guarantees - they're just one signal among many. This is not financial advice. Always combine multiple indicators and do your own research.

_Type /query followed by your question for detailed AI analysis_"""

QUERY_RESPONSE_TEMPLATE = """📊 *Crypto Pulse Analysis: ${coin}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 *Current Pulse Score:* {score:.1f}/10
_{score_interpretation}_

🔥 *Trending Phrases:* {phrases}
_These are the most common terms being used in discussions_

👥 *Influencer Sentiment:* {consensus}
_What accounts with large followings are saying_

📉 *Price-Sentiment Alignment:* {divergence}
_Whether price and sentiment are moving together_

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*AI Analysis:*
{answer}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ _This analysis is for informational purposes only and is not financial advice. Always DYOR._

_Last updated: {timestamp}_"""


# =============================================================================
# TELEGRAM ALERT BOT CLASS
# =============================================================================


class TelegramAlertBot:
    """
    Telegram bot for sending alerts and handling queries.

    Sends momentum alerts when pulse score reaches thresholds:
    - Score >= 7: 🚀 Strong Buy Signal
    - Score <= 3: ❄️ Cooling Off

    Also sends divergence warnings and handles /query commands.

    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
    """

    # Alert thresholds
    STRONG_BUY_THRESHOLD = 7.0
    COOLING_OFF_THRESHOLD = 3.0

    def __init__(
        self,
        token: str,
        channel_id: str,
        coin_symbol: str = "MEME",
        rag_system: Optional[Any] = None,
    ):
        """
        Initialize the Telegram Alert Bot.

        Args:
            token: Telegram bot token
            channel_id: Channel ID or username to send alerts to
            coin_symbol: Cryptocurrency symbol being tracked
            rag_system: Optional CryptoRAG instance for query handling
        """
        self.token = token
        self.channel_id = channel_id
        self.coin = coin_symbol
        self.rag_system = rag_system

        # Initialize bot for sending messages
        self.bot = Bot(token=token)

        # Application for handling commands (initialized lazily)
        self._application: Optional[Application] = None

        # Track last alert to avoid spam
        self._last_alert_score: Optional[float] = None
        self._last_divergence_alert: Optional[str] = None

        logger.info(f"TelegramAlertBot initialized for ${coin_symbol} -> {channel_id}")

    @property
    def application(self) -> Application:
        """Get or create the Application instance."""
        if self._application is None:
            self._application = self._create_application()
        return self._application

    def _create_application(self) -> Application:
        """Create and configure the Telegram Application."""
        app = Application.builder().token(self.token).build()

        # Add command handlers
        app.add_handler(CommandHandler("start", self._handle_start))
        app.add_handler(CommandHandler("help", self._handle_help))
        app.add_handler(CommandHandler("query", self._handle_query))
        app.add_handler(CommandHandler("status", self._handle_status))

        # Add message handler for direct queries
        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._handle_message,
            )
        )

        return app

    # =========================================================================
    # ALERT METHODS (Requirements 9.1, 9.2, 9.5)
    # =========================================================================

    async def send_alert(
        self,
        score: float,
        phrases: list[str],
        divergence: str = "aligned",
    ) -> bool:
        """
        Send a momentum alert based on pulse score.

        Sends alert only if score crosses threshold:
        - Score >= 7: Strong Buy Signal (🚀)
        - Score <= 3: Cooling Off (❄️)

        Args:
            score: Current pulse score (1-10)
            phrases: List of trending phrases
            divergence: Divergence status

        Returns:
            True if alert was sent, False otherwise

        Requirements: 9.1, 9.2, 9.5
        """
        # Check if we should send an alert
        should_alert = False
        template = None

        if score >= self.STRONG_BUY_THRESHOLD:
            if (
                self._last_alert_score is None
                or self._last_alert_score < self.STRONG_BUY_THRESHOLD
            ):
                should_alert = True
                template = STRONG_BUY_TEMPLATE
        elif score <= self.COOLING_OFF_THRESHOLD:
            if (
                self._last_alert_score is None
                or self._last_alert_score > self.COOLING_OFF_THRESHOLD
            ):
                should_alert = True
                template = COOLING_OFF_TEMPLATE

        if not should_alert or template is None:
            return False

        # Format divergence warning if present
        divergence_warning = ""
        if divergence != "aligned":
            divergence_warning = f"\n⚠️ *Warning:* {self._format_divergence(divergence)}"

        # Format phrases
        phrases_str = ", ".join(phrases[:3]) if phrases else "None detected"

        # Build message
        message = template.format(
            coin=self.coin,
            score=score,
            phrases=phrases_str,
            divergence_warning=divergence_warning,
        )

        # Send the alert
        try:
            await self.bot.send_message(
                chat_id=self.channel_id,
                text=message,
                parse_mode="Markdown",
            )
            self._last_alert_score = score
            logger.info(
                f"Sent alert: score={score}, signal={'buy' if score >= 7 else 'cooling'}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False

    def send_alert_sync(
        self,
        score: float,
        phrases: list[str],
        divergence: str = "aligned",
    ) -> bool:
        """
        Synchronous wrapper for send_alert.

        Args:
            score: Current pulse score (1-10)
            phrases: List of trending phrases
            divergence: Divergence status

        Returns:
            True if alert was sent, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new task if we're already in an async context
                asyncio.ensure_future(self.send_alert(score, phrases, divergence))
                return False  # Can't wait for result in running loop
            else:
                return loop.run_until_complete(
                    self.send_alert(score, phrases, divergence)
                )
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.send_alert(score, phrases, divergence))

    # =========================================================================
    # DIVERGENCE WARNING (Requirement 9.3)
    # =========================================================================

    async def send_divergence_warning(
        self,
        score: float,
        divergence_type: str,
        phrases: list[str],
    ) -> bool:
        """
        Send a divergence warning alert.

        Args:
            score: Current pulse score
            divergence_type: Type of divergence detected
            phrases: Trending phrases

        Returns:
            True if warning was sent, False otherwise

        Requirements: 9.3
        """
        # Avoid duplicate warnings
        if divergence_type == "aligned":
            self._last_divergence_alert = None
            return False

        if self._last_divergence_alert == divergence_type:
            return False

        # Format the warning
        explanation = self._get_divergence_explanation(divergence_type)
        phrases_str = ", ".join(phrases[:3]) if phrases else "None detected"

        message = DIVERGENCE_WARNING_TEMPLATE.format(
            coin=self.coin,
            score=score,
            divergence_type=self._format_divergence(divergence_type),
            explanation=explanation,
            phrases=phrases_str,
        )

        try:
            await self.bot.send_message(
                chat_id=self.channel_id,
                text=message,
                parse_mode="Markdown",
            )
            self._last_divergence_alert = divergence_type
            logger.info(f"Sent divergence warning: {divergence_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to send divergence warning: {e}")
            return False

    def send_divergence_warning_sync(
        self,
        score: float,
        divergence_type: str,
        phrases: list[str],
    ) -> bool:
        """Synchronous wrapper for send_divergence_warning."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    self.send_divergence_warning(score, divergence_type, phrases)
                )
                return False
            else:
                return loop.run_until_complete(
                    self.send_divergence_warning(score, divergence_type, phrases)
                )
        except RuntimeError:
            return asyncio.run(
                self.send_divergence_warning(score, divergence_type, phrases)
            )

    # =========================================================================
    # COMMAND HANDLERS (Requirement 9.4)
    # =========================================================================

    async def _handle_start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /start command."""
        welcome_message = f"""🚀 *Welcome to Crypto Pulse Tracker!*

I help you track the social media buzz and sentiment around *${self.coin}* in real-time.

*What is Pulse Score?*
It's a number from 1-10 that measures how much excitement and positive sentiment there is on social media. Higher = more bullish buzz!

*What I'll alert you about:*
📈 Score ≥ 7 → High momentum (usually bullish)
📉 Score ≤ 3 → Low momentum (usually bearish)
⚠️ Divergence → Price and sentiment moving opposite

*Commands you can use:*
/status - See current pulse score and metrics
/query <question> - Ask me anything about the market
/help - Show detailed help

⚠️ *Disclaimer:* This bot provides sentiment analysis only. It is NOT financial advice. Always do your own research (DYOR) before making any investment decisions."""

        await update.message.reply_text(welcome_message, parse_mode="Markdown")

    async def _handle_help(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /help command."""
        help_message = """📚 *Crypto Pulse Tracker - Help Guide*

*Understanding the Pulse Score:*
The Pulse Score (1-10) measures social media sentiment:
• 8-10: Very High - Extreme excitement, peak hype
• 7-8: High - Strong positive momentum
• 4-6: Moderate - Mixed feelings, undecided
• 3-4: Low - Declining interest
• 1-3: Very Low - Fear or capitulation

*Available Commands:*

📊 `/status`
Shows current metrics including pulse score, trending phrases, and what influencers are saying.

❓ `/query <your question>`
Ask anything! Examples:
• `/query What's the current sentiment?`
• `/query Are influencers bullish or bearish?`
• `/query Should I be worried about this dip?`
• `/query What are people talking about?`

*Alert Types Explained:*

🚀 *High Momentum Alert*
Sent when score reaches 7+. This usually indicates strong bullish sentiment - but remember, high excitement can also mean we're near a local top!

❄️ *Cooling Off Alert*
Sent when score drops to 3 or below. This usually indicates fear or declining interest - but it can also mean accumulation opportunities.

⚠️ *Divergence Warning*
Sent when price and sentiment move in opposite directions. This is often an early warning sign of a potential trend reversal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ *Important Disclaimer:*
This bot analyzes social sentiment only. It does NOT provide financial advice. Crypto is highly volatile and risky. Never invest more than you can afford to lose. Always DYOR!"""

        await update.message.reply_text(help_message, parse_mode="Markdown")

    async def _handle_query(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """
        Handle /query command for RAG-based Q&A.

        Requirements: 9.4
        """
        # Extract query text
        query_text = " ".join(context.args) if context.args else ""

        if not query_text:
            await update.message.reply_text(
                "Please provide a question. Example:\n`/query What's the current sentiment?`",
                parse_mode="Markdown",
            )
            return

        # Get response from RAG system
        response = await self._get_rag_response(query_text)

        await update.message.reply_text(response, parse_mode="Markdown")

    async def _handle_status(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /status command."""
        # Get current metrics
        metrics = self._get_current_metrics()
        score = metrics["score"]
        score_interpretation = self._get_score_interpretation(score)

        status_message = f"""📊 *Current Status: ${self.coin}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 *Pulse Score:* {score:.1f}/10
_{score_interpretation}_

🔥 *Trending Phrases:* {metrics["phrases"]}
_What the community is talking about most_

👥 *Influencer Sentiment:* {metrics["consensus"]}
_What large accounts are saying_

📉 *Price-Sentiment Status:* {metrics["divergence"]}

⚡ *Sentiment Velocity:* {metrics["velocity"]:.2f}
_How fast sentiment is changing (+ = improving, - = declining)_

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 *Quick Interpretation:*
{"🟢 Momentum is strong - community is excited!" if score >= 7 else "🔴 Momentum is weak - community sentiment is low" if score <= 3 else "🟡 Momentum is moderate - mixed signals"}

_Use /query <question> for detailed AI analysis_"""

        await update.message.reply_text(status_message, parse_mode="Markdown")

    async def _handle_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle direct text messages as queries."""
        query_text = update.message.text

        if query_text:
            response = await self._get_rag_response(query_text)
            await update.message.reply_text(response, parse_mode="Markdown")

    # =========================================================================
    # RAG INTEGRATION (Requirement 9.4)
    # =========================================================================

    async def _get_rag_response(self, query: str) -> str:
        """
        Get response from RAG system.

        Args:
            query: User's question

        Returns:
            Formatted response string

        Requirements: 9.4
        """
        from datetime import datetime

        metrics = self._get_current_metrics()
        score = metrics["score"]
        score_interpretation = self._get_score_interpretation(score)

        if self.rag_system is not None:
            try:
                # Use the RAG system for intelligent response
                rag_response = self.rag_system.answer(query)

                return QUERY_RESPONSE_TEMPLATE.format(
                    coin=self.coin,
                    score=rag_response.pulse_score,
                    score_interpretation=self._get_score_interpretation(
                        rag_response.pulse_score
                    ),
                    phrases=", ".join(rag_response.trending_phrases[:3])
                    or "None detected",
                    consensus=rag_response.influencer_consensus,
                    divergence=rag_response.divergence_status,
                    answer=rag_response.answer,
                    timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                )

            except Exception as e:
                logger.error(f"RAG query failed: {e}")
                # Fall through to basic response

        # Basic response without RAG
        return QUERY_RESPONSE_TEMPLATE.format(
            coin=self.coin,
            score=score,
            score_interpretation=score_interpretation,
            phrases=metrics["phrases"],
            consensus=metrics["consensus"],
            divergence=metrics["divergence"],
            answer=f"_AI analysis is currently unavailable. Here are the current metrics for your query: {query}_",
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_current_metrics(self) -> dict[str, Any]:
        """Get current metrics from live metrics system."""
        try:
            from rag.live_metrics import get_metrics_for_alert

            metrics = get_metrics_for_alert()
            return {
                "score": metrics.get("score", 5.0),
                "phrases": ", ".join(metrics.get("phrases", [])[:3]) or "None detected",
                "consensus": metrics.get("consensus", "neutral"),
                "divergence": self._format_divergence(
                    metrics.get("divergence", "aligned")
                ),
                "velocity": metrics.get("velocity", 0.0),
            }
        except ImportError:
            return {
                "score": 5.0,
                "phrases": "None detected",
                "consensus": "neutral",
                "divergence": "✅ Aligned",
                "velocity": 0.0,
            }

    def _format_divergence(self, divergence: str) -> str:
        """Format divergence status for display."""
        status_map = {
            "aligned": "✅ Aligned (Price and sentiment moving together)",
            "bearish_divergence": "⚠️ Bearish Divergence (Sentiment high, price falling)",
            "bullish_divergence": "📈 Bullish Divergence (Sentiment low, price rising)",
        }
        return status_map.get(divergence, divergence)

    def _get_divergence_explanation(self, divergence_type: str) -> str:
        """Get explanation for divergence type."""
        explanations = {
            "bearish_divergence": "Social sentiment is very positive, but the price is actually falling. This mismatch is called a 'bearish divergence' - it sometimes happens before a price drop, as the crowd may be overly optimistic while smart money exits.",
            "bullish_divergence": "Social sentiment is negative or fearful, but the price is actually rising. This mismatch is called a 'bullish divergence' - it sometimes happens before a price increase, as the crowd may be overly pessimistic while smart money accumulates.",
        }
        return explanations.get(
            divergence_type,
            "An unusual pattern between price and sentiment has been detected.",
        )

    def _get_score_interpretation(self, score: float) -> str:
        """Get beginner-friendly interpretation of pulse score."""
        if score >= 8:
            return "Very High - Extreme bullish sentiment, usually indicates peak excitement"
        elif score >= 7:
            return "High - Strong positive momentum, typically seen during rallies"
        elif score >= 5:
            return "Moderate - Mixed or neutral sentiment, market is undecided"
        elif score >= 3:
            return "Low - Declining interest, sentiment turning negative"
        else:
            return "Very Low - Minimal buzz, often seen during fear or capitulation"

    # =========================================================================
    # BOT LIFECYCLE
    # =========================================================================

    def set_rag_system(self, rag_system: Any) -> None:
        """
        Set the RAG system for query handling.

        Args:
            rag_system: CryptoRAG instance
        """
        self.rag_system = rag_system
        logger.info("RAG system configured for Telegram bot")

    async def start_polling(self) -> None:
        """Start the bot in polling mode."""
        logger.info("Starting Telegram bot polling...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()

    async def stop(self) -> None:
        """Stop the bot."""
        logger.info("Stopping Telegram bot...")
        if self._application is not None:
            await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()

    def run(self) -> None:
        """Run the bot (blocking)."""
        logger.info("Running Telegram bot...")
        self.application.run_polling()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_telegram_bot(
    token: Optional[str] = None,
    channel_id: Optional[str] = None,
    coin_symbol: Optional[str] = None,
    rag_system: Optional[Any] = None,
) -> TelegramAlertBot:
    """
    Factory function to create a TelegramAlertBot.

    Reads configuration from environment variables if not provided.

    Args:
        token: Telegram bot token (default: TELEGRAM_TOKEN env var)
        channel_id: Channel ID (default: TELEGRAM_CHANNEL_ID env var)
        coin_symbol: Coin symbol (default: TRACKED_COIN env var)
        rag_system: Optional CryptoRAG instance

    Returns:
        Configured TelegramAlertBot instance
    """
    token = token or os.getenv("TELEGRAM_TOKEN")
    channel_id = channel_id or os.getenv("TELEGRAM_CHANNEL_ID")
    coin_symbol = coin_symbol or os.getenv("TRACKED_COIN", "MEME")

    if not token:
        raise ValueError("TELEGRAM_TOKEN is required")
    if not channel_id:
        raise ValueError("TELEGRAM_CHANNEL_ID is required")

    return TelegramAlertBot(
        token=token,
        channel_id=channel_id,
        coin_symbol=coin_symbol,
        rag_system=rag_system,
    )


# =============================================================================
# PIPELINE SUBSCRIPTION
# =============================================================================


def subscribe_bot_to_alerts(
    bot: TelegramAlertBot,
    pulse_scores: Optional[Any] = None,
    price_correlation: Optional[Any] = None,
) -> None:
    """
    Subscribe the Telegram bot to pipeline outputs for automatic alerts.

    Args:
        bot: TelegramAlertBot instance
        pulse_scores: Pathway table with pulse score data
        price_correlation: Pathway table with divergence data

    Requirements: 9.1, 9.2, 9.3
    """
    try:
        import pathway as pw

        from rag.live_metrics import get_live_metrics

        def create_alert_handler():
            """Create handler for pulse score alerts."""

            def handler(key, row, time, is_addition):
                if not is_addition or row is None:
                    return

                try:
                    score = float(row.get("score", row.get("pulse_score", 5.0)))
                    divergence = row.get(
                        "divergence_type", row.get("divergence", "aligned")
                    )

                    # Get trending phrases from live metrics
                    metrics = get_live_metrics()
                    phrases = metrics.trending_phrases[:3]

                    # Send alert if threshold crossed
                    bot.send_alert_sync(score, phrases, divergence)

                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")

            return handler

        def create_divergence_handler():
            """Create handler for divergence warnings."""

            def handler(key, row, time, is_addition):
                if not is_addition or row is None:
                    return

                try:
                    divergence_type = row.get(
                        "divergence_type", row.get("divergence", "aligned")
                    )

                    if divergence_type != "aligned":
                        metrics = get_live_metrics()
                        score = metrics.pulse_score
                        phrases = metrics.trending_phrases[:3]

                        bot.send_divergence_warning_sync(
                            score, divergence_type, phrases
                        )

                except Exception as e:
                    logger.error(f"Error in divergence handler: {e}")

            return handler

        if pulse_scores is not None:
            pw.io.subscribe(pulse_scores, on_change=create_alert_handler())
            logger.info("Subscribed Telegram bot to pulse_scores for alerts")

        if price_correlation is not None:
            pw.io.subscribe(price_correlation, on_change=create_divergence_handler())
            logger.info(
                "Subscribed Telegram bot to price_correlation for divergence warnings"
            )

    except ImportError:
        logger.warning("Pathway not available, skipping pipeline subscription")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and run the bot
    try:
        bot = create_telegram_bot()
        logger.info(f"Starting Telegram bot for ${bot.coin}...")
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Bot failed to start: {e}")
        sys.exit(1)
