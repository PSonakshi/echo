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

STRONG_BUY_TEMPLATE = """🚀 *Momentum Alert: ${coin}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 *Pulse Score:* {score:.1f}/10
📈 *Signal:* Strong Buy Signal
🔥 *Trending:* {phrases}
{divergence_warning}
_Reply /query for detailed analysis_"""

COOLING_OFF_TEMPLATE = """❄️ *Momentum Alert: ${coin}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 *Pulse Score:* {score:.1f}/10
📉 *Signal:* Cooling Off
🔥 *Trending:* {phrases}
{divergence_warning}
_Reply /query for detailed analysis_"""

DIVERGENCE_WARNING_TEMPLATE = """⚠️ *Divergence Warning: ${coin}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 *Pulse Score:* {score:.1f}/10
⚠️ *Warning:* {divergence_type}
📈 *Price vs Sentiment:* {explanation}
🔥 *Trending:* {phrases}

_This may indicate a potential reversal. Exercise caution._"""

QUERY_RESPONSE_TEMPLATE = """📊 *Crypto Pulse Analysis*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 *Pulse Score:* {score:.1f}/10
🔥 *Trending:* {phrases}
👥 *Influencer Consensus:* {consensus}
📉 *Divergence Status:* {divergence}

*Analysis:*
{answer}

_Data updated: {timestamp}_"""


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

I'm tracking *${self.coin}* momentum in real-time.

*Commands:*
/query <question> - Ask about current sentiment
/status - Get current pulse score
/help - Show this help message

I'll send alerts when:
📈 Pulse Score >= 7 (Strong Buy Signal)
📉 Pulse Score <= 3 (Cooling Off)
⚠️ Price-Sentiment Divergence detected"""

        await update.message.reply_text(welcome_message, parse_mode="Markdown")

    async def _handle_help(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /help command."""
        help_message = """*Crypto Pulse Tracker Help*

*Commands:*
• `/query <question>` - Ask about sentiment, trends, or analysis
• `/status` - Get current pulse score and metrics
• `/help` - Show this help message

*Example queries:*
• `/query What's the current sentiment?`
• `/query Are influencers bullish or bearish?`
• `/query What phrases are trending?`

*Alert Types:*
🚀 Strong Buy Signal (score >= 7)
❄️ Cooling Off (score <= 3)
⚠️ Divergence Warning"""

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

        status_message = f"""📊 *Current Status: ${self.coin}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 *Pulse Score:* {metrics["score"]:.1f}/10
🔥 *Trending:* {metrics["phrases"]}
👥 *Influencer Consensus:* {metrics["consensus"]}
📉 *Divergence:* {metrics["divergence"]}
⚡ *Velocity:* {metrics["velocity"]:.2f}"""

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

        if self.rag_system is not None:
            try:
                # Use the RAG system for intelligent response
                rag_response = self.rag_system.answer(query)

                return QUERY_RESPONSE_TEMPLATE.format(
                    score=rag_response.pulse_score,
                    phrases=", ".join(rag_response.trending_phrases[:3]) or "None",
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
            score=metrics["score"],
            phrases=metrics["phrases"],
            consensus=metrics["consensus"],
            divergence=metrics["divergence"],
            answer=f"_RAG system not available. Showing current metrics for query: {query}_",
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
            "aligned": "✅ Aligned",
            "bearish_divergence": "⚠️ Bearish Divergence",
            "bullish_divergence": "📈 Bullish Divergence",
        }
        return status_map.get(divergence, divergence)

    def _get_divergence_explanation(self, divergence_type: str) -> str:
        """Get explanation for divergence type."""
        explanations = {
            "bearish_divergence": "Sentiment is high but price is falling. This may indicate a potential top.",
            "bullish_divergence": "Sentiment is low but price is rising. This may indicate a potential bottom.",
        }
        return explanations.get(divergence_type, "Unknown divergence pattern")

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
