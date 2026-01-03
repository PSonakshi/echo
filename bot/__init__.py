"""
Telegram Bot module for the Crypto Narrative Pulse Tracker.

Provides alert notifications and query handling via Telegram.
"""

from bot.telegram_bot import TelegramAlertBot, create_telegram_bot

__all__ = ["TelegramAlertBot", "create_telegram_bot"]
