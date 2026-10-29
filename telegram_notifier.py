"""
Ampolin AI Trading Agent - Telegram Notifier Module
Dispatches rich notifications for setups, backtest approvals, Gemini AI Brain decisions,
DhanHQ order executions, and Daily P&L / EOD post-mortem reports.
"""

import asyncio
import logging
import os
import sys
from typing import Optional, Dict, Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from ai_brain import TradeDecision
from broker import OrderResult
from sandbox import BacktestResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AmpolinTelegram")

try:
    import telegram
    from telegram import Bot
    TELEGRAM_LIB_AVAILABLE = True
except ImportError:
    TELEGRAM_LIB_AVAILABLE = False
    logger.warning("python-telegram-bot not installed.")


class TelegramNotifier:
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or TELEGRAM_BOT_TOKEN or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or TELEGRAM_CHAT_ID or os.getenv("TELEGRAM_CHAT_ID", "")
        self.is_configured = bool(self.bot_token and self.chat_id and TELEGRAM_LIB_AVAILABLE)

        if not self.is_configured:
            logger.info("Telegram notifier running in local console-logger mode (bot token or chat ID not set).")

    async def _send_async(self, message: str) -> bool:
        if not self.is_configured:
            print(f"\n[TELEGRAM BROADCAST CONSOLE DUMP]\n{message}\n" + "-" * 50)
            return True

        try:
            bot = Bot(token=self.bot_token)
            await bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            logger.info("Telegram notification sent successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to dispatch Telegram message: {e}")
            # Fallback print
            print(f"\n[TELEGRAM FALLBACK]\n{message}\n" + "-" * 50)
            return False

    def send_message(self, message: str) -> bool:
        """Synchronous wrapper for message dispatch."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If running inside an existing loop
                task = asyncio.create_task(self._send_async(message))
                return True
            else:
                return asyncio.run(self._send_async(message))
        except RuntimeError:
            return asyncio.run(self._send_async(message))

    def notify_setup_approved(self, ticker: str, decision: TradeDecision, backtest: BacktestResult, order: OrderResult) -> bool:
        """Sends comprehensive trade approval and order execution report."""
        mode_badge = "🟢 LIVE ORDER" if not order.is_paper_trade else "🟡 PAPER TRADING"
        action_emoji = "🚀 BUY" if decision.action == "BUY" else "🔻 SELL"

        msg = (
            f"<b>⚡ AMPOLIN AI TRADING AGENT - TRADE ALERT ⚡</b>\n"
            f"<b>Status:</b> {mode_badge}\n\n"
            f"<b>Ticker:</b> <code>{ticker}</code> (NSE CNC Delivery)\n"
            f"<b>Action:</b> {action_emoji}\n"
            f"<b>Confidence:</b> {decision.confidence * 100:.1f}%\n"
            f"<b>Entry Price:</b> ₹{decision.entry_price}\n"
            f"<b>Stop Loss:</b> ₹{decision.stop_loss}\n"
            f"<b>Take Profit:</b> ₹{decision.take_profit}\n"
            f"<b>Risk / Reward:</b> 1:{decision.risk_reward_ratio}\n"
            f"<b>Shares Quantity:</b> {order.quantity}\n"
            f"<b>Capital Deployed:</b> ₹{order.capital_allocated:,.2f}\n"
            f"<b>Risk Amount:</b> ₹{order.total_risk_amount:.2f} ({order.risk_percentage}% account risk)\n\n"
            f"<b>📊 90-Day VectorBT Verification:</b>\n"
            f"• Win Rate: {backtest.win_rate * 100:.1f}%\n"
            f"• Expectancy: ₹{backtest.expectancy:.2f}\n"
            f"• Profit Factor: {backtest.profit_factor:.2f}\n"
            f"• Latency: {backtest.execution_time_ms} ms\n\n"
            f"<b>🧠 AI Rationale:</b>\n{decision.rationale}\n\n"
            f"<b>⚠️ Invalidation:</b> {decision.key_risks}"
        )
        return self.send_message(msg)

    def notify_daily_pnl(self, summary: Dict[str, Any]) -> bool:
        """Sends daily P&L and portfolio performance update."""
        pnl_val = summary.get("realized_pnl", 0.0)
        pnl_emoji = "🟢" if pnl_val >= 0 else "🔴"

        msg = (
            f"<b>📈 AMPOLIN AI - DAILY P&L REPORT 📈</b>\n"
            f"<b>Date:</b> {summary.get('date')}\n\n"
            f"<b>Realized P&L:</b> {pnl_emoji} ₹{pnl_val:,.2f}\n"
            f"<b>Total Trades Today:</b> {summary.get('total_trades', 0)}\n"
            f"<b>Open Positions:</b> {summary.get('open_positions', 0)}\n"
            f"<b>Closed Positions:</b> {summary.get('closed_positions', 0)}\n"
            f"<b>Win Rate:</b> {summary.get('win_rate', 0.0) * 100:.1f}%\n"
            f"<b>Wins / Losses:</b> {summary.get('win_count', 0)}W - {summary.get('loss_count', 0)}L"
        )
        return self.send_message(msg)

    def notify_eod_reflection(self, reflection: Dict[str, Any]) -> bool:
        """Sends EOD Post-Mortem summary and learned rule adjustments."""
        adjustments = reflection.get("adjustments", [])
        adj_str = "\n".join([f"• {a}" for a in adjustments]) if adjustments else "None. Active rules maintained."

        msg = (
            f"<b>🔬 AMPOLIN AI - EOD POST-MORTEM & ADAPTIVE RULES 🔬</b>\n"
            f"<b>Timestamp:</b> {reflection.get('timestamp')}\n"
            f"<b>Win Rate Sample:</b> {reflection.get('win_rate', 0.0) * 100:.1f}%\n\n"
            f"<b>Self-Refined Rules / Insights:</b>\n{adj_str}"
        )
        return self.send_message(msg)


if __name__ == "__main__":
    notifier = TelegramNotifier()
    sample_summary = {
        "date": "2026-10-29",
        "realized_pnl": 2450.50,
        "total_trades": 3,
        "open_positions": 2,
        "closed_positions": 1,
        "win_count": 1,
        "loss_count": 0,
        "win_rate": 1.0,
    }
    notifier.notify_daily_pnl(sample_summary)
