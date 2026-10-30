"""
Ampolin AI Trading Agent - Interactive 2-Way Telegram Bot
Allows the user to chat directly with Ampolin AI on Telegram:
- Ask natural language trading and portfolio questions (powered by Gemini AI)
- Check live /positions, /status, /pnl, and /rules
- Trigger on-demand /scan (NSE) and /scan_us (Wall Street)
"""

import asyncio
import logging
import os
import sys
import threading
from typing import Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY
from reflection import TradeLogger
from market_regime import MarketRegimeAnalyzer
from llm_client import MultiProviderLLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AmpolinTelegramBot")

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class InteractiveTelegramBot:
    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.authorized_chat_id = str(TELEGRAM_CHAT_ID or os.getenv("TELEGRAM_CHAT_ID", ""))
        self.logger = TradeLogger()
        self.regime_analyzer = MarketRegimeAnalyzer(sample_size=20)
        self.app = None
        self.llm = MultiProviderLLM()
        self.chat_memories = {}  # Stores recent dialogue per chat_id

    def is_authorized(self, update: Update) -> bool:
        """Ensures only the account owner can interact with the bot."""
        chat_id = str(update.effective_chat.id) if update.effective_chat else ""
        if not self.authorized_chat_id:
            self.authorized_chat_id = chat_id
            return True
        is_auth = (chat_id == self.authorized_chat_id)
        if not is_auth:
            logger.warning(f"Unauthorized chat attempt from chat_id: {chat_id} (expected: {self.authorized_chat_id})")
        return is_auth

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized(update):
            await update.message.reply_text("⛔ Unauthorized access.")
            return

        msg = (
            "🤖 <b>Welcome Kaustuv! Ampolin AI is online & ready.</b>\n\n"
            "Here is what you can do:\n"
            "• <b>/positions</b> — View active open swing trades\n"
            "• <b>/status</b> — System health, India VIX & market regime\n"
            "• <b>/scan</b> — Trigger an instant NSE Nifty 50 scan\n"
            "• <b>/scan_us</b> — Trigger an instant Wall Street scan\n"
            "• <b>/pnl</b> — Today's realized P&L & win/loss stats\n"
            "• <b>/rules</b> — Active risk & trailing stop rules\n\n"
            "💬 <i>Or just type any question in plain English (e.g., 'How is the market looking?', 'What is our stop loss rule?') to chat with me!</i>"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized(update):
            return

        regime = self.regime_analyzer.analyze_regime()
        open_trades = self.logger.get_open_trades()

        msg = (
            "📊 <b>AMPOLIN AI SYSTEM STATUS</b>\n\n"
            f"• <b>Market Regime:</b> <code>{regime.breadth_regime}</code>\n"
            f"• <b>India VIX:</b> {regime.india_vix} ({regime.vix_regime})\n"
            f"• <b>Nifty Breadth:</b> {regime.market_breadth_pct:.1f}% stocks > 50 EMA\n"
            f"• <b>Can Trade Longs:</b> {'✅ Yes' if regime.can_trade_longs else '⛔ Paused'}\n"
            f"• <b>Dynamic Risk/Trade:</b> {regime.dynamic_risk_pct * 100:.2f}%\n"
            f"• <b>Active Positions:</b> {len(open_trades)}\n"
            f"• <b>Mode:</b> Paper Trading (Simulated CNC Delivery)"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized(update):
            return

        open_trades = self.logger.get_open_trades()
        if not open_trades:
            await update.message.reply_text("📭 <b>No active open positions.</b>\nSystem is currently in cash.", parse_mode="HTML")
            return

        lines = [f"📈 <b>ACTIVE OPEN POSITIONS ({len(open_trades)})</b>\n"]
        for t in open_trades:
            sym = "$" if not t["ticker"].endswith(".NS") else "₹"
            stage = t.get("stage", "INITIAL")
            lines.append(
                f"• <b>{t['ticker']}</b>\n"
                f"  Entry: {sym}{t['entry_price']} | Qty: {t['quantity']}\n"
                f"  SL: {sym}{t['stop_loss']} | Target: {sym}{t['take_profit']}\n"
                f"  Stage: <code>{stage}</code>\n"
                f"  Risk: {sym}{t['risk_amount']:.2f}\n"
            )

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized(update):
            return

        summary = self.logger.calculate_daily_summary()
        pnl = summary.get("realized_pnl", 0.0)
        pnl_sign = "+" if pnl >= 0 else ""
        icon = "🟢" if pnl >= 0 else "🔴"

        msg = (
            f"💰 <b>TODAY'S P&L SUMMARY ({summary.get('date')})</b>\n\n"
            f"• <b>Realized P&L:</b> {icon} <b>{pnl_sign}₹{pnl:,.2f}</b>\n"
            f"• <b>Win Rate:</b> {summary.get('win_rate', 0.0)*100:.1f}%\n"
            f"• <b>Wins:</b> {summary.get('win_count', 0)} | <b>Losses:</b> {summary.get('loss_count', 0)}\n"
            f"• <b>Open Positions:</b> {summary.get('open_positions', 0)}"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized(update):
            return

        await update.message.reply_text("🔎 <b>Starting on-demand Nifty 50 scan...</b>\nChecking 50 stocks with VectorBT & Gemini AI. Please wait 15-20 seconds.", parse_mode="HTML")
        try:
            from main import AmpolinPipeline
            pipeline = AmpolinPipeline(dry_run=True)
            res = pipeline.run_pipeline(market="NSE", max_trades_to_execute=2)
            trades = res.get("executed_trades", [])
            if trades:
                msg = f"✅ <b>Scan Complete:</b> Approved & executed {len(trades)} paper trades! Details were sent above."
            else:
                msg = f"ℹ️ <b>Scan Complete:</b> No setups passed the strict institutional filters (Status: {res.get('status')})."
            await update.message.reply_text(msg, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"❌ Scan error: {e}")

    async def cmd_scan_us(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized(update):
            return

        await update.message.reply_text("🇺🇸 <b>Starting on-demand Wall Street scan...</b>\nChecking US tech leaders (NVDA, AAPL, TSLA...). Please wait 15-20 seconds.", parse_mode="HTML")
        try:
            from main import AmpolinPipeline
            pipeline = AmpolinPipeline(dry_run=True)
            res = pipeline.run_pipeline(market="US", max_trades_to_execute=2)
            trades = res.get("executed_trades", [])
            if trades:
                msg = f"✅ <b>Wall Street Scan Complete:</b> Approved {len(trades)} trades! Details sent above."
            else:
                msg = f"ℹ️ <b>Wall Street Scan Complete:</b> No setups satisfied strict VectorBT / Weekly confluence filters."
            await update.message.reply_text(msg, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"❌ Scan error: {e}")

    async def cmd_rules(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized(update):
            return

        msg = (
            "📜 <b>ACTIVE TRADING & RISK RULES</b>\n\n"
            "• <b>Risk per Trade:</b> 1.0% max (0.75% if India VIX > 15)\n"
            "• <b>Weekly Confluence:</b> 20 EMA > 50 EMA on 1wk required\n"
            "• <b>Anti-Chasing:</b> Max 2.2% distance from 20 EMA\n"
            "• <b>VectorBT Filter:</b> Win Rate >= 52% | Profit Factor >= 1.50\n"
            "• <b>1.0R Gain:</b> Stop Loss raised to Breakeven (0% risk)\n"
            "• <b>+3.5% Gain:</b> SL locked at +2.0% guaranteed profit floor\n"
            "• <b>2.0R Gain:</b> 50% partial exit taken, remainder trails 20 EMA"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def handle_natural_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Processes conversational trading questions with multi-turn memory and instant Groq failover."""
        if not self.is_authorized(update):
            return

        user_text = (update.message.text or "").strip()
        if not user_text:
            return

        chat_id = str(update.effective_chat.id)
        logger.info(f"Incoming Telegram chat message: '{user_text}' from chat_id: {chat_id}")

        # Show typing status
        await update.message.chat.send_action("typing")

        open_trades = self.logger.get_open_trades()
        positions_str = ", ".join([f"{t['ticker']} (entry: {t['entry_price']}, stage: {t.get('stage','INITIAL')})" for t in open_trades]) or "None (100% Cash)"
        summary = self.logger.calculate_daily_summary()

        system_prompt = f"""
You are Ampolin AI, an institutional-grade quantitative swing trading partner for Indian Equities (NSE) and Wall Street (US).

Current Portfolio & System Context:
- Active Open Positions: {positions_str}
- Realized P&L Today: ₹{summary.get('realized_pnl', 0.0):,.2f}
- Core Strategy: 20/50 EMA breakouts, RSI 48-68 momentum, VectorBT backtest verification, 1R breakeven, 3.5% profit ratchet (locks +2.0%), and 20 EMA trailing.
- Risk Budget: Hard 0.75%-1.0% max account risk per trade. Zero leverage (Cash Delivery only).

CONVERSATION & TONE RULES:
1. NEVER start messages by repeatedly greeting "Hey Kaustuv" or "Hello Kaustuv". Dive straight into the core answer or insight like a sharp, professional hedge-fund partner.
2. MULTI-TURN CONTINUITY: Follow up naturally on what was previously discussed in the dialogue thread. If Kaustuv asks about a previously mentioned stock or idea, seamlessly connect the dots without asking him to repeat himself.
3. PROACTIVE FOLLOW-UPS: Always end your answers with an actionable, intelligent follow-up question or offer (e.g. offering to run an on-demand scan on Nifty 50 or Wall Street, inspect a specific chart, or adjust profit lock parameters).
4. READABILITY: Use clean bullet points, bolding, and occasional trading emojis (📈, 🛡️, 💰). Keep it under 2-3 short, easily scannable paragraphs on mobile Telegram.
"""

        if chat_id not in self.chat_memories:
            self.chat_memories[chat_id] = []

        chat_history = self.chat_memories[chat_id]

        try:
            response_text = self.llm.generate_chat_response(
                user_prompt=user_text,
                system_prompt=system_prompt,
                history=chat_history,
            )

            if response_text:
                # Update memory
                chat_history.append({"role": "user", "content": user_text})
                chat_history.append({"role": "assistant", "content": response_text})
                # Keep last 10 messages
                if len(chat_history) > 10:
                    self.chat_memories[chat_id] = chat_history[-10:]

                await update.message.reply_text(response_text)
            else:
                await update.message.reply_text("I'm reviewing the live market data feeds right now. Could you ask that once more?")

        except Exception as e:
            logger.error(f"Error handling Telegram chat: {e}")
            await update.message.reply_text(f"Oops, error processing your message: {e}")

    def build_application(self):
        """Constructs the telegram Application with handlers."""
        app = ApplicationBuilder().token(self.bot_token).build()

        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_start))
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("positions", self.cmd_positions))
        app.add_handler(CommandHandler("pnl", self.cmd_pnl))
        app.add_handler(CommandHandler("scan", self.cmd_scan))
        app.add_handler(CommandHandler("scan_us", self.cmd_scan_us))
        app.add_handler(CommandHandler("rules", self.cmd_rules))

        # Natural language handler
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_natural_chat))

        self.app = app
        return app

    def start_polling_in_background(self):
        """Runs the telegram bot in a separate background thread with clean signal isolation."""
        def _runner():
            try:
                app = self.build_application()
                logger.info("Interactive Telegram Bot polling started in background thread.")
                app.run_polling(drop_pending_updates=False, stop_signals=None)
            except Exception as e:
                logger.error(f"Error in Telegram Bot background runner: {e}")

        t = threading.Thread(target=_runner, name="AmpolinTelegramBotThread", daemon=True)
        t.start()
        return t


if __name__ == "__main__":
    print("Testing Interactive Telegram Bot standalone...")
    bot = InteractiveTelegramBot()
    app = bot.build_application()
    print("Bot polling active. Message @AmpolinTradingBot on Telegram to test!")
    app.run_polling()
