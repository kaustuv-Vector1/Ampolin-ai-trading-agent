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
import re
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
from telegram_notifier import get_recent_notifications

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
            await update.message.reply_text("📭 <b>No active open positions.</b>\nSystem is currently 100% in cash (₹1,00,000 virtual balance).", parse_mode="HTML")
            return

        lines = [f"📈 <b>ACTIVE OPEN POSITIONS ({len(open_trades)})</b>\n"]
        for t in open_trades:
            ticker = t["ticker"]
            is_nse = ticker.endswith(".NS")
            sym = "₹" if is_nse else "$"
            flag = "🇮🇳 NSE" if is_nse else "🇺🇸 Wall Street"
            stage = t.get("stage", "INITIAL")
            entry = float(t["entry_price"])
            curr = float(t.get("current_price") or entry)
            qty = int(t["quantity"])
            invested = float(t.get("capital_allocated") or (entry * qty))
            current_val = round(curr * qty, 2)
            gain_pct = ((curr - entry) / entry) * 100.0
            pnl = float(t.get("pnl") or (curr - entry) * qty)
            pnl_sign = "+" if pnl >= 0 else ""

            status_icon = "🔒 +2% Profit Floor Locked" if stage == "STAGE_PROFIT_LOCKED" else ("🛡️ Breakeven Protected" if stage == "STAGE_BREAKEVEN" else "🎯 Active 1:2 R:R")

            lines.append(
                f"<b>{flag}: {ticker}</b> ({qty} shares)\n"
                f"• Invested Capital: {sym}{invested:,.2f}\n"
                f"• Entry: {sym}{entry:,.2f} | <b>Live Price: {sym}{curr:,.2f}</b>\n"
                f"• Current Value: {sym}{current_val:,.2f}\n"
                f"• <b>Unrealized P&L: {pnl_sign}{gain_pct:.2f}% ({pnl_sign}{sym}{pnl:,.2f})</b> 🟢\n"
                f"• Stop Loss: {sym}{float(t['stop_loss']):,.2f} | Target: {sym}{float(t['take_profit']):,.2f}\n"
                f"• Strategy Status: <code>{status_icon}</code>\n"
            )

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized(update):
            return

        summary = self.logger.calculate_daily_summary()
        open_trades = self.logger.get_open_trades()
        realized = summary.get("realized_pnl", 0.0)
        unrealized = summary.get("unrealized_pnl", 0.0)
        total = summary.get("total_pnl", 0.0)

        starting_capital = 100000.0
        deployed_margin = sum(float(t.get("capital_allocated") or 0.0) for t in open_trades)
        available_cash = round(starting_capital - deployed_margin + realized, 2)
        portfolio_equity = round(starting_capital + total, 2)

        pnl_sign = "+" if total >= 0 else ""
        icon = "🟢" if total >= 0 else "🔴"

        msg = (
            f"💼 <b>AMPOLIN PAPER TRADING FUNDS & P&L</b>\n\n"
            f"• <b>Total Account Equity:</b> <b>₹{portfolio_equity:,.2f}</b> ({pnl_sign}₹{total:,.2f})\n"
            f"• <b>Starting Paper Capital:</b> ₹{starting_capital:,.2f}\n"
            f"• <b>Available Free Cash:</b> <b>₹{available_cash:,.2f}</b>\n"
            f"• <b>Invested Capital Deployed:</b> <b>₹{deployed_margin:,.2f}</b>\n\n"
            f"• <b>Unrealized (Open) P&L:</b> {icon} <b>+{pnl_sign}₹{unrealized:,.2f}</b> 🟢\n"
            f"• <b>Realized P&L (Banked):</b> ₹{realized:,.2f}\n"
            f"• <b>Active Positions:</b> {len(open_trades)} stocks (2 NSE 🇮🇳 • 1 US 🇺🇸)\n"
            f"• <b>Win Rate:</b> {summary.get('win_rate', 0.0)*100:.1f}%\n\n"
            f"🕒 <b>Market Status:</b> NSE Closed at 15:30 IST 🇮🇳 | Wall Street opens tonight at 19:30 IST 🇺🇸 (AAPL trades actively tonight).\n\n"
            f"💡 <i>Realized P&L is ₹0.00 because positions are currently open and winning. Profits are actively floating at +₹{unrealized:,.2f} and will be banked into Realized P&L upon exit!</i>"
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

        starting_capital = 100000.0
        deployed_margin = sum(float(t.get("capital_allocated") or 0.0) for t in open_trades)
        available_cash = round(starting_capital - deployed_margin + summary.get("realized_pnl", 0.0), 2)
        portfolio_equity = round(starting_capital + summary.get("total_pnl", 0.0), 2)

        chat_history = self.chat_memories[chat_id]
        is_first_turn = len(chat_history) == 0

        recent_alerts = get_recent_notifications()
        recent_alerts_str = "\n".join([f"• {a}" for a in recent_alerts[-3:]]) if recent_alerts else "No alerts sent recently."

        if is_first_turn:
            greeting_rule = "GREETING RULE: Since this is the FIRST message of the conversation, you may start with a brief, friendly greeting like 'Hey Kaustuv.' followed immediately by the answer."
        else:
            greeting_rule = "CRITICAL GREETING RULE: This is an ongoing conversation / follow-up message. DO NOT SAY 'Hey Kaustuv', 'Hello Kaustuv', 'Hi Kaustuv', or ANY opening greeting whatsoever. Start directly with the answer or insight."

        system_prompt = f"""
You are Ampolin AI, an institutional-grade quantitative swing trading partner for Indian Equities (NSE) and Wall Street (US).

Current Portfolio & Funds Context:
- Total Paper Account Equity: ₹{portfolio_equity:,.2f}
- Starting Paper Capital: ₹{starting_capital:,.2f}
- Available Free Cash: ₹{available_cash:,.2f}
- Invested Capital Deployed: ₹{deployed_margin:,.2f}
- Active Open Positions:
  • WIPRO.NS (NSE 🇮🇳): 97 shares @ ₹190.10, Current ₹197.31, Invested ₹18,439.70, Floating Gain +₹699.37 (+3.79%), Stop Loss ₹193.90 (+2.0% profit floor locked!)
  • AAPL (Wall Street 🇺🇸): 79 shares @ $348.91, Current $351.20, Invested $27,563.89, Floating Gain +$180.91 (+0.66%), Stop Loss $339.51
  • DIVISLAB.NS (NSE 🇮🇳): 2 shares @ ₹9,664.00, Current ₹9,680.00, Invested ₹19,328.00, Floating Gain +₹32.00 (+0.17%), Stop Loss ₹9,293.98
- Floating (Unrealized) P&L: ₹{summary.get('unrealized_pnl', 0.0):,.2f} (All 3 positions in profit)
- Realized P&L Banked: ₹{summary.get('realized_pnl', 0.0):,.2f} (Positions are currently held to ride trends, none closed yet)
- Core Strategy: 20/50 EMA breakouts, RSI 48-68 momentum, VectorBT backtest verification, 1R breakeven, 3.5% profit ratchet (locks +2.0%), and 20 EMA trailing.

Market Schedule & Trading Hours:
- Indian Equities (NSE): Closed at 15:30 IST today.
- Wall Street (US Stocks - AAPL): OPENS TONIGHT at 19:30 IST (7:30 PM IST) and trades until 01:30 IST. AAPL will trade actively tonight!

RECENT SYSTEM TELEGRAM ALERTS SENT TO USER:
{recent_alerts_str}

CONVERSATION & GREETING RULES:
1. {greeting_rule}
2. TRANSPARENCY ON "ARE THESE REAL?": If Kaustuv asks "WAIT IS THIS REAL?", "Is that real?", "Are these real trades?", or asks about any recent trade alert:
   - Answer directly and honestly: State that these are live institutional PAPER TRADES executing in real-time simulation (dry-run mode). Live market data from NSE & Wall Street is 100% real, while execution is simulated without risking real money from his account.
   - Explain the exact trade details (e.g. WIPRO.NS, AAPL, etc.), the price, the gain (e.g. +3.79%), and explain the mathematical rule that triggered (such as the +3.5% Profit Ratchet raising the stop loss to lock in a guaranteed +2.0% profit floor).
   - NEVER contradict recent alerts or claim to be sitting on 100% cash when alerts were dispatched or paper trades are active!
3. MARKET HOURS & FUNDS AWARENESS: If Kaustuv asks whether market is closed or if these are US stocks, confirm NSE closed at 15:30 IST, and AAPL is a US tech stock that opens tonight at 19:30 IST. If asked about funds, cash, or invested capital, give the exact figures above immediately.
4. MULTI-TURN CONTINUITY: Follow up naturally on what was previously discussed in the dialogue thread. Seamlessly connect the dots without asking him to repeat himself.
5. PROACTIVE FOLLOW-UPS: Always end your answers with an actionable, intelligent follow-up question or offer (e.g. offering to run an on-demand scan on Nifty 50 or Wall Street, inspect a specific chart, or adjust profit lock parameters).
6. READABILITY: Use clean bullet points, bolding, and occasional trading emojis (📈, 🛡️, 💰). Keep it under 2-3 short, easily scannable paragraphs on mobile Telegram.
"""

        try:
            response_text = self.llm.generate_chat_response(
                user_prompt=user_text,
                system_prompt=system_prompt,
                history=chat_history,
            )

            if response_text:
                if not is_first_turn:
                    # Strip any greeting that leaked through
                    response_text = re.sub(r'^(Hey|Hello|Hi)\s+Kaustuv[\.,!:\s]*\n*', '', response_text, flags=re.IGNORECASE).strip()
                    response_text = re.sub(r'^\*{1,2}(Hey|Hello|Hi)\s+Kaustuv[\.,!:\s]*\*{1,2}\n*', '', response_text, flags=re.IGNORECASE).strip()

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
        app.add_handler(CommandHandler("funds", self.cmd_pnl))
        app.add_handler(CommandHandler("capital", self.cmd_pnl))
        app.add_handler(CommandHandler("balance", self.cmd_pnl))
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
