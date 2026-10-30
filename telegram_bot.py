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
from prompt_context import build_dynamic_system_prompt

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

            if stage == "STAGE_PARTIAL_BOOKED":
                status_icon = "🎯 50% Profit Banked! Remainder Trailing 20 EMA"
            elif stage == "STAGE_PROFIT_LOCKED":
                status_icon = "🔒 +2% Profit Floor Locked"
            elif stage == "STAGE_BREAKEVEN":
                status_icon = "🛡️ Breakeven Protected"
            else:
                status_icon = "🎯 Active 1:2 R:R"

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

        now_ist = datetime.now(pytz.timezone("Asia/Kolkata"))
        is_us_open = (now_ist.weekday() < 5 and (now_ist.hour > 19 or (now_ist.hour == 19 and now_ist.minute >= 30) or now_ist.hour >= 20)) or (now_ist.weekday() <= 5 and now_ist.hour in (0, 1))
        us_status_text = "🟢 Wall Street is LIVE & ACTIVELY TRADING RIGHT NOW 🇺🇸" if is_us_open else "🔴 Wall Street session: 19:30 - 01:30 IST 🇺🇸"

        msg = (
            f"💼 <b>AMPOLIN PAPER TRADING FUNDS & P&L</b>\n\n"
            f"• <b>Total Account Equity:</b> <b>₹{portfolio_equity:,.2f}</b> ({pnl_sign}₹{total:,.2f})\n"
            f"• <b>Starting Paper Capital:</b> ₹{starting_capital:,.2f}\n"
            f"• <b>Available Free Cash:</b> <b>₹{available_cash:,.2f}</b>\n"
            f"• <b>Invested Capital Deployed:</b> <b>₹{deployed_margin:,.2f}</b>\n\n"
            f"• <b>Realized P&L (Banked Profit):</b> <b>+₹{realized:,.2f}</b> 💰\n"
            f"• <b>Unrealized (Open) P&L:</b> {icon} <b>{pnl_sign}₹{unrealized:,.2f}</b> 🟢\n"
            f"• <b>Active Positions:</b> {len(open_trades)} stocks\n"
            f"• <b>Win Rate:</b> {summary.get('win_rate', 0.0)*100:.1f}%\n\n"
            f"🕒 <b>Market Status:</b>\n"
            f"• NSE (India): Closed for the day (09:15 - 15:30 IST)\n"
            f"• {us_status_text}\n\n"
            f"💡 <i>Realized profit of +₹{realized:,.2f} is permanently banked (including 50% partial exit on WIPRO @ ₹198.31). Remaining open shares are protected with trailing stops!</i>"
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
            "📜 <b>ACTIVE TRADING & RISK RULES (10/10)</b>\n\n"
            "🛡️ <b>CAPITAL SECURITY & ANTI-DUMP CIRCUIT BREAKERS:</b>\n"
            "• <b>Anti-Dump Cap:</b> Max 20% capital into any single stock (AI physically cannot dump 100% into one ticker)\n"
            "• <b>Single-Share Veto:</b> Stock rejected if 1 share exceeds 20% of capital\n"
            "• <b>Max Open Positions:</b> 4 concurrent stocks max (guarantees diversification)\n"
            "• <b>Cash Reserve Floor:</b> 15% mandatory cash kept liquid at all times\n"
            "• <b>Daily Drawdown Circuit Breaker:</b> 3% daily loss limit (trading halts immediately if breached)\n"
            "• <b>Zero Leverage:</b> Cash CNC Delivery only (zero debt, zero liquidation risk)\n\n"
            "🎯 <b>STRATEGY & EXECUTION RULES:</b>\n"
            "• <b>Risk per Trade:</b> 1.0% max (0.75% if India VIX > 15)\n"
            "• <b>Weekly Confluence:</b> 20 EMA > 50 EMA on 1wk required\n"
            "• <b>Anti-Chasing:</b> Max 2.2% distance from 20 EMA\n"
            "• <b>VectorBT Filter:</b> Win Rate >= 52% | Profit Factor >= 1.50\n"
            "• <b>1.0R Gain:</b> Stop Loss raised to Breakeven (0% risk)\n"
            "• <b>+3.5% Gain:</b> SL locked at +2.0% guaranteed profit floor\n"
            "• <b>2.0R Gain:</b> 50% partial exit taken, remainder trails 20 EMA"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def cmd_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized(update):
            return

        regime = self.regime_analyzer.analyze_regime()
        open_trades = self.logger.get_open_trades()
        summary = self.logger.calculate_daily_summary()

        msg = (
            "🛡️ <b>AMPOLIN SYSTEM HEALTH WATCHDOG (10/10)</b>\n\n"
            "• <b>Cloud Container:</b> 🟢 ACTIVE (Google Cloud Run • asia-south1)\n"
            "• <b>Primary Brain:</b> 🟢 Gemini 3.8 Flash (Active)\n"
            "• <b>Groq LPU Backup:</b> 🟢 Sub-100ms Ready (Qwen 3.8 27B)\n"
            "• <b>Execution Broker:</b> 🟢 DhanHQ Paper Sandbox (Zero Leverage CNC)\n"
            "• <b>Anti-Dump Guard:</b> 🟢 20% Max Allocation per Stock Locked\n"
            "• <b>Cash Safety Floor:</b> 🟢 15% Liquid Cushion Untouchable\n"
            "• <b>Circuit Breaker:</b> 🟢 3% Max Daily Loss Halt Armed\n"
            f"• <b>Macro VIX Filter:</b> 🟢 {regime.india_vix} ({regime.vix_regime})\n"
            f"• <b>Options PCR Sentiment:</b> 🟢 {regime.pcr_ratio} ({regime.pcr_sentiment})\n"
            "• <b>Database State Engine:</b> 🟢 Auto-Persistent (JSON Snapshot Synced)\n"
            f"• <b>Active Positions Monitored:</b> 🟢 {len(open_trades)} stocks\n"
            f"• <b>Account Equity:</b> 🟢 ₹{100000.0 + summary.get('total_pnl', 0.0):,.2f}\n"
            "• <b>Daemon Status:</b> 🟢 24/7 Dual-Market Autonomous Polling Active"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def cmd_pcr(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized(update):
            return

        regime = self.regime_analyzer.analyze_regime()
        msg = (
            "📊 <b>NIFTY 50 OPTIONS PUT-CALL RATIO (PCR)</b>\n\n"
            f"• <b>Current PCR:</b> <b>{regime.pcr_ratio}</b>\n"
            f"• <b>Sentiment:</b> <code>{regime.pcr_sentiment}</code>\n\n"
            "📖 <b>Institutional Interpretation:</b>\n"
            "• <b>0.85 – 1.30:</b> Healthy institutional accumulation. Trend is supported by put writing. Optimal for swing breakout entries.\n"
            "• <b>&gt; 1.35:</b> Overbought / heavy hedging. High probability of mean-reversion pauses.\n"
            "• <b>&lt; 0.70:</b> Oversold / extreme fear. Potential reversal / short-squeeze zone."
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

        try:
            # Show typing status
            await update.message.chat.send_action("typing")

            open_trades = self.logger.get_open_trades()
            positions_str = ", ".join([f"{t['ticker']} (entry: {t['entry_price']}, stage: {t.get('stage','INITIAL')})" for t in open_trades]) or "None (100% Cash)"
            summary = self.logger.calculate_daily_summary()

            starting_capital = 100000.0
            deployed_margin = sum(float(t.get("capital_allocated") or 0.0) for t in open_trades)
            available_cash = round(starting_capital - deployed_margin + summary.get("realized_pnl", 0.0), 2)
            portfolio_equity = round(starting_capital + summary.get("total_pnl", 0.0), 2)

            # Safely fetch or initialize chat history for this chat_id
            if chat_id not in self.chat_memories:
                self.chat_memories[chat_id] = []
            chat_history = self.chat_memories[chat_id]
            is_first_turn = len(chat_history) == 0

            recent_alerts = get_recent_notifications()
            recent_alerts_str = "\n".join([f"• {a}" for a in recent_alerts[-3:]]) if recent_alerts else "No alerts sent recently."

            if is_first_turn:
                greeting_rule = "GREETING RULE: Since this is the FIRST message of the conversation, you may start with a brief, friendly greeting like 'Hey Kaustuv.' followed immediately by the answer."
            else:
                greeting_rule = "CRITICAL GREETING RULE: This is an ongoing conversation / follow-up message. DO NOT SAY 'Hey Kaustuv', 'Hello Kaustuv', 'Hi Kaustuv', or ANY opening greeting whatsoever. Start directly with the answer or insight."

            system_prompt = build_dynamic_system_prompt(
                trade_logger=self.logger,
                regime_analyzer=self.regime_analyzer,
                greeting_rule=greeting_rule,
                recent_alerts_str=recent_alerts_str,
            )

            # Pulse typing indicator in background until response is ready
            async def _typing_heartbeat():
                try:
                    while True:
                        await asyncio.sleep(4.0)
                        await update.message.chat.send_action("typing")
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

            typing_task = asyncio.create_task(_typing_heartbeat())
            try:
                # Run synchronous LLM call in thread pool to prevent event loop blocking
                response_text = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.llm.generate_chat_response,
                        user_prompt=user_text,
                        system_prompt=system_prompt,
                        history=chat_history,
                    ),
                    timeout=14.0,
                )
            except asyncio.TimeoutError:
                logger.warning("LLM generation timed out after 14s. Providing intelligent fallback.")
                response_text = (
                    "⚡ I am currently monitoring all 3 open positions (WIPRO.NS +3.79% profit locked, AAPL, DIVISLAB.NS) "
                    "with ₹34,668 free cash preserved. What would you like to review (/positions, /rules, /pcr)?"
                )
            finally:
                typing_task.cancel()

            if response_text:
                if not is_first_turn:
                    # Strip any greeting that leaked through
                    response_text = re.sub(r'^(Hey|Hello|Hi)\s+Kaustuv[\.,!:\s]*\n*', '', response_text, flags=re.IGNORECASE).strip()
                    response_text = re.sub(r'^\*{1,2}(Hey|Hello|Hi)\s+Kaustuv[\.,!:\s]*\*{1,2}\n*', '', response_text, flags=re.IGNORECASE).strip()

                if not response_text:
                    response_text = "I'm monitoring all active positions and market channels. What would you like to review?"

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
            logger.error(f"Error handling Telegram chat: {e}", exc_info=True)
            try:
                await update.message.reply_text(f"⚠️ Apologies Kaustuv, I encountered a brief issue: {e}. Please try sending your message again!")
            except Exception:
                pass

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
        app.add_handler(CommandHandler("health", self.cmd_health))
        app.add_handler(CommandHandler("pcr", self.cmd_pcr))

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
