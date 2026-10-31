"""
Ampolin AI Trading Agent - Autonomous Cloud Scheduler
Executes automated trading workflows aligned with National Stock Exchange of India (NSE) market hours:
- 09:15 IST: Market Open Position Sweep & Macro Health Check
- 10:30 IST: Morning Swing Scan (Nifty 50 Breakouts)
- 13:45 IST: Afternoon Swing Scan (Late-Session Breakouts)
- 15:10 IST: Pre-Close Trailing Stop & Breakeven Sweep (1R / 2R scaling)
- 15:45 IST: End-of-Day (EOD) Reflection, Post-Mortem, & Telegram Daily P&L
"""

import logging
import sys
import time
from datetime import datetime
import pytz
import schedule

from main import AmpolinPipeline
from market_regime import MarketRegimeAnalyzer
from position_manager import PositionManager
from telegram_notifier import TelegramNotifier
from reflection import TradeLogger, EODReflection
from market_hours import is_us_market_open, is_nse_market_open

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AmpolinCloud")

IST = pytz.timezone("Asia/Kolkata")


class CloudTradingDaemon:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.pipeline = AmpolinPipeline(dry_run=self.dry_run)
        self.notifier = TelegramNotifier()
        self.logger = TradeLogger()
        self.position_mgr = PositionManager(trade_logger=self.logger, notifier=self.notifier)
        self.reflection = EODReflection(self.logger)

    def is_market_day(self) -> bool:
        """Returns True if today is Monday to Friday."""
        return is_nse_market_open() or is_us_market_open()

    def is_nse_market_open(self) -> bool:
        """Returns True if current time is within NSE trading hours (09:15 - 15:30 IST) excluding holidays."""
        return is_nse_market_open()

    def is_us_market_open(self) -> bool:
        """Returns True if current time is within Wall Street hours (09:30 - 16:00 ET) excluding holidays."""
        return is_us_market_open()

    def periodic_market_check(self):
        """Runs every 15 minutes: sweeps open positions and executes scans during active market sessions."""
        if self.is_nse_market_open():
            logger.info("--- [AUTONOMOUS 15-MIN CYCLE] NSE MARKET ACTIVE ---")
            self.position_mgr.manage_open_positions()
            self.scan_cycle("INTRADAY NSE AUTONOMOUS SCAN")
        elif self.is_us_market_open():
            logger.info("--- [AUTONOMOUS 15-MIN CYCLE] WALL STREET MARKET ACTIVE ---")
            self.position_mgr.manage_open_positions()
            self.scan_us_cycle("INTRADAY US AUTONOMOUS SCAN")
        else:
            self.position_mgr.manage_open_positions()

    def morning_check(self):
        if not self.is_market_day():
            logger.info("Market is closed today (Weekend). Skipping morning check.")
            return

        logger.info("--- [09:15 IST] EXECUTING MORNING MARKET OPEN SWEEP ---")
        actions = self.position_mgr.manage_open_positions()
        summary = self.logger.calculate_daily_summary()
        self.notifier.send_message(
            f"🔔 <b>NSE MARKET OPEN (09:15 IST)</b>\n"
            f"Ampolin Cloud Daemon active.\n"
            f"Open Positions: {summary.get('open_positions', 0)}\n"
            f"Position adjustments: {len(actions)}"
        )

    def scan_cycle(self, session_label: str):
        if not self.is_nse_market_open():
            return

        logger.info(f"--- [{session_label}] EXECUTING NIFTY 50 AUTONOMOUS SCAN ---")
        result = self.pipeline.run_pipeline(market="NSE", max_trades_to_execute=2)
        logger.info(f"Scan cycle complete: {result.get('status')} | Trades: {len(result.get('executed_trades', []))}")

    def scan_us_cycle(self, session_label: str):
        if not self.is_us_market_open():
            return

        logger.info(f"--- [{session_label}] EXECUTING WALL STREET (US) AUTONOMOUS SCAN ---")
        result = self.pipeline.run_pipeline(market="US", max_trades_to_execute=2)
        logger.info(f"Wall Street scan cycle complete: {result.get('status')} | Trades: {len(result.get('executed_trades', []))}")

    def pre_close_sweep(self):
        if not self.is_market_day():
            return

        logger.info("--- [POSITION SWEEP] EXECUTING TRAILING SL & PROFIT LOCK SWEEP ---")
        actions = self.position_mgr.manage_open_positions()
        logger.info(f"Position sweep completed with {len(actions)} adjustments.")

    def eod_cycle(self):
        if not self.is_market_day():
            return

        logger.info("--- [15:45 IST] EXECUTING EOD POST-MORTEM & ADAPTIVE RULES ---")
        post_mortem = self.reflection.run_post_mortem()
        summary = self.logger.calculate_daily_summary()

        self.notifier.notify_eod_reflection(post_mortem)
        self.notifier.notify_daily_pnl(summary)
        logger.info("EOD Cycle complete. Daily report dispatched.")

    def run_scheduler(self):
        logger.info("Starting Ampolin Cloud Autonomous Trading Daemon (Schedule: IST)...")

        # Indian Equities (NSE) Schedule in IST
        schedule.every().day.at("09:15", "Asia/Kolkata").do(self.morning_check)
        schedule.every().day.at("10:30", "Asia/Kolkata").do(lambda: self.scan_cycle("10:30 IST MORNING"))
        schedule.every().day.at("13:45", "Asia/Kolkata").do(lambda: self.scan_cycle("13:45 IST AFTERNOON"))
        schedule.every().day.at("15:10", "Asia/Kolkata").do(self.pre_close_sweep)
        schedule.every().day.at("15:45", "Asia/Kolkata").do(self.eod_cycle)

        # Wall Street (US Equities) Schedule in America/New_York (Automatic DST adjustment)
        schedule.every().day.at("09:35", "America/New_York").do(lambda: self.scan_us_cycle("09:35 ET US OPEN"))
        schedule.every().day.at("12:30", "America/New_York").do(lambda: self.scan_us_cycle("12:30 ET US MID-DAY"))
        schedule.every().day.at("15:45", "America/New_York").do(self.pre_close_sweep)

        # Recurring Autonomous Monitoring: sweeps positions & auto-executes setups every 15 minutes during active sessions
        schedule.every(15).minutes.do(self.periodic_market_check)

        # Launch interactive 2-way Telegram chat listener in background thread
        try:
            from telegram_bot import InteractiveTelegramBot
            tg_bot = InteractiveTelegramBot()
            tg_bot.start_polling_in_background()
            logger.info("Interactive 2-Way Telegram Chat Bot initialized and polling.")
        except Exception as e:
            logger.warning(f"Could not start Telegram chat bot: {e}")

        self.notifier.send_message(
            "☁️ <b>AMPOLIN CLOUD DAEMON ACTIVE (24/7 DUAL-MARKET)</b>\n\n"
            "🇮🇳 <b>National Stock Exchange (NSE India - IST):</b>\n"
            "• 09:15 IST: Open Sweep & Health Check\n"
            "• 10:30 IST: Morning Swing Scan (Nifty 50)\n"
            "• 13:45 IST: Afternoon Swing Scan\n"
            "• 15:10 IST: Pre-Close 3.5% Profit Lock Sweep\n"
            "• 15:45 IST: EOD Reflection & Daily P&L\n\n"
            "🇺🇸 <b>Wall Street (NYSE/NASDAQ - America/New_York):</b>\n"
            "• 09:35 ET: Wall Street Open Scan (NVDA, AAPL...)\n"
            "• 12:30 ET: US Mid-Session Momentum Scan\n"
            "• 15:45 ET: US Pre-Close Trailing SL Sweep\n\n"
            "💬 <b>2-Way AI Chat Online:</b> Message me anytime or type <b>/help</b>, <b>/positions</b>, or <b>/scan</b>!"
        )

        while True:
            schedule.run_pending()
            time.sleep(10)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ampolin Autonomous Trading Daemon")
    parser.add_argument("--live", action="store_true", help="Run with live broker orders instead of paper trading")
    parser.add_argument("--test-sweep", action="store_true", help="Run a one-shot position sweep test and exit")
    args = parser.parse_args()

    is_dry_run = not args.live
    daemon = CloudTradingDaemon(dry_run=is_dry_run)

    if args.test_sweep:
        print(f"Testing one-shot position sweep (Dry-Run: {is_dry_run})...")
        daemon.pre_close_sweep()
    else:
        print(f"Starting autonomous daily schedule (Dry-Run: {is_dry_run})...")
        daemon.run_scheduler()
