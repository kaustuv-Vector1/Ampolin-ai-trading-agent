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
        now_ist = datetime.now(IST)
        return now_ist.weekday() < 5  # Mon=0, Fri=4

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
        if not self.is_market_day():
            return

        logger.info(f"--- [{session_label}] EXECUTING NIFTY 50 AUTONOMOUS SCAN ---")
        result = self.pipeline.run_pipeline(max_trades_to_execute=2)
        logger.info(f"Scan cycle complete: {result.get('status')} | Trades: {len(result.get('executed_trades', []))}")

    def pre_close_sweep(self):
        if not self.is_market_day():
            return

        logger.info("--- [15:10 IST] EXECUTING PRE-CLOSE POSITION SWEEP ---")
        actions = self.position_mgr.manage_open_positions()
        logger.info(f"Pre-close sweep completed with {len(actions)} adjustments.")

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

        # Schedule daily events in IST
        schedule.every().day.at("09:15").do(self.morning_check)
        schedule.every().day.at("10:30").do(lambda: self.scan_cycle("10:30 IST MORNING"))
        schedule.every().day.at("13:45").do(lambda: self.scan_cycle("13:45 IST AFTERNOON"))
        schedule.every().day.at("15:10").do(self.pre_close_sweep)
        schedule.every().day.at("15:45").do(self.eod_cycle)

        self.notifier.send_message(
            "☁️ <b>AMPOLIN CLOUD DAEMON STARTED</b>\n"
            "Autonomous schedule active for Indian Equities (NSE):\n"
            "• 09:15 IST: Open Sweep\n"
            "• 10:30 IST: Morning Scan\n"
            "• 13:45 IST: Afternoon Scan\n"
            "• 15:10 IST: Pre-Close Trailing SL Check\n"
            "• 15:45 IST: EOD Reflection & Daily P&L"
        )

        while True:
            schedule.run_pending()
            time.sleep(10)


if __name__ == "__main__":
    daemon = CloudTradingDaemon(dry_run=True)
    # Quick verification that daemon methods can be called
    print("CloudTradingDaemon loaded successfully. Testing one-shot position sweep...")
    daemon.pre_close_sweep()
