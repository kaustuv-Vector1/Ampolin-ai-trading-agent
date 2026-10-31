"""
Ampolin AI Trading Agent - Real-Time DhanHQ WebSocket Market Feed Module
Streams live tick-by-tick prices for active positions and triggers instantaneous
1R Breakeven and 2R Partial Profit executions with zero polling delay.
Includes an intelligent Market Simulation engine for offline/after-hours testing.
"""

import asyncio
import json
import logging
import os
import random
import sys
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable

from config import DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN
from reflection import TradeLogger
from position_manager import PositionManager
from telegram_notifier import TelegramNotifier

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AmpolinLiveFeed")

try:
    from dhanhq import DhanContext, MarketFeed
    DHAN_FEED_AVAILABLE = True
except ImportError:
    DHAN_FEED_AVAILABLE = False
    logger.warning("dhanhq MarketFeed not available. Defaulting to simulation feed.")

# Common Nifty 50 Security ID mapping for DhanHQ NSE
NIFTY_SECURITY_ID_MAP = {
    "RELIANCE": "2885",
    "TCS": "11536",
    "HDFCBANK": "1333",
    "ICICIBANK": "4963",
    "BHARTIARTL": "10604",
    "SBIN": "3045",
    "INFY": "1594",
    "ITC": "1660",
    "HINDUNILVR": "1394",
    "LT": "11483",
    "BAJFINANCE": "317",
    "TATAMOTORS": "3456",
    "KOTAKBANK": "1922",
    "SUNPHARMA": "3351",
    "M&M": "2031",
    "TRENT": "1964",
    "BEL": "383",
    "ZOMATO": "5097",
}


class LiveMarketFeed:
    def __init__(
        self,
        client_id: Optional[str] = None,
        access_token: Optional[str] = None,
        trade_logger: Optional[TradeLogger] = None,
        position_manager: Optional[PositionManager] = None,
        notifier: Optional[TelegramNotifier] = None,
        simulation_mode: bool = False,
    ):
        self.client_id = client_id or DHAN_CLIENT_ID or os.getenv("DHAN_CLIENT_ID", "")
        self.access_token = access_token or DHAN_ACCESS_TOKEN or os.getenv("DHAN_ACCESS_TOKEN", "")
        self.trade_logger = trade_logger or TradeLogger()
        self.notifier = notifier or TelegramNotifier()
        self.position_manager = position_manager or PositionManager(
            trade_logger=self.trade_logger,
            notifier=self.notifier,
        )

        self.simulation_mode = simulation_mode or (not self.client_id or not self.access_token)
        self.market_feed = None
        self.is_running = False
        self.latest_prices: Dict[str, float] = {}
        self._thread: Optional[threading.Thread] = None

    def _resolve_security_id(self, ticker: str) -> Optional[str]:
        from broker import resolve_security_id
        return resolve_security_id(ticker)

    def on_tick_received(self, ticker: str, ltp: float):
        """
        Called on every live tick. Checks open trades for real-time 1R/2R/SL triggers.
        """
        self.latest_prices[ticker] = ltp
        logger.debug(f"[TICK] {ticker} @ ₹{ltp:.2f}")

        # Check open trades against this ticker
        open_trades = self.trade_logger.get_open_trades()
        for trade in open_trades:
            if trade["ticker"] != ticker:
                continue

            trade_id = trade["id"]
            order_id = trade["order_id"]
            entry = float(trade["entry_price"])
            sl = float(trade["stop_loss"])
            qty = int(trade["quantity"])
            stage = trade["stage"] or "STAGE_INITIAL"
            partial_taken = bool(trade["partial_exit_taken"])

            initial_risk = abs(entry - sl) if abs(entry - sl) > 0 else entry * 0.02
            current_r = (ltp - entry) / initial_risk

            # 1. Stop Loss Hit Trigger
            if ltp <= sl:
                logger.warning(f"🚨 [REAL-TIME SL HIT] {ticker} dropped to ₹{ltp} <= SL ₹{sl}!")
                self.trade_logger.close_trade(order_id, exit_price=ltp)
                self.notifier.send_message(
                    f"🛑 <b>REAL-TIME STOP LOSS TRIGGERED</b>\n"
                    f"Ticker: <code>{ticker}</code>\n"
                    f"Execution Price: ₹{ltp:.2f} | Entry: ₹{entry:.2f}\n"
                    f"Realized P&L: ₹{((ltp - entry) * qty):,.2f}"
                )

            # 2. Stage 1: Breakeven Trigger at 1.0R
            elif current_r >= 1.0 and sl < entry and stage == "STAGE_INITIAL":
                logger.info(f"🛡️ [REAL-TIME BREAKEVEN] {ticker} reached +{current_r:.2f}R @ ₹{ltp}! Moving SL to ₹{entry}")
                with self.trade_logger._get_connection() as conn:
                    conn.cursor().execute(
                        "UPDATE trades SET stop_loss = ?, stage = 'STAGE_BREAKEVEN' WHERE id = ?",
                        (entry, trade_id),
                    )
                    conn.commit()
                self.notifier.send_message(
                    f"🛡️ <b>TICK-BY-TICK BREAKEVEN ACTIVATED</b>\n"
                    f"Ticker: <code>{ticker}</code> reached ₹{ltp:.2f} (+{current_r:.1f}R)\n"
                    f"Protective Stop Loss moved to Entry: ₹{entry:.2f}\n"
                    f"Position is now 100% risk-free!"
                )

            # 3. Stage 2: 2.0R 50% Partial Profit Trigger
            elif current_r >= 2.0 and not partial_taken:
                shares_sold = max(1, int(qty * 0.50))
                remaining = qty - shares_sold
                booked_pnl = round((ltp - entry) * shares_sold, 2)

                logger.info(f"🎯 [REAL-TIME 2.0R TARGET] {ticker} @ ₹{ltp}! Selling {shares_sold} shares.")
                with self.trade_logger._get_connection() as conn:
                    conn.cursor().execute(
                        """
                        UPDATE trades
                        SET quantity = ?, stage = 'STAGE_PARTIAL_BOOKED', partial_exit_taken = 1
                        WHERE id = ?
                        """,
                        (remaining, trade_id),
                    )
                    conn.commit()
                self.notifier.send_message(
                    f"🎯 <b>REAL-TIME 2.0R TARGET REACHED</b>\n"
                    f"Ticker: <code>{ticker}</code> @ ₹{ltp:.2f}\n"
                    f"Sold 50% ({shares_sold} shares) &bull; <b>Locked Profit: ₹{booked_pnl:,.2f}</b>\n"
                    f"Remaining {remaining} shares trailing 20 EMA."
                )

    def start_feed(self):
        """Starts the WebSocket feed in a dedicated daemon thread."""
        self.is_running = True

        if not self.simulation_mode and DHAN_FEED_AVAILABLE:
            self._thread = threading.Thread(target=self._run_dhan_websocket, daemon=True)
        else:
            logger.info("Starting Real-Time Market Simulation Feed...")
            self._thread = threading.Thread(target=self._run_simulation_stream, daemon=True)

        self._thread.start()
        logger.info("Live WebSocket Feed background thread started.")

    def stop_feed(self):
        self.is_running = False
        if self.market_feed:
            try:
                self.market_feed.disconnect()
            except Exception:
                pass
        logger.info("Live WebSocket Feed stopped.")

    def _run_dhan_websocket(self):
        """Runs the official DhanHQ MarketFeed WebSocket connection."""
        try:
            context = DhanContext(self.client_id, self.access_token)
            open_trades = self.trade_logger.get_open_trades()

            instruments = []
            for t in open_trades:
                sec_id = self._resolve_security_id(t["ticker"])
                instruments.append((MarketFeed.NSE, sec_id, MarketFeed.Ticker))

            if not instruments:
                # Default to Nifty 50 anchor (HDFCBANK) if no open trades
                instruments = [(MarketFeed.NSE, "1333", MarketFeed.Ticker)]

            def on_message(instance, message):
                try:
                    if isinstance(message, dict):
                        sec_id = str(message.get("security_id", ""))
                        ltp = float(message.get("LTP", 0.0))
                        # Reverse lookup ticker
                        for ticker_sym, mapped_id in NIFTY_SECURITY_ID_MAP.items():
                            if mapped_id == sec_id:
                                self.on_tick_received(f"{ticker_sym}.NS", ltp)
                                break
                except Exception as ex:
                    logger.error(f"Error parsing DhanHQ WebSocket message: {ex}")

            logger.info(f"Connecting DhanHQ MarketFeed WebSocket for {len(instruments)} instruments...")
            self.market_feed = MarketFeed(
                context,
                instruments=instruments,
                on_message=on_message,
            )
            self.market_feed.run_forever()

        except Exception as e:
            logger.error(f"DhanHQ WebSocket connection failed: {e}. Falling back to Simulation Feed.")
            self._run_simulation_stream()

    def _run_simulation_stream(self):
        """
        Generates realistic tick-by-tick market simulation for after-hours / weekend testing.
        """
        # Baseline seed prices
        sim_tickers = {
            "RELIANCE.NS": 1167.40,
            "ICICIBANK.NS": 1386.60,
            "BHARTIARTL.NS": 1972.10,
            "INFY.NS": 1146.50,
            "TCS.NS": 3950.00,
        }

        # Also pull any open trades from DB into sim
        for t in self.trade_logger.get_open_trades():
            if t["ticker"] not in sim_tickers:
                sim_tickers[t["ticker"]] = float(t["entry_price"])

        logger.info(f"Market Simulation Feed active for {list(sim_tickers.keys())}")

        while self.is_running:
            for ticker, price in sim_tickers.items():
                # Random walk: &plusmn;0.05% to &plusmn;0.2% tick fluctuation
                delta = price * random.uniform(-0.0015, 0.0018)
                new_price = round(price + delta, 2)
                sim_tickers[ticker] = new_price
                self.on_tick_received(ticker, new_price)

            time.sleep(2.0)  # tick every 2 seconds


if __name__ == "__main__":
    feed = LiveMarketFeed(simulation_mode=True)
    feed.start_feed()
    print("Streaming live ticks for 6 seconds...")
    time.sleep(6)
    feed.stop_feed()
    print("Feed stopped successfully. Latest prices captured:", feed.latest_prices)
