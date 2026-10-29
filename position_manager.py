"""
Ampolin AI Trading Agent - Position Manager Module
Executes Multi-Stage Profit Scaling & Trailing Stops:
- Stage 1: Move Stop Loss to Breakeven at 1.0R profit
- Stage 2: Book 50% partial profit at 2.0R target & attach 20 EMA trailing stop
- Stage 3: Trail remaining 50% position with 20 EMA until technical exit
"""

import logging
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import pandas_ta as ta
import yfinance as yf

from config import (
    STAGE1_BREAKEVEN_R,
    STAGE2_PARTIAL_EXIT_R,
    STAGE2_PARTIAL_EXIT_RATIO,
    STAGE3_TRAILING_EMA,
)
from reflection import TradeLogger
from telegram_notifier import TelegramNotifier
from broker import DhanBroker

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AmpolinPositionMgr")


class PositionManager:
    def __init__(
        self,
        trade_logger: Optional[TradeLogger] = None,
        notifier: Optional[TelegramNotifier] = None,
        broker: Optional[DhanBroker] = None,
    ):
        self.trade_logger = trade_logger or TradeLogger()
        self.notifier = notifier or TelegramNotifier()
        self.broker = broker or DhanBroker(dry_run=True)

    def fetch_live_price_and_ema(self, ticker: str) -> tuple[Optional[float], Optional[float]]:
        """
        Fetches current market price and 20 EMA for a given ticker.
        """
        try:
            df = yf.download(ticker, period="3mo", interval="1d", progress=False, multi_level_index=False)
            if df.empty or len(df) < 20:
                return None, None

            curr_price = float(df["Close"].iloc[-1])
            ema20_series = ta.ema(df["Close"], length=STAGE3_TRAILING_EMA)
            curr_ema20 = float(ema20_series.iloc[-1]) if ema20_series is not None and not ema20_series.empty else curr_price

            return round(curr_price, 2), round(curr_ema20, 2)
        except Exception as e:
            logger.error(f"Error fetching live price for {ticker}: {e}")
            return None, None

    def manage_open_positions(self) -> List[Dict[str, Any]]:
        """
        Inspects all open positions in trades.db and applies multi-stage management rules.
        """
        open_trades = self.trade_logger.get_open_trades()
        if not open_trades:
            logger.info("No open positions found in database to manage.")
            return []

        logger.info(f"Managing {len(open_trades)} active open positions...")
        actions_taken = []

        with self.trade_logger._get_connection() as conn:
            cursor = conn.cursor()

            for trade in open_trades:
                trade_id = trade["id"]
                order_id = trade["order_id"]
                ticker = trade["ticker"]
                entry_price = float(trade["entry_price"])
                stop_loss = float(trade["stop_loss"])
                take_profit = float(trade["take_profit"])
                quantity = int(trade["quantity"])
                initial_qty = int(trade["initial_quantity"] or quantity)
                current_stage = trade["stage"] or "STAGE_INITIAL"
                partial_taken = bool(trade["partial_exit_taken"])

                curr_price, ema20 = self.fetch_live_price_and_ema(ticker)
                if curr_price is None:
                    continue

                # Calculate R-multiple metrics
                initial_risk = abs(entry_price - stop_loss)
                if initial_risk <= 0:
                    initial_risk = entry_price * 0.02

                current_r = (curr_price - entry_price) / initial_risk

                logger.info(
                    f"[{ticker}] Entry: ₹{entry_price} | Price: ₹{curr_price} | R: {current_r:+.2f}R | "
                    f"SL: ₹{stop_loss} | 20EMA: ₹{ema20} | Stage: {current_stage}"
                )

                # Rule 0: Hard Stop Loss Hit
                if curr_price <= stop_loss:
                    self.trade_logger.close_trade(order_id, exit_price=curr_price)
                    actions_taken.append({
                        "ticker": ticker,
                        "action": "STOP_LOSS_EXIT",
                        "exit_price": curr_price,
                        "pnl": round((curr_price - entry_price) * quantity, 2),
                    })
                    msg = (
                        f"🛑 <b>STOP LOSS HIT - {ticker}</b>\n"
                        f"Exit Price: ₹{curr_price} | Entry: ₹{entry_price}\n"
                        f"Realized P&L: ₹{round((curr_price - entry_price) * quantity, 2):,.2f}"
                    )
                    self.notifier.send_message(msg)
                    continue

                # Stage 1: Breakeven Trigger at 1.0R
                if current_r >= STAGE1_BREAKEVEN_R and stop_loss < entry_price and current_stage == "STAGE_INITIAL":
                    new_sl = entry_price  # Risk is now exactly zero!
                    cursor.execute(
                        "UPDATE trades SET stop_loss = ?, stage = 'STAGE_BREAKEVEN' WHERE id = ?",
                        (new_sl, trade_id),
                    )
                    conn.commit()
                    logger.info(f"-> [STAGE 1] Breakeven triggered for {ticker}! SL moved from ₹{stop_loss} to ₹{new_sl}")
                    actions_taken.append({
                        "ticker": ticker,
                        "action": "MOVED_TO_BREAKEVEN",
                        "new_stop_loss": new_sl,
                    })
                    msg = (
                        f"🛡️ <b>BREAKEVEN TRIGGERED (+1.0R) - {ticker}</b>\n"
                        f"Current Price: ₹{curr_price} (+{current_r:.1f}R)\n"
                        f"Protective Stop Loss raised to Entry Price: ₹{new_sl}\n"
                        f"Trade is now 100% risk-free!"
                    )
                    self.notifier.send_message(msg)
                    current_stage = "STAGE_BREAKEVEN"

                # Stage 2: 50% Partial Exit at 2.0R + Attach 20 EMA Trailing Stop
                if current_r >= STAGE2_PARTIAL_EXIT_R and not partial_taken:
                    shares_to_sell = max(1, int(quantity * STAGE2_PARTIAL_EXIT_RATIO))
                    remaining_shares = quantity - shares_to_sell
                    partial_pnl = round((curr_price - entry_price) * shares_to_sell, 2)

                    # Update remaining shares and set trailing EMA stop
                    trailing_sl = max(entry_price, ema20 or entry_price)
                    cursor.execute(
                        """
                        UPDATE trades
                        SET quantity = ?, stop_loss = ?, stage = 'STAGE_PARTIAL_BOOKED', partial_exit_taken = 1
                        WHERE id = ?
                        """,
                        (remaining_shares, trailing_sl, trade_id),
                    )
                    conn.commit()
                    logger.info(
                        f"-> [STAGE 2] 2.0R Target Hit for {ticker}! Sold {shares_to_sell} shares (50%) @ ₹{curr_price}. "
                        f"Booked P&L: ₹{partial_pnl}. Remaining: {remaining_shares} shares trailing 20 EMA (₹{trailing_sl})"
                    )
                    actions_taken.append({
                        "ticker": ticker,
                        "action": "PARTIAL_PROFIT_BOOKED",
                        "shares_sold": shares_to_sell,
                        "booked_pnl": partial_pnl,
                        "remaining_shares": remaining_shares,
                        "trailing_sl": trailing_sl,
                    })
                    msg = (
                        f"🎯 <b>2.0R TARGET ACHIEVED - {ticker}</b>\n"
                        f"Sold 50% ({shares_to_sell} shares) @ ₹{curr_price}\n"
                        f"<b>Realized Profit:</b> ₹{partial_pnl:,.2f}\n"
                        f"Remaining {remaining_shares} shares trailing 20 EMA @ ₹{trailing_sl}"
                    )
                    self.notifier.send_message(msg)
                    current_stage = "STAGE_PARTIAL_BOOKED"

                # Stage 3: Trailing Stop along 20 EMA
                if current_stage == "STAGE_PARTIAL_BOOKED":
                    if ema20 and curr_price < ema20:
                        # Price broke below 20 EMA -> Close remaining half
                        self.trade_logger.close_trade(order_id, exit_price=curr_price)
                        final_pnl = round((curr_price - entry_price) * quantity, 2)
                        actions_taken.append({
                            "ticker": ticker,
                            "action": "TRAILING_EMA_EXIT",
                            "exit_price": curr_price,
                            "pnl": final_pnl,
                        })
                        msg = (
                            f"🏁 <b>20 EMA TRAILING EXIT - {ticker}</b>\n"
                            f"Closed remaining {quantity} shares @ ₹{curr_price} (fell below 20 EMA ₹{ema20})\n"
                            f"Final Leg P&L: ₹{final_pnl:,.2f}"
                        )
                        self.notifier.send_message(msg)
                    elif ema20 and ema20 > stop_loss:
                        # Ratchet trailing stop upward with rising 20 EMA
                        cursor.execute("UPDATE trades SET stop_loss = ? WHERE id = ?", (ema20, trade_id))
                        conn.commit()
                        logger.info(f"-> [STAGE 3] Trailing SL for {ticker} ratcheted up to 20 EMA: ₹{ema20}")

        logger.info(f"Position management sweep complete. {len(actions_taken)} adjustments enacted.")
        return actions_taken


if __name__ == "__main__":
    pm = PositionManager()
    actions = pm.manage_open_positions()
    print("\nPosition Manager Results:", actions)
