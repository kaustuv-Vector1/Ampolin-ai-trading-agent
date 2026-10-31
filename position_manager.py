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
    PROFIT_TRIGGER_PCT,
    PROFIT_LOCK_PCT,
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

                # Calculate R-multiple metrics using stored one_r (never recalculating from moved stop loss)
                one_r = float(trade.get("one_r") or 0.0)
                if one_r <= 0:
                    tp = float(trade.get("take_profit") or 0.0)
                    if tp > entry_price:
                        one_r = round((tp - entry_price) / 2.0, 4)
                    else:
                        one_r = round(abs(entry_price - stop_loss), 4)
                    if one_r <= 0:
                        one_r = round(entry_price * 0.02, 4)

                current_r = (curr_price - entry_price) / one_r

                logger.info(
                    f"[{ticker}] Entry: ₹{entry_price} | Price: ₹{curr_price} | R: {current_r:+.2f}R (1R: ₹{one_r:.2f}) | "
                    f"SL: ₹{stop_loss} | 20EMA: ₹{ema20} | Stage: {current_stage}"
                )

                pct_gain = ((curr_price - entry_price) / entry_price) * 100.0
                floating_pnl = round((curr_price - entry_price) * quantity, 2)

                # Persist live mark-to-market price, floating P&L, and one_r to database
                cursor.execute(
                    "UPDATE trades SET current_price = ?, pnl = ?, pnl_percentage = ?, one_r = ? WHERE id = ?",
                    (curr_price, floating_pnl, round(pct_gain, 2), one_r, trade_id),
                )
                conn.commit()

                # Rule 0: Hard Stop Loss or Trailing Floor Hit
                if curr_price <= stop_loss:
                    # Cancel broker stop-loss to avoid double selling
                    if trade.get("broker_sl_order_id"):
                        self.broker.cancel_broker_stop_loss(trade["broker_sl_order_id"])
                    self.trade_logger.close_trade(order_id, exit_price=curr_price)
                    realized_pnl = round((curr_price - entry_price) * quantity, 2)
                    is_profit_exit = curr_price >= entry_price
                    action_type = "PROFIT_FLOOR_EXIT" if is_profit_exit else "STOP_LOSS_EXIT"
                    logger.info(
                        f"[STAGE TRANSITION] {ticker}: {current_stage} -> CLOSED | Exit: ₹{curr_price} | "
                        f"P&L: ₹{realized_pnl} | R: {current_r:+.2f}R | Trigger: Stop Loss ₹{stop_loss} reached ({action_type})"
                    )
                    actions_taken.append({
                        "ticker": ticker,
                        "action": action_type,
                        "exit_price": curr_price,
                        "pnl": realized_pnl,
                    })
                    if is_profit_exit:
                        msg = (
                            f"💰 <b>PROFIT FLOOR SECURED - {ticker}</b>\n"
                            f"Exit Price: ₹{curr_price} | Entry: ₹{entry_price}\n"
                            f"<b>Realized Profit:</b> ₹{realized_pnl:,.2f} (+{pct_gain:.2f}%)"
                        )
                    else:
                        msg = (
                            f"🛑 <b>STOP LOSS HIT - {ticker}</b>\n"
                            f"Exit Price: ₹{curr_price} | Entry: ₹{entry_price}\n"
                            f"Realized P&L: ₹{realized_pnl:,.2f}"
                        )
                    self.notifier.send_message(msg)
                    continue

                # Rule 0.5 (FIX 4): Profit Ratchet (+3.5% gain -> lock in +2.0% stop floor)
                # Must run for positions in ANY stage, and stops may ONLY move up, never down.
                locked_sl = round(entry_price * (1.0 + (PROFIT_LOCK_PCT / 100.0)), 2)
                if pct_gain >= PROFIT_TRIGGER_PCT and locked_sl > stop_loss:
                    new_stage = "STAGE_PROFIT_LOCKED" if current_stage in ("STAGE_INITIAL", "STAGE_BREAKEVEN") else current_stage
                    cursor.execute(
                        "UPDATE trades SET stop_loss = ?, stage = ? WHERE id = ?",
                        (locked_sl, new_stage, trade_id),
                    )
                    conn.commit()
                    old_sl = trade.get("broker_sl_order_id")
                    sl_id = self.broker.place_broker_stop_loss(order_id, ticker, quantity, locked_sl, previous_sl_order_id=old_sl)
                    if sl_id:
                        self.trade_logger.update_broker_sl(order_id, sl_id, locked_sl, quantity)
                    logger.info(
                        f"[STAGE TRANSITION] {ticker}: Stop Loss ratcheted to ₹{locked_sl} (+{PROFIT_LOCK_PCT}% floor) | "
                        f"Stage: {current_stage} -> {new_stage} | R-Multiple: {current_r:+.2f}R | "
                        f"Trigger: Gain +{pct_gain:.2f}% >= +{PROFIT_TRIGGER_PCT}% (Broker SL: {sl_id})"
                    )
                    actions_taken.append({
                        "ticker": ticker,
                        "action": "PROFIT_LOCKED_2PCT",
                        "new_stop_loss": locked_sl,
                        "current_gain_pct": pct_gain,
                        "broker_sl_id": sl_id,
                    })
                    msg = (
                        f"🔒 <b>PROFIT LOCKED IN (+{PROFIT_LOCK_PCT}% GUARANTEED) - {ticker}</b>\n"
                        f"Current Price: ₹{curr_price} (<b>+{pct_gain:.2f}%</b> | {current_r:+.2f}R)\n"
                        f"Protective Stop Loss raised to ₹{locked_sl} (+{PROFIT_LOCK_PCT}% profit floor).\n"
                        f"If price retraces back to ₹{locked_sl}, it sells automatically to guarantee +{PROFIT_LOCK_PCT}% profit!"
                    )
                    self.notifier.send_message(msg)
                    stop_loss = locked_sl
                    current_stage = new_stage

                # Stage 1: Breakeven Trigger at 1.0R
                if current_r >= STAGE1_BREAKEVEN_R and stop_loss < entry_price and current_stage == "STAGE_INITIAL":
                    new_sl = entry_price  # Risk is now exactly zero!
                    cursor.execute(
                        "UPDATE trades SET stop_loss = ?, stage = 'STAGE_BREAKEVEN' WHERE id = ?",
                        (new_sl, trade_id),
                    )
                    conn.commit()
                    old_sl = trade.get("broker_sl_order_id")
                    sl_id = self.broker.place_broker_stop_loss(order_id, ticker, quantity, new_sl, previous_sl_order_id=old_sl)
                    if sl_id:
                        self.trade_logger.update_broker_sl(order_id, sl_id, new_sl, quantity)
                    logger.info(
                        f"[STAGE TRANSITION] {ticker}: STAGE_INITIAL -> STAGE_BREAKEVEN | "
                        f"SL moved from ₹{stop_loss} to ₹{new_sl} | R-Multiple: {current_r:+.2f}R >= {STAGE1_BREAKEVEN_R}R | "
                        f"Trigger: 1.0R target reached (Broker SL: {sl_id})"
                    )
                    actions_taken.append({
                        "ticker": ticker,
                        "action": "MOVED_TO_BREAKEVEN",
                        "new_stop_loss": new_sl,
                        "broker_sl_id": sl_id,
                    })
                    msg = (
                        f"🛡️ <b>BREAKEVEN TRIGGERED (+1.0R) - {ticker}</b>\n"
                        f"Current Price: ₹{curr_price} (+{current_r:.1f}R)\n"
                        f"Protective Stop Loss raised to Entry Price: ₹{new_sl}\n"
                        f"Trade is now 100% risk-free!"
                    )
                    self.notifier.send_message(msg)
                    stop_loss = new_sl
                    current_stage = "STAGE_BREAKEVEN"

                # Stage 2 (FIX 5): 50% Partial Exit at 2.0R (true initial risk) + Attach 20 EMA Trailing Stop
                if current_r >= STAGE2_PARTIAL_EXIT_R and not partial_taken:
                    shares_to_sell = max(1, int(quantity * STAGE2_PARTIAL_EXIT_RATIO))
                    remaining_shares = quantity - shares_to_sell
                    partial_pnl = round((curr_price - entry_price) * shares_to_sell, 2)
                    trailing_sl = max(entry_price, ema20 or entry_price)
                    if trailing_sl < stop_loss:
                        trailing_sl = stop_loss

                    # Atomically update DB, create closed partial trade record, and persist snapshot
                    self.trade_logger.record_partial_exit(
                        order_id=order_id,
                        shares_sold=shares_to_sell,
                        exit_price=curr_price,
                        booked_pnl=partial_pnl,
                        remaining_shares=remaining_shares,
                        trailing_sl=trailing_sl,
                    )
                    old_sl = trade.get("broker_sl_order_id")
                    sl_id = self.broker.place_broker_stop_loss(order_id, ticker, remaining_shares, trailing_sl, previous_sl_order_id=old_sl)
                    if sl_id:
                        self.trade_logger.update_broker_sl(order_id, sl_id, trailing_sl, remaining_shares)
                    logger.info(
                        f"[STAGE TRANSITION] {ticker}: {current_stage} -> STAGE_PARTIAL_BOOKED | "
                        f"Sold {shares_to_sell} shares @ ₹{curr_price} | Booked P&L: ₹{partial_pnl} | "
                        f"Remaining: {remaining_shares} shares trailing 20 EMA (₹{trailing_sl}) | "
                        f"R-Multiple: {current_r:+.2f}R >= {STAGE2_PARTIAL_EXIT_R}R | "
                        f"Trigger: True 2.0R target hit (Broker SL: {sl_id})"
                    )
                    actions_taken.append({
                        "ticker": ticker,
                        "action": "PARTIAL_PROFIT_BOOKED",
                        "shares_sold": shares_to_sell,
                        "booked_pnl": partial_pnl,
                        "remaining_shares": remaining_shares,
                        "trailing_sl": trailing_sl,
                        "broker_sl_id": sl_id,
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
                        # Cancel broker stop-loss to avoid double selling
                        if trade.get("broker_sl_order_id"):
                            self.broker.cancel_broker_stop_loss(trade["broker_sl_order_id"])
                        self.trade_logger.close_trade(order_id, exit_price=curr_price)
                        final_pnl = round((curr_price - entry_price) * quantity, 2)
                        logger.info(
                            f"[STAGE TRANSITION] {ticker}: STAGE_PARTIAL_BOOKED -> CLOSED | "
                            f"Closed remaining {quantity} shares @ ₹{curr_price} | P&L: ₹{final_pnl} | "
                            f"Trigger: Price fell below 20 EMA (₹{ema20})"
                        )
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
                        old_sl = trade.get("broker_sl_order_id")
                        sl_id = self.broker.place_broker_stop_loss(order_id, ticker, quantity, ema20, previous_sl_order_id=old_sl)
                        if sl_id:
                            self.trade_logger.update_broker_sl(order_id, sl_id, ema20, quantity)
                        logger.info(
                            f"[STAGE TRANSITION] {ticker}: Trailing SL ratcheted from ₹{stop_loss} to 20 EMA ₹{ema20} | "
                            f"R-Multiple: {current_r:+.2f}R | Trigger: Rising 20 EMA (Broker SL: {sl_id})"
                        )

        logger.info(f"Position management sweep complete. {len(actions_taken)} adjustments enacted.")
        if actions_taken:
            try:
                self.trade_logger.calculate_daily_summary()
                self.trade_logger.backup_mgr.create_snapshot()
            except Exception:
                pass
        return actions_taken


if __name__ == "__main__":
    pm = PositionManager()
    actions = pm.manage_open_positions()
    print("\nPosition Manager Results:", actions)
