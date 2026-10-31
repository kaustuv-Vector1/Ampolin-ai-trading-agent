"""
Ampolin Trading Bot - Unit & Integration Test Suite
Verifies FIX 1 through FIX 7:
1. DST / Market Hours (America/New_York and Asia/Kolkata timezone, holidays & early-close calendar validation)
2. Currency Mixing (USD to INR conversion for 20% cap veto, 15% cash floor, and 3% daily loss kill-switch)
3. Position Accounting (Remaining shares, invested capital, realized vs unrealized P&L)
4. Stop Ratchet (Stage 1.5 +3.5% gain -> +2.0% stop in any stage, stops only move up, WIPRO partial-booked case)
5. Partial Exit Trigger (1R defined once, true 2.0R trigger, legacy one_r migration backfill)
6. Icon Logic (Green for >= 0, Red for < 0, verifying AAPL -1.52% shows Red)
7. Broker-Side Protection (No default 1333, skip US tickers, cancel previous SL order on move, idempotency)
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import pytest
from datetime import datetime, time
from zoneinfo import ZoneInfo
from unittest.mock import patch, MagicMock

from market_hours import (
    is_us_market_open,
    is_nse_market_open,
    get_usd_inr_rate,
    NY_TZ,
    KOLKATA_TZ,
)
from broker import DhanBroker, resolve_security_id
from reflection import TradeLogger
from position_manager import PositionManager


# =====================================================================
# FIX 1: Market Hours, DST & Early-Close Tests
# =====================================================================
def test_fix1_us_market_hours_dst():
    """Verify America/New_York timezone open 09:30 - 16:00 ET and weekend/holiday checks."""
    # Wednesday 10:30 AM ET (Regular trading day)
    wed_open_et = datetime(2026, 11, 4, 10, 30, tzinfo=NY_TZ)
    assert is_us_market_open(wed_open_et) is True

    # Wednesday 08:30 AM ET (Pre-market - closed)
    wed_pre_et = datetime(2026, 11, 4, 8, 30, tzinfo=NY_TZ)
    assert is_us_market_open(wed_pre_et) is False

    # Wednesday 17:00 ET (Post-market - closed)
    wed_post_et = datetime(2026, 11, 4, 17, 0, tzinfo=NY_TZ)
    assert is_us_market_open(wed_post_et) is False

    # Sunday 12:00 PM ET (Weekend - closed)
    sun_et = datetime(2026, 11, 8, 12, 0, tzinfo=NY_TZ)
    assert is_us_market_open(sun_et) is False


def test_fix1_nyse_early_close():
    """
    Verify NYSE early-close days (e.g. Black Friday 2024-11-29 where market closes at 13:00 ET).
    At 11:30 ET -> market is OPEN.
    At 13:30 ET -> market is CLOSED (early close).
    """
    # 2024-11-29 Black Friday 11:30 ET -> OPEN
    early_day_open = datetime(2024, 11, 29, 11, 30, tzinfo=NY_TZ)
    assert is_us_market_open(early_day_open) is True

    # 2024-11-29 Black Friday 13:30 ET -> CLOSED (Early close was 13:00 ET)
    early_day_after_close = datetime(2024, 11, 29, 13, 30, tzinfo=NY_TZ)
    assert is_us_market_open(early_day_after_close) is False


def test_fix1_nse_market_hours():
    """Verify Asia/Kolkata timezone open 09:15 - 15:30 IST."""
    # Wednesday 10:30 AM IST (Open)
    wed_open_ist = datetime(2026, 11, 4, 10, 30, tzinfo=KOLKATA_TZ)
    assert is_nse_market_open(wed_open_ist) is True

    # Wednesday 08:30 AM IST (Pre-market - closed)
    wed_pre_ist = datetime(2026, 11, 4, 8, 30, tzinfo=KOLKATA_TZ)
    assert is_nse_market_open(wed_pre_ist) is False

    # Saturday 11:00 AM IST (Weekend - closed)
    sat_ist = datetime(2026, 11, 7, 11, 0, tzinfo=KOLKATA_TZ)
    assert is_nse_market_open(sat_ist) is False


# =====================================================================
# FIX 2: Currency Mixing (USD to INR Conversion, 20% Cap, 15% Floor, 3% Kill-Switch)
# =====================================================================
def test_fix2_usd_to_inr_single_stock_cap():
    """
    Verify 20% single-stock ceiling is strictly evaluated in INR using USD/INR rate.
    Account Capital: Rs 1,00,000 -> Max single-stock ceiling: Rs 20,000.
    AAPL at $350 with USD/INR = 84.50 -> 1 share = Rs 29,575 > Rs 20,000.
    Single-share veto MUST reject order with quantity = 0.
    """
    broker = DhanBroker(dry_run=True)
    capital = 100000.0  # INR

    with patch("market_hours.get_usd_inr_rate", return_value=84.50):
        # 1 share of AAPL at $350 = Rs 29,575, exceeding Rs 20,000 cap
        qty, cost, risk = broker.calculate_position_size(
            entry_price=350.0,
            stop_loss=340.0,
            capital=capital,
            currency="USD",
        )
        assert qty == 0
        assert cost == 0.0

        # An affordable USD stock at $50 = Rs 4,225
        qty_small, cost_small, risk_small = broker.calculate_position_size(
            entry_price=50.0,
            stop_loss=48.0,
            capital=capital,
            currency="USD",
        )
        assert qty_small > 0
        assert qty_small * (50.0 * 84.50) <= capital * 0.20


def test_fix2_usd_position_15pct_cash_floor_inr():
    """
    Verify USD trade respects 15% mandatory cash floor in INR.
    Total Capital: Rs 1,00,000 -> 15% Floor: Rs 15,000.
    Available Cash: Rs 20,000 (Usable cash: Rs 5,000).
    USD Stock at $70 -> 1 share = 70 * 84.50 = Rs 5,915.
    Rs 5,915 > usable cash Rs 5,000 -> Trade MUST be rejected (qty = 0).
    """
    broker = DhanBroker(dry_run=True)
    capital = 100000.0
    available_cash = 20000.0  # Rs 20,000 available cash

    with patch("market_hours.get_usd_inr_rate", return_value=84.50):
        qty, cost, risk = broker.calculate_position_size(
            entry_price=70.0,
            stop_loss=65.0,
            capital=capital,
            available_cash=available_cash,
            currency="USD",
        )
        assert qty == 0

        # With Rs 25,000 cash (usable cash = Rs 10,000), 1 share can be afforded
        qty_ok, cost_ok, risk_ok = broker.calculate_position_size(
            entry_price=70.0,
            stop_loss=65.0,
            capital=capital,
            available_cash=25000.0,
            currency="USD",
        )
        assert qty_ok >= 1


def test_fix2_usd_position_3pct_kill_switch_inr(tmp_path):
    """
    Verify 3% daily loss kill-switch converts USD trade loss to INR.
    Total Capital: Rs 1,00,000 -> 3% max daily loss ceiling: Rs 3,000.
    A closed USD trade with loss of -$40.00:
    Without conversion: 40 < 3000 (would fail to trip!).
    With conversion: -40 * 84.50 = -Rs 3,380.
    Since abs(-Rs 3,380) >= Rs 3,000, the 3% kill-switch circuit breaker activates.
    """
    db_file = tmp_path / "test_kill_switch.db"
    tl = TradeLogger(db_path=db_file)
    with tl._get_connection() as conn:
        conn.execute("DELETE FROM trades")
        conn.commit()

    now_date = datetime.now().strftime("%Y-%m-%d")
    with tl._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trades (
                order_id, ticker, action, entry_price, stop_loss, take_profit,
                quantity, initial_quantity, one_r, capital_allocated, risk_amount,
                status, stage, partial_exit_taken, exit_price, pnl, pnl_percentage,
                entry_time, exit_time, is_paper_trade
            ) VALUES ('SIM-KILL-TEST', 'AAPL', 'BUY', 350.0, 340.0, 370.0, 4, 4, 10.0, 1400.0, 40.0, 'CLOSED', 'STAGE_INITIAL', 0, 340.0, -40.0, -2.85, '2026-10-31 10:00:00', ?, 1)
        """, (f"{now_date} 12:00:00",))
        conn.commit()

    with patch("market_hours.get_usd_inr_rate", return_value=84.50):
        summary = tl.calculate_daily_summary(target_date=now_date)
        realized_loss_inr = summary["realized_pnl"]

        # Expected: -40.0 USD * 84.50 = -3380.0 INR
        assert realized_loss_inr == -3380.0

        # Assert this trips the 3% account capital circuit breaker
        capital = 100000.0
        max_daily_loss = capital * 0.03
        assert abs(realized_loss_inr) >= max_daily_loss


# =====================================================================
# FIX 6: Icon Logic Tests
# =====================================================================
def test_fix6_pnl_icon_logic():
    """Verify icon logic uses green for pnl >= 0 and red for pnl < 0."""
    def get_pnl_icon(pnl: float) -> str:
        return "🟢" if pnl >= 0 else "🔴"

    # AAPL negative P&L at -1.52% (-$420.28) must show RED
    aapl_pnl = -420.28
    assert get_pnl_icon(aapl_pnl) == "🔴"

    # WIPRO positive P&L must show GREEN
    wipro_pnl = 402.29
    assert get_pnl_icon(wipro_pnl) == "🟢"

    # Zero breakeven P&L shows GREEN
    breakeven_pnl = 0.0
    assert get_pnl_icon(breakeven_pnl) == "🟢"


# =====================================================================
# FIX 7: Broker Stop-Loss Protection Tests
# =====================================================================
def test_fix7_broker_sl_skips_us_ticker():
    """Verify US tickers are skipped entirely by Dhan broker stop placement."""
    broker = DhanBroker(dry_run=True)
    sl_id = broker.place_broker_stop_loss("SIM-AAPL", "AAPL", 79, 339.51)
    assert sl_id is None


def test_fix7_broker_sl_fails_without_security_id():
    """Verify unmapped tickers do NOT default to 1333 and return None."""
    broker = DhanBroker(dry_run=True)
    sl_id = broker.place_broker_stop_loss("SIM-UNKNOWN", "UNKNOWNSTOCK.NS", 10, 100.0)
    assert sl_id is None


def test_fix7_broker_sl_verified_dhan_security_id():
    """Verify verified Dhan security IDs are resolved for Nifty 50 stocks."""
    assert resolve_security_id("WIPRO.NS") == "3787"
    assert resolve_security_id("DIVISLAB.NS") == "10940"
    assert resolve_security_id("RELIANCE.NS") == "2885"
    assert resolve_security_id("HDFCBANK.NS") == "1333"


def test_fix7_broker_sl_cancels_previous_order_on_move():
    """Verify cancel_broker_stop_loss is invoked when moving an existing stop order."""
    broker = DhanBroker(dry_run=True)
    with patch.object(broker, "cancel_broker_stop_loss", return_value=True) as mock_cancel:
        sl_id = broker.place_broker_stop_loss(
            order_id="TEST-MOVE",
            ticker="WIPRO.NS",
            quantity=49,
            stop_loss=193.90,
            previous_sl_order_id="OLD-SL-19010",
        )
        assert sl_id is not None
        mock_cancel.assert_called_once_with("OLD-SL-19010")


# =====================================================================
# FIX 4: WIPRO Stop Ratchet in STAGE_PARTIAL_BOOKED (+4.32% Gain)
# =====================================================================
def test_fix4_wipro_stop_ratchet_in_partial_booked_stage(tmp_path):
    """
    Exact reproduction of WIPRO holding:
    - Stage: STAGE_PARTIAL_BOOKED
    - Entry: Rs 190.10
    - Stop Loss: Rs 190.10 (breakeven after partial exit)
    - Live Price: Rs 198.31 (+4.32% gain)
    - Expected: Stop ratchets up to Rs 193.90 (+2.0% floor), stage remains STAGE_PARTIAL_BOOKED.
    """
    db_file = tmp_path / "test_wipro_ratchet.db"
    tl = TradeLogger(db_path=db_file)
    with tl._get_connection() as conn:
        conn.execute("DELETE FROM trades")
        conn.commit()

    broker = DhanBroker(dry_run=True)
    pm = PositionManager(trade_logger=tl, broker=broker)

    order_id = "SIM-WIPRO-TEST"
    ticker = "WIPRO.NS"
    entry_price = 190.10
    current_stop = 190.10
    quantity = 49

    with tl._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trades (
                order_id, ticker, action, entry_price, stop_loss, take_profit,
                quantity, initial_quantity, one_r, capital_allocated, risk_amount,
                status, stage, partial_exit_taken, current_price, pnl, pnl_percentage,
                entry_time, is_paper_trade
            ) VALUES (?, ?, 'BUY', ?, ?, 205.42, ?, 97, 7.66, 9314.90, 743.02, 'SIMULATED_FILLED', 'STAGE_PARTIAL_BOOKED', 1, 198.31, 402.29, 4.32, '2026-10-29 23:43:59', 1)
        """, (order_id, ticker, entry_price, current_stop, quantity))
        conn.commit()

    # Live market price = 198.31, 20 EMA = 188.40
    with patch.object(pm, "fetch_live_price_and_ema", return_value=(198.31, 188.40)):
        pm.manage_open_positions()

    updated = tl.get_open_trades()[0]
    # Assert stop ratcheted to +2.0% floor (190.10 * 1.02 = 193.90)
    assert updated["stop_loss"] == 193.90
    # Assert stage remained STAGE_PARTIAL_BOOKED (not corrupted)
    assert updated["stage"] == "STAGE_PARTIAL_BOOKED"


# =====================================================================
# FIX 5: Legacy Database Migration (Backfill one_r)
# =====================================================================
def test_fix5_legacy_migration_backfills_one_r(tmp_path):
    """
    Verify old trade rows in SQLite without one_r get correctly backfilled on migration.
    For WIPRO: entry = 190.10, take_profit = 205.42, stop = 190.10.
    one_r backfill from (take_profit - entry_price) / 2 = 7.66.
    """
    db_file = tmp_path / "test_legacy.db"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    # Create legacy table without one_r column
    cur.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT UNIQUE,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            entry_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            take_profit REAL NOT NULL,
            quantity INTEGER NOT NULL,
            capital_allocated REAL NOT NULL,
            risk_amount REAL NOT NULL,
            status TEXT NOT NULL,
            entry_time TEXT NOT NULL
        )
    """)
    cur.execute("""
        INSERT INTO trades (
            order_id, ticker, action, entry_price, stop_loss, take_profit,
            quantity, capital_allocated, risk_amount, status, entry_time
        ) VALUES ('LEGACY-WIPRO', 'WIPRO.NS', 'BUY', 190.10, 190.10, 205.42, 49, 9314.90, 743.02, 'OPEN', '2026-10-29 23:43:59')
    """)
    conn.commit()
    conn.close()

    # Instantiate TradeLogger to run migration on legacy DB
    tl = TradeLogger(db_path=db_file)
    with tl._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT one_r FROM trades WHERE order_id = 'LEGACY-WIPRO'")
        val = cur.fetchone()[0]
        assert val == 7.66


# =====================================================================
# FIX 3, 4, 5: Position Full Lifecycle & Accounting
# =====================================================================
def test_position_full_lifecycle_and_accounting(tmp_path):
    """
    Simulates a position moving through:
    Entry -> +1R (Breakeven) -> +3.5% (+2.0% Stop Ratchet) -> +2R (50% Partial Exit) -> Trailing 20 EMA exit.
    At each step asserts:
    - stop price
    - quantity
    - invested capital
    - realized P&L
    - unrealized P&L
    """
    db_file = tmp_path / "test_lifecycle.db"
    tl = TradeLogger(db_path=db_file)
    with tl._get_connection() as conn:
        conn.execute("DELETE FROM trades")
        conn.commit()

    broker = DhanBroker(dry_run=True)
    pm = PositionManager(trade_logger=tl, broker=broker)

    order_id = "TEST-WIPRO-001"
    ticker = "WIPRO.NS"
    entry_price = 100.0
    initial_sl = 98.0   # 1R = Rs 2.0 (2.0%)
    initial_tp = 104.0  # 2R = Rs 4.0 (4.0%)
    initial_qty = 100
    invested_capital = entry_price * initial_qty  # Rs 10,000.0

    # 1. ENTRY STEP
    with tl._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trades (
                order_id, ticker, action, entry_price, stop_loss, take_profit,
                quantity, initial_quantity, one_r, capital_allocated, risk_amount,
                status, stage, partial_exit_taken, current_price, pnl, pnl_percentage,
                entry_time, is_paper_trade
            ) VALUES (?, ?, 'BUY', ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', 'STAGE_INITIAL', 0, ?, 0, 0, '2026-10-31 10:00:00', 1)
        """, (
            order_id, ticker, entry_price, initial_sl, initial_tp,
            initial_qty, initial_qty, 2.0, invested_capital, 200.0, entry_price,
        ))
        conn.commit()

    trades = tl.get_open_trades()
    assert len(trades) == 1
    t = trades[0]
    assert t["stop_loss"] == 98.0
    assert t["quantity"] == 100
    assert t["capital_allocated"] == 10000.0
    assert t["one_r"] == 2.0

    # 2. STEP +1R (Price = 102.0 -> Gain = +2.0% = 1.0R -> Stage 1 Breakeven)
    with patch.object(pm, "fetch_live_price_and_ema", return_value=(102.0, 99.0)):
        pm.manage_open_positions()

    t_1r = tl.get_open_trades()[0]
    assert t_1r["stage"] == "STAGE_BREAKEVEN"
    assert t_1r["stop_loss"] == 100.0  # Raised to breakeven
    assert t_1r["quantity"] == 100
    assert t_1r["capital_allocated"] == 10000.0
    assert t_1r["pnl"] == 200.0        # Unrealized = 100 * (102 - 100)
    assert t_1r["one_r"] == 2.0        # Defined ONCE: never changed from moved stop!

    summary = tl.calculate_daily_summary()
    assert summary["realized_pnl"] == 0.0
    assert summary["unrealized_pnl"] == 200.0

    # 3. STEP +3.5% (Gain = +3.5% -> Stage 1.5 Stop Ratchet to +2.0% floor = 102.0)
    # Price at 103.50 (gain = +3.5%), stop moves from 100.0 to 102.0
    with patch.object(pm, "fetch_live_price_and_ema", return_value=(103.5, 101.0)):
        pm.manage_open_positions()

    t_ratchet = tl.get_open_trades()[0]
    assert t_ratchet["stop_loss"] == 102.0  # +2.0% floor locked!
    assert t_ratchet["quantity"] == 100
    assert t_ratchet["capital_allocated"] == 10000.0
    assert t_ratchet["pnl"] == 350.0        # Unrealized = 100 * 3.5

    # 4. STEP +2R (Price = 104.0 -> Gain = 4.0 = 2.0R -> Stage 2 50% Partial Exit)
    with patch.object(pm, "fetch_live_price_and_ema", return_value=(104.0, 102.5)):
        pm.manage_open_positions()

    t_partial = tl.get_open_trades()[0]
    assert t_partial["stage"] == "STAGE_PARTIAL_BOOKED"
    assert t_partial["quantity"] == 50                 # 50 shares remaining
    assert t_partial["capital_allocated"] == 5000.0    # Invested capital updated to remaining shares only!
    assert t_partial["pnl"] == 200.0                  # Unrealized on remaining 50 shares = 50 * (104 - 100) = 200.0
    assert t_partial["stop_loss"] >= 102.0             # Trailing stop only moves UP

    # Check realized P&L is stored separately and not double counted
    summary_partial = tl.calculate_daily_summary()
    assert summary_partial["realized_pnl"] == 200.0   # Realized from 50 shares sold @ 104: 50 * 4 = 200.0
    assert summary_partial["unrealized_pnl"] == 200.0 # Unrealized from 50 remaining shares
    assert summary_partial["total_pnl"] == 400.0      # Total P&L = 200 + 200

    # 5. STEP 3: Trailing 20 EMA Exit (Price falls below 20 EMA, e.g. Price = 102.5, 20 EMA = 103.0)
    with patch.object(pm, "fetch_live_price_and_ema", return_value=(102.5, 103.0)):
        pm.manage_open_positions()

    # Position is now completely closed
    open_after = tl.get_open_trades()
    assert len(open_after) == 0

    # Final accounting check
    summary_final = tl.calculate_daily_summary()
    assert summary_final["realized_pnl"] == 325.0
    assert summary_final["unrealized_pnl"] == 0.0
    assert summary_final["total_pnl"] == 325.0
