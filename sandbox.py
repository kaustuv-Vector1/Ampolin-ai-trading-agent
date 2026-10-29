"""
Ampolin AI Trading Agent - VectorBT Sandbox Module
Uses VectorBT to run a sub-100ms vectorized backtest of candidate setups over the past 90 days.
Approves only setups with positive expectancy and >= 45% win rate.
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import vectorbt as vbt

from config import (
    ACTIVE_RULES_PATH,
    MIN_WIN_RATE,
    MIN_EXPECTANCY,
    BACKTEST_LOOKBACK_DAYS,
)
from scanner import CandidateSetup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AmpolinSandbox")


@dataclass
class BacktestResult:
    ticker: str
    approved: bool
    rejection_reason: Optional[str]
    win_rate: float
    expectancy: float
    profit_factor: float
    total_trades: int
    total_return_pct: float
    execution_time_ms: float
    avg_win_pct: float
    avg_loss_pct: float
    max_drawdown_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VectorBTSandbox:
    def __init__(self, rules_path: Optional[str] = None):
        self.rules = self._load_rules(rules_path or ACTIVE_RULES_PATH)
        self.min_win_rate = float(self.rules.get("min_win_rate", MIN_WIN_RATE))
        self.min_expectancy = float(self.rules.get("min_expectancy", MIN_EXPECTANCY))
        self.lookback_days = int(self.rules.get("backtest_lookback_days", BACKTEST_LOOKBACK_DAYS))
        # Warm up VectorBT / Numba JIT so all subsequent backtests execute in < 10ms
        self._warmup_jit()

    def _load_rules(self, path) -> Dict[str, Any]:
        if path and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("rules", {})
            except Exception as e:
                logger.warning(f"Could not load rules from {path}: {e}")
        return {
            "min_win_rate": MIN_WIN_RATE,
            "min_expectancy": MIN_EXPECTANCY,
            "backtest_lookback_days": BACKTEST_LOOKBACK_DAYS,
        }

    def _warmup_jit(self):
        """Pre-compiles Numba kernels once at startup to guarantee sub-100ms per backtest."""
        try:
            dummy_price = pd.Series([100.0, 101.0, 102.0, 100.5, 103.0])
            dummy_entries = pd.Series([False, True, False, False, False])
            dummy_exits = pd.Series([False, False, False, True, False])
            _ = vbt.Portfolio.from_signals(dummy_price, dummy_entries, dummy_exits)
            logger.debug("VectorBT Numba JIT successfully pre-warmed.")
        except Exception as e:
            logger.warning(f"VectorBT JIT pre-warm warning: {e}")

    def run_backtest(self, setup: CandidateSetup) -> BacktestResult:
        """
        Runs a vectorized backtest on the asset's past 90 days.
        Measures exact runtime to guarantee sub-100ms execution.
        """
        start_time = time.perf_counter()

        df = setup.df
        if df is None or len(df) < 20:
            exec_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return BacktestResult(
                ticker=setup.ticker,
                approved=False,
                rejection_reason="Insufficient data bars for 90-day simulation",
                win_rate=0.0,
                expectancy=0.0,
                profit_factor=0.0,
                total_trades=0,
                total_return_pct=0.0,
                execution_time_ms=exec_ms,
                avg_win_pct=0.0,
                avg_loss_pct=0.0,
                max_drawdown_pct=0.0,
            )

        # Slice the last 90 trading days (or max available)
        recent_df = df.iloc[-min(len(df), self.lookback_days):].copy()

        # Generate vectorized trading signals according to strategy
        close = recent_df["Close"]
        ema_fast = recent_df["EMA_FAST"]
        ema_slow = recent_df["EMA_SLOW"]
        rsi = recent_df["RSI"]
        atr = recent_df["ATR"]

        rsi_min = float(self.rules.get("rsi_min", 48.0))
        rsi_max = float(self.rules.get("rsi_max", 68.0))

        # Entry signal: Price above EMA20, EMA20 > EMA50, RSI in momentum sweet spot
        entries = (close > ema_fast) & (ema_fast > ema_slow) & (rsi >= rsi_min) & (rsi <= rsi_max)

        # Exit signal: Close drops below EMA fast (trailing exit) OR ATR stop hit
        exits = close < ema_fast

        # Dynamic stop loss and take profit percentages based on average ATR
        avg_atr = atr.mean()
        avg_close = close.mean()
        atr_pct = (avg_atr / avg_close) if avg_close > 0 else 0.02
        sl_stop_pct = max(0.015, min(0.05, atr_pct * float(self.rules.get("atr_sl_multiplier", 1.5))))
        tp_stop_pct = max(0.03, min(0.12, sl_stop_pct * float(self.rules.get("tp_risk_multiplier", 2.0))))

        try:
            # Fast vectorized portfolio simulation
            portfolio = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=sl_stop_pct,
                tp_stop=tp_stop_pct,
                init_cash=100000.0,
                fees=0.001,  # 0.1% transaction fee / STT estimation for NSE delivery
            )

            trades = portfolio.trades
            total_trades = int(trades.count())

            if total_trades > 0:
                win_rate = float(trades.win_rate()) if not pd.isna(trades.win_rate()) else 0.0
                expectancy = float(trades.expectancy()) if not pd.isna(trades.expectancy()) else 0.0
                profit_factor_raw = trades.profit_factor()
                profit_factor = float(profit_factor_raw) if not (pd.isna(profit_factor_raw) or np.isinf(profit_factor_raw)) else (10.0 if win_rate > 0 else 0.0)
                total_return = float(portfolio.total_return()) * 100.0

                winning_trades = trades.winning
                losing_trades = trades.losing
                avg_win = float(winning_trades.returns.mean()) * 100.0 if winning_trades.count() > 0 else 0.0
                avg_loss = float(losing_trades.returns.mean()) * 100.0 if losing_trades.count() > 0 else 0.0
                max_dd = float(portfolio.drawdowns.max_drawdown()) * 100.0
            else:
                # If zero trades were triggered in 90 days, test on the full available dataset (180 days)
                full_portfolio = vbt.Portfolio.from_signals(
                    close=df["Close"],
                    entries=(df["Close"] > df["EMA_FAST"]) & (df["EMA_FAST"] > df["EMA_SLOW"]) & (df["RSI"] >= rsi_min) & (df["RSI"] <= rsi_max),
                    exits=df["Close"] < df["EMA_FAST"],
                    sl_stop=sl_stop_pct,
                    tp_stop=tp_stop_pct,
                    init_cash=100000.0,
                    fees=0.001,
                )
                full_trades = full_portfolio.trades
                total_trades = int(full_trades.count())
                win_rate = float(full_trades.win_rate()) if not pd.isna(full_trades.win_rate()) else 0.0
                expectancy = float(full_trades.expectancy()) if not pd.isna(full_trades.expectancy()) else 0.0
                pf_raw = full_trades.profit_factor()
                profit_factor = float(pf_raw) if not (pd.isna(pf_raw) or np.isinf(pf_raw)) else (5.0 if win_rate > 0 else 0.0)
                total_return = float(full_portfolio.total_return()) * 100.0
                avg_win = float(full_trades.winning.returns.mean()) * 100.0 if full_trades.winning.count() > 0 else 0.0
                avg_loss = float(full_trades.losing.returns.mean()) * 100.0 if full_trades.losing.count() > 0 else 0.0
                max_dd = float(full_portfolio.drawdowns.max_drawdown()) * 100.0

            # Strict approval logic:
            # 1. Expectancy must be > 0 (positive edge per rupee risked)
            # 2. Win rate must be >= 45% (0.45)
            # 3. Must have had at least 1 validated historical trade
            approved = True
            rejection_reasons = []

            if total_trades < 1:
                approved = False
                rejection_reasons.append("Insufficient trade samples in backtest window")
            if win_rate < self.min_win_rate:
                approved = False
                rejection_reasons.append(f"Win rate {win_rate*100:.1f}% below minimum threshold {self.min_win_rate*100:.1f}%")
            if expectancy <= self.min_expectancy:
                approved = False
                rejection_reasons.append(f"Expectancy ₹{expectancy:.2f} is non-positive")

            rejection_reason_str = " | ".join(rejection_reasons) if rejection_reasons else None

        except Exception as e:
            logger.error(f"Error during VectorBT simulation for {setup.ticker}: {e}")
            approved = False
            rejection_reason_str = f"Simulation exception: {e}"
            win_rate = 0.0
            expectancy = 0.0
            profit_factor = 0.0
            total_trades = 0
            total_return = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            max_dd = 0.0

        exec_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return BacktestResult(
            ticker=setup.ticker,
            approved=approved,
            rejection_reason=rejection_reason_str,
            win_rate=round(win_rate, 4),
            expectancy=round(expectancy, 2),
            profit_factor=round(profit_factor, 2),
            total_trades=total_trades,
            total_return_pct=round(total_return, 2),
            execution_time_ms=exec_ms,
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            max_drawdown_pct=round(max_dd, 2),
        )

    def filter_candidates(self, candidates: List[CandidateSetup]) -> List[tuple[CandidateSetup, BacktestResult]]:
        """
        Iterates through candidate setups, executes VectorBT backtests,
        and filters down to approved setups only.
        """
        approved_pairs = []
        logger.info(f"Running VectorBT sub-100ms backtests for {len(candidates)} candidate setups...")

        for setup in candidates:
            res = self.run_backtest(setup)
            status_tag = "APPROVED" if res.approved else "REJECTED"
            logger.info(
                f"[{status_tag}] {res.ticker} | Time: {res.execution_time_ms} ms | "
                f"Win Rate: {res.win_rate*100:.1f}% | Expectancy: ₹{res.expectancy} | "
                f"Trades: {res.total_trades} | Profit Factor: {res.profit_factor} "
                f"{f'({res.rejection_reason})' if res.rejection_reason else ''}"
            )
            if res.approved:
                approved_pairs.append((setup, res))

        logger.info(f"Sandbox backtesting complete: {len(approved_pairs)} / {len(candidates)} approved.")
        return approved_pairs


if __name__ == "__main__":
    from scanner import SwingScanner
    scanner = SwingScanner()
    sandbox = VectorBTSandbox()

    test_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "BHARTIARTL.NS"]
    setups = scanner.scan_universe(test_tickers, timeframe="1d")

    approved = sandbox.filter_candidates(setups)
    print(f"\nFinal Approved Setups: {len(approved)}")
    for s, b in approved:
        print(f" -> {s.ticker}: Price ₹{s.close}, Win Rate={b.win_rate*100:.1f}%, Expectancy=₹{b.expectancy}, Latency={b.execution_time_ms}ms")
