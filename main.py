"""
Ampolin AI Trading Agent - Main Orchestrator Pipeline
Connects Scanner -> VectorBT Sandbox -> Gemini AI Brain -> DhanHQ Broker -> SQLite Logger -> Telegram Alerts -> EOD Reflection.
"""

import argparse
import logging
import sys
from typing import List, Optional, Dict, Any

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from config import (
    NIFTY_50_TICKERS,
    US_STOCKS_TICKERS,
    DEFAULT_TIMEFRAME,
    ACTIVE_RULES_PATH,
    MAX_CAPITAL_PER_STOCK_PCT,
    MAX_OPEN_POSITIONS,
    MIN_CASH_RESERVE_PCT,
    MAX_DAILY_LOSS_PCT,
)
from scanner import SwingScanner, CandidateSetup
from sandbox import VectorBTSandbox, BacktestResult
from ai_brain import GeminiBrain, TradeDecision
from broker import DhanBroker, OrderResult
from reflection import TradeLogger, EODReflection
from telegram_notifier import TelegramNotifier
from market_regime import MarketRegimeAnalyzer, MarketRegimeReport
from position_manager import PositionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AmpolinMain")


class AmpolinPipeline:
    def __init__(self, dry_run: bool = True, timeframe: str = DEFAULT_TIMEFRAME):
        self.dry_run = dry_run
        self.timeframe = timeframe

        logger.info(f"Initializing Ampolin AI Trading Agent (Dry-Run={self.dry_run}, Timeframe={self.timeframe})...")
        self.trade_logger = TradeLogger()
        self.scanner = SwingScanner()
        self.sandbox = VectorBTSandbox()
        self.brain = GeminiBrain(trade_logger=self.trade_logger)
        self.broker = DhanBroker(dry_run=self.dry_run)
        self.reflection = EODReflection(self.trade_logger)
        self.notifier = TelegramNotifier()
        self.regime_analyzer = MarketRegimeAnalyzer(sample_size=20)
        self.position_manager = PositionManager(
            trade_logger=self.trade_logger,
            notifier=self.notifier,
            broker=self.broker,
        )

    def run_pipeline(
        self,
        tickers: Optional[List[str]] = None,
        market: str = "NSE",
        max_trades_to_execute: int = 3,
    ) -> Dict[str, Any]:
        """
        Executes complete autonomous swing trading cycle:
        1. Market Scan (Nifty 50 or Wall Street US Mega-Caps)
        2. VectorBT Backtest Filter (Expectancy > 0, Win Rate >= 45%)
        3. Gemini Flash AI Brain Evaluation (Strict Pydantic JSON with Weekly Confluence)
        4. DhanHQ / Simulated Execution (1% Max Account Risk)
        5. SQLite Trade Logging
        6. Telegram Dispatch
        """
        if tickers:
            scan_tickers = tickers
        elif market.upper() == "US":
            scan_tickers = US_STOCKS_TICKERS
        else:
            scan_tickers = NIFTY_50_TICKERS

        market_label = "WALL STREET (US)" if market.upper() == "US" else "NSE (INDIA)"
        print("\n" + "=" * 70)
        print(f"🚀 STARTING AMPOLIN AI TRADING PIPELINE [{market_label}] ({len(scan_tickers)} stocks, TF: {self.timeframe})")
        print("=" * 70)

        # Step 0: Active Position Management (1R Breakeven, 2R Scaling & 3.5% Profit Ratchet)
        print("\n" + "-" * 50)
        print("🔍 STEP 0: MANAGING ACTIVE OPEN POSITIONS (1R Breakeven, 2R Scaling & Profit Lock)")
        print("-" * 50)
        position_actions = self.position_manager.manage_open_positions()

        # Step 0.2: Capital Security Guard - Daily Loss Circuit Breaker
        total_cap = self.broker.get_account_capital()
        daily_summary = self.trade_logger.calculate_daily_summary()
        max_daily_loss = total_cap * MAX_DAILY_LOSS_PCT
        today_loss = float(daily_summary.get("realized_pnl", 0.0))

        if today_loss <= -max_daily_loss:
            logger.critical(
                f"🚨 [CIRCUIT BREAKER ACTIVATED] Realized loss today (₹{abs(today_loss):,.2f}) "
                f"reached max tolerance (₹{max_daily_loss:,.2f} / {MAX_DAILY_LOSS_PCT*100}%). "
                f"Halting all new trading for capital protection!"
            )
            self.notifier.send_message(
                f"🚨 <b>DAILY LOSS CIRCUIT BREAKER TRIPPED</b>\n\n"
                f"• <b>Realized Loss Today:</b> -₹{abs(today_loss):,.2f}\n"
                f"• <b>Max Loss Ceiling:</b> ₹{max_daily_loss:,.2f} (3.0%)\n"
                f"• <b>Status:</b> All new trades halted for the day to protect your capital. Remaining positions are guarded by trailing stops."
            )
            return {
                "status": "CIRCUIT_BREAKER_HALTED",
                "realized_loss": today_loss,
                "max_daily_loss": max_daily_loss,
                "position_actions": position_actions,
                "executed_trades": [],
            }

        # Step 0.3: Capital Security Guard - Max Open Positions Ceiling
        current_open = self.trade_logger.get_open_trades()
        if len(current_open) >= MAX_OPEN_POSITIONS:
            logger.info(
                f"[PORTFOLIO GUARD] Maximum open positions limit reached ({len(current_open)}/{MAX_OPEN_POSITIONS}). "
                f"Halting new scans to preserve diversification and cash reserve."
            )
            return {
                "status": "MAX_POSITIONS_REACHED",
                "open_positions_count": len(current_open),
                "max_open_positions": MAX_OPEN_POSITIONS,
                "position_actions": position_actions,
                "executed_trades": [],
            }

        # Step 0.5: Macro Regime Assessment
        if market.upper() == "NSE":
            print("\n" + "-" * 50)
            print("🌐 STEP 0.5: EVALUATING MARKET REGIME (India VIX & Market Breadth)")
            print("-" * 50)
            regime = self.regime_analyzer.analyze_regime()
            print(f"-> Regime: {regime.breadth_regime} | Breadth: {regime.market_breadth_pct}% | VIX: {regime.india_vix} ({regime.vix_regime})")
            print(f"-> Dynamic Risk per Trade: {regime.dynamic_risk_pct * 100:.2f}% | Trade Permitted: {regime.can_trade_longs}")

            if not regime.can_trade_longs:
                logger.warning(f"Market Regime Check: {regime.regime_rationale}. Halting new long entries.")
                return {
                    "status": "MARKET_REGIME_PAUSED",
                    "regime": regime.to_dict(),
                    "position_actions": position_actions,
                    "candidates_count": 0,
                    "executed_trades": [],
                }
            dynamic_risk = regime.dynamic_risk_pct
        else:
            # Wall Street Market Scan
            print("\n" + "-" * 50)
            print("🌐 STEP 0.5: EVALUATING WALL STREET REGIME")
            print("-" * 50)
            print("-> Market: Wall Street (NYSE/NASDAQ) | Default Base Risk: 0.75%")
            dynamic_risk = 0.0075

        # Step 1: Scanner
        candidates = self.scanner.scan_universe(tickers=scan_tickers, market=market, timeframe=self.timeframe)
        if not candidates:
            logger.info("No candidates satisfied EMA 20/50 + RSI (48-68) + ATR breakout criteria.")
            return {"status": "NO_CANDIDATES", "candidates_count": 0, "approved_count": 0, "executed_trades": []}

        # Step 2: VectorBT Sandbox Filter
        approved_pairs = self.sandbox.filter_candidates(candidates)
        if not approved_pairs:
            logger.info("None of the candidate setups passed VectorBT 90-day backtest criteria (Win Rate >= 45%, Expectancy > 0).")
            return {"status": "BACKTEST_REJECTED", "candidates_count": len(candidates), "approved_count": 0, "executed_trades": []}

        executed_records = []

        # Sort candidates by backtest expectancy and win rate
        approved_pairs.sort(key=lambda pair: (pair[1].expectancy, pair[1].win_rate), reverse=True)

        # Step 3, 4, 5, 6: AI Brain, Execution, Logging & Alerts
        for setup, backtest in approved_pairs[:max_trades_to_execute]:
            print(f"\n--- Processing {setup.ticker} ---")

            # 3.0 Capital Security Check: Dynamic Open Positions Count
            active_open = self.trade_logger.get_open_trades()
            if len(active_open) >= MAX_OPEN_POSITIONS:
                logger.info(f"[PORTFOLIO GUARD] Maximum {MAX_OPEN_POSITIONS} positions reached. Halting new entries.")
                break

            # 3.1 Capital Security Check: Anti-Duplicate Position Guard
            open_tickers = {t["ticker"].replace(".NS", "").upper() for t in active_open}
            clean_ticker = setup.ticker.replace(".NS", "").upper()
            if clean_ticker in open_tickers:
                logger.info(f"[PORTFOLIO GUARD] {setup.ticker} is already active in open positions. Skipping duplicate entry.")
                continue

            # 3.2 Capital Security Check: Cash Reserve Floor
            invested_capital = sum(float(t.get("capital_allocated") or (float(t.get("entry_price") or 0) * int(t.get("quantity") or 0))) for t in active_open)
            available_cash = max(0.0, total_cap - invested_capital)
            min_cash_reserve = total_cap * MIN_CASH_RESERVE_PCT
            usable_cash = available_cash - min_cash_reserve
            if usable_cash <= 0:
                logger.warning(
                    f"[CASH GUARD] Available cash (₹{available_cash:,.2f}) is at or below the 15% reserve floor "
                    f"(₹{min_cash_reserve:,.2f}). Halting new entries to preserve liquidity."
                )
                break

            # 3.3 Gemini AI Brain Evaluation
            decision = self.brain.evaluate_setup(setup, backtest)

            if decision.action != "BUY":
                logger.info(f"AI Brain chose to {decision.action} on {setup.ticker}. Skipping execution.")
                continue

            if decision.confidence < 0.55:
                logger.info(f"AI Brain confidence ({decision.confidence*100:.1f}%) below minimum conviction threshold. Skipping.")
                continue

            # 4. DhanHQ Broker Execution (Dynamic Risk + Anti-Dump 20% Cap + 15% Cash Buffer)
            order_result = self.broker.place_cnc_bracket_order(
                decision,
                custom_risk_pct=dynamic_risk,
                available_cash=available_cash,
            )

            # Block if rejected or 0 quantity
            if order_result.quantity <= 0 or order_result.status.startswith("REJECTED"):
                logger.warning(f"[CAPITAL GUARD] Order rejected/blocked for {setup.ticker}: {order_result.status}")
                continue

            # 5. SQLite Logging
            trade_id = self.trade_logger.log_trade(
                order=order_result,
                decision=decision,
                backtest=backtest,
            )

            # 6. Telegram Alert Dispatch
            self.notifier.notify_setup_approved(
                ticker=setup.ticker,
                decision=decision,
                backtest=backtest,
                order=order_result,
            )

            executed_records.append({
                "trade_id": trade_id,
                "ticker": setup.ticker,
                "action": decision.action,
                "confidence": decision.confidence,
                "entry_price": decision.entry_price,
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit,
                "quantity": order_result.quantity,
                "capital_allocated": order_result.capital_allocated,
                "total_risk": order_result.total_risk_amount,
                "risk_percentage": order_result.risk_percentage,
                "is_paper_trade": order_result.is_paper_trade,
                "backtest_win_rate": backtest.win_rate,
                "backtest_expectancy": backtest.expectancy,
                "backtest_ms": backtest.execution_time_ms,
            })

        print("\n" + "=" * 70)
        print(f"✅ PIPELINE COMPLETE: {len(executed_records)} trades executed (Dry-Run: {self.dry_run})")
        print("=" * 70)

        # Compute Daily Summary
        daily_summary = self.trade_logger.calculate_daily_summary()
        if executed_records:
            self.notifier.notify_daily_pnl(daily_summary)

        return {
            "status": "SUCCESS",
            "candidates_count": len(candidates),
            "approved_count": len(approved_pairs),
            "executed_count": len(executed_records),
            "executed_trades": executed_records,
            "daily_summary": daily_summary,
        }

    def run_eod(self) -> Dict[str, Any]:
        """Runs daily EOD post-mortem and self-refinement."""
        result = self.reflection.run_post_mortem()
        self.notifier.notify_eod_reflection(result)
        return result


def main():
    parser = argparse.ArgumentParser(description="Ampolin AI Trading Agent Master Runner")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Run in paper trading mode (default: True)")
    parser.add_argument("--live", action="store_true", help="Run with real live broker orders on DhanHQ")
    parser.add_argument("--timeframe", type=str, default="1d", choices=["1d", "4h"], help="Scanning timeframe (1d or 4h)")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated custom tickers (e.g. RELIANCE.NS,TCS.NS)")
    parser.add_argument("--eod", action="store_true", help="Run daily EOD post-mortem and exit")

    args = parser.parse_args()
    dry_run = not args.live

    pipeline = AmpolinPipeline(dry_run=dry_run, timeframe=args.timeframe)

    if args.eod:
        pipeline.run_eod()
        return

    custom_tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None
    pipeline.run_pipeline(tickers=custom_tickers)


if __name__ == "__main__":
    main()
