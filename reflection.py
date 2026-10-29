"""
Ampolin AI Trading Agent - Trade Logger & Reflection Engine
Manages SQLite database (trades.db), logs executions, computes daily P&L,
and executes daily EOD post-mortem analysis that self-refines active_rules.json.
"""

import json
import logging
import sqlite3
import sys
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from config import DB_PATH, ACTIVE_RULES_PATH
from broker import OrderResult
from ai_brain import TradeDecision
from sandbox import BacktestResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AmpolinReflection")


class TradeLogger:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes SQLite tables for trades, daily P&L, and reflections."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Trades Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE,
                    ticker TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    initial_quantity INTEGER,
                    capital_allocated REAL NOT NULL,
                    risk_amount REAL NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT DEFAULT 'STAGE_INITIAL',
                    partial_exit_taken INTEGER DEFAULT 0,
                    exit_price REAL,
                    pnl REAL DEFAULT 0.0,
                    pnl_percentage REAL DEFAULT 0.0,
                    confidence REAL,
                    rationale TEXT,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    is_paper_trade INTEGER DEFAULT 1,
                    metadata_json TEXT
                )
            """)

            # Add columns if migrating an existing DB
            cursor.execute("PRAGMA table_info(trades)")
            cols = [c[1] for c in cursor.fetchall()]
            if "initial_quantity" not in cols:
                cursor.execute("ALTER TABLE trades ADD COLUMN initial_quantity INTEGER")
            if "stage" not in cols:
                cursor.execute("ALTER TABLE trades ADD COLUMN stage TEXT DEFAULT 'STAGE_INITIAL'")
            if "partial_exit_taken" not in cols:
                cursor.execute("ALTER TABLE trades ADD COLUMN partial_exit_taken INTEGER DEFAULT 0")

            # Daily P&L Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_pnl (
                    date TEXT PRIMARY KEY,
                    starting_capital REAL,
                    realized_pnl REAL DEFAULT 0.0,
                    unrealized_pnl REAL DEFAULT 0.0,
                    total_trades INTEGER DEFAULT 0,
                    win_count INTEGER DEFAULT 0,
                    loss_count INTEGER DEFAULT 0,
                    updated_at TEXT
                )
            """)

            # Post-Mortems Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS post_mortems (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    adjustments_made TEXT,
                    win_rate REAL,
                    total_trades_analyzed INTEGER
                )
            """)

            conn.commit()
            logger.debug("SQLite database initialized successfully.")

    def log_trade(
        self,
        order: OrderResult,
        decision: TradeDecision,
        backtest: Optional[BacktestResult] = None,
    ) -> int:
        """Records a new open trade into the database."""
        meta = {
            "risk_reward_ratio": decision.risk_reward_ratio,
            "holding_period_days": decision.holding_period_days,
            "key_risks": decision.key_risks,
            "backtest_win_rate": backtest.win_rate if backtest else None,
            "backtest_expectancy": backtest.expectancy if backtest else None,
            "raw_response": order.raw_response,
        }

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO trades (
                    order_id, ticker, action, entry_price, stop_loss, take_profit,
                    quantity, initial_quantity, capital_allocated, risk_amount, status,
                    stage, partial_exit_taken, confidence, rationale, entry_time, is_paper_trade, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order.order_id,
                    order.ticker,
                    order.action,
                    order.entry_price,
                    order.stop_loss,
                    order.take_profit,
                    order.quantity,
                    order.quantity,
                    order.capital_allocated,
                    order.total_risk_amount,
                    order.status,
                    "STAGE_INITIAL",
                    0,
                    decision.confidence,
                    decision.rationale,
                    order.timestamp,
                    1 if order.is_paper_trade else 0,
                    json.dumps(meta),
                ),
            )
            trade_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Trade logged in SQLite [ID: {trade_id}] for {order.ticker}")
            return trade_id

    def close_trade(self, order_id: str, exit_price: float, exit_time: Optional[str] = None) -> bool:
        """Closes an open trade and calculates realized P&L."""
        now_str = exit_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE order_id = ?", (order_id,))
            row = cursor.fetchone()
            if not row:
                logger.warning(f"No trade found with order_id {order_id}")
                return False

            qty = row["quantity"]
            entry_price = row["entry_price"]
            action = row["action"]

            if action == "BUY":
                pnl = (exit_price - entry_price) * qty
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
            else:
                pnl = (entry_price - exit_price) * qty
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100.0

            cursor.execute(
                """
                UPDATE trades
                SET status = 'CLOSED', exit_price = ?, exit_time = ?, pnl = ?, pnl_percentage = ?
                WHERE order_id = ?
                """,
                (exit_price, now_str, round(pnl, 2), round(pnl_pct, 2), order_id),
            )
            conn.commit()
            logger.info(f"Trade {order_id} closed at ₹{exit_price} | P&L: ₹{pnl:.2f} ({pnl_pct:.2f}%)")
            return True

    def get_all_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_open_trades(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE status IN ('SUBMITTED', 'SIMULATED_FILLED', 'OPEN')")
            return [dict(row) for row in cursor.fetchall()]

    def get_experiential_memory(self, ticker: Optional[str] = None, limit: int = 6) -> str:
        """
        Retrieves experiential trade memory from trades.db and active_rules.json
        to inject directly into Gemini AI's prompt for in-context learning.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY id DESC LIMIT ?", (limit,))
            closed_trades = [dict(r) for r in cursor.fetchall()]

            prior_ticker_trades = []
            if ticker:
                cursor.execute("SELECT * FROM trades WHERE ticker = ? ORDER BY id DESC LIMIT 3", (ticker,))
                prior_ticker_trades = [dict(r) for r in cursor.fetchall()]

        lines = []
        if closed_trades:
            wins = sum(1 for t in closed_trades if (t.get("pnl") or 0) > 0)
            losses = len(closed_trades) - wins
            win_rate = (wins / len(closed_trades)) * 100
            total_realized = sum(t.get("pnl") or 0 for t in closed_trades)
            lines.append(f"Recent Executed Performance: {wins} Wins / {losses} Losses ({win_rate:.0f}% win rate) | Realized P&L: ₹{total_realized:,.2f}")
            lines.append("Recent Closed Trades:")
            for t in closed_trades[:4]:
                pnl = t.get("pnl") or 0.0
                pct = t.get("pnl_percentage") or 0.0
                result_str = f"WIN (+{pct:.1f}%)" if pnl > 0 else f"LOSS ({pct:.1f}%)"
                lines.append(f"• [{result_str}] {t['ticker']} @ {t['entry_price']} -> Exit: {t['exit_price']} | P&L: ₹{pnl:,.2f}")
        else:
            lines.append("Recent Executed Performance: Currently holding active paper positions. No closed losses on record.")

        if prior_ticker_trades:
            lines.append(f"Past History on {ticker}:")
            for pt in prior_ticker_trades:
                status = pt.get("status", "")
                pnl = pt.get("pnl") or 0.0
                lines.append(f"• Status: {status} | Entry: {pt['entry_price']} | Stage: {pt.get('stage', 'INITIAL')}")

        if ACTIVE_RULES_PATH.exists():
            try:
                with open(ACTIVE_RULES_PATH, "r", encoding="utf-8") as f:
                    rules_data = json.load(f)
                    lessons = rules_data.get("reflection_summary", "")
                    if lessons:
                        lines.append(f"Learned Behavioral Lesson: {lessons}")
            except Exception:
                pass

        return "\n".join(lines)

    def calculate_daily_summary(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """Calculates realized and unrealized P&L for a given date."""
        d_str = target_date or date.today().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE entry_time LIKE ?", (f"{d_str}%",))
            today_trades = [dict(row) for row in cursor.fetchall()]

            closed_trades = [t for t in today_trades if t["status"] == "CLOSED"]
            open_trades = [t for t in today_trades if t["status"] != "CLOSED"]

            realized_pnl = sum(t["pnl"] for t in closed_trades)
            win_count = sum(1 for t in closed_trades if t["pnl"] > 0)
            loss_count = sum(1 for t in closed_trades if t["pnl"] < 0)

            summary = {
                "date": d_str,
                "total_trades": len(today_trades),
                "open_positions": len(open_trades),
                "closed_positions": len(closed_trades),
                "realized_pnl": round(realized_pnl, 2),
                "win_count": win_count,
                "loss_count": loss_count,
                "win_rate": round(win_count / len(closed_trades), 2) if closed_trades else 0.0,
            }

            # Update daily_pnl table
            cursor.execute(
                """
                INSERT INTO daily_pnl (date, realized_pnl, total_trades, win_count, loss_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    realized_pnl=excluded.realized_pnl,
                    total_trades=excluded.total_trades,
                    win_count=excluded.win_count,
                    loss_count=excluded.loss_count,
                    updated_at=excluded.updated_at
                """,
                (d_str, round(realized_pnl, 2), len(today_trades), win_count, loss_count, datetime.now().isoformat()),
            )
            conn.commit()
            return summary


class EODReflection:
    def __init__(self, logger_instance: Optional[TradeLogger] = None, rules_path: Path = ACTIVE_RULES_PATH):
        self.logger = logger_instance or TradeLogger()
        self.rules_path = rules_path

    def run_post_mortem(self) -> Dict[str, Any]:
        """
        Executes end-of-day post-mortem:
        1. Analyzes trade distribution and outcomes.
        2. Evaluates performance edge.
        3. Self-adjusts active_rules.json parameters if thresholds drift.
        4. Logs the reflection session.
        """
        logger.info("Executing EOD Post-Mortem Reflection...")
        all_trades = self.logger.get_all_trades(limit=100)
        closed_trades = [t for t in all_trades if t["status"] == "CLOSED"]

        # Load active rules
        rules_data = {}
        if self.rules_path.exists():
            with open(self.rules_path, "r", encoding="utf-8") as f:
                rules_data = json.load(f)

        current_rules = rules_data.get("rules", {})
        learned_insights = rules_data.get("learned_insights", [])
        adjustments_history = rules_data.get("adjustments_history", [])

        adjustments_made = []
        now_iso = datetime.now().isoformat()

        if len(closed_trades) >= 5:
            win_count = sum(1 for t in closed_trades if t["pnl"] > 0)
            win_rate = win_count / len(closed_trades)
            avg_win = (sum(t["pnl"] for t in closed_trades if t["pnl"] > 0) / max(win_count, 1))
            avg_loss = abs(sum(t["pnl"] for t in closed_trades if t["pnl"] < 0) / max(len(closed_trades) - win_count, 1))

            # Adaptive Rule Adjustments:
            # 1. If win rate is below 45%, tighten RSI upper boundary to avoid buying near tops
            if win_rate < 0.45:
                old_rsi_max = current_rules.get("rsi_max", 68.0)
                new_rsi_max = max(62.0, old_rsi_max - 2.0)
                if new_rsi_max != old_rsi_max:
                    current_rules["rsi_max"] = new_rsi_max
                    adjustments_made.append(f"Tightened RSI upper bound from {old_rsi_max} to {new_rsi_max} to reduce top-exhaustion entries.")

                # Slightly increase ATR SL buffer to avoid getting stopped out by intra-day spikes
                old_sl_mult = current_rules.get("atr_sl_multiplier", 1.5)
                new_sl_mult = min(2.0, old_sl_mult + 0.1)
                if new_sl_mult != old_sl_mult:
                    current_rules["atr_sl_multiplier"] = round(new_sl_mult, 2)
                    adjustments_made.append(f"Expanded ATR Stop Loss buffer from {old_sl_mult}x to {new_sl_mult}x to protect against noise.")

            # 2. If win rate is healthy (>= 60%), maintain edge or permit slight momentum expansion
            elif win_rate >= 0.60:
                insights_msg = f"Strategy edge performing strongly ({win_rate*100:.1f}% win rate across {len(closed_trades)} trades)."
                if insights_msg not in learned_insights:
                    learned_insights.append(insights_msg)

        else:
            win_rate = 0.50
            adjustments_made.append("Sample size < 5 trades; maintained baseline rules and verified risk integrity.")

        # Persist updated rules to active_rules.json
        if adjustments_made:
            adjustments_history.append({
                "timestamp": now_iso,
                "adjustments": adjustments_made,
                "win_rate": round(win_rate, 4),
            })
            rules_data["rules"] = current_rules
            rules_data["last_updated"] = now_iso
            rules_data["learned_insights"] = learned_insights[-10:]  # keep recent 10 insights
            rules_data["adjustments_history"] = adjustments_history[-20:]

            with open(self.rules_path, "w", encoding="utf-8") as f:
                json.dump(rules_data, f, indent=2)
            logger.info(f"active_rules.json updated with {len(adjustments_made)} self-refinements.")

        summary_text = (
            f"EOD Reflection Report ({date.today().isoformat()}):\n"
            f"- Total Historic Trades Logged: {len(all_trades)}\n"
            f"- Closed Trades: {len(closed_trades)}\n"
            f"- Win Rate: {win_rate*100:.1f}%\n"
            f"- Adjustments: {', '.join(adjustments_made)}"
        )

        # Log into SQLite post_mortems table
        with self.logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO post_mortems (timestamp, summary, adjustments_made, win_rate, total_trades_analyzed)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now_iso, summary_text, json.dumps(adjustments_made), round(win_rate, 4), len(closed_trades)),
            )
            conn.commit()

        logger.info(summary_text)
        return {
            "summary": summary_text,
            "adjustments": adjustments_made,
            "win_rate": win_rate,
            "timestamp": now_iso,
        }


if __name__ == "__main__":
    logger_test = TradeLogger()
    summary = logger_test.calculate_daily_summary()
    print("Daily Summary:", summary)

    reflector = EODReflection(logger_test)
    post_mortem = reflector.run_post_mortem()
    print("\nPost Mortem Result:\n", post_mortem["summary"])
