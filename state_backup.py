"""
Ampolin AI Trading Agent - Automated State Snapshot & Persistence Engine
Provides automatic continuous backup, atomic snapshot persistence,
and instant multi-region recovery across ephemeral Cloud Run container boots.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from config import BASE_DIR, DB_PATH, ACTIVE_RULES_PATH

logger = logging.getLogger("AmpolinBackup")

SNAPSHOT_FILE = BASE_DIR / "portfolio_state_snapshot.json"


class StateBackupManager:
    """
    Automated State Guardian:
    - Atomically captures live SQLite trades, open stages, locked profit ratchets, and realized P&L.
    - Persists snapshots to portfolio_state_snapshot.json.
    - Automatically restores state on fresh cloud container initialization.
    """

    def __init__(self, db_path: Path = DB_PATH, snapshot_path: Path = SNAPSHOT_FILE):
        self.db_path = db_path
        self.snapshot_path = snapshot_path

    def create_snapshot(self) -> Dict[str, Any]:
        """Creates a complete JSON export of the current portfolio state."""
        import sqlite3

        if not self.db_path.exists():
            return {"status": "NO_DATABASE", "trades": [], "timestamp": datetime.now().isoformat()}

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM trades ORDER BY id ASC")
        trades = [dict(row) for row in cursor.fetchall()]

        cursor.execute("SELECT * FROM daily_pnl ORDER BY date DESC LIMIT 30")
        daily_pnl = [dict(row) for row in cursor.fetchall()]

        cursor.execute("SELECT * FROM post_mortems ORDER BY id DESC LIMIT 10")
        reflections = [dict(row) for row in cursor.fetchall()]
        conn.close()

        rules_data = {}
        if ACTIVE_RULES_PATH.exists():
            try:
                with open(ACTIVE_RULES_PATH, "r", encoding="utf-8") as f:
                    rules_data = json.load(f)
            except Exception:
                pass

        snapshot = {
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "open_positions_count": sum(1 for t in trades if t.get("status") in ("SUBMITTED", "SIMULATED_FILLED", "OPEN")),
            "total_trades_count": len(trades),
            "trades": trades,
            "daily_pnl": daily_pnl,
            "post_mortems": reflections,
            "active_rules": rules_data,
        }

        # Atomically write snapshot to disk
        try:
            temp_path = self.snapshot_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
            temp_path.replace(self.snapshot_path)
            logger.info(f"Portfolio state snapshot successfully saved ({len(trades)} trades).")
        except Exception as e:
            logger.error(f"Failed to save portfolio snapshot: {e}")

        return snapshot

    def restore_from_snapshot(self, snapshot_data: Optional[Dict[str, Any]] = None) -> bool:
        """Restores SQLite database from a JSON snapshot if DB is empty or missing."""
        import sqlite3

        data = snapshot_data
        if not data and self.snapshot_path.exists():
            try:
                with open(self.snapshot_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Could not read snapshot file: {e}")
                return False

        if not data or "trades" not in data:
            return False

        trades = data.get("trades", [])
        if not trades:
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check existing count
        cursor.execute("SELECT COUNT(*) FROM trades")
        existing_count = cursor.fetchone()[0]

        if existing_count > 0 and not snapshot_data:
            # Already populated, no need to overwrite
            conn.close()
            return True

        logger.info(f"Restoring {len(trades)} trades from state snapshot...")
        for t in trades:
            one_r = t.get("one_r")
            if one_r is None or one_r == 0:
                entry_p = float(t.get("entry_price") or 0)
                sl_p = float(t.get("stop_loss") or 0)
                tp_p = float(t.get("take_profit") or 0)
                one_r = round(abs(tp_p - entry_p) / 2.0, 4) if tp_p > entry_p else round(abs(entry_p - sl_p), 4)

            cursor.execute("""
                INSERT OR REPLACE INTO trades (
                    order_id, ticker, action, entry_price, stop_loss, take_profit,
                    quantity, initial_quantity, one_r, capital_allocated, risk_amount,
                    status, stage, partial_exit_taken, exit_price, pnl, pnl_percentage,
                    confidence, rationale, current_price, entry_time, exit_time, is_paper_trade, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                t.get("order_id"), t.get("ticker"), t.get("action"),
                t.get("entry_price"), t.get("stop_loss"), t.get("take_profit"),
                t.get("quantity"), t.get("initial_quantity"), one_r, t.get("capital_allocated"),
                t.get("risk_amount"), t.get("status"), t.get("stage", "STAGE_INITIAL"),
                t.get("partial_exit_taken", 0), t.get("exit_price"), t.get("pnl", 0.0),
                t.get("pnl_percentage", 0.0), t.get("confidence"), t.get("rationale"),
                t.get("current_price"), t.get("entry_time"), t.get("exit_time"),
                t.get("is_paper_trade", 1), t.get("metadata_json")
            ))

        conn.commit()
        conn.close()
        logger.info("Portfolio state restoration complete.")
        return True
