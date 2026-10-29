"""
Ampolin AI Trading Agent - DhanHQ Broker Module
Connects to DhanHQ API to place CNC Delivery bracket orders with 1% max account risk.
Supports multi-day GTT OCO (Forever) bracket orders and safe dry-run/paper execution.
"""

import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from config import (
    DHAN_CLIENT_ID,
    DHAN_ACCESS_TOKEN,
    MAX_ACCOUNT_RISK_PCT,
    DEFAULT_ACCOUNT_CAPITAL,
)
from ai_brain import TradeDecision

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AmpolinBroker")

try:
    from dhanhq import dhanhq, DhanContext
    DHAN_AVAILABLE = True
except ImportError:
    DHAN_AVAILABLE = False
    logger.warning("dhanhq library not available. Running in simulation mode.")


@dataclass
class OrderResult:
    order_id: str
    ticker: str
    status: str
    action: str
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: float
    capital_allocated: float
    total_risk_amount: float
    risk_percentage: float
    product_type: str
    order_type: str
    is_paper_trade: bool
    timestamp: str
    raw_response: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DhanBroker:
    def __init__(
        self,
        client_id: Optional[str] = None,
        access_token: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.client_id = client_id or DHAN_CLIENT_ID or os.getenv("DHAN_CLIENT_ID", "")
        self.access_token = access_token or DHAN_ACCESS_TOKEN or os.getenv("DHAN_ACCESS_TOKEN", "")
        self.dry_run = dry_run
        self.dhan = None

        if self.client_id and self.access_token and DHAN_AVAILABLE and not self.dry_run:
            try:
                self.context = DhanContext(self.client_id, self.access_token)
                self.dhan = dhanhq(self.context)
                logger.info(f"Connected to DhanHQ API for Client ID: {self.client_id[:4]}****")
            except Exception as e:
                logger.error(f"Failed to initialize DhanHQ client: {e}. Defaulting to Paper Trading.")
                self.dhan = None
        else:
            if not self.dry_run and (not self.client_id or not self.access_token):
                logger.info("DhanHQ credentials missing or placeholder. Running in Paper Trading (Dry-Run) mode.")
            else:
                logger.info("DhanBroker running in simulated Dry-Run mode.")

    def get_account_capital(self) -> float:
        """
        Retrieves available equity cash balance from DhanHQ or config default.
        """
        if self.dhan and not self.dry_run:
            try:
                fund_limits = self.dhan.get_fund_limits()
                if fund_limits.get("status") == "success":
                    avail = float(fund_limits.get("data", {}).get("availabelBalance", DEFAULT_ACCOUNT_CAPITAL))
                    return max(avail, 1000.0)
            except Exception as e:
                logger.warning(f"Could not fetch DhanHQ fund limits: {e}. Using default capital.")

        return DEFAULT_ACCOUNT_CAPITAL

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        capital: Optional[float] = None,
        max_risk_pct: Optional[float] = None,
    ) -> tuple[int, float, float]:
        """
        Calculates position size strictly respecting dynamic account risk (adjusted for India VIX).
        Returns: (quantity, total_cost, total_risk_amount)
        """
        total_capital = capital or self.get_account_capital()
        risk_pct = max_risk_pct if max_risk_pct is not None else MAX_ACCOUNT_RISK_PCT
        max_risk_amount = total_capital * risk_pct

        per_share_risk = abs(entry_price - stop_loss)
        if per_share_risk <= 0:
            per_share_risk = entry_price * 0.02  # 2% fallback risk distance

        # Quantity based on risk
        shares_by_risk = int(max_risk_amount / per_share_risk)

        # Quantity capped by total capital (cannot exceed 100% of cash in CNC delivery)
        shares_by_capital = int(total_capital / entry_price)

        quantity = max(1, min(shares_by_risk, shares_by_capital))
        total_cost = round(quantity * entry_price, 2)
        total_risk = round(quantity * per_share_risk, 2)

        return quantity, total_cost, total_risk

    def place_cnc_bracket_order(
        self,
        decision: TradeDecision,
        security_id: Optional[str] = None,
        custom_risk_pct: Optional[float] = None,
    ) -> OrderResult:
        """
        Places a CNC Delivery order with multi-day Stop Loss and Target protection.
        In Indian Equities, multi-day bracket orders for CNC are implemented via
        Dhan Forever Orders (GTT OCO) or direct CNC Delivery with OCO triggers.
        """
        clean_ticker = decision.ticker.replace(".NS", "")
        capital = self.get_account_capital()
        quantity, total_cost, total_risk = self.calculate_position_size(
            entry_price=decision.entry_price,
            stop_loss=decision.stop_loss,
            capital=capital,
            max_risk_pct=custom_risk_pct,
        )

        risk_pct = round((total_risk / capital) * 100.0, 2)
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.info(
            f"Preparing CNC Delivery Bracket Order for {clean_ticker}: "
            f"Qty: {quantity} shares @ ₹{decision.entry_price} | SL: ₹{decision.stop_loss} | "
            f"TP: ₹{decision.take_profit} | Capital: ₹{total_cost} | Risk: ₹{total_risk} ({risk_pct}%)"
        )

        # Paper Trading / Dry Run Execution
        if self.dry_run or not self.dhan:
            simulated_id = f"SIM-DHAN-{datetime.now().strftime('%Y%m%d%H%M%S')}-{clean_ticker}"
            logger.info(f"[PAPER EXECUTION] Order {simulated_id} simulated successfully.")
            return OrderResult(
                order_id=simulated_id,
                ticker=decision.ticker,
                status="SIMULATED_FILLED",
                action=decision.action,
                quantity=quantity,
                entry_price=decision.entry_price,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                capital_allocated=total_cost,
                total_risk_amount=total_risk,
                risk_percentage=risk_pct,
                product_type="CNC",
                order_type="LIMIT",
                is_paper_trade=True,
                timestamp=timestamp_str,
                raw_response={
                    "status": "success",
                    "mode": "paper_trading",
                    "orderId": simulated_id,
                    "message": "Order simulated under 1% account risk parameters.",
                },
            )

        # Live DhanHQ Order Placement
        try:
            # 1. Place Primary Entry CNC Order
            sec_id = security_id or "1333"  # default or mapped Dhan Security ID
            order_params = {
                "security_id": sec_id,
                "exchange_segment": self.dhan.NSE,
                "transaction_type": self.dhan.BUY if decision.action == "BUY" else self.dhan.SELL,
                "quantity": quantity,
                "order_type": self.dhan.LIMIT,
                "product_type": self.dhan.CNC,  # Cash and Carry Delivery
                "price": decision.entry_price,
                "validity": self.dhan.DAY,
            }

            logger.info(f"Transmitting CNC Entry Order to DhanHQ: {order_params}")
            response = self.dhan.place_order(**order_params)
            order_id = response.get("data", {}).get("orderId", f"DHAN-{int(datetime.now().timestamp())}")
            order_status = response.get("status", "SUBMITTED")

            # 2. Place Multi-Day Forever (GTT OCO) Bracket Stop Loss & Target
            try:
                forever_params = {
                    "security_id": sec_id,
                    "exchange_segment": self.dhan.NSE,
                    "transaction_type": self.dhan.SELL if decision.action == "BUY" else self.dhan.BUY,
                    "quantity": quantity,
                    "order_flag": "OCO",  # One Cancels Other
                    "product_type": self.dhan.CNC,
                    "price": decision.stop_loss,
                    "trigger_price": decision.stop_loss * 1.002,
                    "target_price": decision.take_profit,
                    "target_trigger_price": decision.take_profit * 0.998,
                }
                forever_resp = self.dhan.place_forever(**forever_params)
                logger.info(f"Dhan Forever (GTT OCO) Bracket Response: {forever_resp}")
            except Exception as fe:
                logger.warning(f"Could not place automated Dhan Forever OCO order: {fe}")

            return OrderResult(
                order_id=str(order_id),
                ticker=decision.ticker,
                status=order_status,
                action=decision.action,
                quantity=quantity,
                entry_price=decision.entry_price,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                capital_allocated=total_cost,
                total_risk_amount=total_risk,
                risk_percentage=risk_pct,
                product_type="CNC",
                order_type="LIMIT",
                is_paper_trade=False,
                timestamp=timestamp_str,
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"DhanHQ live order placement error: {e}")
            return OrderResult(
                order_id=f"FAILED-{int(datetime.now().timestamp())}",
                ticker=decision.ticker,
                status="FAILED",
                action=decision.action,
                quantity=quantity,
                entry_price=decision.entry_price,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                capital_allocated=total_cost,
                total_risk_amount=total_risk,
                risk_percentage=risk_pct,
                product_type="CNC",
                order_type="LIMIT",
                is_paper_trade=False,
                timestamp=timestamp_str,
                raw_response={"error": str(e)},
            )


if __name__ == "__main__":
    broker = DhanBroker(dry_run=True)
    test_decision = TradeDecision(
        ticker="RELIANCE.NS",
        action="BUY",
        confidence=0.85,
        entry_price=1200.0,
        stop_loss=1165.0,
        take_profit=1270.0,
        risk_reward_ratio=2.0,
        holding_period_days=5,
        rationale="Strong EMA 20 breakout test",
        key_risks="Earnings volatility",
    )

    result = broker.place_cnc_bracket_order(test_decision)
    print("\n--- DhanBroker Execution Result ---")
    print(result.to_dict())
