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
    MAX_CAPITAL_PER_STOCK_PCT,
    MIN_CASH_RESERVE_PCT,
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


DHAN_SECURITY_MAP: Dict[str, str] = {
    "BHARTIARTL": "10604", "DIVISLAB": "10940", "MARUTI": "10999", "LT": "11483",
    "ULTRACEMCO": "11532", "TCS": "11536", "NTPC": "11630", "JSWSTEEL": "11723",
    "GRASIM": "1232", "HDFCBANK": "1333", "HEROMOTOCO": "1348", "TECHM": "13538",
    "HINDALCO": "1363", "HINDUNILVR": "1394", "POWERGRID": "14977", "ADANIPORTS": "15083",
    "APOLLOHOSP": "157", "INFY": "1594", "ITC": "1660", "BAJAJ-AUTO": "16669",
    "BAJAJFINSV": "16675", "NESTLEIND": "17963", "KOTAKBANK": "1922", "TRENT": "1964",
    "M&M": "2031", "COALINDIA": "20374", "SBILIFE": "21808", "ASIANPAINT": "236",
    "ONGC": "2475", "ADANIENT": "25", "RELIANCE": "2885", "SBIN": "3045",
    "SHREECEM": "3103", "BAJFINANCE": "317", "SUNPHARMA": "3351", "TATACONSUM": "3432",
    "TATAMOTORS": "3456", "TATASTEEL": "3499", "TITAN": "3506", "WIPRO": "3787",
    "BEL": "383", "ICICIBANK": "4963", "ZOMATO": "5097", "INDUSINDBK": "5258",
    "BPCL": "526", "BRITANNIA": "547", "AXISBANK": "5900", "CIPLA": "694",
    "HCLTECH": "7229", "DRREDDY": "881", "EICHERMOT": "910"
}


def resolve_security_id(ticker: str) -> Optional[str]:
    """Resolves Dhan security ID for NSE equity instruments. Returns None for US or unmapped tickers."""
    clean = ticker.replace(".NS", "").replace(".BO", "").strip().upper()
    return DHAN_SECURITY_MAP.get(clean)


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
        available_cash: Optional[float] = None,
        currency: str = "INR",
    ) -> tuple[int, float, float]:
        """
        Calculates position size strictly respecting dynamic account risk (adjusted for India VIX)
        and hard anti-concentration / anti-dump guards.
        Converts all USD prices to INR for portfolio risk checks, the 20% cap, and 15% cash floor.
        Returns: (quantity, total_cost_native, total_risk_native)
        """
        from market_hours import get_usd_inr_rate
        usd_rate = get_usd_inr_rate() if currency.upper() == "USD" else 1.0

        total_capital = capital or self.get_account_capital()
        risk_pct = max_risk_pct if max_risk_pct is not None else MAX_ACCOUNT_RISK_PCT
        max_risk_amount = total_capital * risk_pct

        entry_price_inr = entry_price * usd_rate
        per_share_risk = abs(entry_price - stop_loss)
        if per_share_risk <= 0:
            per_share_risk = entry_price * 0.02  # 2% fallback risk distance
        per_share_risk_inr = per_share_risk * usd_rate

        # 1. Quantity strictly limited by 1% account risk budget (in INR)
        shares_by_risk = int(max_risk_amount / per_share_risk_inr)

        # 2. Hard Anti-Concentration Guard: AI can NEVER allocate > 20% of account into any single stock (in INR)
        max_stock_capital = total_capital * MAX_CAPITAL_PER_STOCK_PCT
        shares_by_concentration = int(max_stock_capital / entry_price_inr)

        # Hard Veto: If even 1 single share exceeds the 20% capital ceiling in INR, reject order immediately
        if entry_price_inr > max_stock_capital:
            logger.warning(
                f"[CAPITAL GUARD] Share price ({currency} {entry_price:,.2f} = ₹{entry_price_inr:,.2f}) exceeds single-stock capital ceiling "
                f"(₹{max_stock_capital:,.2f} / {MAX_CAPITAL_PER_STOCK_PCT*100}%). Order rejected by Anti-Dump Guard."
            )
            return 0, 0.0, 0.0

        # 3. Cash Reserve Floor Guard: Ensure trade never breaches 15% mandatory cash liquidity (in INR)
        if available_cash is not None:
            min_cash_reserve = total_capital * MIN_CASH_RESERVE_PCT
            usable_cash = max(0.0, available_cash - min_cash_reserve)
            shares_by_cash = int(usable_cash / entry_price_inr)
            if shares_by_cash <= 0:
                logger.warning(
                    f"[CAPITAL GUARD] Free cash (₹{available_cash:,.2f}) after reserving 15% buffer "
                    f"(₹{min_cash_reserve:,.2f}) cannot afford 1 share of ₹{entry_price_inr:,.2f}. Trade blocked."
                )
                return 0, 0.0, 0.0
            quantity = min(shares_by_risk, shares_by_concentration, shares_by_cash)
        else:
            quantity = min(shares_by_risk, shares_by_concentration)

        if quantity <= 0:
            logger.warning(f"[CAPITAL GUARD] Calculated quantity is 0 for entry price {currency} {entry_price:,.2f}. Trade blocked.")
            return 0, 0.0, 0.0

        total_cost = round(quantity * entry_price, 2)
        total_risk = round(quantity * per_share_risk, 2)

        return quantity, total_cost, total_risk

    def place_broker_stop_loss(
        self,
        order_id: str,
        ticker: str,
        quantity: int,
        stop_loss: float,
        security_id: Optional[str] = None,
        previous_sl_order_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Places a broker-side stop loss order (GTT / Forever on DhanHQ or registered paper broker order).
        - Skips US tickers entirely (Dhan is NSE/BSE only).
        - Requires a valid security_id from instrument map; logs ERROR and returns None if missing (no default 1333).
        - Cancels previous SL order if previous_sl_order_id is passed when stop ratchets up.
        - Does NOT swallow live order errors: returns None on failure so caller knows no stop exists.
        """
        # 1. Skip US tickers entirely
        is_usd = not ticker.endswith(".NS") and not ticker.endswith(".BO")
        if is_usd:
            logger.debug(f"[BROKER SL SKIPPED] {ticker} is a US stock; skipping Dhan broker stop.")
            return None

        # 2. Resolve security_id (no hardcoded fallback!)
        sec_id = security_id or resolve_security_id(ticker)
        if not sec_id:
            logger.error(
                f"[BROKER SL ERROR] No Dhan security_id found for {ticker}. "
                f"Skipping broker stop placement to avoid placing order on wrong stock!"
            )
            return None

        # 3. Cancel previous SL order if moving stop
        if previous_sl_order_id:
            logger.info(f"[BROKER SL MOVE] Cancelling previous broker SL order {previous_sl_order_id} before ratcheting stop.")
            self.cancel_broker_stop_loss(previous_sl_order_id)

        # 4. Dry Run / Paper trading execution
        if self.dry_run or not self.dhan:
            sl_order_id = f"SIM-SL-{order_id}-{int(stop_loss*100)}"
            logger.info(f"[BROKER SL REGISTERED] Paper stop-loss {sl_order_id} set for {quantity} {ticker} @ {stop_loss}")
            return sl_order_id

        # 5. Live DhanHQ placement with exact SDK signature
        try:
            params = {
                "security_id": str(sec_id),
                "exchange_segment": self.dhan.NSE,
                "transaction_type": self.dhan.SELL,
                "product_type": self.dhan.CNC,
                "order_type": self.dhan.LIMIT,
                "quantity": quantity,
                "price": round(stop_loss, 2),
                "trigger_Price": round(stop_loss * 1.002, 2),
                "order_flag": "SINGLE",
            }
            logger.info(f"[BROKER SL SUBMIT] Transmitting Forever SL to DhanHQ for {ticker} (SecID {sec_id}): {params}")
            resp = self.dhan.place_forever(**params)
            if not resp or resp.get("status") != "success":
                err_msg = resp.get("remarks") or resp.get("message") or str(resp)
                logger.error(f"[BROKER SL FAILED] DhanHQ rejected forever SL order for {ticker}: {err_msg}")
                return None

            real_id = str(resp.get("data", {}).get("orderId"))
            logger.info(f"[BROKER SL PLACED] Live DhanHQ stop-loss order placed successfully: {real_id}")
            return real_id
        except Exception as e:
            logger.error(f"[BROKER SL ERROR] Exception placing live Dhan stop-loss for {ticker}: {e}")
            return None

    def cancel_broker_stop_loss(self, sl_order_id: str) -> bool:
        """Cancels an active broker-side forever stop-loss order."""
        if not sl_order_id:
            return False
        if self.dry_run or not self.dhan:
            logger.info(f"[BROKER SL CANCEL] Paper stop-loss order {sl_order_id} cancelled.")
            return True
        try:
            resp = self.dhan.cancel_forever(order_id=str(sl_order_id))
            logger.info(f"[BROKER SL CANCEL] DhanHQ cancel_forever({sl_order_id}) response: {resp}")
            return True
        except Exception as e:
            logger.error(f"[BROKER SL CANCEL ERROR] Failed to cancel Dhan forever order {sl_order_id}: {e}")
            return False

    def place_cnc_bracket_order(
        self,
        decision: TradeDecision,
        security_id: Optional[str] = None,
        custom_risk_pct: Optional[float] = None,
        available_cash: Optional[float] = None,
    ) -> OrderResult:
        """
        Places a CNC Delivery order with multi-day Stop Loss and Target protection.
        In Indian Equities, multi-day bracket orders for CNC are implemented via
        Dhan Forever Orders (GTT OCO) or direct CNC Delivery with OCO triggers.
        """
        clean_ticker = decision.ticker.replace(".NS", "").replace(".BO", "")
        is_usd = not decision.ticker.endswith(".NS") and not decision.ticker.endswith(".BO")
        currency = "USD" if is_usd else "INR"
        capital = self.get_account_capital()
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        quantity, total_cost, total_risk = self.calculate_position_size(
            entry_price=decision.entry_price,
            stop_loss=decision.stop_loss,
            capital=capital,
            max_risk_pct=custom_risk_pct,
            available_cash=available_cash,
            currency=currency,
        )

        if quantity <= 0:
            logger.warning(f"[CAPITAL GUARD] Blocked trade for {clean_ticker}: Single-stock concentration or cash buffer limit reached.")
            return OrderResult(
                order_id=f"REJECTED-{clean_ticker}",
                ticker=decision.ticker,
                status="REJECTED_CAPITAL_GUARD",
                action="REJECTED",
                quantity=0,
                entry_price=decision.entry_price,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                capital_allocated=0.0,
                total_risk_amount=0.0,
                risk_percentage=0.0,
                product_type="CNC",
                order_type="LIMIT",
                timestamp=timestamp_str,
                is_paper_trade=True,
                raw_response={"status": "rejected", "message": "Position blocked by Anti-Dump & Capital Protection Guards (max 20% allocation / 15% cash reserve)."}
            )

        risk_pct = round((total_risk / capital) * 100.0, 2)
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sym_char = "$" if is_usd else "₹"
        logger.info(
            f"Preparing CNC Delivery Bracket Order for {clean_ticker}: "
            f"Qty: {quantity} shares @ {sym_char}{decision.entry_price} | SL: {sym_char}{decision.stop_loss} | "
            f"TP: {sym_char}{decision.take_profit} | Capital: {sym_char}{total_cost} | Risk: {sym_char}{total_risk} ({risk_pct}%)"
        )

        # Paper Trading / Dry Run Execution
        if self.dry_run or not self.dhan:
            simulated_id = f"SIM-DHAN-{datetime.now().strftime('%Y%m%d%H%M%S')}-{clean_ticker}"
            sec_id = security_id or resolve_security_id(decision.ticker)
            sl_id = self.place_broker_stop_loss(simulated_id, decision.ticker, quantity, decision.stop_loss, security_id=sec_id)
            logger.info(f"[PAPER EXECUTION] Order {simulated_id} simulated successfully with broker stop-loss {sl_id}.")
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
                    "broker_sl_order_id": sl_id,
                    "message": "Order simulated under 1% account risk parameters.",
                },
            )

        # Live DhanHQ Order Placement
        try:
            # 1. Place Primary Entry CNC Order
            sec_id = security_id or resolve_security_id(decision.ticker)
            if not sec_id:
                logger.error(f"[LIVE REJECTED] No Dhan security ID mapped for {decision.ticker}. Order aborted.")
                return OrderResult(
                    order_id=f"REJECTED-NO-SEC-ID-{clean_ticker}",
                    ticker=decision.ticker,
                    status="REJECTED_NO_SECURITY_ID",
                    action="REJECTED",
                    quantity=0,
                    entry_price=decision.entry_price,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                    capital_allocated=0.0,
                    total_risk_amount=0.0,
                    risk_percentage=0.0,
                    product_type="CNC",
                    order_type="LIMIT",
                    is_paper_trade=False,
                    timestamp=timestamp_str,
                    raw_response={"status": "rejected", "message": f"No Dhan security ID mapped for {decision.ticker}"}
                )
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
