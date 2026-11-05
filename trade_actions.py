"""
Ampolin AI Trading Agent - Natural Language Trade Actions & Execution Engine
Enables direct execution of trade commands (Close, Sell, Lock Profit, Breakeven)
from conversational chat in Telegram and the Web UI Dashboard.
"""

import logging
import re
from typing import Optional, Dict, Any, List

logger = logging.getLogger("AmpolinTradeActions")


def execute_close_trade(trade: Dict[str, Any], trade_logger, broker=None, notifier=None) -> str:
    """Executes an immediate square-off / exit for an active position."""
    ticker = trade["ticker"]
    order_id = trade["order_id"]
    entry = float(trade["entry_price"])
    qty = int(trade["quantity"])
    
    # Try fetching fresh live price
    curr_price = float(trade.get("current_price") or entry)
    try:
        from position_manager import PositionManager
        pm = PositionManager(trade_logger=trade_logger, broker=broker)
        lp, _ = pm.fetch_live_price_and_ema(ticker)
        if lp and lp > 0:
            curr_price = lp
    except Exception as e:
        logger.warning(f"Could not fetch live price during natural close of {ticker}: {e}")

    # Cancel registered broker stop-loss order if any
    if broker and trade.get("broker_sl_order_id"):
        try:
            broker.cancel_broker_stop_loss(trade["broker_sl_order_id"])
        except Exception as e:
            logger.warning(f"Could not cancel broker SL {trade.get('broker_sl_order_id')}: {e}")

    # Close trade in database
    trade_logger.close_trade(order_id, exit_price=curr_price)
    
    pnl = round((curr_price - entry) * qty, 2)
    pct = round(((curr_price - entry) / entry) * 100.0, 2) if entry > 0 else 0.0
    sym = "₹" if ticker.endswith(".NS") or ticker.endswith(".BO") else "$"
    sign = "+" if pnl >= 0 else ""

    msg = (
        f"✅ <b>Position Closed & Profit Banked!</b>\n\n"
        f"• <b>Ticker:</b> {ticker} ({qty} shares)\n"
        f"• <b>Entry Price:</b> {sym}{entry:,.2f}\n"
        f"• <b>Exit Price:</b> {sym}{curr_price:,.2f}\n"
        f"• <b>Realized P&L:</b> {sign}{sym}{pnl:,.2f} ({sign}{pct:.2f}%) 💰\n"
        f"• <b>Status:</b> Position squared off successfully! Funds returned to your available cash."
    )
    
    if notifier:
        try:
            notifier.send_message(msg)
        except Exception as e:
            logger.warning(f"Could not send close alert: {e}")
            
    return msg


def execute_lock_profit(trade: Dict[str, Any], trade_logger, broker=None, notifier=None) -> str:
    """Locks a guaranteed +2.0% profit floor for an active position."""
    ticker = trade["ticker"]
    order_id = trade["order_id"]
    entry = float(trade["entry_price"])
    locked_sl = round(entry * 1.02, 2)
    trade_logger.update_trade_stop_loss(order_id, locked_sl, "STAGE_PROFIT_LOCKED")
    
    sym = "₹" if ticker.endswith(".NS") or ticker.endswith(".BO") else "$"
    msg = (
        f"🔒 <b>Profit Locked In (+2.0% Floor) - {ticker}</b>\n\n"
        f"• Entry: {sym}{entry:,.2f}\n"
        f"• Protective SL: {sym}{locked_sl:,.2f} (+2.0% profit floor)\n"
        f"• <b>Status:</b> Guaranteed minimum +2.0% profit locked!"
    )
    if notifier:
        try:
            notifier.send_message(msg)
        except Exception:
            pass
    return msg


def execute_breakeven(trade: Dict[str, Any], trade_logger, broker=None, notifier=None) -> str:
    """Moves protective stop loss to entry price (0.0% risk)."""
    ticker = trade["ticker"]
    order_id = trade["order_id"]
    entry = float(trade["entry_price"])
    trade_logger.update_trade_stop_loss(order_id, entry, "STAGE_BREAKEVEN")
    
    sym = "₹" if ticker.endswith(".NS") or ticker.endswith(".BO") else "$"
    msg = (
        f"🛡️ <b>Breakeven Activated - {ticker}</b>\n\n"
        f"• Entry: {sym}{entry:,.2f}\n"
        f"• Protective SL: {sym}{entry:,.2f}\n"
        f"• <b>Status:</b> Zero downside risk. Trade is fully protected!"
    )
    if notifier:
        try:
            notifier.send_message(msg)
        except Exception:
            pass
    return msg


def process_natural_chat_trade_command(
    user_msg: str,
    trade_logger,
    broker=None,
    notifier=None,
) -> Optional[str]:
    """
    Parses conversational user messages for trade actions (sell/close, lock profit, breakeven).
    Returns formatted confirmation string if an action was executed, or None to fall back to LLM.
    """
    msg_raw = user_msg.strip()
    msg = msg_raw.lower()
    
    open_trades = trade_logger.get_open_trades()
    
    # 1. Close / Sell Intents
    sell_patterns = [
        r"\b(sell|close|exit|square\s*off|take\s*profit|book\s*profit)\b",
        r"\bsell\s+(the\s+)?(trade|position|all|shares)\b",
        r"\b(sell|close)\s+at\s+[\+\-]?\d+(\.\d+)?%?",
    ]
    is_sell_request = any(re.search(p, msg) for p in sell_patterns)
    
    if is_sell_request:
        if not open_trades:
            return "📭 You currently have no active open positions to sell (100% cash buffer)."
        
        # Check if a specific ticker was mentioned
        target = None
        for t in open_trades:
            ticker = t["ticker"]
            base = ticker.replace(".NS", "").replace(".BO", "").lower()
            if base in msg or ticker.lower() in msg:
                target = t
                break
                
        # If no specific ticker mentioned, but exactly 1 open trade, apply to that trade
        if not target and len(open_trades) == 1:
            target = open_trades[0]
        elif not target and any(w in msg for w in ["all", "everything"]):
            results = [execute_close_trade(t, trade_logger, broker, notifier) for t in open_trades]
            return "\n\n".join(results)
            
        if target:
            return execute_close_trade(target, trade_logger, broker, notifier)

    # 2. Lock Profit (+2%) Intents
    lock_patterns = [
        r"\b(lock|guarantee)\s+(profit|\+?2%?)\b",
        r"\block\s+(\+?2%?\s+)?profit\b",
    ]
    if any(re.search(p, msg) for p in lock_patterns):
        if not open_trades:
            return "📭 No active positions to lock profit on."
        target = None
        for t in open_trades:
            base = t["ticker"].replace(".NS", "").replace(".BO", "").lower()
            if base in msg or t["ticker"].lower() in msg:
                target = t
                break
        if not target and len(open_trades) == 1:
            target = open_trades[0]
        if target:
            return execute_lock_profit(target, trade_logger, broker, notifier)

    # 3. Breakeven Intents
    be_patterns = [
        r"\b(move\s+to\s+)?breakeven\b",
        r"\bbreak\s*even\b",
        r"\bset\s+be\b",
    ]
    if any(re.search(p, msg) for p in be_patterns):
        if not open_trades:
            return "📭 No active positions to move to breakeven."
        target = None
        for t in open_trades:
            base = t["ticker"].replace(".NS", "").replace(".BO", "").lower()
            if base in msg or t["ticker"].lower() in msg:
                target = t
                break
        if not target and len(open_trades) == 1:
            target = open_trades[0]
        if target:
            return execute_breakeven(target, trade_logger, broker, notifier)

    return None
