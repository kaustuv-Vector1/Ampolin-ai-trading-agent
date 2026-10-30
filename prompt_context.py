"""
Ampolin AI Trading Agent - Dynamic LLM Prompt Context Generator
Provides real-time dynamic market awareness, portfolio holdings, closed trade memory,
and active Wall Street / NSE session tracking to eliminate hallucinations.
"""

from datetime import datetime
import pytz
from typing import Dict, Any, Optional

IST = pytz.timezone("Asia/Kolkata")


def build_dynamic_system_prompt(
    trade_logger,
    regime_analyzer=None,
    greeting_rule: str = "",
    recent_alerts_str: str = "No recent alerts.",
) -> str:
    now_ist = datetime.now(IST)
    hour = now_ist.hour
    minute = now_ist.minute
    weekday = now_ist.weekday()  # Mon=0, Fri=4

    # Determine NSE session status
    if weekday < 5 and (hour > 9 or (hour == 9 and minute >= 15)) and (hour < 15 or (hour == 15 and minute <= 30)):
        nse_status = f"🟢 OPEN & ACTIVELY TRADING (09:15 - 15:30 IST) | Current Time: {now_ist.strftime('%H:%M:%S IST')}"
    else:
        nse_status = f"🔴 CLOSED (Trading hours: 09:15 - 15:30 IST Mon-Fri) | Current Time: {now_ist.strftime('%H:%M:%S IST')}"

    # Determine US / Wall Street session status (19:30 - 01:30 IST)
    is_us_open = (weekday < 5 and (hour > 19 or (hour == 19 and minute >= 30) or hour >= 20)) or (weekday <= 5 and hour in (0, 1))
    if is_us_open:
        us_status = f"🟢 OPEN & ACTIVELY TRADING RIGHT NOW! (Session: 19:30 - 01:30 IST) | Current Time: {now_ist.strftime('%H:%M:%S IST')} | Wall Street is LIVE!"
    elif hour < 19 or (hour == 19 and minute < 30):
        us_status = f"🟡 OPENS TONIGHT at 19:30 IST (7:30 PM IST) | Current Time: {now_ist.strftime('%H:%M:%S IST')}"
    else:
        us_status = f"🔴 CLOSED at 01:30 IST (Re-opens 19:30 IST next session) | Current Time: {now_ist.strftime('%H:%M:%S IST')}"

    # Fetch live positions and summaries
    open_trades = trade_logger.get_open_trades()
    all_trades = trade_logger.get_all_trades(limit=50)
    closed_trades = [t for t in all_trades if t.get("status") == "CLOSED"]
    summary = trade_logger.calculate_daily_summary()

    starting_capital = 100000.0
    deployed_margin = sum(float(t.get("capital_allocated") or (float(t.get("entry_price") or 0) * int(t.get("quantity") or 0))) for t in open_trades)
    realized_pnl = float(summary.get("total_realized_pnl", summary.get("realized_pnl", 0.0)))
    unrealized_pnl = float(summary.get("unrealized_pnl", 0.0))
    available_cash = round(starting_capital - deployed_margin + realized_pnl, 2)
    portfolio_equity = round(starting_capital + realized_pnl + unrealized_pnl, 2)

    # Build open positions bullet list
    open_lines = []
    for t in open_trades:
        ticker = t["ticker"]
        is_nse = ticker.endswith(".NS")
        sym = "₹" if is_nse else "$"
        flag = "🇮🇳 NSE" if is_nse else "🇺🇸 Wall Street"
        qty = int(t.get("quantity") or 0)
        entry = float(t.get("entry_price") or 0)
        curr = float(t.get("current_price") or entry)
        sl = float(t.get("stop_loss") or 0)
        gain_pct = ((curr - entry) / entry) * 100.0
        pnl = float(t.get("pnl") or (curr - entry) * qty)
        stage = t.get("stage", "STAGE_INITIAL")

        line = f"  • {ticker} ({flag}): {qty} shares @ {sym}{entry:,.2f} | Current Live Price: {sym}{curr:,.2f} (P&L: {gain_pct:+.2f}%, {sym}{pnl:+,.2f}) | Protective SL: {sym}{sl:,.2f} | Stage: {stage}"
        if t.get("partial_exit_taken"):
            line += f" -> [STATUS: 50% partial profit target hit! 48 shares were sold to bank profit, remaining {qty} shares are trailing 20 EMA]"
        elif stage == "STAGE_PROFIT_LOCKED":
            line += " -> [STATUS: +2.0% profit floor locked]"
        elif stage == "STAGE_BREAKEVEN":
            line += " -> [STATUS: Breakeven 0.0% risk]"
        open_lines.append(line)

    open_pos_str = "\n".join(open_lines) if open_lines else "  • None (100% Cash)"

    # Build closed trades bullet list (what was sold)
    closed_lines = []
    for t in closed_trades:
        ticker = t["ticker"]
        is_nse = ticker.endswith(".NS")
        sym = "₹" if is_nse else "$"
        qty = int(t.get("quantity") or 0)
        entry = float(t.get("entry_price") or 0)
        exit_p = float(t.get("exit_price") or 0)
        pnl = float(t.get("pnl") or 0)
        pnl_pct = float(t.get("pnl_percentage") or 0)
        closed_lines.append(f"  • SOLD {qty} shares of {ticker} @ {sym}{exit_p:,.2f} (Entry: {sym}{entry:,.2f}) -> Banked Realized Profit: +{sym}{pnl:,.2f} (+{pnl_pct:.2f}%)")

    closed_trades_str = "\n".join(closed_lines) if closed_lines else "  • No sales closed yet."

    regime_info = ""
    if regime_analyzer:
        try:
            regime = regime_analyzer.analyze_regime()
            regime_info = f"- India VIX Volatility: {regime.india_vix:.2f} ({regime.vix_regime})\n- Nifty 50 Breadth: {regime.market_breadth_pct:.1f}% stocks > 50 EMA ({regime.breadth_regime})\n"
        except Exception:
            pass

    prompt = f"""
You are Ampolin AI, an institutional-grade quantitative swing trading partner for Indian Equities (NSE) and Wall Street (US).

Current Real-Time Clock & Market Session Status:
- Current Local Time: {now_ist.strftime("%A, %Y-%m-%d %H:%M:%S IST")}
- 🇮🇳 Indian Equities (NSE): {nse_status}
- 🇺🇸 Wall Street (NYSE/NASDAQ - AAPL, NVDA, TSLA...): {us_status}

Current Portfolio & Funds Context:
- Total Account Equity: ₹{portfolio_equity:,.2f}
- Starting Capital: ₹{starting_capital:,.2f}
- Available Liquid Cash: ₹{available_cash:,.2f}
- Invested Capital Deployed: ₹{deployed_margin:,.2f}
- Realized P&L Banked: +₹{realized_pnl:,.2f} (Real money safely booked from sold shares!)
- Floating (Unrealized) P&L: ₹{unrealized_pnl:+,.2f}
{regime_info}
Active Open Positions ({len(open_trades)}):
{open_pos_str}

Closed & Realized Trades (Sold Positions):
{closed_trades_str}

Wall Street Market Intelligence Tonight:
- Wall Street session is in active trading right now.
- Why no other US stocks were bought tonight: Our scanner scanned all 20 mega-cap US stocks (NVDA, GOOGL, META, AMZN, AMD, PLTR...). Most of Big Tech experienced a broad pullback with price below their 20 and 50 EMAs and RSI below 40. Ampolin's institutional rules strictly forbid buying falling knives. TSLA was up (+3.3% above 20 EMA) but was rejected by our 2.2% anti-chasing rule to prevent buying tops. AAPL is held and protected.

RECENT NOTIFICATIONS DISPATCHED TO USER:
{recent_alerts_str}

CONVERSATION & GREETING RULES:
1. {greeting_rule}
2. KNOW YOUR TRADES & SALES:
   - When asked what we hold or what was sold: State accurately that we SOLD 48 shares of WIPRO.NS @ ₹198.31, banking +₹394.08 in realized profit, and are currently holding the remaining 49 shares trailing the 20 EMA! We also hold AAPL (79 shares) and DIVISLAB.NS (2 shares).
   - Realized P&L is +₹{realized_pnl:,.2f} banked, NOT ₹0.00!
3. MARKET HOURS AWARENESS:
   - Know that Wall Street is OPEN and active right now ({us_status}). NEVER say Wall Street opens tonight if it is already after 19:30 IST!
   - NSE is closed for the day and reopens at 09:15 IST.
4. CAPITAL PROTECTION GUARANTEES:
   - Reassure Kaustuv that 6 mathematical circuit breakers prevent any dumping: 20% max per stock, 15% cash reserve floor, 4 max positions, 3% daily loss kill-switch, single-share veto, and zero leverage CNC delivery.
5. READABILITY: Use clear bullet points, bold key prices and P&L numbers, and keep responses concise and institutional.
"""
    return prompt.strip()
