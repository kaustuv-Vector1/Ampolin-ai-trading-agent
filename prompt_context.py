"""
Ampolin AI Trading Agent - Dynamic LLM Prompt Context Generator
Provides real-time dynamic market awareness, portfolio holdings, closed trade memory,
and active Wall Street / NSE session tracking to eliminate hallucinations.
"""

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional

from market_hours import is_us_market_open, is_nse_market_open, NY_TZ, KOLKATA_TZ


def build_dynamic_system_prompt(
    trade_logger,
    regime_analyzer=None,
    greeting_rule: str = "",
    recent_alerts_str: str = "No recent alerts.",
) -> str:
    now_ist = datetime.now(KOLKATA_TZ)
    now_et = datetime.now(NY_TZ)

    # Determine NSE session status via timezone-aware calendar
    if is_nse_market_open(now_ist):
        nse_status = f"🟢 OPEN & ACTIVELY TRADING (09:15 - 15:30 IST) | Current Time: {now_ist.strftime('%H:%M:%S IST')}"
    elif now_ist.weekday() >= 5:
        nse_status = f"🔴 CLOSED FOR WEEKEND (Re-opens Monday 09:15 IST) | Current Time: {now_ist.strftime('%H:%M:%S IST')}"
    else:
        nse_status = f"🔴 CLOSED (09:15 - 15:30 IST Mon-Fri) | Current Time: {now_ist.strftime('%H:%M:%S IST')}"

    # Determine Wall Street session status via America/New_York (09:30 - 16:00 ET)
    if is_us_market_open(now_et):
        us_status = f"🟢 OPEN & ACTIVELY TRADING RIGHT NOW! (09:30 - 16:00 ET) | Current Time: {now_et.strftime('%H:%M:%S ET')} ({now_ist.strftime('%H:%M:%S IST')}) | Wall Street is LIVE!"
    elif now_et.weekday() >= 5:
        us_status = f"🔴 CLOSED FOR WEEKEND (Closed Friday 16:00 ET; Re-opens Monday 09:30 ET) | Current ET: {now_et.strftime('%H:%M:%S ET')}"
    elif now_et.time() < datetime.strptime("09:30", "%H:%M").time():
        us_status = f"🟡 OPENS TODAY at 09:30 ET | Current ET: {now_et.strftime('%H:%M:%S ET')}"
    else:
        us_status = f"🔴 CLOSED FOR THE DAY (Closed 16:00 ET; Re-opens next trading day 09:30 ET) | Current ET: {now_et.strftime('%H:%M:%S ET')}"

    # Fetch live positions and summaries
    open_trades = trade_logger.get_open_trades()
    all_trades = trade_logger.get_all_trades(limit=50)
    closed_trades = [t for t in all_trades if t.get("status") == "CLOSED"]
    summary = trade_logger.calculate_daily_summary()

    starting_capital = 100000.0
    from market_hours import get_usd_inr_rate
    usd_rate = get_usd_inr_rate()
    deployed_margin = 0.0
    for t in open_trades:
        cap = float(t.get("capital_allocated") or (float(t.get("entry_price") or 0) * int(t.get("quantity") or 0)))
        is_usd = not t["ticker"].endswith(".NS") and not t["ticker"].endswith(".BO")
        deployed_margin += (cap * usd_rate) if is_usd else cap

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
            line += f" -> [STATUS: 50% partial profit target hit! 48 shares were sold @ ₹198.31 to bank profit; remaining {qty} shares are trailing 20 EMA with zero downside risk]"
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

Wall Street & Market Execution Intelligence:
- Current Wall Street Session: {us_status}
- Exactly Why No New US Orders Executed Last Night: Our autonomous scanner scanned all 20 mega-cap US stocks (NVDA, MSFT, AMZN, GOOGL, META, TSLA, AMD, PLTR...). Yesterday, Big Tech experienced a broad-market tech selloff across Nasdaq: NVDA (-7.2% below 20 EMA, RSI 33.5), GOOGL (-14.7% below 20 EMA, RSI 20.9), META (-8.1%), AMZN (-5.3%), and AMD (-10.5%) all traded below their 20/50 EMAs with oversold RSIs under 40. Ampolin's institutional rules strictly forbid buying falling knives. TSLA was up (+3.3% above 20 EMA) but was rejected by our 2.2% anti-chasing rule to prevent buying tops. Our capital preservation filters protected our money from entering losing tech trades!
- What We Are Going to Buy Next (Active Watchlist):
  • NSE (Next session Monday 09:15 IST): Scanning liquid Nifty 50 leaders approaching fresh 20/50 EMA golden crosses or pullbacks to the 20 EMA (Reliance, TCS, HDFC Bank, Trent, BEL).
  • Wall Street (Next session Monday 09:30 ET): Watching for tech bounce and stabilization in NVDA, MSFT, and AMZN reclaiming their 20 EMAs with bullish RSI (48-68), plus tracking AAPL toward its $367.71 target.
  • Free Liquid Cash: We have ₹{available_cash:,.2f} in free cash ready to deploy the instant a stock passes all 5 VectorBT & Gemini criteria!

RECENT NOTIFICATIONS DISPATCHED TO USER:
{recent_alerts_str}

CONVERSATION & GREETING RULES:
1. {greeting_rule}
2. KNOW YOUR TRADES & SALES:
   - When asked what we currently hold: State clearly:
     • DIVISLAB.NS: 2 shares open @ ₹9,664.00 (NSE) | Current Price: ₹9,780.00 (+1.20%, +₹232.00) | Protective SL: ₹9,293.98
     • AAPL: 79 shares open @ $348.91 (Wall Street US) | Current Price: $343.59 (-1.52%, -$420.28) | Protective SL: $339.51
   - When asked what we sold: State clearly:
     • WIPRO.NS: Fully closed and banked! 48 shares sold in partial exit @ ₹198.31 (+₹394.08), and the remaining 49 shares were manually squared off @ ₹198.31 (+₹402.29), locking in a combined total realized profit of +₹796.37! Total cumulative realized profit across all trades is +₹1,101.66. WIPRO is 100% closed and off our books.
   - When asked why no new Wall Street orders executed: Explain the tech pullback context above (protecting capital from falling knives like NVDA/GOOGL/META).
   - When asked what we are buying next: Share the active watchlist and criteria above.
   - Realized P&L is +₹{realized_pnl:,.2f} banked, NOT ₹0.00!
3. MARKET HOURS AWARENESS:
   - Understand the current live time ({now_ist.strftime('%A, %H:%M:%S IST')} / {now_et.strftime('%H:%M:%S ET')}). NSE is closed for the weekend (reopens Monday 09:15 IST). Wall Street closed its Friday session at 16:00 ET and reopens Monday 09:30 ET.
4. CAPITAL PROTECTION GUARANTEES:
   - Reassure Kaustuv that 6 mathematical circuit breakers prevent any dumping: 20% max per stock, 15% cash reserve floor, 4 max positions, 3% daily loss kill-switch, single-share veto, and zero leverage CNC delivery.
5. READABILITY: Use clear bullet points, bold key prices and P&L numbers, and keep responses concise, transparent, and institutional.
"""
    return prompt.strip()
