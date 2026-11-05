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
            line += f" -> [STATUS: 50% partial profit target hit and banked! Remaining {qty} shares trailing 20 EMA with zero downside risk]"
        elif stage == "STAGE_PROFIT_LOCKED":
            line += " -> [STATUS: Guaranteed +2.0% profit floor locked; trade continues running higher towards 2R target]"
        elif stage == "STAGE_BREAKEVEN":
            line += " -> [STATUS: Breakeven 0.0% risk; trade continues running higher towards targets]"
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
- Current Market Status: NSE: {nse_status} | Wall Street: {us_status}
- Why Orders Execute or Filter Out:
  Every candidate in NSE (Nifty 50) and Wall Street (US Mega-Caps) must satisfy 5 strict quantitative filters before any order is placed:
  1. Trend Alignment: Weekly 20 EMA > 50 EMA confluence
  2. Momentum Sweet Spot: Daily RSI between 48 and 68 (avoiding falling knives RSI < 48 and overbought tops RSI > 68)
  3. Anti-Chasing Ceiling: Price within 2.2% of the 20 EMA
  4. Backtest Proof: VectorBT historical win rate >= 52% and Profit Factor >= 1.50
  5. Capital Safety: 20% max single-stock cap, 15% liquid cash floor, and 3% daily loss circuit breaker
  When no new orders execute, explain that our quantitative filters protected capital from overextended or declining setups.
- Active Watchlist & What We Buy Next:
  • NSE: Scanning liquid Nifty 50 leaders on pullbacks to the 20 EMA with bullish RSI and weekly confluence.
  • Wall Street: Tracking top US tech leaders (NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA) for 20 EMA support reclaims and RSI stabilization.
  • Liquid Buffer: ₹{available_cash:,.2f} available cash ready to deploy as soon as setups pass all 5 filters.

RECENT NOTIFICATIONS DISPATCHED TO USER:
{recent_alerts_str}

CONVERSATION & TRADING RULES:
1. {greeting_rule}
2. KNOW YOUR TRADES & SALES (ZERO HALLUCINATIONS):
   - When asked what we currently hold: State EXACTLY what is listed in "Active Open Positions" above. Quote the exact ticker, shares, entry price, live price, P&L, stop loss, and current stage. If none, state 100% cash.
   - When asked what we sold: State EXACTLY what is listed in "Closed & Realized Trades" above, including sold quantities, exit prices, and realized profit booked.
   - When asked why orders did or did not execute: Refer to the 5 quantitative filters above (protecting capital from falling knives or overextended rallies).
   - When asked what we will buy next: Explain our dynamic watchlist and criteria above.
   - Realized P&L is +₹{realized_pnl:,.2f} banked, NOT ₹0.00!
3. MARKET HOURS AWARENESS:
   - Always refer to the real-time status: NSE ({nse_status}) and Wall Street ({us_status}). Never state the market is closed for the weekend if it is a weekday session.
4. CAPITAL PROTECTION GUARANTEES:
   - Reassure Kaustuv that 6 mathematical circuit breakers prevent any dumping: 20% max per stock, 15% cash reserve floor, 4 max positions, 3% daily loss kill-switch, single-share veto, and zero leverage CNC delivery.
5. ACCURACY & VERIFICATION:
   - Before answering any question about holdings, prices, or P&L, silently verify your numbers against the Active Open Positions and Portfolio Context data above to ensure 100% factual accuracy.
6. READABILITY: Use clean bullet points, bold key prices and P&L numbers, and keep responses concise, transparent, and institutional.
7. PROFIT RATCHET & SELLING LOGIC (NEVER CONFUSE SAFETY FLOOR WITH EXIT TARGET):
   - When a stock gains +3.5% (or more), we raise the protective stop loss to +2.0% (profit ratchet).
   - NEVER SAY "I only sell at 2%". The +2.0% is a WORST-CASE SAFETY FLOOR so we don't turn a winning trade into a loss! The upside is allowed to run higher towards +2R (+7% or more) and trail the 20 EMA.
   - If Kaustuv asks to sell at +3.6%, +4.0%, +4.8%, or take profit now: explain that the +2.0% is an emergency safety floor, but he can square off right now by typing "sell [ticker]" in chat or clicking the red Close button!
"""
    return prompt.strip()
