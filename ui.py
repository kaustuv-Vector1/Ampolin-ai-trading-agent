"""
Ampolin AI Trading Agent - Modern Web UI & API Server
FastAPI interactive dashboard for real-time scanning, VectorBT backtesting,
Gemini AI Brain decision inspection, DhanHQ trade execution, and SQLite portfolio review.
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import uvicorn
from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    NIFTY_50_TICKERS,
    ACTIVE_RULES_PATH,
    DB_PATH,
    DEFAULT_TIMEFRAME,
)
from scanner import SwingScanner
from sandbox import VectorBTSandbox
from ai_brain import GeminiBrain
from broker import DhanBroker
from reflection import TradeLogger, EODReflection
from telegram_notifier import TelegramNotifier, get_recent_notifications
from market_regime import MarketRegimeAnalyzer
from position_manager import PositionManager
from main import AmpolinPipeline
from llm_client import MultiProviderLLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AmpolinUI")

app = FastAPI(title="Ampolin AI Trading Agent", version="1.0.0")

# Mount assets directory for PWA icons
assets_path = Path(__file__).resolve().parent / "assets"
if assets_path.exists():
    app.mount("/static", StaticFiles(directory=str(assets_path)), name="static")

@app.get("/manifest.json")
def pwa_manifest():
    return JSONResponse({
        "name": "Ampolin AI Trading Agent",
        "short_name": "Ampolin AI",
        "description": "Autonomous Swing Trading Terminal for NSE & Wall Street",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#000000",
        "theme_color": "#000000",
        "orientation": "portrait-primary",
        "icons": [
            {
                "src": "/static/ampolin_icon_sq.png",
                "sizes": "192x192",
                "type": "image/png",
                "purpose": "any maskable"
            },
            {
                "src": "/static/ampolin_icon_sq.png",
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "any maskable"
            }
        ]
    })

# Pipeline singletons
scanner = SwingScanner()
sandbox = VectorBTSandbox()
brain = GeminiBrain()
broker = DhanBroker(dry_run=True)
trade_logger = TradeLogger()
reflection = EODReflection(trade_logger)
notifier = TelegramNotifier()
regime_analyzer = MarketRegimeAnalyzer(sample_size=15)
position_manager = PositionManager(trade_logger=trade_logger, notifier=notifier, broker=broker)
llm_brain = MultiProviderLLM()

# Cache for latest scan results in memory
latest_scan_results = {
    "timestamp": None,
    "timeframe": "1d",
    "candidates": [],
    "approved_setups": [],
    "ai_decisions": [],
    "executed_trades": [],
}


class ScanRequest(BaseModel):
    timeframe: str = "1d"
    tickers: Optional[List[str]] = None
    dry_run: bool = True


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None


@app.post("/api/chat")
def in_app_chat(req: ChatRequest):
    """Direct in-app conversation with Ampolin AI with multi-turn memory and instant Groq fallback."""
    user_msg = (req.message or "").strip()
    if not user_msg:
        return {"reply": "Please type a message to start chatting!"}

    open_trades = trade_logger.get_open_trades()
    positions_str = ", ".join([f"{t['ticker']} (entry: {t['entry_price']}, stage: {t.get('stage','INITIAL')})" for t in open_trades]) or "None (100% Cash)"
    summary = trade_logger.calculate_daily_summary()
    regime = regime_analyzer.analyze_regime()

    recent_alerts = get_recent_notifications()
    recent_alerts_str = "\n".join([f"• {a}" for a in recent_alerts[-3:]]) if recent_alerts else "No alerts sent recently."

    is_first_turn = not req.history or len(req.history) == 0
    if is_first_turn:
        greeting_rule = "GREETING RULE: Since this is the FIRST message of the session, you may start with a brief, friendly greeting like 'Hey Kaustuv.' followed immediately by the answer."
    else:
        greeting_rule = "CRITICAL GREETING RULE: This is an ongoing conversation / follow-up message. DO NOT SAY 'Hey Kaustuv', 'Hello Kaustuv', 'Hi Kaustuv', or ANY opening greeting whatsoever. Start directly with the answer or insight."

    system_prompt = f"""
You are Ampolin AI, an institutional-grade quantitative swing trading partner for Indian Equities (NSE) and Wall Street (US).

Current System Context:
- Active Open Positions: {positions_str}
- Floating (Unrealized) P&L: ₹{summary.get('unrealized_pnl', 0.0):,.2f}
- Realized P&L Banked: ₹{summary.get('realized_pnl', 0.0):,.2f}
- Total Portfolio P&L: ₹{summary.get('total_pnl', 0.0):,.2f}
- India VIX Volatility: {regime.india_vix:.2f} ({regime.vix_regime})
- Market Breadth: {regime.market_breadth_pct:.1f}% stocks > 50 EMA ({regime.breadth_regime})
- Trading Rules: 1% risk budgeting, 20/50 EMA breakouts, RSI 48-68 momentum, VectorBT statistical edge (Win Rate >= 52%, Profit Factor >= 1.50), 1R breakeven, 3.5% profit ratchet (locks +2.0%), and 20 EMA trailing.

RECENT SYSTEM ALERTS SENT TO USER:
{recent_alerts_str}

CONVERSATION & GREETING RULES:
1. {greeting_rule}
2. TRANSPARENCY ON "ARE THESE REAL?": If Kaustuv asks "WAIT IS THIS REAL?", "Is that real?", "Are these real trades?", or asks about any recent trade alert:
   - Answer directly and honestly: Explain that these are live institutional PAPER TRADES executing in real-time simulation (dry-run mode). Live market data from NSE & Wall Street is 100% real, while execution is simulated without risking real money from his account.
   - Explain the exact trade details (e.g. WIPRO.NS, AAPL, etc.), current prices, and explain the mathematical rule that triggered (such as the +3.5% Profit Ratchet raising the stop loss to ₹193.9 to lock in a guaranteed +2.0% profit floor).
   - NEVER contradict recent alerts or claim to be sitting on 100% cash when alerts were dispatched or paper trades are active!
3. MULTI-TURN CONTINUITY: Follow up naturally on what was previously discussed in the dialogue thread. Reference earlier stocks, rules, or questions seamlessly without asking him to repeat himself.
4. PROACTIVE ENGAGEMENT: Always end your answers with an actionable, intelligent follow-up question or offer (e.g. offering to run an on-demand scan on Nifty 50 or Wall Street, review trailing stops, or inspect a specific chart).
5. READABILITY: Use clean bullet points, bolding, and occasional trading emojis (📈, 🛡️, 💰). Keep it scannable on mobile.
"""
    reply = llm_brain.generate_chat_response(user_prompt=user_msg, system_prompt=system_prompt, history=req.history)
    if not reply:
        reply = "I'm currently updating live feeds. Please ask again in a few seconds!"
    elif not is_first_turn:
        reply = re.sub(r'^(Hey|Hello|Hi)\s+Kaustuv[\.,!:\s]*\n*', '', reply, flags=re.IGNORECASE).strip()
        reply = re.sub(r'^\*{1,2}(Hey|Hello|Hi)\s+Kaustuv[\.,!:\s]*\*{1,2}\n*', '', reply, flags=re.IGNORECASE).strip()

    return {"reply": reply}


@app.get("/api/status")
def get_system_status():
    open_trades = trade_logger.get_open_trades()
    summary = trade_logger.calculate_daily_summary()
    regime = regime_analyzer.analyze_regime()

    starting_capital = 100000.0
    deployed_margin = sum(float(t.get("capital_allocated") or 0.0) for t in open_trades)
    realized_pnl = summary.get("realized_pnl", 0.0)
    unrealized_pnl = summary.get("unrealized_pnl", 0.0)
    total_pnl = summary.get("total_pnl", 0.0)
    available_cash = round(starting_capital - deployed_margin + realized_pnl, 2)
    portfolio_equity = round(starting_capital + total_pnl, 2)

    return {
        "status": "ONLINE",
        "system_name": "Ampolin AI Trading Agent",
        "starting_capital": starting_capital,
        "available_cash": available_cash,
        "deployed_margin": round(deployed_margin, 2),
        "portfolio_equity": portfolio_equity,
        "open_positions": len(open_trades),
        "daily_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "win_rate": summary.get("win_rate", 0.0),
        "broker_mode": "Paper Trading (Simulation)" if broker.dry_run or not broker.dhan else "DhanHQ Live",
        "ai_model": brain.model_name,
        "nifty_50_count": len(NIFTY_50_TICKERS),
        "regime": regime.to_dict(),
    }


@app.post("/api/manage-positions")
def manage_positions():
    actions = position_manager.manage_open_positions()
    return {"status": "SUCCESS", "actions_taken": actions}


@app.get("/api/rules")
def get_rules():
    if ACTIVE_RULES_PATH.exists():
        with open(ACTIVE_RULES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@app.post("/api/rules")
def update_rules(payload: dict):
    if ACTIVE_RULES_PATH.exists():
        with open(ACTIVE_RULES_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return {"status": "SUCCESS", "message": "active_rules.json updated successfully"}


@app.get("/api/trades")
def get_trades(limit: int = 50):
    trades = trade_logger.get_all_trades(limit=limit)
    summary = trade_logger.calculate_daily_summary()
    return {"trades": trades, "summary": summary}


@app.post("/api/scan")
def trigger_scan(req: ScanRequest):
    selected_tickers = req.tickers or NIFTY_50_TICKERS[:12]  # top liquid batch for fast response
    setups = scanner.scan_universe(selected_tickers, timeframe=req.timeframe)
    approved_pairs = sandbox.filter_candidates(setups)

    serialized_approved = []
    ai_decisions = []
    for s, b in approved_pairs:
        dec = brain.evaluate_setup(s, b)
        ai_decisions.append(dec.model_dump())
        serialized_approved.append({
            "setup": s.to_dict(),
            "backtest": b.to_dict(),
            "decision": dec.model_dump(),
        })

    latest_scan_results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    latest_scan_results["timeframe"] = req.timeframe
    latest_scan_results["candidates"] = [s.to_dict() for s in setups]
    latest_scan_results["approved_setups"] = serialized_approved

    return {
        "status": "SUCCESS",
        "candidates_count": len(setups),
        "approved_count": len(approved_pairs),
        "data": serialized_approved,
    }


@app.post("/api/run-pipeline")
def run_full_pipeline(req: ScanRequest):
    pipeline = AmpolinPipeline(dry_run=req.dry_run, timeframe=req.timeframe)
    tickers = req.tickers or NIFTY_50_TICKERS[:15]
    result = pipeline.run_pipeline(tickers=tickers, max_trades_to_execute=3)
    return result


@app.post("/api/eod-reflection")
def run_eod_reflection():
    result = reflection.run_post_mortem()
    notifier.notify_eod_reflection(result)
    return result


@app.get("/", response_class=HTMLResponse)
def dashboard_html():
    return DASHBOARD_HTML_PAGE


DASHBOARD_HTML_PAGE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Ampolin AI Trading Agent | Terminal</title>
  <link rel="manifest" href="/manifest.json">
  <link rel="icon" type="image/png" href="/static/ampolin_icon_sq.png">
  <link rel="apple-touch-icon" href="/static/ampolin_icon_sq.png">
  <meta name="mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta name="apple-mobile-web-app-title" content="Ampolin AI">
  <meta name="theme-color" content="#000000">
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <style>
    body { background-color: #000000; font-family: -apple-system, BlinkMacSystemFont, "Geist", "Inter", "Segoe UI", Roboto, sans-serif; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: #000000; }
    ::-webkit-scrollbar-thumb { background: #27272a; border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: #3f3f46; }
    .border-subtle { border-color: #27272a; }
  </style>
</head>
<body class="bg-black text-zinc-300 min-h-screen flex flex-col antialiased selection:bg-white selection:text-black">

  <!-- Top Minimalist Header -->
  <header class="bg-black/95 backdrop-blur border-b border-zinc-850 px-4 sm:px-6 py-3.5 flex justify-between items-center sticky top-0 z-50">
    <div class="flex items-center space-x-3">
      <img src="/static/ampolin_icon_sq.png" alt="Ampolin AI" class="w-8 h-8 rounded-lg border border-zinc-800 object-cover shrink-0 shadow-sm">
      <div>
        <div class="flex items-center gap-2">
          <span class="text-sm sm:text-base font-bold tracking-tight text-white mono">AMPOLIN</span>
          <span class="text-zinc-400 font-normal text-[10px] px-1.5 py-0.5 rounded border border-zinc-800 bg-zinc-900 mono">QUANT</span>
        </div>
        <p class="text-[10px] sm:text-[11px] text-zinc-500 tracking-tight">Autonomous Dual-Market Terminal &bull; NSE &amp; Wall Street</p>
      </div>
    </div>

    <!-- Unified Segmented Status Capsule -->
    <div class="flex items-center bg-zinc-950 border border-zinc-800 rounded-lg px-2.5 sm:px-3 py-1.5 text-[11px] mono divide-x divide-zinc-800 shadow-inner">
      <div class="flex items-center gap-1.5 pr-2.5 sm:pr-3">
        <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
        <span class="text-white font-semibold" id="badge-engine">ACTIVE</span>
      </div>
      <div class="px-2.5 sm:px-3 text-zinc-400 hidden sm:block">
        <span id="badge-brain">Gemini &bull; Groq</span>
      </div>
      <div class="px-2.5 sm:px-3 text-zinc-400 hidden md:block">
        <span>Risk: 1.0% Max</span>
      </div>
      <div class="pl-2.5 sm:pl-3 text-zinc-300 font-medium">
        <span id="live-time">--:--:-- IST</span>
      </div>
    </div>
  </header>

  <!-- Metric Overview Bar (Minimalist Cards) -->
  <div class="max-w-7xl mx-auto w-full px-4 sm:px-6 pt-5 sm:pt-6 grid grid-cols-2 md:grid-cols-6 gap-2.5 sm:gap-3.5">
    <div class="bg-zinc-950 p-3.5 sm:p-4 rounded-xl border border-zinc-800/80">
      <div class="flex justify-between items-center text-zinc-500 text-[10px] sm:text-[11px] font-medium tracking-wider uppercase mb-1">
        <span>Portfolio Equity</span>
        <i class="fa-solid fa-wallet text-zinc-600"></i>
      </div>
      <div class="text-xl sm:text-2xl font-bold text-white mono" id="metric-equity">₹100,000.00</div>
      <div class="text-[10px] sm:text-[11px] text-zinc-400 mt-1" id="metric-funds-breakdown">Cash: ₹100,000 | Invested: ₹0</div>
    </div>

    <div class="bg-zinc-950 p-3.5 sm:p-4 rounded-xl border border-zinc-800/80">
      <div class="flex justify-between items-center text-zinc-500 text-[10px] sm:text-[11px] font-medium tracking-wider uppercase mb-1">
        <span>Total P&amp;L (Floating)</span>
        <i class="fa-solid fa-chart-line text-zinc-600"></i>
      </div>
      <div class="text-xl sm:text-2xl font-bold text-white mono" id="metric-pnl">₹0.00</div>
      <div class="text-[10px] sm:text-[11px] text-zinc-400 mt-1" id="metric-pnl-breakdown">Floating: ₹0.00 | Realized: ₹0.00</div>
    </div>

    <div class="bg-zinc-950 p-3.5 sm:p-4 rounded-xl border border-zinc-800/80">
      <div class="flex justify-between items-center text-zinc-500 text-[10px] sm:text-[11px] font-medium tracking-wider uppercase mb-1">
        <span>Market Breadth</span>
        <i class="fa-solid fa-bars-progress text-zinc-600"></i>
      </div>
      <div class="text-xl sm:text-2xl font-bold text-white mono" id="metric-breadth">--%</div>
      <div class="text-[10px] sm:text-[11px] text-zinc-600 mt-1" id="metric-breadth-regime">Nifty 50 &gt; 50 EMA</div>
    </div>

    <div class="bg-zinc-950 p-3.5 sm:p-4 rounded-xl border border-zinc-800/80">
      <div class="flex justify-between items-center text-zinc-500 text-[10px] sm:text-[11px] font-medium tracking-wider uppercase mb-1">
        <span>India VIX</span>
        <i class="fa-solid fa-bolt text-zinc-600"></i>
      </div>
      <div class="text-xl sm:text-2xl font-bold text-white mono" id="metric-vix">--</div>
      <div class="text-[10px] sm:text-[11px] text-zinc-600 mt-1" id="metric-dynamic-risk">Risk: --%</div>
    </div>

    <div class="bg-zinc-950 p-3.5 sm:p-4 rounded-xl border border-zinc-800/80">
      <div class="flex justify-between items-center text-zinc-500 text-[10px] sm:text-[11px] font-medium tracking-wider uppercase mb-1">
        <span>Win Rate</span>
        <i class="fa-solid fa-bullseye text-zinc-600"></i>
      </div>
      <div class="text-xl sm:text-2xl font-bold text-white mono" id="metric-winrate">0.0%</div>
      <div class="text-[10px] sm:text-[11px] text-zinc-600 mt-1">VectorBT Filtered (&ge;52%)</div>
    </div>

    <div class="bg-zinc-950 p-3.5 sm:p-4 rounded-xl border border-zinc-800/80 col-span-2 md:col-span-1">
      <div class="flex justify-between items-center text-zinc-500 text-[10px] sm:text-[11px] font-medium tracking-wider uppercase mb-1">
        <span>Active Positions</span>
        <i class="fa-solid fa-layer-group text-zinc-600"></i>
      </div>
      <div class="text-xl sm:text-2xl font-bold text-white mono" id="metric-open-trades">0</div>
      <div class="text-[10px] sm:text-[11px] text-zinc-600 mt-1">1R Breakeven &bull; 3.5% Lock</div>
    </div>
  </div>

  <!-- Primary Action & Control Toolbar (Mobile Thumb-Friendly Grid) -->
  <div class="max-w-7xl mx-auto w-full px-4 sm:px-6 py-4 flex flex-col sm:flex-row gap-3 items-stretch sm:items-center justify-between">
    <div class="grid grid-cols-2 sm:flex sm:flex-wrap gap-2">
      <button onclick="runPipeline()" id="btn-pipeline" class="bg-white hover:bg-zinc-200 text-black font-semibold px-3.5 py-2.5 sm:py-2 rounded-lg text-xs tracking-wide flex items-center justify-center gap-2 transition active:scale-95 shadow-sm">
        <i class="fa-solid fa-play text-[10px]"></i> Run Pipeline
      </button>

      <button onclick="runScan()" id="btn-scan" class="bg-zinc-900 hover:bg-zinc-800 text-zinc-200 font-medium px-3.5 py-2.5 sm:py-2 rounded-lg border border-zinc-800 flex items-center justify-center gap-2 transition text-xs">
        <i class="fa-solid fa-magnifying-glass text-[10px] text-zinc-400"></i> Scan NSE
      </button>

      <button onclick="managePositions()" id="btn-manage" class="bg-zinc-900 hover:bg-zinc-800 text-zinc-200 font-medium px-3.5 py-2.5 sm:py-2 rounded-lg border border-zinc-800 flex items-center justify-center gap-2 transition text-xs">
        <i class="fa-solid fa-arrows-split-up-and-left text-[10px] text-zinc-400"></i> Profit Lock
      </button>

      <button onclick="runEOD()" id="btn-eod" class="bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 font-medium px-3 py-2.5 sm:py-2 rounded-lg border border-zinc-800 flex items-center justify-center gap-2 transition text-xs">
        <i class="fa-solid fa-microscope text-[10px]"></i> EOD Reflection
      </button>
    </div>

    <div class="flex items-center justify-between sm:justify-end gap-2 text-xs">
      <div class="flex items-center bg-zinc-950 px-3 py-1.5 rounded-lg border border-zinc-800 gap-2 flex-1 sm:flex-initial">
        <span class="text-zinc-500 text-[11px]">Timeframe:</span>
        <select id="select-tf" class="bg-transparent border-0 text-white mono text-xs focus:outline-none cursor-pointer">
          <option value="1d" selected class="bg-zinc-900">1D (Daily)</option>
          <option value="4h" class="bg-zinc-900">4H (Swing)</option>
        </select>
      </div>

      <div class="flex items-center bg-zinc-950 px-3 py-1.5 rounded-lg border border-zinc-800 gap-2 flex-1 sm:flex-initial justify-center">
        <span class="text-zinc-500 text-[11px]">Mode:</span>
        <span class="text-zinc-300 border border-zinc-700/80 bg-zinc-900/80 px-2 py-0.5 rounded text-[11px] mono font-medium">Paper</span>
      </div>
    </div>
  </div>

  <!-- Main Content Layout Grid -->
  <main class="max-w-7xl mx-auto w-full px-6 pb-12 flex-1 grid grid-cols-1 lg:grid-cols-3 gap-5">

    <!-- Left 2 Columns: Live Candidates, VectorBT Verification & AI Decisions -->
    <div class="lg:col-span-2 space-y-5">

      <!-- Candidate Setups & Backtests Section -->
      <section class="bg-zinc-950 rounded-xl border border-zinc-800/80 p-5">
        <div class="flex justify-between items-center mb-3.5">
          <div>
            <h2 class="text-sm font-semibold text-white tracking-wide flex items-center gap-2">
              <i class="fa-solid fa-crosshairs text-zinc-400"></i> Candidate Breakouts &bull; VectorBT Verification
            </h2>
            <p class="text-[11px] text-zinc-500 mt-0.5">EMA 20/50 + RSI (48-68) + Multi-Timeframe Weekly Confluence</p>
          </div>
          <span class="text-[11px] mono bg-zinc-900 px-2 py-0.5 rounded text-zinc-400 border border-zinc-800" id="scan-count">0 candidates</span>
        </div>

        <div id="candidates-container" class="space-y-2.5 max-h-[460px] overflow-y-auto pr-1">
          <div class="text-center py-12 text-zinc-600 text-xs">
            <i class="fa-solid fa-compass text-2xl mb-2 text-zinc-700 block"></i>
            Click "Run Autonomous Pipeline" or "Scan Candidates" to analyze the universe.
          </div>
        </div>
      </section>

      <!-- Active SQLite Trades Log -->
      <section class="bg-zinc-950 rounded-xl border border-zinc-800/80 p-5">
        <div class="flex justify-between items-center mb-3.5">
          <div>
            <h2 class="text-sm font-semibold text-white tracking-wide flex items-center gap-2">
              <i class="fa-solid fa-database text-zinc-400"></i> Executed Trades &bull; SQLite Audit
            </h2>
            <p class="text-[11px] text-zinc-500 mt-0.5">Real-time database tracking of paper &amp; live swing positions</p>
          </div>
          <button onclick="loadTrades()" class="text-xs text-zinc-400 hover:text-white flex items-center gap-1 transition">
            <i class="fa-solid fa-arrows-rotate text-[10px]"></i> Refresh
          </button>
        </div>

        <div class="overflow-x-auto">
          <table class="w-full text-xs text-left text-zinc-300">
            <thead class="bg-zinc-900/80 text-zinc-500 uppercase mono text-[10px] tracking-wider border-b border-zinc-800">
              <tr>
                <th class="py-2 px-3">Ticker</th>
                <th class="py-2 px-3">Action</th>
                <th class="py-2 px-3">Qty</th>
                <th class="py-2 px-3">Entry</th>
                <th class="py-2 px-3">Live Price</th>
                <th class="py-2 px-3">Stop Loss</th>
                <th class="py-2 px-3">Target</th>
                <th class="py-2 px-3">P&amp;L (Gain)</th>
                <th class="py-2 px-3">Stage / Status</th>
              </tr>
            </thead>
            <tbody id="trades-tbody" class="divide-y divide-zinc-850">
              <tr>
                <td colspan="9" class="text-center py-8 text-zinc-600">No trades recorded in database yet.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

    </div>

    <!-- Right 1 Column: Gemini AI Brain Decision & Active Rules -->
    <div class="space-y-5">

      <!-- Gemini AI Brain Highlight Card -->
      <section class="bg-zinc-950 rounded-xl border border-zinc-800/80 p-5">
        <div class="flex items-center justify-between mb-2">
          <h2 class="text-sm font-semibold text-white tracking-wide flex items-center gap-2">
            <i class="fa-solid fa-brain text-zinc-400"></i> Gemini AI Decision
          </h2>
          <span class="text-[10px] bg-zinc-900 text-zinc-300 border border-zinc-800 px-2 py-0.5 rounded mono uppercase">Strict Pydantic</span>
        </div>
        <p class="text-[11px] text-zinc-500 mb-3.5">Evaluates news catalysts, insider filings, and VectorBT edge.</p>

        <div id="ai-decision-card" class="bg-black rounded-lg p-3.5 border border-zinc-800/80 text-xs space-y-2.5">
          <div class="text-zinc-600 text-center py-8">
            <i class="fa-solid fa-microchip text-xl text-zinc-700 block mb-2"></i>
            Awaiting setup evaluation from pipeline.
          </div>
        </div>
      </section>

      <!-- Active Rules & Self-Refining Parameters -->
      <section class="bg-zinc-950 rounded-xl border border-zinc-800/80 p-5">
        <div class="flex items-center justify-between mb-2">
          <h2 class="text-sm font-semibold text-white tracking-wide flex items-center gap-2">
            <i class="fa-solid fa-sliders text-zinc-400"></i> Active Parameters
          </h2>
          <span class="text-[10px] bg-zinc-900 text-zinc-400 border border-zinc-800 px-2 py-0.5 rounded mono">active_rules.json</span>
        </div>
        <p class="text-[11px] text-zinc-500 mb-3">Self-tuned by daily 15:45 IST post-mortem reflections.</p>

        <div id="rules-container" class="space-y-1.5 text-xs">
          <div class="flex justify-between py-1 border-b border-zinc-850">
            <span class="text-zinc-500">Trend Filter:</span>
            <span class="mono text-zinc-200">20 EMA &bull; 50 EMA</span>
          </div>
          <div class="flex justify-between py-1 border-b border-zinc-850">
            <span class="text-zinc-500">RSI Momentum Zone:</span>
            <span class="mono text-zinc-200">48.0 &ndash; 68.0</span>
          </div>
          <div class="flex justify-between py-1 border-b border-zinc-850">
            <span class="text-zinc-500">Min VectorBT Win Rate:</span>
            <span class="mono text-white font-medium">&ge; 52.0%</span>
          </div>
          <div class="flex justify-between py-1 border-b border-zinc-850">
            <span class="text-zinc-500">Min Profit Factor:</span>
            <span class="mono text-white font-medium">&ge; 1.50</span>
          </div>
          <div class="flex justify-between py-1 border-b border-zinc-850">
            <span class="text-zinc-500">Anti-Chasing Ceiling:</span>
            <span class="mono text-zinc-200">&le; 2.2% of 20 EMA</span>
          </div>
          <div class="flex justify-between py-1 border-b border-zinc-850">
            <span class="text-zinc-500">Profit Ratchet:</span>
            <span class="mono text-white">+3.5% (locks +2.0%)</span>
          </div>
          <div class="flex justify-between py-1 border-b border-zinc-850">
            <span class="text-zinc-500">Execution Segment:</span>
            <span class="mono text-zinc-300">CNC Delivery (Zero Leverage)</span>
          </div>
        </div>

        <div class="mt-4 pt-3 border-t border-zinc-850">
          <h3 class="text-xs font-medium text-zinc-400 mb-1 flex items-center gap-1.5">
            <i class="fa-regular fa-lightbulb text-zinc-400"></i> Experiential Learnings:
          </h3>
          <ul id="insights-list" class="text-[11px] text-zinc-500 space-y-1 list-disc list-inside">
            <li>Baseline active. Multi-timeframe weekly alignment active.</li>
          </ul>
        </div>
      </section>

      <!-- Telegram & Broker Health Card -->
      <section class="bg-zinc-950 rounded-xl border border-zinc-800/80 p-5 text-xs space-y-2">
        <h2 class="text-sm font-semibold text-white mb-2 flex items-center gap-2">
          <i class="fa-solid fa-link text-zinc-400"></i> Connected Services
        </h2>
        <div class="flex justify-between py-1 border-b border-zinc-850">
          <span class="text-zinc-500">Broker:</span>
          <span class="font-medium text-white mono">DhanHQ API (Client: 1109569577)</span>
        </div>
        <div class="flex justify-between py-1 border-b border-zinc-850">
          <span class="text-zinc-500">AI Intelligence:</span>
          <span class="font-medium text-zinc-200 mono">Google Gemini 3.5</span>
        </div>
        <div class="flex justify-between py-1 border-b border-zinc-850">
          <span class="text-zinc-500">Alerts &amp; 2-Way Chat:</span>
          <span class="font-medium text-white mono">@AmpolinTradingBot</span>
        </div>
        <div class="flex justify-between py-1">
          <span class="text-zinc-500">Cloud Infrastructure:</span>
          <span class="font-medium text-zinc-300 mono">Google Cloud Run (asia-south1)</span>
        </div>
      </section>

    </div>
  </main>

  <footer class="bg-black border-t border-zinc-850 py-4 text-center text-xs text-zinc-600 mono">
    AMPOLIN AI TRADING AGENT &bull; MINIMALIST HIGH-PRECISION SYSTEM &bull; 24/7 DUAL-MARKET
  </footer>

  <script>
    // Live IST Clock
    function updateClock() {
      const now = new Date();
      const options = { timeZone: 'Asia/Kolkata', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' };
      document.getElementById('live-time').innerText = now.toLocaleTimeString('en-GB', options) + ' IST';
    }
    setInterval(updateClock, 1000);
    updateClock();

    // Load Initial Data
    async function initDashboard() {
      await loadStatus();
      await loadRules();
      await loadTrades();
    }

    async function loadStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const realized = data.daily_pnl || 0.0;
        const unrealized = data.unrealized_pnl || 0.0;
        const total = data.total_pnl !== undefined ? data.total_pnl : (realized + unrealized);
        const pnlSign = total >= 0 ? '+' : '';
        const uSign = unrealized >= 0 ? '+' : '';
        const rSign = realized >= 0 ? '+' : '';

        const pnlEl = document.getElementById('metric-pnl');
        pnlEl.innerText = pnlSign + '₹' + total.toFixed(2);
        if (total > 0) {
          pnlEl.className = 'text-xl sm:text-2xl font-bold text-emerald-400 mono';
        } else if (total < 0) {
          pnlEl.className = 'text-xl sm:text-2xl font-bold text-rose-400 mono';
        } else {
          pnlEl.className = 'text-xl sm:text-2xl font-bold text-white mono';
        }
        document.getElementById('metric-pnl-breakdown').innerText = `Floating: ${uSign}₹${unrealized.toFixed(2)} | Realized: ${rSign}₹${realized.toFixed(2)}`;

        if (data.portfolio_equity !== undefined) {
          document.getElementById('metric-equity').innerText = '₹' + Number(data.portfolio_equity).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
          document.getElementById('metric-funds-breakdown').innerText = `Cash: ₹${Number(data.available_cash || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })} | Invested: ₹${Number(data.deployed_margin || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
        }

        document.getElementById('metric-winrate').innerText = (data.win_rate * 100).toFixed(1) + '%';
        document.getElementById('metric-open-trades').innerText = data.open_positions;
        document.getElementById('badge-brain').innerText = data.ai_model;

        if (data.regime) {
          document.getElementById('metric-breadth').innerText = data.regime.market_breadth_pct.toFixed(1) + '%';
          document.getElementById('metric-breadth-regime').innerText = data.regime.breadth_regime;
          document.getElementById('metric-vix').innerText = data.regime.india_vix.toFixed(2);
          document.getElementById('metric-dynamic-risk').innerText = 'Risk: ' + (data.regime.dynamic_risk_pct * 100).toFixed(2) + '%';
        }
      } catch (e) {
        console.error('Status fetch failed:', e);
      }
    }

    async function managePositions() {
      const btn = document.getElementById('btn-manage');
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Sweeping...';

      try {
        const res = await fetch('/api/manage-positions', { method: 'POST' });
        const data = await res.json();
        alert(`Position Manager executed! ${data.actions_taken ? data.actions_taken.length : 0} adjustments made.`);
        await loadTrades();
        await loadStatus();
      } catch (e) {
        alert('Position management error: ' + e);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-arrows-split-up-and-left text-[10px] text-zinc-400"></i> Sweep Stops & Profit Lock';
      }
    }

    async function loadRules() {
      try {
        const res = await fetch('/api/rules');
        const data = await res.json();
        if (data.learned_insights && data.learned_insights.length > 0) {
          const list = document.getElementById('insights-list');
          list.innerHTML = data.learned_insights.map(i => `<li>${i}</li>`).join('');
        }
      } catch (e) {
        console.error('Rules fetch failed:', e);
      }
    }

    async function loadTrades() {
      try {
        const res = await fetch('/api/trades');
        const data = await res.json();
        const tbody = document.getElementById('trades-tbody');
        if (!data.trades || data.trades.length === 0) {
          tbody.innerHTML = '<tr><td colspan="9" class="text-center py-8 text-zinc-600">No trades recorded in database yet.</td></tr>';
          return;
        }

        tbody.innerHTML = data.trades.map(t => {
          const sym = t.ticker.endsWith('.NS') ? '₹' : '$';
          const entry = parseFloat(t.entry_price || 0);
          const live = parseFloat(t.current_price || entry);
          const qty = parseInt(t.quantity || 1);
          const pnlVal = t.pnl !== null && t.pnl !== undefined ? parseFloat(t.pnl) : (live - entry) * qty;
          const pnlPct = t.pnl_percentage !== null && t.pnl_percentage !== undefined ? parseFloat(t.pnl_percentage) : ((live - entry) / (entry || 1) * 100);
          const pnlSign = pnlVal >= 0 ? '+' : '';
          const pnlColor = pnlVal > 0 ? 'text-emerald-400 font-semibold' : (pnlVal < 0 ? 'text-rose-400 font-semibold' : 'text-zinc-400');

          let stageBadge = `<span class="px-1.5 py-0.5 rounded text-[10px] border border-zinc-800 bg-zinc-900 text-zinc-300">${t.status}</span>`;
          if (t.stage === 'STAGE_PROFIT_LOCKED') {
            stageBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] border border-emerald-800/80 bg-emerald-950/60 text-emerald-300 font-medium">🔒 +2% LOCKED</span>';
          } else if (t.stage === 'STAGE_BREAKEVEN') {
            stageBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] border border-blue-800/80 bg-blue-950/60 text-blue-300 font-medium">🛡️ BREAKEVEN</span>';
          } else if (t.status === 'CLOSED') {
            stageBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] border border-zinc-700 bg-zinc-800 text-zinc-400 font-medium">CLOSED</span>';
          }

          return `
            <tr class="hover:bg-zinc-900/60 transition mono text-xs">
              <td class="py-2.5 px-3 font-semibold text-white">${t.ticker}</td>
              <td class="py-2.5 px-3"><span class="font-bold text-white">${t.action}</span></td>
              <td class="py-2.5 px-3 text-zinc-300">${t.quantity}</td>
              <td class="py-2.5 px-3 text-zinc-300">${sym}${entry.toFixed(2)}</td>
              <td class="py-2.5 px-3 text-white font-medium">${sym}${live.toFixed(2)}</td>
              <td class="py-2.5 px-3 text-zinc-400">${sym}${parseFloat(t.stop_loss).toFixed(2)}</td>
              <td class="py-2.5 px-3 text-zinc-300">${sym}${parseFloat(t.take_profit).toFixed(2)}</td>
              <td class="py-2.5 px-3 ${pnlColor}">${pnlSign}${sym}${pnlVal.toFixed(2)} (${pnlSign}${pnlPct.toFixed(2)}%)</td>
              <td class="py-2.5 px-3">${stageBadge}</td>
            </tr>
          `;
        }).join('');
      } catch (e) {
        console.error('Trades fetch failed:', e);
      }
    }

    async function runPipeline() {
      const btn = document.getElementById('btn-pipeline');
      const tf = document.getElementById('select-tf').value;
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin text-black"></i> Executing Pipeline...';

      try {
        const res = await fetch('/api/run-pipeline', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ timeframe: tf, dry_run: true })
        });
        const data = await res.json();
        await runScan();
        await loadStatus();
        await loadTrades();
      } catch (e) {
        alert('Pipeline execution error: ' + e);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-play text-[10px]"></i> Run Autonomous Pipeline';
      }
    }

    async function runScan() {
      const btn = document.getElementById('btn-scan');
      const tf = document.getElementById('select-tf').value;
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Scanning...';

      try {
        const res = await fetch('/api/scan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ timeframe: tf })
        });
        const result = await res.json();
        renderCandidates(result.data || []);
      } catch (e) {
        alert('Scan error: ' + e);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-magnifying-glass text-[10px] text-zinc-400"></i> Scan Candidates';
      }
    }

    function renderCandidates(approvedList) {
      const container = document.getElementById('candidates-container');
      document.getElementById('scan-count').innerText = `${approvedList.length} approved setups`;

      if (!approvedList || approvedList.length === 0) {
        container.innerHTML = `
          <div class="text-center py-12 text-zinc-600 text-xs">
            <i class="fa-solid fa-shield text-xl mb-2 text-zinc-700 block"></i>
            No candidate setups satisfied the strict criteria: EMA 20/50 + RSI 48-68 + VectorBT Win Rate &ge; 52% + Weekly Confluence.
          </div>`;
        return;
      }

      container.innerHTML = approvedList.map(item => {
        const s = item.setup;
        const b = item.backtest;
        const d = item.decision;
        const sym = s.currency === 'USD' ? '$' : '₹';

        return `
          <div class="bg-black rounded-lg p-3.5 border border-zinc-800/80 hover:border-zinc-700 transition">
            <div class="flex justify-between items-start mb-2">
              <div>
                <span class="text-sm font-bold text-white mono">${s.ticker}</span>
                <span class="ml-2 text-[10px] bg-zinc-900 text-zinc-300 border border-zinc-800 px-1.5 py-0.5 rounded mono">${s.breakout_type}</span>
                ${s.weekly_trend === 'BULLISH_CONFLUENCE' ? '<span class="ml-1 text-[10px] bg-zinc-850 text-white border border-zinc-700 px-1.5 py-0.5 rounded mono">1W CONFLUENCE</span>' : ''}
              </div>
              <div class="text-right">
                <div class="text-sm font-bold text-white mono">${sym}${s.close}</div>
                <div class="text-[10px] text-zinc-500">ATR: ${sym}${s.atr}</div>
              </div>
            </div>

            <!-- Technical Gauges (Minimalist Monochrome) -->
            <div class="grid grid-cols-4 gap-2 text-[11px] bg-zinc-950 p-2 rounded border border-zinc-850 mb-2.5 mono">
              <div>
                <span class="text-zinc-600 block text-[9px] uppercase tracking-wider">EMA 20/50</span>
                <span class="text-zinc-200">${sym}${s.ema_fast} &gt; ${sym}${s.ema_slow}</span>
              </div>
              <div>
                <span class="text-zinc-600 block text-[9px] uppercase tracking-wider">RSI (14)</span>
                <span class="text-white font-bold">${s.rsi}</span>
              </div>
              <div>
                <span class="text-zinc-600 block text-[9px] uppercase tracking-wider">VBT Win%</span>
                <span class="text-white font-bold">${(b.win_rate * 100).toFixed(1)}%</span>
              </div>
              <div>
                <span class="text-zinc-600 block text-[9px] uppercase tracking-wider">Expectancy</span>
                <span class="text-white font-bold">${sym}${b.expectancy.toFixed(2)}</span>
              </div>
            </div>

            <!-- Latency & AI Conviction -->
            <div class="flex justify-between items-center text-xs pt-1 border-t border-zinc-850">
              <span class="text-zinc-500 text-[10px] mono">
                <i class="fa-solid fa-bolt text-zinc-600"></i> ${b.execution_time_ms}ms vectorized backtest
              </span>
              <button onclick='displayAIDecision(${JSON.stringify(d).replace(/'/g, "&#39;")})' class="text-zinc-300 hover:text-white text-[11px] font-medium flex items-center gap-1 transition">
                AI Evaluation (${(d.confidence * 100).toFixed(0)}% Conviction) <i class="fa-solid fa-arrow-right text-[9px]"></i>
              </button>
            </div>
          </div>
        `;
      }).join('');

      if (approvedList.length > 0) {
        displayAIDecision(approvedList[0].decision);
      }
    }

    function displayAIDecision(d) {
      const card = document.getElementById('ai-decision-card');
      const sym = d.ticker.endsWith('.NS') ? '₹' : '$';
      card.innerHTML = `
        <div class="flex justify-between items-center border-b border-zinc-850 pb-2">
          <span class="mono font-bold text-white text-sm">${d.ticker}</span>
          <span class="px-2 py-0.5 rounded text-xs font-bold border border-white bg-white text-black mono">${d.action}</span>
        </div>
        <div class="grid grid-cols-3 gap-2 text-center py-2 bg-zinc-950 rounded border border-zinc-850 mono">
          <div>
            <span class="text-[9px] text-zinc-500 block uppercase tracking-wider">ENTRY</span>
            <span class="font-bold text-white">${sym}${d.entry_price}</span>
          </div>
          <div>
            <span class="text-[9px] text-zinc-500 block uppercase tracking-wider">STOP LOSS</span>
            <span class="font-bold text-zinc-400">${sym}${d.stop_loss}</span>
          </div>
          <div>
            <span class="text-[9px] text-zinc-500 block uppercase tracking-wider">TARGET</span>
            <span class="font-bold text-white">${sym}${d.take_profit}</span>
          </div>
        </div>
        <div class="text-[11px] text-zinc-400 leading-relaxed pt-1">
          <span class="text-white font-medium">Thesis:</span> ${d.rationale}
        </div>
        <div class="text-[10px] text-zinc-500 mono pt-1 border-t border-zinc-850 flex justify-between">
          <span>Confidence: ${(d.confidence * 100).toFixed(0)}%</span>
          <span>R:R: 1:${d.risk_reward_ratio}</span>
          <span>Hold: ~${d.holding_period_days}d</span>
        </div>
      `;
    }

    async function runEOD() {
      const btn = document.getElementById('btn-eod');
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Reflecting...';

      try {
        const res = await fetch('/api/eod-reflection', { method: 'POST' });
        const data = await res.json();
        alert('EOD Post-Mortem Reflection Finished! Active rules verified and updated.');
        await loadRules();
      } catch (e) {
        alert('EOD reflection error: ' + e);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-microscope text-[10px]"></i> EOD Reflection';
      }
    }

    function toggleChatDrawer() {
      const drawer = document.getElementById('chat-drawer');
      drawer.classList.toggle('hidden');
      if (!drawer.classList.contains('hidden')) {
        document.getElementById('chat-input').focus();
        const msgs = document.getElementById('chat-messages');
        msgs.scrollTop = msgs.scrollHeight;
      }
    }

    function sendQuickPrompt(txt) {
      document.getElementById('chat-input').value = txt;
      handleSendChatMessage(new Event('submit'));
    }

    async function handleSendChatMessage(e) {
      if (e) e.preventDefault();
      const input = document.getElementById('chat-input');
      const msg = input.value.trim();
      if (!msg) return;

      input.value = '';
      const container = document.getElementById('chat-messages');

      // User Bubble
      container.innerHTML += `
        <div class="flex justify-end">
          <div class="bg-white text-black font-medium p-3 rounded-xl max-w-[85%] text-xs shadow-sm">
            ${escapeHtml(msg)}
          </div>
        </div>
      `;
      container.scrollTop = container.scrollHeight;

      // Typing indicator
      const typingId = 'typing-' + Date.now();
      container.innerHTML += `
        <div id="${typingId}" class="flex gap-2 items-start">
          <div class="w-6 h-6 rounded bg-zinc-900 border border-zinc-800 flex items-center justify-center text-[10px] text-white shrink-0 mono font-bold">A</div>
          <div class="bg-zinc-900/90 text-zinc-400 border border-zinc-800 p-2.5 rounded-xl max-w-[85%] text-xs flex items-center gap-1.5 mono">
            <span class="w-1.5 h-1.5 bg-zinc-400 rounded-full animate-bounce"></span>
            <span class="w-1.5 h-1.5 bg-zinc-400 rounded-full animate-bounce" style="animation-delay: 0.15s"></span>
            <span class="w-1.5 h-1.5 bg-zinc-400 rounded-full animate-bounce" style="animation-delay: 0.3s"></span>
          </div>
        </div>
      `;
      container.scrollTop = container.scrollHeight;

      if (!window.chatSessionHistory) window.chatSessionHistory = [];
      window.chatSessionHistory.push({ role: 'user', content: msg });

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: msg, history: window.chatSessionHistory })
        });
        const data = await res.json();
        document.getElementById(typingId)?.remove();

        if (data.reply) {
          window.chatSessionHistory.push({ role: 'assistant', content: data.reply });
        }

        const formatted = (data.reply || "No reply generated.")
          .replace(/\\n\\n/g, '<br><br>')
          .replace(/\\n/g, '<br>')
          .replace(/\\*\\*(.*?)\\*\\*/g, '<b>$1</b>');

        container.innerHTML += `
          <div class="flex gap-2 items-start">
            <div class="w-6 h-6 rounded bg-zinc-900 border border-zinc-800 flex items-center justify-center text-[10px] text-white shrink-0 mono font-bold">A</div>
            <div class="bg-zinc-900/90 text-zinc-200 border border-zinc-800 p-3 rounded-xl max-w-[85%] text-xs leading-relaxed">
              ${formatted}
            </div>
          </div>
        `;
      } catch (err) {
        document.getElementById(typingId)?.remove();
        container.innerHTML += `
          <div class="flex gap-2 items-start">
            <div class="w-6 h-6 rounded bg-zinc-900 border border-zinc-800 flex items-center justify-center text-[10px] text-white shrink-0 mono font-bold">A</div>
            <div class="bg-rose-950/40 text-rose-300 border border-rose-900/50 p-3 rounded-xl max-w-[85%] text-xs">
              Error: ${err.message || 'Could not connect to AI'}
            </div>
          </div>
        `;
      }
      container.scrollTop = container.scrollHeight;
    }

    function escapeHtml(text) {
      const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
      return text.replace(/[&<>"']/g, function(m) { return map[m]; });
    }

    window.onload = initDashboard;
  </script>

  <!-- Floating In-App AI Chat Trigger -->
  <button onclick="toggleChatDrawer()" id="btn-open-chat" class="fixed bottom-4 right-4 sm:bottom-6 sm:right-6 z-40 bg-white hover:bg-zinc-200 text-black font-semibold px-4 py-2.5 rounded-full shadow-2xl flex items-center gap-2 text-xs transition active:scale-95 border border-zinc-200 cursor-pointer">
    <i class="fa-solid fa-sparkles text-black"></i>
    <span>Chat with AI</span>
  </button>

  <!-- In-App AI Chat Drawer (Mobile Bottom Sheet + Desktop Floating Window) -->
  <div id="chat-drawer" class="fixed inset-x-0 bottom-0 sm:inset-auto sm:bottom-6 sm:right-6 z-50 w-full sm:w-[400px] h-[85vh] sm:h-[540px] bg-black border-t sm:border border-zinc-800 sm:rounded-2xl shadow-2xl flex flex-col hidden transition-all">
    <!-- Chat Header -->
    <div class="px-4 py-3 bg-zinc-950 border-b border-zinc-800 flex justify-between items-center sm:rounded-t-2xl">
      <div class="flex items-center gap-2.5">
        <img src="/static/ampolin_icon_sq.png" alt="Ampolin AI" class="w-6 h-6 rounded border border-zinc-800 object-cover">
        <div>
          <div class="text-xs font-bold text-white mono flex items-center gap-1.5">
            AMPOLIN AI <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
          </div>
          <p class="text-[10px] text-zinc-500">Dual-Engine &bull; Gemini Flash &amp; Groq</p>
        </div>
      </div>
      <button onclick="toggleChatDrawer()" class="text-zinc-400 hover:text-white text-base p-1.5 rounded-lg hover:bg-zinc-900 transition cursor-pointer">
        <i class="fa-solid fa-xmark"></i>
      </button>
    </div>

    <!-- Quick Prompt Chips -->
    <div class="px-3 py-2 bg-zinc-950/70 border-b border-zinc-850 flex gap-1.5 overflow-x-auto text-[11px] mono">
      <button onclick="sendQuickPrompt('How is the market regime and India VIX today?')" class="bg-zinc-900 hover:bg-zinc-800 text-zinc-300 px-2.5 py-1 rounded-md border border-zinc-800 whitespace-nowrap text-[10px] cursor-pointer">📊 Regime</button>
      <button onclick="sendQuickPrompt('What are our active open positions and stages?')" class="bg-zinc-900 hover:bg-zinc-800 text-zinc-300 px-2.5 py-1 rounded-md border border-zinc-800 whitespace-nowrap text-[10px] cursor-pointer">📈 Positions</button>
      <button onclick="sendQuickPrompt('Explain the 3.5% profit lock rule')" class="bg-zinc-900 hover:bg-zinc-800 text-zinc-300 px-2.5 py-1 rounded-md border border-zinc-800 whitespace-nowrap text-[10px] cursor-pointer">🛡️ Profit Lock</button>
      <button onclick="sendQuickPrompt('What is our VectorBT filter criteria?')" class="bg-zinc-900 hover:bg-zinc-800 text-zinc-300 px-2.5 py-1 rounded-md border border-zinc-800 whitespace-nowrap text-[10px] cursor-pointer">⚡ VectorBT</button>
    </div>

    <!-- Message History Stream -->
    <div id="chat-messages" class="flex-1 p-4 overflow-y-auto space-y-3 text-xs leading-relaxed">
      <div class="flex gap-2 items-start">
        <div class="w-6 h-6 rounded bg-zinc-900 border border-zinc-800 flex items-center justify-center text-[10px] text-white shrink-0 mono font-bold">A</div>
        <div class="bg-zinc-900/90 text-zinc-200 border border-zinc-800 p-3 rounded-xl max-w-[85%] text-xs">
          Hey Kaustuv! I'm your autonomous quant partner. Ask me about open trades, market breadth, scanning setups, or risk rules.
        </div>
      </div>
    </div>

    <!-- Chat Input Form -->
    <form onsubmit="handleSendChatMessage(event)" class="p-3 bg-zinc-950 border-t border-zinc-800 flex items-center gap-2 sm:rounded-b-2xl">
      <input type="text" id="chat-input" placeholder="Ask Ampolin AI..." autocomplete="off" class="flex-1 bg-zinc-900 text-white placeholder-zinc-500 text-xs px-3.5 py-2.5 rounded-lg border border-zinc-800 focus:outline-none focus:border-zinc-500 mono">
      <button type="submit" id="chat-send-btn" class="bg-white hover:bg-zinc-200 text-black px-3.5 py-2.5 rounded-lg text-xs font-semibold shrink-0 transition active:scale-95 cursor-pointer">
        <i class="fa-solid fa-arrow-up"></i>
      </button>
    </form>
  </div>
</body>
</html>
"""

@app.on_event("startup")
def on_startup():
    """Starts the 24/7 autonomous trading scheduler & Telegram bot in the cloud."""
    import threading
    try:
        from cloud_scheduler import CloudTradingDaemon
        daemon = CloudTradingDaemon(dry_run=True)
        t = threading.Thread(target=daemon.run_scheduler, daemon=True, name="CloudDaemonThread")
        t.start()
        logger.info("Autonomous Dual-Market Scheduler & Telegram Bot started inside Cloud Container.")
    except Exception as e:
        logger.warning(f"Could not auto-start background daemon: {e}")

def start_server(host: Optional[str] = None, port: Optional[int] = None):
    server_host = host or os.getenv("HOST", "0.0.0.0")
    server_port = port or int(os.getenv("PORT", "8000"))
    logger.info(f"Starting Ampolin AI Trading Agent Web UI on http://{server_host}:{server_port}")
    uvicorn.run(app, host=server_host, port=server_port)


if __name__ == "__main__":
    start_server()
