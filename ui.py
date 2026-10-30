"""
Ampolin AI Trading Agent - Modern Web UI & API Server
FastAPI interactive dashboard for real-time scanning, VectorBT backtesting,
Gemini AI Brain decision inspection, DhanHQ trade execution, and SQLite portfolio review.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List

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
from telegram_notifier import TelegramNotifier
from market_regime import MarketRegimeAnalyzer
from position_manager import PositionManager
from main import AmpolinPipeline

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
        "description": "Automated Swing Trading System for NSE & Wall Street",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0b0f17",
        "theme_color": "#0b0f17",
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


@app.get("/api/status")
def get_system_status():
    open_trades = trade_logger.get_open_trades()
    summary = trade_logger.calculate_daily_summary()
    regime = regime_analyzer.analyze_regime()
    return {
        "status": "ONLINE",
        "system_name": "Ampolin AI Trading Agent",
        "open_positions": len(open_trades),
        "daily_pnl": summary.get("realized_pnl", 0.0),
        "win_rate": summary.get("win_rate", 0.0),
        "broker_mode": "Paper Trading (Dry-Run)" if broker.dry_run or not broker.dhan else "DhanHQ Live",
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
  <header class="bg-black border-b border-zinc-800/80 px-6 py-4 flex flex-wrap justify-between items-center sticky top-0 z-50">
    <div class="flex items-center space-x-3.5">
      <img src="/static/ampolin_icon_sq.png" alt="Ampolin AI" class="w-8 h-8 rounded-lg border border-zinc-700 object-cover">
      <div>
        <h1 class="text-base font-bold tracking-tight text-white flex items-center gap-2">
          AMPOLIN <span class="text-zinc-400 font-normal text-xs px-2 py-0.5 rounded border border-zinc-800 bg-zinc-900/60 mono">AI TRADING</span>
        </h1>
        <p class="text-[11px] text-zinc-500 tracking-wide">Autonomous Dual-Market Terminal &bull; NSE &amp; Wall Street</p>
      </div>
    </div>

    <!-- Live Status Badges (Monochrome) -->
    <div class="flex items-center gap-3 text-xs mt-2 sm:mt-0">
      <div class="flex items-center gap-2 bg-zinc-950 px-3 py-1.5 rounded-md border border-zinc-800">
        <span class="w-1.5 h-1.5 rounded-full bg-white animate-pulse"></span>
        <span class="text-zinc-500 text-[11px]">Engine:</span>
        <span class="font-medium text-white mono text-[11px]" id="badge-engine">ACTIVE</span>
      </div>
      <div class="flex items-center gap-2 bg-zinc-950 px-3 py-1.5 rounded-md border border-zinc-800">
        <span class="text-zinc-500 text-[11px]">Brain:</span>
        <span class="font-medium text-zinc-200 mono text-[11px]" id="badge-brain">gemini-3.5-flash-lite</span>
      </div>
      <div class="flex items-center gap-2 bg-zinc-950 px-3 py-1.5 rounded-md border border-zinc-800">
        <span class="text-zinc-500 text-[11px]">Risk:</span>
        <span class="font-medium text-zinc-200 mono text-[11px]">1.0% Max</span>
      </div>
      <div class="flex items-center gap-2 bg-zinc-950 px-3 py-1.5 rounded-md border border-zinc-800">
        <span id="live-time" class="mono text-zinc-400 text-[11px]">--:--:-- IST</span>
      </div>
    </div>
  </header>

  <!-- Metric Overview Bar (Minimalist Cards) -->
  <div class="max-w-7xl mx-auto w-full px-6 pt-6 grid grid-cols-2 md:grid-cols-5 gap-3.5">
    <div class="bg-zinc-950 p-4 rounded-xl border border-zinc-800/80">
      <div class="flex justify-between items-center text-zinc-500 text-[11px] font-medium tracking-wider uppercase mb-1">
        <span>Daily Realized P&amp;L</span>
        <i class="fa-solid fa-chart-line text-zinc-600"></i>
      </div>
      <div class="text-2xl font-bold text-white mono" id="metric-pnl">₹0.00</div>
      <div class="text-[11px] text-zinc-600 mt-1">Cash Delivery Positions</div>
    </div>

    <div class="bg-zinc-950 p-4 rounded-xl border border-zinc-800/80">
      <div class="flex justify-between items-center text-zinc-500 text-[11px] font-medium tracking-wider uppercase mb-1">
        <span>Market Breadth</span>
        <i class="fa-solid fa-bars-progress text-zinc-600"></i>
      </div>
      <div class="text-2xl font-bold text-white mono" id="metric-breadth">--%</div>
      <div class="text-[11px] text-zinc-600 mt-1" id="metric-breadth-regime">Nifty 50 &gt; 50 EMA</div>
    </div>

    <div class="bg-zinc-950 p-4 rounded-xl border border-zinc-800/80">
      <div class="flex justify-between items-center text-zinc-500 text-[11px] font-medium tracking-wider uppercase mb-1">
        <span>India VIX</span>
        <i class="fa-solid fa-bolt text-zinc-600"></i>
      </div>
      <div class="text-2xl font-bold text-white mono" id="metric-vix">--</div>
      <div class="text-[11px] text-zinc-600 mt-1" id="metric-dynamic-risk">Risk: --%</div>
    </div>

    <div class="bg-zinc-950 p-4 rounded-xl border border-zinc-800/80">
      <div class="flex justify-between items-center text-zinc-500 text-[11px] font-medium tracking-wider uppercase mb-1">
        <span>Strategy Win Rate</span>
        <i class="fa-solid fa-bullseye text-zinc-600"></i>
      </div>
      <div class="text-2xl font-bold text-white mono" id="metric-winrate">0.0%</div>
      <div class="text-[11px] text-zinc-600 mt-1">VectorBT Filtered (&ge;52%)</div>
    </div>

    <div class="bg-zinc-950 p-4 rounded-xl border border-zinc-800/80">
      <div class="flex justify-between items-center text-zinc-500 text-[11px] font-medium tracking-wider uppercase mb-1">
        <span>Active Positions</span>
        <i class="fa-solid fa-layer-group text-zinc-600"></i>
      </div>
      <div class="text-2xl font-bold text-white mono" id="metric-open-trades">0</div>
      <div class="text-[11px] text-zinc-600 mt-1">1R Breakeven &bull; 3.5% Lock</div>
    </div>
  </div>

  <!-- Primary Action & Control Toolbar -->
  <div class="max-w-7xl mx-auto w-full px-6 py-5 flex flex-wrap gap-3 items-center justify-between">
    <div class="flex items-center gap-2.5 flex-wrap">
      <button onclick="runPipeline()" id="btn-pipeline" class="bg-white hover:bg-zinc-200 text-black font-semibold px-4 py-2 rounded-lg text-xs tracking-wide flex items-center gap-2 transition active:scale-95 shadow-sm">
        <i class="fa-solid fa-play text-[10px]"></i> Run Autonomous Pipeline
      </button>

      <button onclick="runScan()" id="btn-scan" class="bg-zinc-900 hover:bg-zinc-800 text-zinc-200 font-medium px-3.5 py-2 rounded-lg border border-zinc-800 flex items-center gap-2 transition text-xs">
        <i class="fa-solid fa-magnifying-glass text-[10px] text-zinc-400"></i> Scan Candidates
      </button>

      <button onclick="managePositions()" id="btn-manage" class="bg-zinc-900 hover:bg-zinc-800 text-zinc-200 font-medium px-3.5 py-2 rounded-lg border border-zinc-800 flex items-center gap-2 transition text-xs">
        <i class="fa-solid fa-arrows-split-up-and-left text-[10px] text-zinc-400"></i> Sweep Stops &amp; Profit Lock
      </button>

      <button onclick="runEOD()" id="btn-eod" class="bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 font-medium px-3 py-2 rounded-lg border border-zinc-800 flex items-center gap-2 transition text-xs">
        <i class="fa-solid fa-microscope text-[10px]"></i> EOD Reflection
      </button>
    </div>

    <div class="flex items-center gap-2.5 text-xs">
      <div class="flex items-center bg-zinc-950 px-3 py-1.5 rounded-lg border border-zinc-800 gap-2">
        <span class="text-zinc-500 text-[11px]">Timeframe:</span>
        <select id="select-tf" class="bg-transparent border-0 text-white mono text-xs focus:outline-none cursor-pointer">
          <option value="1d" selected class="bg-zinc-900">1D (Daily)</option>
          <option value="4h" class="bg-zinc-900">4H (Swing)</option>
        </select>
      </div>

      <div class="flex items-center bg-zinc-950 px-3 py-1.5 rounded-lg border border-zinc-800 gap-2">
        <span class="text-zinc-500 text-[11px]">Mode:</span>
        <span class="text-zinc-300 border border-zinc-700/80 bg-zinc-900/80 px-2 py-0.5 rounded text-[11px] mono font-medium">Paper Trading</span>
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
                <th class="py-2 px-3">Stop Loss</th>
                <th class="py-2 px-3">Target</th>
                <th class="py-2 px-3">Risk</th>
                <th class="py-2 px-3">Status</th>
              </tr>
            </thead>
            <tbody id="trades-tbody" class="divide-y divide-zinc-850">
              <tr>
                <td colspan="8" class="text-center py-8 text-zinc-600">No trades recorded in database yet.</td>
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
        const pnl = data.daily_pnl || 0.0;
        const pnlSign = pnl >= 0 ? '+' : '';
        document.getElementById('metric-pnl').innerText = pnlSign + '₹' + pnl.toFixed(2);
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
          tbody.innerHTML = '<tr><td colspan="8" class="text-center py-8 text-zinc-600">No trades recorded in database yet.</td></tr>';
          return;
        }

        tbody.innerHTML = data.trades.map(t => {
          const sym = t.ticker.endsWith('.NS') ? '₹' : '$';
          return `
            <tr class="hover:bg-zinc-900/60 transition mono text-xs">
              <td class="py-2.5 px-3 font-semibold text-white">${t.ticker}</td>
              <td class="py-2.5 px-3"><span class="font-bold text-white">${t.action}</span></td>
              <td class="py-2.5 px-3 text-zinc-300">${t.quantity}</td>
              <td class="py-2.5 px-3 text-zinc-300">${sym}${t.entry_price}</td>
              <td class="py-2.5 px-3 text-zinc-400">${sym}${t.stop_loss}</td>
              <td class="py-2.5 px-3 text-white font-medium">${sym}${t.take_profit}</td>
              <td class="py-2.5 px-3 text-zinc-400">${sym}${t.risk_amount}</td>
              <td class="py-2.5 px-3"><span class="px-1.5 py-0.5 rounded text-[10px] border border-zinc-800 bg-zinc-900 text-zinc-300">${t.status}</span></td>
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

    window.onload = initDashboard;
  </script>
</body>
</html>
"""
          </div>
        </div>
        <div class="text-[11px]">
          <span class="text-gray-400 block font-semibold mb-1">Conviction &amp; R:R Ratio:</span>
          <div class="flex justify-between text-gray-300">
            <span>Score: <b class="text-purple-300">${(d.confidence * 100).toFixed(1)}%</b></span>
            <span>R:R: <b class="text-blue-300">1:${d.risk_reward_ratio}</b></span>
            <span>Horizon: <b class="text-amber-300">${d.holding_period_days} Days</b></span>
          </div>
        </div>
        <div class="text-[11px]">
          <span class="text-gray-400 block font-semibold mb-1">AI Thesis:</span>
          <p class="text-gray-300 bg-dark-800 p-2 rounded leading-relaxed">${d.rationale}</p>
        </div>
        <div class="text-[11px]">
          <span class="text-gray-400 block font-semibold mb-1">Invalidation / Risks:</span>
          <p class="text-rose-300/80 bg-rose-950/20 border border-rose-900/30 p-2 rounded">${d.key_risks}</p>
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
        btn.innerHTML = '<i class="fa-solid fa-microscope"></i> EOD Reflection';
      }
    }

    window.onload = initDashboard;
  </script>
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
