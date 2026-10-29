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
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ampolin AI Trading Agent | NSE Equities Terminal</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: { 500: '#3b82f6', 600: '#2563eb', 700: '#1d4ed8' },
            dark: { 900: '#0b0f17', 800: '#111827', 700: '#1f2937', 600: '#374151' }
          }
        }
      }
    }
  </script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <style>
    body { background-color: #0b0f17; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .card-glow { box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5); }
    .neon-border { border: 1px solid rgba(59, 130, 246, 0.25); }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-thumb { background: #374151; border-radius: 4px; }
  </style>
</head>
<body class="text-gray-200 min-h-screen flex flex-col">

  <!-- Top Navigation Header -->
  <header class="bg-dark-800 border-b border-dark-700 px-6 py-4 flex flex-wrap justify-between items-center sticky top-0 z-50">
    <div class="flex items-center space-x-3">
      <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-blue-500/30">
        <i class="fa-solid fa-chart-line"></i>
      </div>
      <div>
        <h1 class="text-xl font-black tracking-wide text-white flex items-center gap-2">
          AMPOLIN <span class="text-blue-500 font-medium text-sm px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/20">AI TRADING AGENT</span>
        </h1>
        <p class="text-xs text-gray-400">Automated Indian Equities (NSE) Swing System &bull; DhanHQ &bull; Gemini 3.8 Flash</p>
      </div>
    </div>

    <!-- Live Status Badges -->
    <div class="flex items-center gap-4 text-xs mt-2 sm:mt-0">
      <div class="flex items-center gap-2 bg-dark-900/80 px-3 py-1.5 rounded-lg border border-dark-700">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span class="text-gray-400">Engine:</span>
        <span class="font-semibold text-emerald-400" id="badge-engine">ACTIVE</span>
      </div>
      <div class="flex items-center gap-2 bg-dark-900/80 px-3 py-1.5 rounded-lg border border-dark-700">
        <i class="fa-solid fa-brain text-purple-400"></i>
        <span class="text-gray-400">Brain:</span>
        <span class="font-semibold text-purple-300" id="badge-brain">Gemini Flash</span>
      </div>
      <div class="flex items-center gap-2 bg-dark-900/80 px-3 py-1.5 rounded-lg border border-dark-700">
        <i class="fa-solid fa-shield-halved text-blue-400"></i>
        <span class="text-gray-400">Risk Rule:</span>
        <span class="font-semibold text-blue-300">1% Max Account Risk</span>
      </div>
      <div class="flex items-center gap-2 bg-dark-900/80 px-3 py-1.5 rounded-lg border border-dark-700">
        <i class="fa-solid fa-clock text-amber-400"></i>
        <span id="live-time" class="font-mono text-gray-300">--:--:-- IST</span>
      </div>
    </div>
  </header>

  <!-- Metric Overview Bar -->
  <div class="max-w-7xl mx-auto w-full px-6 pt-6 grid grid-cols-2 md:grid-cols-5 gap-4">
    <div class="bg-dark-800 p-4 rounded-xl border border-dark-700 card-glow">
      <div class="flex justify-between items-center text-gray-400 text-xs mb-1">
        <span>DAILY REALIZED P&L</span>
        <i class="fa-solid fa-indian-rupee-sign text-emerald-400"></i>
      </div>
      <div class="text-2xl font-bold text-emerald-400" id="metric-pnl">₹0.00</div>
      <div class="text-[11px] text-gray-500 mt-1">NSE CNC Swing Positions</div>
    </div>

    <div class="bg-dark-800 p-4 rounded-xl border border-dark-700 card-glow">
      <div class="flex justify-between items-center text-gray-400 text-xs mb-1">
        <span>MARKET BREADTH</span>
        <i class="fa-solid fa-chart-simple text-blue-400"></i>
      </div>
      <div class="text-2xl font-bold text-blue-400" id="metric-breadth">--%</div>
      <div class="text-[11px] text-gray-500 mt-1" id="metric-breadth-regime">Nifty 50 > 50 EMA</div>
    </div>

    <div class="bg-dark-800 p-4 rounded-xl border border-dark-700 card-glow">
      <div class="flex justify-between items-center text-gray-400 text-xs mb-1">
        <span>INDIA VIX VOLATILITY</span>
        <i class="fa-solid fa-gauge-high text-amber-400"></i>
      </div>
      <div class="text-2xl font-bold text-amber-400" id="metric-vix">--</div>
      <div class="text-[11px] text-gray-500 mt-1" id="metric-dynamic-risk">Dynamic Risk: --%</div>
    </div>

    <div class="bg-dark-800 p-4 rounded-xl border border-dark-700 card-glow">
      <div class="flex justify-between items-center text-gray-400 text-xs mb-1">
        <span>STRATEGY WIN RATE</span>
        <i class="fa-solid fa-percent text-purple-400"></i>
      </div>
      <div class="text-2xl font-bold text-purple-400" id="metric-winrate">0.0%</div>
      <div class="text-[11px] text-gray-500 mt-1">VectorBT Filtered (&ge;45%)</div>
    </div>

    <div class="bg-dark-800 p-4 rounded-xl border border-dark-700 card-glow">
      <div class="flex justify-between items-center text-gray-400 text-xs mb-1">
        <span>ACTIVE POSITIONS</span>
        <i class="fa-solid fa-layer-group text-emerald-400"></i>
      </div>
      <div class="text-2xl font-bold text-white" id="metric-open-trades">0</div>
      <div class="text-[11px] text-gray-500 mt-1">1R Breakeven & 2R Scaling</div>
    </div>
  </div>

  <!-- Primary Action & Control Toolbar -->
  <div class="max-w-7xl mx-auto w-full px-6 py-6 flex flex-wrap gap-4 items-center justify-between">
    <div class="flex items-center gap-3">
      <button onclick="runPipeline()" id="btn-pipeline" class="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold px-5 py-2.5 rounded-xl shadow-lg shadow-blue-500/25 flex items-center gap-2 transition active:scale-95 text-sm">
        <i class="fa-solid fa-play"></i> Run Full Autonomous Pipeline
      </button>

      <button onclick="runScan()" id="btn-scan" class="bg-dark-700 hover:bg-dark-600 text-gray-200 font-medium px-4 py-2.5 rounded-xl border border-dark-600 flex items-center gap-2 transition text-sm">
        <i class="fa-solid fa-magnifying-glass"></i> Scan Candidates
      </button>

      <button onclick="managePositions()" id="btn-manage" class="bg-dark-700 hover:bg-dark-600 text-emerald-400 font-medium px-4 py-2.5 rounded-xl border border-dark-600 flex items-center gap-2 transition text-sm">
        <i class="fa-solid fa-arrows-split-up-and-left"></i> Manage Positions (1R/2R)
      </button>

      <button onclick="runEOD()" id="btn-eod" class="bg-dark-700 hover:bg-dark-600 text-amber-300 font-medium px-4 py-2.5 rounded-xl border border-dark-600 flex items-center gap-2 transition text-sm">
        <i class="fa-solid fa-microscope"></i> EOD Reflection
      </button>
    </div>

    <div class="flex items-center gap-3 text-xs">
      <div class="flex items-center bg-dark-800 px-3 py-2 rounded-xl border border-dark-700 gap-2">
        <span class="text-gray-400">Timeframe:</span>
        <select id="select-tf" class="bg-dark-900 border border-dark-600 rounded px-2 py-1 text-white font-mono focus:outline-none">
          <option value="1d" selected>1D (Daily Candles)</option>
          <option value="4h">4H (Swing Momentum)</option>
        </select>
      </div>

      <div class="flex items-center bg-dark-800 px-3 py-2 rounded-xl border border-dark-700 gap-2">
        <span class="text-gray-400">Mode:</span>
        <span class="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded font-medium">Paper Trading (Safe)</span>
      </div>
    </div>
  </div>

  <!-- Main Content Layout Grid -->
  <main class="max-w-7xl mx-auto w-full px-6 pb-12 flex-1 grid grid-cols-1 lg:grid-cols-3 gap-6">

    <!-- Left 2 Columns: Live Candidates, VectorBT Verification & AI Decisions -->
    <div class="lg:col-span-2 space-y-6">

      <!-- Candidate Setups & Backtests Section -->
      <section class="bg-dark-800 rounded-2xl border border-dark-700 p-5 card-glow">
        <div class="flex justify-between items-center mb-4">
          <div>
            <h2 class="text-base font-bold text-white flex items-center gap-2">
              <i class="fa-solid fa-crosshairs text-blue-400"></i> Candidate Breakout Setups & VectorBT Sandbox
            </h2>
            <p class="text-xs text-gray-400">EMA 20/50 + RSI (48-68) + ATR Breakouts verified over past 90 days</p>
          </div>
          <span class="text-xs font-mono bg-dark-900 px-2 py-1 rounded text-gray-400" id="scan-count">0 candidates</span>
        </div>

        <div id="candidates-container" class="space-y-3 max-h-[460px] overflow-y-auto pr-1">
          <div class="text-center py-10 text-gray-500 text-sm">
            <i class="fa-solid fa-radar fa-spin text-3xl mb-3 text-dark-600 block"></i>
            Click "Run Full Autonomous Pipeline" or "Scan Candidates" to analyze Nifty 50 stocks.
          </div>
        </div>
      </section>

      <!-- Active SQLite Trades Log -->
      <section class="bg-dark-800 rounded-2xl border border-dark-700 p-5 card-glow">
        <div class="flex justify-between items-center mb-4">
          <div>
            <h2 class="text-base font-bold text-white flex items-center gap-2">
              <i class="fa-solid fa-database text-purple-400"></i> Trade History (SQLite: trades.db)
            </h2>
            <p class="text-xs text-gray-400">Audit trail of automated CNC Delivery orders with 1% risk budgeting</p>
          </div>
          <button onclick="loadTrades()" class="text-xs text-blue-400 hover:underline flex items-center gap-1">
            <i class="fa-solid fa-arrows-rotate"></i> Refresh
          </button>
        </div>

        <div class="overflow-x-auto">
          <table class="w-full text-xs text-left text-gray-300">
            <thead class="bg-dark-900 text-gray-400 uppercase font-mono border-b border-dark-700">
              <tr>
                <th class="py-2.5 px-3">Ticker</th>
                <th class="py-2.5 px-3">Action</th>
                <th class="py-2.5 px-3">Qty</th>
                <th class="py-2.5 px-3">Entry</th>
                <th class="py-2.5 px-3">Stop Loss</th>
                <th class="py-2.5 px-3">Take Profit</th>
                <th class="py-2.5 px-3">Risk (1%)</th>
                <th class="py-2.5 px-3">Status</th>
              </tr>
            </thead>
            <tbody id="trades-tbody" class="divide-y divide-dark-700/50">
              <tr>
                <td colspan="8" class="text-center py-6 text-gray-500">No trades recorded in database yet.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

    </div>

    <!-- Right 1 Column: Gemini AI Brain Decision & Active Rules -->
    <div class="space-y-6">

      <!-- Gemini AI Brain Highlight Card -->
      <section class="bg-dark-800 rounded-2xl border border-dark-700 p-5 card-glow">
        <div class="flex items-center justify-between mb-3">
          <h2 class="text-base font-bold text-white flex items-center gap-2">
            <i class="fa-solid fa-brain text-purple-400"></i> Gemini 3.8 Flash Brain
          </h2>
          <span class="text-[10px] bg-purple-500/10 text-purple-300 border border-purple-500/20 px-2 py-0.5 rounded uppercase font-mono">Pydantic JSON</span>
        </div>
        <p class="text-xs text-gray-400 mb-4">Evaluates backtest edge, technical structure, and validates R:R &ge; 1:1.5 before order dispatch.</p>

        <div id="ai-decision-card" class="bg-dark-900 rounded-xl p-4 border border-dark-700 text-xs space-y-3">
          <div class="text-gray-500 text-center py-6">
            <i class="fa-solid fa-microchip text-2xl text-dark-600 block mb-2"></i>
            Awaiting setup evaluation from pipeline.
          </div>
        </div>
      </section>

      <!-- Active Rules & Self-Refining Parameters -->
      <section class="bg-dark-800 rounded-2xl border border-dark-700 p-5 card-glow">
        <div class="flex items-center justify-between mb-3">
          <h2 class="text-base font-bold text-white flex items-center gap-2">
            <i class="fa-solid fa-sliders text-amber-400"></i> Active Rules & Insights
          </h2>
          <span class="text-[10px] bg-amber-500/10 text-amber-300 border border-amber-500/20 px-2 py-0.5 rounded font-mono">active_rules.json</span>
        </div>
        <p class="text-xs text-gray-400 mb-3">Dynamically tuned by daily EOD reflection post-mortems.</p>

        <div id="rules-container" class="space-y-2 text-xs">
          <div class="flex justify-between py-1 border-b border-dark-700/50">
            <span class="text-gray-400">Moving Averages:</span>
            <span class="font-mono text-white">EMA 20 &bull; EMA 50</span>
          </div>
          <div class="flex justify-between py-1 border-b border-dark-700/50">
            <span class="text-gray-400">RSI Momentum Zone:</span>
            <span class="font-mono text-white">48.0 &ndash; 68.0</span>
          </div>
          <div class="flex justify-between py-1 border-b border-dark-700/50">
            <span class="text-gray-400">Min VectorBT Win Rate:</span>
            <span class="font-mono text-emerald-400">&ge; 45.0%</span>
          </div>
          <div class="flex justify-between py-1 border-b border-dark-700/50">
            <span class="text-gray-400">Max Risk Per Trade:</span>
            <span class="font-mono text-blue-400">1.0% of Capital</span>
          </div>
          <div class="flex justify-between py-1 border-b border-dark-700/50">
            <span class="text-gray-400">Stop Loss Distance:</span>
            <span class="font-mono text-white">1.5x ATR</span>
          </div>
          <div class="flex justify-between py-1 border-b border-dark-700/50">
            <span class="text-gray-400">Execution Product:</span>
            <span class="font-mono text-amber-300">CNC Delivery (GTT OCO)</span>
          </div>
        </div>

        <div class="mt-4 pt-3 border-t border-dark-700">
          <h3 class="text-xs font-semibold text-gray-300 mb-1 flex items-center gap-1">
            <i class="fa-solid fa-lightbulb text-amber-400"></i> Learned Insights:
          </h3>
          <ul id="insights-list" class="text-[11px] text-gray-400 space-y-1 list-disc list-inside">
            <li>Baseline active. EMA 20/50 + RSI 48-68 avoids tops.</li>
          </ul>
        </div>
      </section>

      <!-- Telegram & Broker Health Card -->
      <section class="bg-dark-800 rounded-2xl border border-dark-700 p-5 card-glow text-xs space-y-2">
        <h2 class="text-base font-bold text-white mb-2 flex items-center gap-2">
          <i class="fa-solid fa-network-wired text-blue-400"></i> Integrations
        </h2>
        <div class="flex justify-between py-1">
          <span class="text-gray-400">Broker:</span>
          <span class="font-semibold text-blue-400">DhanHQ API (NSE CNC)</span>
        </div>
        <div class="flex justify-between py-1">
          <span class="text-gray-400">AI SDK:</span>
          <span class="font-semibold text-purple-300">official google-genai 2.34</span>
        </div>
        <div class="flex justify-between py-1">
          <span class="text-gray-400">Alerts:</span>
          <span class="font-semibold text-emerald-400">Telegram Bot (Daily P&L)</span>
        </div>
        <div class="flex justify-between py-1">
          <span class="text-gray-400">Backtester:</span>
          <span class="font-semibold text-amber-300">VectorBT &amp; Numba JIT</span>
        </div>
      </section>

    </div>
  </main>

  <footer class="bg-dark-800 border-t border-dark-700 py-3 text-center text-xs text-gray-500">
    Ampolin AI Trading Agent &bull; Indian Equities Automated Swing System &bull; Active &amp; Verified
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
        document.getElementById('metric-pnl').innerText = '₹' + data.daily_pnl.toFixed(2);
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
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Managing...';

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
        btn.innerHTML = '<i class="fa-solid fa-arrows-split-up-and-left"></i> Manage Positions (1R/2R)';
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
          tbody.innerHTML = '<tr><td colspan="8" class="text-center py-6 text-gray-500">No trades recorded in database yet.</td></tr>';
          return;
        }

        tbody.innerHTML = data.trades.map(t => `
          <tr class="hover:bg-dark-700/30 transition font-mono">
            <td class="py-2 px-3 font-bold text-white">${t.ticker}</td>
            <td class="py-2 px-3"><span class="${t.action === 'BUY' ? 'text-emerald-400' : 'text-rose-400'} font-semibold">${t.action}</span></td>
            <td class="py-2 px-3">${t.quantity}</td>
            <td class="py-2 px-3">₹${t.entry_price}</td>
            <td class="py-2 px-3 text-rose-400">₹${t.stop_loss}</td>
            <td class="py-2 px-3 text-emerald-400">₹${t.take_profit}</td>
            <td class="py-2 px-3 text-blue-300">₹${t.risk_amount}</td>
            <td class="py-2 px-3"><span class="px-1.5 py-0.5 rounded text-[10px] ${t.status === 'CLOSED' ? 'bg-gray-700 text-gray-300' : 'bg-emerald-500/20 text-emerald-300'}">${t.status}</span></td>
          </tr>
        `).join('');
      } catch (e) {
        console.error('Trades fetch failed:', e);
      }
    }

    async function runPipeline() {
      const btn = document.getElementById('btn-pipeline');
      const tf = document.getElementById('select-tf').value;
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Autonomous Pipeline...';

      try {
        const res = await fetch('/api/run-pipeline', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ timeframe: tf, dry_run: true })
        });
        const data = await res.json();
        await runScan(); // refresh candidate view
        await loadStatus();
        await loadTrades();
      } catch (e) {
        alert('Pipeline execution error: ' + e);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-play"></i> Run Full Autonomous Pipeline';
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
        btn.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> Scan Candidates';
      }
    }

    function renderCandidates(approvedList) {
      const container = document.getElementById('candidates-container');
      document.getElementById('scan-count').innerText = `${approvedList.length} approved setups`;

      if (!approvedList || approvedList.length === 0) {
        container.innerHTML = `
          <div class="text-center py-10 text-gray-400 text-xs">
            <i class="fa-solid fa-shield-cat text-2xl mb-2 text-dark-600 block"></i>
            No candidate stocks met the combined criteria: EMA 20/50 + RSI (48-68) + ATR Breakout + VectorBT Expectancy &gt; 0 and Win Rate &ge; 45%.
          </div>`;
        return;
      }

      container.innerHTML = approvedList.map(item => {
        const s = item.setup;
        const b = item.backtest;
        const d = item.decision;

        return `
          <div class="bg-dark-900 rounded-xl p-4 border border-dark-700 hover:border-blue-500/40 transition">
            <div class="flex justify-between items-start mb-2">
              <div>
                <span class="text-base font-bold text-white font-mono">${s.ticker}</span>
                <span class="ml-2 text-[10px] bg-blue-500/10 text-blue-400 border border-blue-500/20 px-2 py-0.5 rounded">${s.breakout_type}</span>
              </div>
              <div class="text-right">
                <div class="text-base font-bold text-white font-mono">₹${s.close}</div>
                <div class="text-[10px] text-gray-400">ATR: ₹${s.atr}</div>
              </div>
            </div>

            <!-- Technical Gauges -->
            <div class="grid grid-cols-4 gap-2 text-[11px] bg-dark-800/80 p-2.5 rounded-lg border border-dark-700/50 mb-3">
              <div>
                <span class="text-gray-400 block text-[10px]">EMA 20 / 50</span>
                <span class="font-mono text-emerald-400">₹${s.ema_fast} &gt; ₹${s.ema_slow}</span>
              </div>
              <div>
                <span class="text-gray-400 block text-[10px]">RSI (14)</span>
                <span class="font-mono text-blue-300 font-bold">${s.rsi}</span>
              </div>
              <div>
                <span class="text-gray-400 block text-[10px]">VectorBT Win%</span>
                <span class="font-mono text-emerald-400 font-bold">${(b.win_rate * 100).toFixed(1)}%</span>
              </div>
              <div>
                <span class="text-gray-400 block text-[10px]">Expectancy</span>
                <span class="font-mono text-amber-300 font-bold">₹${b.expectancy.toFixed(2)}</span>
              </div>
            </div>

            <!-- Latency & AI Conviction -->
            <div class="flex justify-between items-center text-xs pt-1 border-t border-dark-700/40">
              <span class="text-gray-400 flex items-center gap-1 text-[11px]">
                <i class="fa-solid fa-stopwatch text-amber-400"></i> ${b.execution_time_ms} ms backtest
              </span>
              <button onclick='displayAIDecision(${JSON.stringify(d).replace(/'/g, "&#39;")})' class="text-purple-400 hover:text-purple-300 text-[11px] font-semibold flex items-center gap-1">
                View AI Decision (${(d.confidence * 100).toFixed(0)}% Conviction) <i class="fa-solid fa-arrow-right"></i>
              </button>
            </div>
          </div>
        `;
      }).join('');

      // Auto display first AI decision
      if (approvedList.length > 0) {
        displayAIDecision(approvedList[0].decision);
      }
    }

    function displayAIDecision(d) {
      const card = document.getElementById('ai-decision-card');
      card.innerHTML = `
        <div class="flex justify-between items-center border-b border-dark-700 pb-2">
          <span class="font-mono font-bold text-white text-sm">${d.ticker}</span>
          <span class="px-2 py-0.5 rounded text-xs font-bold ${d.action === 'BUY' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}">${d.action}</span>
        </div>
        <div class="grid grid-cols-3 gap-2 text-center py-2 bg-dark-800 rounded-lg">
          <div>
            <span class="text-[10px] text-gray-400 block">ENTRY</span>
            <span class="font-mono font-bold text-white">₹${d.entry_price}</span>
          </div>
          <div>
            <span class="text-[10px] text-gray-400 block">STOP LOSS</span>
            <span class="font-mono font-bold text-rose-400">₹${d.stop_loss}</span>
          </div>
          <div>
            <span class="text-[10px] text-gray-400 block">TARGET (TP)</span>
            <span class="font-mono font-bold text-emerald-400">₹${d.take_profit}</span>
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

def start_server(host: str = "127.0.0.1", port: int = 8000):
    logger.info(f"Starting Ampolin AI Trading Agent Web UI on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
