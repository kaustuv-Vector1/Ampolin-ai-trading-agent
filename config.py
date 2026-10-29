"""
Ampolin AI Trading Agent - Configuration Module
Automated Swing Trading System for Indian Equities (NSE) using DhanHQ & Gemini AI
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "trades.db"
ACTIVE_RULES_PATH = BASE_DIR / "active_rules.json"
ENV_PATH = BASE_DIR / ".env"

# Load environment variables from .env if present
if ENV_PATH.exists():
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

# DhanHQ API Credentials
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")

# Gemini AI Credentials & Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Configurable model name; defaults to gemini-2.5-flash or gemini-3.8-flash
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Telegram Bot Credentials
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Risk Management Settings
MAX_ACCOUNT_RISK_PCT = float(os.getenv("MAX_ACCOUNT_RISK_PCT", "0.01"))  # 1% base risk per trade
DEFAULT_ACCOUNT_CAPITAL = float(os.getenv("DEFAULT_ACCOUNT_CAPITAL", "100000.0"))  # INR 1,00,000

# Market Regime & Macro Filters (India VIX & Nifty 50 Breadth)
VIX_TICKER = "^INDIAVIX"
INDEX_TICKER = "^NSEI"
VIX_LOW_THRESHOLD = 15.0       # Normal / Aggressive (1.0% risk)
VIX_MODERATE_THRESHOLD = 22.0  # Moderate volatility (0.75% risk)
VIX_HIGH_THRESHOLD = 28.0      # Elevated volatility (0.50% risk)
MIN_MARKET_BREADTH_PCT = 35.0  # Min % of Nifty 50 stocks above 50 EMA required for new longs

# Multi-Stage Profit Scaling & Trailing Stops
STAGE1_BREAKEVEN_R = 1.0       # Move Stop Loss to Breakeven when trade reaches 1.0R profit
STAGE2_PARTIAL_EXIT_R = 2.0    # Book 50% partial exit when trade reaches 2.0R profit
STAGE2_PARTIAL_EXIT_RATIO = 0.50 # 50% of position scaled out at 2R
STAGE3_TRAILING_EMA = 20       # Trail remaining 50% along the 20-period EMA

# Technical Scanner & Strategy Defaults
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "1d")  # "1d" or "4h"
EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_MIN = float(os.getenv("RSI_MIN", "48.0"))
RSI_MAX = float(os.getenv("RSI_MAX", "68.0"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_SL_MULTIPLIER = float(os.getenv("ATR_SL_MULTIPLIER", "1.5"))
TP_RISK_MULTIPLIER = float(os.getenv("TP_RISK_MULTIPLIER", "2.0"))

# VectorBT Sandbox Filter Requirements
MIN_WIN_RATE = float(os.getenv("MIN_WIN_RATE", "0.45"))  # >= 45% win rate required
MIN_EXPECTANCY = float(os.getenv("MIN_EXPECTANCY", "0.0"))  # Positive expectancy required
BACKTEST_LOOKBACK_DAYS = int(os.getenv("BACKTEST_LOOKBACK_DAYS", "90"))

# Nifty 50 Tickers (NSE suffix .NS for yfinance)
NIFTY_50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    "SBIN.NS", "INFY.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "ADANIENT.NS",
    "ZOMATO.NS", "KOTAKBANK.NS", "NTPC.NS", "ONGC.NS", "AXISBANK.NS",
    "TITAN.NS", "POWERGRID.NS", "TATASTEEL.NS", "M&M.NS", "COALINDIA.NS",
    "BAJAJFINSV.NS", "ASIANPAINT.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "JSWSTEEL.NS",
    "GRASIM.NS", "TECHM.NS", "WIPRO.NS", "CIPLA.NS", "HINDALCO.NS",
    "ADANIPORTS.NS", "SBILIFE.NS", "DRREDDY.NS", "BPCL.NS", "EICHERMOT.NS",
    "TATACONSUM.NS", "BRITANNIA.NS", "APOLLOHOSP.NS", "HEROMOTOCO.NS", "SHREECEM.NS",
    "DIVISLAB.NS", "INDUSINDBK.NS", "BAJAJ-AUTO.NS", "TRENT.NS", "BEL.NS"
]
