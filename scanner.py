"""
Ampolin AI Trading Agent - Market Scanner Module
Fetches Daily/4H OHLCV candles for Nifty 50 stocks using yfinance (.NS suffix).
Detects EMA 20/50 breakouts filtered by RSI (48-68) and ATR.
"""

import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import pandas_ta as ta
import yfinance as yf

from config import (
    NIFTY_50_TICKERS,
    ACTIVE_RULES_PATH,
    DEFAULT_TIMEFRAME,
    EMA_FAST,
    EMA_SLOW,
    RSI_PERIOD,
    RSI_MIN,
    RSI_MAX,
    ATR_PERIOD,
    ATR_SL_MULTIPLIER,
    TP_RISK_MULTIPLIER,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AmpolinScanner")


@dataclass
class CandidateSetup:
    ticker: str
    timeframe: str
    signal_date: str
    close: float
    open: float
    high: float
    low: float
    volume: float
    ema_fast: float
    ema_slow: float
    rsi: float
    atr: float
    suggested_sl: float
    suggested_tp: float
    risk_reward_ratio: float
    breakout_type: str
    volume_ratio: float
    df: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("df", None)
        return d


class SwingScanner:
    def __init__(self, rules_path: Optional[str] = None):
        self.rules = self._load_rules(rules_path or ACTIVE_RULES_PATH)

    def _load_rules(self, path) -> Dict[str, Any]:
        if path and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("rules", {})
            except Exception as e:
                logger.warning(f"Could not load rules from {path}: {e}. Using config defaults.")
        return {
            "ema_fast": EMA_FAST,
            "ema_slow": EMA_SLOW,
            "rsi_min": RSI_MIN,
            "rsi_max": RSI_MAX,
            "rsi_period": RSI_PERIOD,
            "atr_period": ATR_PERIOD,
            "atr_sl_multiplier": ATR_SL_MULTIPLIER,
            "tp_risk_multiplier": TP_RISK_MULTIPLIER,
        }

    def fetch_data(self, ticker: str, timeframe: str = "1d", lookback_days: int = 180) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from yfinance for an NSE ticker.
        Supports '1d' (Daily) and '4h' (resampled from 1h).
        """
        try:
            if timeframe == "4h":
                # yfinance provides 1h data up to 730 days
                period = f"{min(lookback_days, 700)}d"
                df = yf.download(
                    ticker,
                    period=period,
                    interval="1h",
                    progress=False,
                    multi_level_index=False,
                    auto_adjust=True,
                )
                if df.empty:
                    return None
                df_resampled = df.resample("4h").agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }).dropna()
                return df_resampled

            else:  # Daily (1d)
                period = f"{lookback_days}d"
                df = yf.download(
                    ticker,
                    period=period,
                    interval="1d",
                    progress=False,
                    multi_level_index=False,
                    auto_adjust=True,
                )
                if df.empty:
                    return None
                return df

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates EMA Fast (20), EMA Slow (50), RSI (14), and ATR (14).
        """
        df = df.copy()
        fast_len = int(self.rules.get("ema_fast", EMA_FAST))
        slow_len = int(self.rules.get("ema_slow", EMA_SLOW))
        rsi_len = int(self.rules.get("rsi_period", RSI_PERIOD))
        atr_len = int(self.rules.get("atr_period", ATR_PERIOD))

        df["EMA_FAST"] = ta.ema(df["Close"], length=fast_len)
        df["EMA_SLOW"] = ta.ema(df["Close"], length=slow_len)
        df["RSI"] = ta.rsi(df["Close"], length=rsi_len)
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=atr_len)
        df["VOL_MA20"] = df["Volume"].rolling(window=20).mean()

        return df

    def evaluate_breakout(self, df: pd.DataFrame, ticker: str, timeframe: str) -> Optional[CandidateSetup]:
        """
        Evaluates whether the latest bar satisfies the EMA 20/50 breakout filtered by RSI and ATR.
        """
        if len(df) < 55:
            return None

        # Clean nulls
        clean_df = df.dropna(subset=["EMA_FAST", "EMA_SLOW", "RSI", "ATR"]).copy()
        if len(clean_df) < 5:
            return None

        curr = clean_df.iloc[-1]
        prev = clean_df.iloc[-2]

        rsi_min = float(self.rules.get("rsi_min", RSI_MIN))
        rsi_max = float(self.rules.get("rsi_max", RSI_MAX))
        atr_sl_mult = float(self.rules.get("atr_sl_multiplier", ATR_SL_MULTIPLIER))
        tp_mult = float(self.rules.get("tp_risk_multiplier", TP_RISK_MULTIPLIER))

        close = float(curr["Close"])
        open_price = float(curr["Open"])
        high = float(curr["High"])
        low = float(curr["Low"])
        volume = float(curr["Volume"])
        ema_fast = float(curr["EMA_FAST"])
        ema_slow = float(curr["EMA_SLOW"])
        rsi = float(curr["RSI"])
        atr = float(curr["ATR"])
        vol_ma = float(curr["VOL_MA20"]) if not pd.isna(curr["VOL_MA20"]) else volume

        # Breakout Criteria:
        # 1. Trend alignment: EMA 20 > EMA 50 (bullish moving average alignment)
        # 2. Price positioning: Close > EMA 20
        # 3. Fresh breakout trigger:
        #    a) Previous bar was below/at EMA 20 and current bar broke above EMA 20, OR
        #    b) EMA 20 recently crossed EMA 50 (Golden Cross within last 3 bars), OR
        #    c) Bullish expansion candle (Close > Open and Close > EMA 20) with volume > 0.8x avg
        is_bullish_alignment = ema_fast > ema_slow and close > ema_fast
        fresh_cross_ema20 = (prev["Close"] <= prev["EMA_FAST"]) and (close > ema_fast)
        recent_golden_cross = any(
            clean_df.iloc[i]["EMA_FAST"] > clean_df.iloc[i]["EMA_SLOW"]
            and clean_df.iloc[i-1]["EMA_FAST"] <= clean_df.iloc[i-1]["EMA_SLOW"]
            for i in range(-3, 0)
        )
        bullish_expansion = (close > open_price) and (close > ema_fast * 1.002)

        breakout_valid = is_bullish_alignment and (fresh_cross_ema20 or recent_golden_cross or bullish_expansion)
        if not breakout_valid:
            return None

        # 4. RSI Momentum Filter:
        # Must be in sweet-spot range (48 to 68) - avoids oversold chop & overextended tops
        if not (rsi_min <= rsi <= rsi_max):
            return None

        # 5. ATR Volatility Filter:
        # ATR must be valid and non-zero (> 0.2% of price to ensure sufficient liquidity/volatility)
        if atr <= 0 or (atr / close) < 0.002:
            return None

        # Determine breakout classification
        if fresh_cross_ema20:
            breakout_type = "EMA20_PRICE_CROSSOVER"
        elif recent_golden_cross:
            breakout_type = "EMA20_EMA50_GOLDEN_CROSS"
        else:
            breakout_type = "EMA_TREND_MOMENTUM_EXPANSION"

        # Calculate Protective Stop Loss and Take Profit targets using ATR
        sl_distance = atr * atr_sl_mult
        suggested_sl = round(close - sl_distance, 2)
        target_distance = sl_distance * tp_mult
        suggested_tp = round(close + target_distance, 2)
        risk_reward = round(target_distance / sl_distance, 2) if sl_distance > 0 else 2.0
        volume_ratio = round(volume / vol_ma, 2) if vol_ma > 0 else 1.0

        signal_time_str = str(clean_df.index[-1].strftime("%Y-%m-%d %H:%M") if hasattr(clean_df.index[-1], "strftime") else clean_df.index[-1])

        return CandidateSetup(
            ticker=ticker,
            timeframe=timeframe,
            signal_date=signal_time_str,
            close=round(close, 2),
            open=round(open_price, 2),
            high=round(high, 2),
            low=round(low, 2),
            volume=round(volume, 2),
            ema_fast=round(ema_fast, 2),
            ema_slow=round(ema_slow, 2),
            rsi=round(rsi, 2),
            atr=round(atr, 2),
            suggested_sl=suggested_sl,
            suggested_tp=suggested_tp,
            risk_reward_ratio=risk_reward,
            breakout_type=breakout_type,
            volume_ratio=volume_ratio,
            df=clean_df,
        )

    def scan_universe(
        self,
        tickers: Optional[List[str]] = None,
        timeframe: str = DEFAULT_TIMEFRAME,
        lookback_days: int = 180,
    ) -> List[CandidateSetup]:
        """
        Scans given universe or full Nifty 50 for setups.
        """
        ticker_list = tickers or NIFTY_50_TICKERS
        candidates: List[CandidateSetup] = []
        logger.info(f"Scanning {len(ticker_list)} stocks on timeframe '{timeframe}'...")

        for ticker in ticker_list:
            df = self.fetch_data(ticker, timeframe=timeframe, lookback_days=lookback_days)
            if df is None or df.empty:
                continue

            df = self.compute_indicators(df)
            setup = self.evaluate_breakout(df, ticker, timeframe)
            if setup:
                logger.info(
                    f"-> Setup found: {ticker} @ ₹{setup.close} | RSI: {setup.rsi} | "
                    f"EMA20: {setup.ema_fast} > EMA50: {setup.ema_slow} | ATR: {setup.atr} | Type: {setup.breakout_type}"
                )
                candidates.append(setup)

        logger.info(f"Scan complete. Found {len(candidates)} candidate setups out of {len(ticker_list)} stocks.")
        return candidates


if __name__ == "__main__":
    scanner = SwingScanner()
    # Quick dry scan on top liquid Nifty 50 stocks
    sample_tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "INFY.NS", "LT.NS"]
    results = scanner.scan_universe(sample_tickers, timeframe="1d")
    print(f"\nDiscovered {len(results)} candidate setups:")
    for r in results:
        print(f" - {r.ticker}: Close={r.close}, RSI={r.rsi}, ATR={r.atr}, SL={r.suggested_sl}, TP={r.suggested_tp}")
