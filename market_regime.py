"""
Ampolin AI Trading Agent - Market Regime & Macro Analysis Module
Calculates Nifty 50 Market Breadth (% above 50 EMA) and India VIX Dynamic Risk Sizing.
"""

import logging
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

import pandas as pd
import pandas_ta as ta
import yfinance as yf

from config import (
    NIFTY_50_TICKERS,
    VIX_TICKER,
    INDEX_TICKER,
    VIX_LOW_THRESHOLD,
    VIX_MODERATE_THRESHOLD,
    VIX_HIGH_THRESHOLD,
    MIN_MARKET_BREADTH_PCT,
    MAX_ACCOUNT_RISK_PCT,
)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AmpolinRegime")


@dataclass
class MarketRegimeReport:
    india_vix: float
    vix_regime: str
    dynamic_risk_pct: float
    market_breadth_pct: float
    breadth_regime: str
    index_price: float
    index_above_50_ema: bool
    can_trade_longs: bool
    regime_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MarketRegimeAnalyzer:
    def __init__(self, sample_size: int = 30):
        # sample_size of Nifty 50 constituents for ultra-fast breadth calculation
        self.sample_size = sample_size
        self._cached_report: Optional[MarketRegimeReport] = None

    def fetch_india_vix(self) -> tuple[float, str, float]:
        """
        Fetches current India VIX and derives dynamic account risk multiplier.
        Returns: (vix_value, vix_regime, dynamic_risk_pct)
        """
        try:
            df = yf.download(VIX_TICKER, period="5d", interval="1d", progress=False, multi_level_index=False)
            if df.empty or "Close" not in df:
                logger.warning("Could not fetch India VIX data. Defaulting to moderate regime.")
                return 16.0, "MODERATE", MAX_ACCOUNT_RISK_PCT * 0.75

            vix = float(df["Close"].dropna().iloc[-1])

            if vix <= VIX_LOW_THRESHOLD:
                vix_regime = "LOW"
                dynamic_risk = MAX_ACCOUNT_RISK_PCT  # 1.0% full risk
            elif vix <= VIX_MODERATE_THRESHOLD:
                vix_regime = "MODERATE"
                dynamic_risk = MAX_ACCOUNT_RISK_PCT * 0.75  # 0.75% risk
            elif vix <= VIX_HIGH_THRESHOLD:
                vix_regime = "ELEVATED"
                dynamic_risk = MAX_ACCOUNT_RISK_PCT * 0.50  # 0.50% risk
            else:
                vix_regime = "PANIC"
                dynamic_risk = MAX_ACCOUNT_RISK_PCT * 0.25  # 0.25% defensive risk

            return round(vix, 2), vix_regime, round(dynamic_risk, 4)

        except Exception as e:
            logger.error(f"Error fetching India VIX: {e}")
            return 16.0, "MODERATE", MAX_ACCOUNT_RISK_PCT * 0.75

    def fetch_market_breadth(self) -> tuple[float, str, float, bool]:
        """
        Computes % of Nifty 50 stocks trading above their 50-day EMA.
        Returns: (breadth_pct, breadth_regime, index_price, index_above_50_ema)
        """
        try:
            # 1. Check benchmark index ^NSEI
            index_df = yf.download(INDEX_TICKER, period="4mo", interval="1d", progress=False, multi_level_index=False)
            if not index_df.empty:
                index_close = float(index_df["Close"].iloc[-1])
                ema50 = float(ta.ema(index_df["Close"], length=50).iloc[-1])
                index_above_50 = index_close > ema50
            else:
                index_close = 23000.0
                index_above_50 = True

            # 2. Fast batch download for breadth sample
            tickers = NIFTY_50_TICKERS[:self.sample_size]
            batch_df = yf.download(tickers, period="4mo", interval="1d", progress=False)

            if batch_df.empty:
                return 50.0, "NEUTRAL_CHOP", index_close, index_above_50

            above_50_count = 0
            total_evaluated = 0

            close_data = batch_df["Close"] if "Close" in batch_df else batch_df

            for ticker in tickers:
                if ticker not in close_data:
                    continue
                series = close_data[ticker].dropna()
                if len(series) < 50:
                    continue

                curr_close = series.iloc[-1]
                ema_50_series = ta.ema(series, length=50)
                if ema_50_series is None or ema_50_series.empty or pd.isna(ema_50_series.iloc[-1]):
                    continue

                curr_ema50 = ema_50_series.iloc[-1]
                if curr_close > curr_ema50:
                    above_50_count += 1
                total_evaluated += 1

            breadth_pct = round((above_50_count / max(total_evaluated, 1)) * 100.0, 1)

            if breadth_pct >= 55.0 and index_above_50:
                breadth_regime = "BULLISH_EXPANSION"
            elif breadth_pct < MIN_MARKET_BREADTH_PCT:
                breadth_regime = "BEARISH_DISTRIBUTION"
            else:
                breadth_regime = "NEUTRAL_CHOP"

            return breadth_pct, breadth_regime, round(index_close, 2), index_above_50

        except Exception as e:
            logger.error(f"Error computing market breadth: {e}")
            return 50.0, "NEUTRAL_CHOP", 23000.0, True

    def analyze_regime(self, force_refresh: bool = False) -> MarketRegimeReport:
        """
        Executes full macro & regime assessment combining India VIX and Nifty 50 Breadth.
        """
        if self._cached_report and not force_refresh:
            return self._cached_report

        logger.info("Analyzing Indian Market Regime (India VIX & Nifty 50 Breadth)...")

        vix, vix_regime, dynamic_risk = self.fetch_india_vix()
        breadth_pct, breadth_regime, index_price, index_above_50 = self.fetch_market_breadth()

        # Decision on whether to permit long swing entries:
        # Severe distribution (breadth < 35%) or Panic VIX (> 28) pauses long breakouts
        if breadth_regime == "BEARISH_DISTRIBUTION" or vix_regime == "PANIC":
            can_trade = False
            rationale = (
                f"Market in {breadth_regime} ({breadth_pct}% above 50 EMA) with India VIX at {vix} ({vix_regime}). "
                f"Long swing breakouts paused to prevent false-break whipsaws."
            )
        else:
            can_trade = True
            rationale = (
                f"Market in {breadth_regime} ({breadth_pct}% above 50 EMA, Nifty @ {index_price}). "
                f"India VIX @ {vix} ({vix_regime}). Dynamic position risk set to {dynamic_risk * 100:.2f}%."
            )

        report = MarketRegimeReport(
            india_vix=vix,
            vix_regime=vix_regime,
            dynamic_risk_pct=dynamic_risk,
            market_breadth_pct=breadth_pct,
            breadth_regime=breadth_regime,
            index_price=index_price,
            index_above_50_ema=index_above_50,
            can_trade_longs=can_trade,
            regime_rationale=rationale,
        )

        self._cached_report = report
        logger.info(f"Regime Analysis: {report.breadth_regime} | VIX: {report.india_vix} | Risk: {report.dynamic_risk_pct*100:.2f}% | Can Trade: {report.can_trade_longs}")
        return report


if __name__ == "__main__":
    analyzer = MarketRegimeAnalyzer(sample_size=15)
    report = analyzer.analyze_regime()
    print("\n--- Market Regime Report ---")
    for k, v in report.to_dict().items():
        print(f" {k}: {v}")
