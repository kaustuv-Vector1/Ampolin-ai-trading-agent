"""
Ampolin Trading Bot - Market Hours and Currency Utility
Handles timezone-aware market session detection, early-close schedules, and live USD/INR conversion.
"""
import logging
import time as time_module
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("AmpolinMarketHours")

NY_TZ = ZoneInfo("America/New_York")
KOLKATA_TZ = ZoneInfo("Asia/Kolkata")

_USD_INR_CACHE = {
    "rate": 84.50,
    "timestamp": 0.0,
}


def get_usd_inr_rate(max_age_seconds: int = 1800, fallback_rate: float = 84.50) -> float:
    """Fetches live USD/INR rate from Yahoo Finance with memory caching and explicit fallback warning."""
    now = time_module.time()
    if _USD_INR_CACHE["timestamp"] > 0 and (now - _USD_INR_CACHE["timestamp"] < max_age_seconds):
        return _USD_INR_CACHE["rate"]

    try:
        import yfinance as yf
        df = yf.download("USDINR=X", period="5d", interval="1d", progress=False)
        if not df.empty and "Close" in df:
            close_col = df["Close"]
            rate = float(close_col.iloc[-1].iloc[0]) if hasattr(close_col.iloc[-1], "iloc") else float(close_col.iloc[-1])
            if rate > 50.0:
                _USD_INR_CACHE["rate"] = round(rate, 2)
                _USD_INR_CACHE["timestamp"] = now
                return _USD_INR_CACHE["rate"]
    except Exception as e:
        logger.warning(
            f"[USD/INR FALLBACK] Could not fetch live USD/INR rate: {e}. "
            f"Using fallback rate ₹{fallback_rate:.2f}/USD. Note: rate may be stale."
        )

    if _USD_INR_CACHE["timestamp"] == 0.0:
        logger.warning(
            f"[USD/INR FALLBACK] No cached exchange rate available. "
            f"Using hardcoded fallback ₹{fallback_rate:.2f}/USD."
        )
        _USD_INR_CACHE["rate"] = fallback_rate

    return _USD_INR_CACHE["rate"]


def is_us_market_open(dt: Optional[datetime] = None) -> bool:
    """
    Checks if US stock market (NYSE/NASDAQ) is actively open:
    - Monday to Friday
    - Skips official US market holidays
    - Supports early-close days (e.g. 13:00 ET on Black Friday / Christmas Eve)
    """
    now_ny = dt.astimezone(NY_TZ) if (dt and dt.tzinfo) else (dt.replace(tzinfo=NY_TZ) if dt else datetime.now(NY_TZ))
    if now_ny.weekday() >= 5:  # Saturday or Sunday
        return False

    date_str = now_ny.strftime("%Y-%m-%d")
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=date_str, end_date=date_str)
        if sched.empty:
            return False

        # Minute-accurate session open and close from calendar
        session_open = sched.iloc[0]["market_open"].tz_convert(NY_TZ)
        session_close = sched.iloc[0]["market_close"].tz_convert(NY_TZ)
        return session_open <= now_ny <= session_close
    except Exception as e:
        logger.warning(f"Could not check NYSE schedule calendar: {e}. Falling back to default hours.")
        current_time = now_ny.time()
        return time(9, 30) <= current_time <= time(16, 0)


def is_nse_market_open(dt: Optional[datetime] = None) -> bool:
    """
    Checks if Indian stock market (NSE) is actively open:
    - Monday to Friday
    - Skips official Indian market holidays
    - Minute-accurate session check via BSE/NSE calendar
    """
    now_in = dt.astimezone(KOLKATA_TZ) if (dt and dt.tzinfo) else (dt.replace(tzinfo=KOLKATA_TZ) if dt else datetime.now(KOLKATA_TZ))
    if now_in.weekday() >= 5:
        return False

    date_str = now_in.strftime("%Y-%m-%d")
    try:
        import pandas_market_calendars as mcal
        bse = mcal.get_calendar("BSE")  # BSE and NSE share equity market calendar
        sched = bse.schedule(start_date=date_str, end_date=date_str)
        if sched.empty:
            return False

        session_open = sched.iloc[0]["market_open"].tz_convert(KOLKATA_TZ)
        session_close = sched.iloc[0]["market_close"].tz_convert(KOLKATA_TZ)
        return session_open <= now_in <= session_close
    except Exception as e:
        logger.warning(f"Could not check BSE/NSE holiday calendar: {e}. Falling back to default hours.")
        current_time = now_in.time()
        return time(9, 15) <= current_time <= time(15, 30)
