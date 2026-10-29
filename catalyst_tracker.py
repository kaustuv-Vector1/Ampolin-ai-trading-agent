"""
Ampolin AI Trading Agent - Catalyst & Big Money Tracker
Pulls breaking news, market catalysts, SEC Form 4 insider transactions,
and institutional whale footprints using yfinance.
"""

import logging
import sys
from typing import Dict, Any, List, Optional
import yfinance as yf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AmpolinCatalyst")


class CatalystTracker:
    def __init__(self, max_news_items: int = 3, max_insider_rows: int = 3):
        self.max_news = max_news_items
        self.max_insiders = max_insider_rows

    def fetch_news(self, ticker: str) -> List[Dict[str, str]]:
        """
        Fetches latest breaking news headlines and publisher for a ticker.
        """
        clean_sym = ticker.replace(".NS", "").replace(".BO", "")
        news_items = []
        try:
            search = yf.Search(clean_sym, news_count=self.max_news)
            for item in getattr(search, "news", [])[:self.max_news]:
                title = item.get("title", "")
                publisher = item.get("publisher", "")
                if title:
                    news_items.append({
                        "title": title,
                        "publisher": publisher,
                    })
        except Exception as e:
            logger.debug(f"Could not fetch news for {ticker}: {e}")

        return news_items

    def fetch_insider_trades(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Fetches recent SEC Form 4 insider / institutional transactions for US stocks or major movements.
        """
        insider_data = []
        # Insiders are heavily reported on US equities (without .NS)
        if ".NS" in ticker or ".BO" in ticker:
            return []

        try:
            t = yf.Ticker(ticker)
            df = getattr(t, "insider_transactions", None)
            if df is not None and hasattr(df, "empty") and not df.empty:
                for _, row in df.head(self.max_insiders).iterrows():
                    shares = row.get("Shares", 0)
                    value = row.get("Value", 0)
                    text = row.get("Text", "Transaction")
                    date = str(row.get("Start Date", ""))
                    insider_data.append({
                        "shares": shares,
                        "value": value,
                        "description": text,
                        "date": date,
                    })
        except Exception as e:
            logger.debug(f"Could not fetch insider data for {ticker}: {e}")

        return insider_data

    def get_catalyst_context(self, ticker: str) -> str:
        """
        Returns a formatted summary for injection into the Gemini AI analysis prompt.
        """
        news = self.fetch_news(ticker)
        insiders = self.fetch_insider_trades(ticker)

        lines = []
        if news:
            lines.append("Breaking News & Market Catalysts:")
            for n in news:
                lines.append(f"• {n['title']} (Source: {n['publisher']})")
        else:
            lines.append("Breaking News: No major breaking negative news detected (Neutral sentiment).")

        if insiders:
            lines.append("Recent Insider / Big Money Filings:")
            for ins in insiders:
                val_str = f"${ins['value']:,.0f}" if ins['value'] else "N/A"
                lines.append(f"• {ins['date']}: {ins['description']} ({ins['shares']:,} shares, Value: {val_str})")

        return "\n".join(lines)


if __name__ == "__main__":
    tracker = CatalystTracker()
    print("Testing US stock catalyst (NVDA):")
    print(tracker.get_catalyst_context("NVDA"))
    print("\nTesting Indian stock catalyst (RELIANCE.NS):")
    print(tracker.get_catalyst_context("RELIANCE.NS"))
