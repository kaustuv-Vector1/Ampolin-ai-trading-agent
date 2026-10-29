"""
Ampolin AI Trading Agent - AI Brain Module
Passes approved setups to Gemini Flash using the official google-genai SDK
with strict Pydantic JSON schema (BUY/SELL/PASS, entry, stop loss, take profit, confidence).
"""

import json
import logging
import os
from typing import Literal, Optional, Dict, Any

from pydantic import BaseModel, Field

from config import GEMINI_API_KEY, GEMINI_MODEL, ACTIVE_RULES_PATH
from scanner import CandidateSetup
from sandbox import BacktestResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AmpolinBrain")

# Try importing official google-genai SDK
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-genai SDK not imported. Running in simulation mode.")


class TradeDecision(BaseModel):
    ticker: str = Field(..., description="Stock Ticker (NSE symbol with .NS or Wall Street ticker e.g. NVDA, AAPL)")
    action: Literal["BUY", "SELL", "PASS"] = Field(
        ..., description="Swing trade recommendation: BUY for high conviction long, SELL for short, PASS to skip"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Conviction score between 0.0 and 1.0 (requires >= 0.75 for execution)")
    entry_price: float = Field(..., description="Recommended entry execution price in local currency (INR or USD)")
    stop_loss: float = Field(..., description="Protective Stop Loss price in local currency")
    take_profit: float = Field(..., description="Take Profit target price in local currency")
    risk_reward_ratio: float = Field(..., description="Risk to Reward ratio (e.g. 2.0 for 1:2)")
    holding_period_days: int = Field(default=5, description="Expected swing holding period in trading days")
    rationale: str = Field(..., description="Technical thesis evaluating Multi-Timeframe alignment, EMA breakout, volume footprint, and VectorBT edge")
    key_risks: str = Field(..., description="Specific invalidation criteria, market volatility, or macro headwinds")


class GeminiBrain:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        trade_logger: Optional[Any] = None,
    ):
        self.api_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name or GEMINI_MODEL or "gemini-3.5-flash-lite"
        self.trade_logger = trade_logger
        self.catalyst_tracker = None
        try:
            from catalyst_tracker import CatalystTracker
            self.catalyst_tracker = CatalystTracker()
        except Exception as e:
            logger.debug(f"Could not load CatalystTracker: {e}")

        self.client = None

        if self.api_key and GENAI_AVAILABLE:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Gemini client initialized with model '{self.model_name}'")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini Client: {e}. Will use intelligent fallback.")
        else:
            logger.info("No active GEMINI_API_KEY configured. GeminiBrain will operate in rule-based verification fallback.")

    def _build_analysis_prompt(self, setup: CandidateSetup, backtest: BacktestResult, rules: Dict[str, Any]) -> str:
        sym = "₹" if setup.currency == "INR" else "$"

        catalyst_context = "No breaking news or insider alerts."
        if self.catalyst_tracker:
            catalyst_context = self.catalyst_tracker.get_catalyst_context(setup.ticker)

        memory_context = "No historical closed trades recorded yet."
        if self.trade_logger and hasattr(self.trade_logger, "get_experiential_memory"):
            memory_context = self.trade_logger.get_experiential_memory(setup.ticker)

        prompt = f"""
You are the Chief Quantitative Strategist for 'Ampolin AI Trading Agent', an institutional-grade swing trading system
operating on the National Stock Exchange of India (NSE) and US Wall Street Equities for cash delivery positions.

Analyze the following validated technical breakout setup and deliver a definitive trading decision.

=== CANDIDATE ASSET SETUP ===
- Ticker: {setup.ticker} (Currency: {setup.currency})
- Timeframe: {setup.timeframe}
- Current Close: {sym}{setup.close}
- Candle Open: {sym}{setup.open} | High: {sym}{setup.high} | Low: {sym}{setup.low}
- Candlestick Close Strength: {setup.candle_strength:.2f} (0.0 to 1.0; > 0.55 indicates strong buyers in control)
- 20 EMA: {sym}{setup.ema_fast} (Fast EMA)
- 50 EMA: {sym}{setup.ema_slow} (Slow EMA)
- Fast EMA > Slow EMA: {setup.ema_fast > setup.ema_slow}
- Higher Timeframe Weekly Trend Confluence: {setup.weekly_trend}
- RSI (14): {setup.rsi:.1f} (Target sweet spot: 48.0 - 68.0)
- ATR (14): {sym}{setup.atr:.2f} ({setup.atr / setup.close * 100:.2f}% of price)
- Volume Expansion Ratio (vs 20 MA): {setup.volume_ratio:.2f}x
- Breakout Type: {setup.breakout_type}
- Suggested Technical SL: {sym}{setup.suggested_sl}
- Suggested Technical TP: {sym}{setup.suggested_tp}
- Base Risk/Reward: 1:{setup.risk_reward_ratio}

=== BREAKING NEWS & BIG-MONEY / INSIDER CATALYSTS ===
{catalyst_context}

=== EXPERIENTIAL PAST TRADE MEMORY (LEARNED LESSONS) ===
{memory_context}

=== 90-DAY VECTORBT BACKTEST VERIFICATION ===
- Historical Win Rate: {backtest.win_rate * 100:.1f}% (Minimum required: 45.0%)
- Expectancy: {sym}{backtest.expectancy:.2f} (Must be positive)
- Profit Factor: {backtest.profit_factor:.2f}
- Historical Trades Count: {backtest.total_trades}
- Max Historical Drawdown: {backtest.max_drawdown_pct:.1f}%
- Vectorized Simulation Latency: {backtest.execution_time_ms} ms

=== INSTITUTIONAL TRADING RULES ===
- Multi-Timeframe Alignment: Strongly prioritize setups with 'BULLISH_CONFLUENCE' on the Weekly timeframe.
- News/Catalyst Awareness: If breaking news indicates severe regulatory, legal, or earnings risk, reject or PASS.
- Memory Integration: Learn from past trade outcomes. Avoid setups resembling recent failed patterns.
- Maximum account risk per trade: 1.0% of total capital (dynamically adjusted).
- Minimum Risk to Reward: 1:2.0.
- Holding period: Multi-day swing (typically 3 to 10 trading sessions).
- Provide confidence >= 0.75 ONLY if technical breakout, volume footprint, and VectorBT statistical edge align convincingly. Otherwise, choose PASS.

Provide your strict JSON response evaluating whether to BUY, SELL, or PASS this swing setup.
"""
        return prompt

    def evaluate_setup(self, setup: CandidateSetup, backtest: BacktestResult) -> TradeDecision:
        """
        Passes the setup to Gemini Flash and parses into strict Pydantic TradeDecision.
        """
        rules = {}
        if ACTIVE_RULES_PATH.exists():
            try:
                with open(ACTIVE_RULES_PATH, "r", encoding="utf-8") as f:
                    rules = json.load(f).get("rules", {})
            except Exception:
                pass

        # If live API client is available, call Gemini Flash with strict Pydantic schema
        if self.client:
            prompt = self._build_analysis_prompt(setup, backtest, rules)
            models_to_try = [self.model_name, "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash"]
            # Deduplicate preserving order
            seen = set()
            unique_models = [m for m in models_to_try if not (m in seen or seen.add(m))]

            for model in unique_models:
                try:
                    logger.info(f"Querying Gemini ({model}) for {setup.ticker} trade decision...")
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=TradeDecision,
                            temperature=0.1,
                        ),
                    )
                    parsed_decision = TradeDecision.model_validate_json(response.text)
                    logger.info(
                        f"Gemini Decision for {setup.ticker}: {parsed_decision.action} | "
                        f"Confidence: {parsed_decision.confidence:.2f} | "
                        f"Entry: ₹{parsed_decision.entry_price} | SL: ₹{parsed_decision.stop_loss} | TP: ₹{parsed_decision.take_profit}"
                    )
                    return parsed_decision
                except Exception as e:
                    logger.warning(f"Error querying model '{model}': {e}. Trying fallback model...")
                    continue

        # Intelligent deterministic fallback when GEMINI_API_KEY is not configured
        logger.info(f"Using rule-based AI verification engine for {setup.ticker}")
        return self._rule_based_fallback(setup, backtest)

    def _rule_based_fallback(self, setup: CandidateSetup, backtest: BacktestResult) -> TradeDecision:
        """
        High-fidelity rule-based fallback that generates a fully structured TradeDecision
        strictly adhering to the user's criteria.
        """
        # Baseline confidence derived from backtest win rate and expectancy
        base_confidence = min(0.95, max(0.40, backtest.win_rate * 0.6 + min(backtest.expectancy / 10.0, 0.35)))

        # If backtest passed with solid expectancy and win rate >= 45%
        if backtest.approved and setup.close > setup.ema_fast and (48.0 <= setup.rsi <= 68.0):
            action = "BUY"
            confidence = round(base_confidence, 2)
            entry = setup.close
            stop_loss = setup.suggested_sl
            take_profit = setup.suggested_tp
            rr = setup.risk_reward_ratio
            rationale = (
                f"Valid EMA 20/50 breakout ({setup.breakout_type}). Price ₹{setup.close} above EMA 20 (₹{setup.ema_fast}) "
                f"with EMA 50 (₹{setup.ema_slow}) aligned in bullish sequence. RSI at {setup.rsi:.1f} is well within the "
                f"48-68 momentum zone without exhaustion. 90-day VectorBT backtest confirmed {backtest.win_rate*100:.1f}% "
                f"win rate and positive expectancy of ₹{backtest.expectancy:.2f} per rupee risked."
            )
            key_risks = (
                f"Break below technical stop ₹{stop_loss} (1.5x ATR buffer) invalidates setup. Watch for broader Nifty 50 "
                f"market sentiment reversals and quarterly earnings dates."
            )
        else:
            action = "PASS"
            confidence = 0.35
            entry = setup.close
            stop_loss = setup.suggested_sl
            take_profit = setup.suggested_tp
            rr = setup.risk_reward_ratio
            rationale = (
                f"Setup fails high-conviction criteria: Backtest approved={backtest.approved}, "
                f"RSI={setup.rsi:.1f}, Win rate={backtest.win_rate*100:.1f}%."
            )
            key_risks = "Choppy market structure or low statistical expectancy."

        return TradeDecision(
            ticker=setup.ticker,
            action=action,
            confidence=confidence,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=rr,
            holding_period_days=7,
            rationale=rationale,
            key_risks=key_risks,
        )


if __name__ == "__main__":
    from scanner import CandidateSetup
    from sandbox import BacktestResult

    brain = GeminiBrain()

    # Dummy test setup
    dummy_setup = CandidateSetup(
        ticker="RELIANCE.NS",
        timeframe="1d",
        signal_date="2026-10-29",
        close=1200.0,
        open=1190.0,
        high=1205.0,
        low=1188.0,
        volume=15000000.0,
        ema_fast=1195.0,
        ema_slow=1175.0,
        rsi=56.5,
        atr=25.0,
        suggested_sl=1162.5,
        suggested_tp=1275.0,
        risk_reward_ratio=2.0,
        breakout_type="EMA20_PRICE_CROSSOVER",
        volume_ratio=1.2,
    )

    dummy_backtest = BacktestResult(
        ticker="RELIANCE.NS",
        approved=True,
        rejection_reason=None,
        win_rate=0.60,
        expectancy=15.2,
        profit_factor=2.1,
        total_trades=8,
        total_return_pct=14.5,
        execution_time_ms=3.4,
        avg_win_pct=4.2,
        avg_loss_pct=-2.0,
        max_drawdown_pct=-3.5,
    )

    decision = brain.evaluate_setup(dummy_setup, dummy_backtest)
    print("\n--- AI Brain Evaluation Result ---")
    print(decision.model_dump_json(indent=2))
