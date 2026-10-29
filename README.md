# Ampolin AI Trading Agent ⚡📈

Automated swing trading system for **Indian Equities (NSE)** engineered with **DhanHQ**, **Gemini 3.8 Flash**, and **VectorBT**.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    A["Nifty 50 Universe (NSE)"] --> B["Scanner Engine (scanner.py)<br/>EMA 20/50 + RSI (48-68) + ATR"]
    B -->|Candidate Setups| C["VectorBT Sandbox (sandbox.py)<br/>Sub-100ms 90-Day Vectorized Backtest"]
    C -->|Approved Setups (Expectancy > 0, Win Rate >= 45%)| D["AI Brain (ai_brain.py)<br/>Gemini 3.8 Flash (Pydantic Schema)"]
    D -->|BUY Conviction with SL/TP| E["DhanHQ Broker (broker.py)<br/>CNC Delivery Bracket (1% Max Account Risk)"]
    E -->|Executed Order| F["Trade Logger (reflection.py)<br/>SQLite (trades.db)"]
    E -->|Instant Alert| G["Telegram Notifier (telegram_notifier.py)<br/>Daily P&L & Trade Alerts"]
    F -->|EOD Post-Mortem| H["Reflection Agent (reflection.py)<br/>Self-Refines active_rules.json"]
    H -.->|Adaptive Rules Feedback| B
    H -.->|Adaptive Rules Feedback| C
```

---

## 🚀 Key Modules

1. **`scanner.py`**:
   - Downloads Daily / 4H OHLCV candles for Nifty 50 stocks using `yfinance` (`.NS` suffix).
   - Identifies EMA 20 / 50 bullish breakouts.
   - Filters out overbought tops and weak chop using **RSI (48.0 - 68.0)**.
   - Computes dynamic stop loss and target buffers using **ATR (14)**.

2. **`sandbox.py`**:
   - Uses **VectorBT** accelerated with **Numba JIT** to run sub-100ms (typically 2-5 ms) vectorized backtests over the past 90 days.
   - Filters and approves only setups with **positive expectancy** (`expectancy > 0`) and **win rate >= 45%**.

3. **`market_regime.py` (New)**:
   - **Nifty 50 Market Breadth Filter**: Evaluates % of Nifty 50 constituents trading above their 50 EMA. When breadth < 35%, pauses long breakouts to prevent whipsaws during market-wide distribution.
   - **India VIX Dynamic Risk Sizing**: Fetches `^INDIAVIX` and throttles risk per trade dynamically:
     - VIX &le; 15: 1.0% risk (Normal/Aggressive)
     - 15 < VIX &le; 22: 0.75% risk (Moderate)
     - 22 < VIX &le; 28: 0.50% risk (Elevated)
     - VIX > 28: 0.25% defensive risk (Panic)

4. **`position_manager.py` (New)**:
   - **Stage 1 (1.0R Reached)**: Protective Stop Loss is ratcheted up to **Breakeven** (entry price). Trade becomes 100% risk-free.
   - **Stage 2 (2.0R Reached)**: Books **50% partial profit** (CNC Delivery) and attaches a trailing stop along the **20 EMA**.
   - **Stage 3 (Trailing 20 EMA)**: Trails remaining 50% shares with the rising 20 EMA until price closes below it.

5. **`ai_brain.py`**:
   - Evaluates approved setups using the official `google-genai` SDK and Gemini Flash.
   - Enforces strict Pydantic JSON schema: `action` (`BUY`/`SELL`/`PASS`), `confidence`, `entry_price`, `stop_loss`, `take_profit`, `risk_reward_ratio`, `rationale`, and `key_risks`.
   - Includes graceful deterministic fallback if no API key is set for offline dry-runs.

6. **`broker.py`**:
   - Connects to **DhanHQ API** (`dhanhq`) for NSE Cash Equity swing trades.
   - Enforces **dynamic account risk** position sizing based on India VIX.
   - Places multi-day swing bracket orders using **Dhan Forever Orders (GTT OCO)** on CNC Delivery positions.
   - Paper trading (dry-run) simulation mode for safe testing without risking capital.

7. **`reflection.py`**:
   - SQLite Trade Logger (`trades.db`) maintaining an audit trail of orders, daily P&L, multi-stage trade lifecycle, and reflections.
   - Daily EOD post-mortem engine that analyzes win/loss clusters and automatically self-adjusts `active_rules.json` parameters.

8. **`telegram_notifier.py`**:
   - Dispatches formatted HTML notifications to Telegram for trade approvals, execution fills, 1R breakeven triggers, 2R partial exits, and daily P&L.

9. **`ui.py`**:
   - Modern dark-theme web dashboard powered by **FastAPI** & **Tailwind CSS**.
   - Live Market Breadth gauge, India VIX meter, candidate scanner, VectorBT backtest inspector, AI conviction viewer, Dhan order book, and rules manager.

10. **`cloud_scheduler.py` (New)**:
    - 24/7 autonomous daemon aligned with IST trading hours (09:15 open sweep, 10:30 & 13:45 scans, 15:10 pre-close trailing stop sweep, 15:45 EOD reflection).

---

## ☁️ Cloud Deployment (24/7 Autonomous Execution)

The system is 100% cloud-ready and can be deployed anywhere with a single command:

### Option A: Docker / Docker Compose (Any VPS, EC2, or GCP Compute)
```bash
docker compose up -d
```
This runs the container with timezone locked to `Asia/Kolkata` (IST) and mounts `trades.db` and `active_rules.json` for persistence.

### Option B: Google Cloud Run
Deploy as a serverless containerized service:
```bash
gcloud run deploy ampolin-trader \
  --source . \
  --platform managed \
  --region asia-south1 \
  --allow-unauthenticated \
  --port 8000
```

---

## 📦 Setup & Installation

### 1. Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your keys:
```powershell
cp .env.example .env
```

```env
DHAN_CLIENT_ID=your_dhan_client_id
DHAN_ACCESS_TOKEN=your_dhan_jwt_token
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## 🖥️ Running the System

### 1. Execute Scanner & VectorBT Sandbox Dry-Run
```powershell
.\venv\Scripts\python.exe scanner.py
.\venv\Scripts\python.exe sandbox.py
```

### 2. Run the Full Autonomous Pipeline (Dry-Run / Paper Mode)
```powershell
.\venv\Scripts\python.exe main.py --dry-run
```

### 3. Run on 4-Hour Timeframe
```powershell
.\venv\Scripts\python.exe main.py --timeframe 4h
```

### 4. Run EOD Post-Mortem & Self-Refine Rules
```powershell
.\venv\Scripts\python.exe main.py --eod
```

### 5. Launch the Web UI Dashboard
```powershell
.\venv\Scripts\python.exe ui.py
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## 💡 Suggested Improvements & Strategic Roadmap

1. **Sectoral Breadth Filter**:
   - Add Nifty 50 Market Breadth filter (% of stocks above 50 EMA) to avoid long swing entries when broader market is under severe distribution.
2. **Dynamic Volatility Bands**:
   - Incorporate India VIX into position sizing: when India VIX > 22, reduce max account risk from 1.0% to 0.5%.
3. **Trailing Stop Multi-Stage Exit**:
   - Trail stop to breakeven when position reaches 1R profit, and scale out 50% at 2R target while letting the remaining 50% ride the 20 EMA.
4. **WebSocket Live Feed**:
   - Integrate DhanHQ WebSocket market feed (`MarketFeed`) for real-time tick-by-tick order fill tracking and GTT execution confirmation.
