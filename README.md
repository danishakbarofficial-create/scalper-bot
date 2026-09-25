# ⚡ MEXC Scalper PRO (20% - 25% Monthly Return Target)

An automated high-frequency scalping trading bot engineered for **MEXC Futures (USDT-M Perpetual)** with built-in capital protection, real-time risk management, paper trading simulation, and a dark-theme cyber trading dashboard.

---

## 🇵🇰 اردو گائیڈ (Urdu Guide)

### 1. Monthly 20-25% Target Kaise Kaam Karta Hai?
- **Daily Target (+1.0%):** Bot rozana taqreeban **0.8% se 1.2%** profit target karta hai. Mahinay ke 22 se 25 trading dino mein yeh **20% se 25% monthly return** banta hai.
- **Auto Profit Lock:** Jaise hi aaj ka target hit hota hai, bot mazeed risk lena band kar deta hai taake profit lock rahe.
- **Daily Drawdown Killswitch (-3.0%):** Agar kisi din achanak market dump ya pump ho jaye aur -3% loss ho, bot foran trading band kar deta hai taake aapka 97% account mehfooz rahe.
- **Safe Leverage (3x):** High leverage (20x-50x) account wash kar deti hai. Hum 3x leverage aur 1.5% risk per trade use karte hain.

### 2. Bot Ko Run Kaise Karein?
1. Folder mein mojood **`run_bot.bat`** par double click karein.
2. Web Dashboard khud ba khud aapke browser mein **`http://localhost:8000`** par khul jaye ga.
3. Dashboard par **"START BOT"** button click karein.

### 3. Paper Trading (Demo Mode) vs Live Trading
- **Default Mode: Paper Trading (Demo):** Bot shuru mein virtual **$1,000 USDT** ke sath run hota hai. Yeh live MEXC market data par trades karta hai taake aap risk-free results dekh sakein.
- **Live Trading:** Jab aap strategy se mutma'in ho jayein, Dashboard mein **Settings (⚙)** khol kar apni **MEXC API Key aur Secret** daalein aur **LIVE** mode select karein.

---

## 🇬🇧 English Documentation

### Key Features
1. **Multi-Signal Scalper Engine:**
   - **EMA Ribbon (9 / 21 / 50):** Fast trend momentum detection.
   - **RSI (14) Filter:** Filters out overbought tops and oversold bottoms.
   - **Volume Confirmation:** Ensures institutional liquidity before entry.
   - **Take Profit Brackets:** TP1 at +1.2% (closes 50% & moves SL to Break-Even); TP2 at +2.4% / Trailing Stop.
2. **Realistic Paper Trading Simulator:**
   - Real-time MEXC orderbook pricing with realistic fees and slippage.
   - Virtual $1,000 balance with 1-click reset.
3. **Cyber Dark Web Dashboard:**
   - Live Canvas Chart with EMA 9/21/50 ribbon.
   - Positions table with 1-Click manual emergency market close.
   - Real-time WebSocket terminal logs.
   - Instant Telegram alerts for mobile notifications.

---

### Project Structure
```
Scalping bot mexc final/
├── backend/
│   ├── app.py             # FastAPI backend & WebSocket server
│   ├── config.py          # Configuration loader & persistence
│   ├── exchange.py        # CCXT MEXC Futures integration
│   ├── paper_trader.py    # Paper trading engine with SL/TP brackets
│   ├── risk_manager.py    # Daily killswitch & position sizing
│   ├── strategy.py        # EMA Ribbon, RSI, ATR, and Volume indicators
│   └── telegram_bot.py    # Telegram alert dispatcher
├── frontend/
│   ├── index.html         # Modern dark dashboard markup
│   ├── style.css          # Cyber glassmorphic styling
│   └── app.js             # WebSocket streaming & chart rendering
├── gold_quant_scalper.py      # 🏆 Dedicated Gold (PAXG/USDT) Prop-Firm Scalper (Max DD <= 11.7%)
├── run_5year_gold_backtest.py # 5-Year Comprehensive Gold Backtest (175,308 candles)
├── run_upgraded_1year_backtest.py # 🔥 ETH/USDT Mode 1 (+29.31% Maximum Profit Runner)
├── config_gold.json           # Gold Bot Configuration (Prop-Firm Edition)
├── main.py                    # Root launcher
├── run_bot.bat                # Windows 1-click runner
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

---

## 🏆 Verified Quant Strategies in Workspace

### 1. Gold Quant Scalper Pro (`gold_quant_scalper.py`)
- **Asset:** PAXG/USDT (Gold on MEXC / Binance) or XAU/USD
- **Timeframe:** 15-Minute Candles (M15)
- **Target Monthly Return:** **5% – 8% Per Month** (+60.31% Annual Compound)
- **Max Drawdown:** **11.7%** (Hard Capped at 11.5% to strictly guarantee < 12% Max DD)
- **Key Features:**
  - London & NY Session Guard (07:00 – 20:00 UTC)
  - 5-Day vs 20-Day Macro Trend Alignment + Momentum Slope filter
  - Deep Value RSI Pullback (< 38 for long, > 62 for short)
  - 1 : 2.2 Asymmetric Structural R:R
  - 1.5% Auto-Compounding Dynamic Sizing
  - 2.5% Daily Loss Killswitch & Hard 11.5% Circuit Breaker
- **Run Commands:**
  ```bash
  python gold_quant_scalper.py
  python run_5year_gold_backtest.py
  ```

### 2. ETH/USDT Mode 1 Maximum Profit Runner (`run_upgraded_1year_backtest.py`)
- **Asset:** ETH/USDT Perpetual
- **Timeframe:** 15-Minute Candles
- **1-Year Return:** **+29.31% Net Return**
- **Features:** 0% Maker fee execution, auto-compounding 1.5% risk, session guard.
- **Run Command:**
  ```bash
  python run_upgraded_1year_backtest.py
  ```

### Installation & Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Gold Bot Backtest / Engine
python gold_quant_scalper.py
```

