"""
=============================================================================
      🏆 GOLD QUANT SCALPER — 24/7 LIVE PAPER TESTING ENGINE (1-WEEK RUNNER)
=============================================================================
Specially tailored for 1-Week live paper testing on AWS EC2.
- Asset: PAXG/USDT (Gold on Binance/MEXC)
- Strategy: 15m Institutional Macro Trend + Deep Value Pullback
- Risk: Strictly 1.5% per trade (Max DD Capped at 11.5%)
- Reward: 1:2.2 RR (1.4x ATR SL)
- Session: London & New York (07:00 - 20:00 UTC)
- Saves full trade history to: data/gold_paper_trades.json
- Telegram notifications built-in with real-time phone alerts
=============================================================================
"""

import os
import sys
import time
import json
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import requests

try:
    import ccxt
except ImportError:
    print("ccxt not installed. Please run: pip install ccxt")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "gold_paper_trades.json"
CONFIG_FILE = BASE_DIR / "config_gold.json"

def get_utc_now():
    return datetime.datetime.now(datetime.timezone.utc)

def get_utc_str():
    return get_utc_now().strftime("%Y-%m-%d %H:%M:%S UTC")

class GoldLiveTester:
    def __init__(self, initial_balance: float = 1000.0):
        self.symbol = "PAXG/USDT"
        self.timeframe = "15m"
        self.risk_pct = 1.5           # 1.5% risk per trade
        self.rr_ratio = 2.2           # 1:2.2 Risk to Reward
        self.sl_atr_mult = 1.4        # 1.4x ATR Structural SL
        self.max_dd_limit = 11.5      # Hard DD cut-off
        self.session_start_utc = 7    # 07:00 UTC London Open
        self.session_end_utc = 20     # 20:00 UTC NY Close
        
        # Load or initialize paper state
        self.load_state(initial_balance)
        
        # Public CCXT Binance (no API keys required for public live ticker & candles)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # Telegram config
        self.tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    cfg = json.load(f)
                    self.tg_token = cfg.get("telegram_bot_token", self.tg_token)
                    self.tg_chat_id = str(cfg.get("telegram_chat_id", self.tg_chat_id))
            except Exception:
                pass

    def send_telegram(self, message: str) -> bool:
        if not self.tg_token or not self.tg_chat_id:
            return False
        try:
            url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
            payload = {
                "chat_id": self.tg_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            res = requests.post(url, json=payload, timeout=8)
            res_json = res.json()
            if res.status_code == 200 and res_json.get("ok"):
                return True
            else:
                desc = res_json.get("description", "Unknown error")
                print(f"[!] Telegram notice: {desc}")
                return False
        except Exception as e:
            print(f"[!] Telegram connection notice: {e}")
            return False

    def load_state(self, initial_balance: float):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                    self.balance = float(data.get("balance", initial_balance))
                    self.initial_balance = float(data.get("initial_balance", initial_balance))
                    self.peak_balance = float(data.get("peak_balance", self.balance))
                    self.position = data.get("position", None)
                    self.trades = data.get("trades", [])
                    print(f"[+] Loaded existing testing state. Balance: ${self.balance:.2f} | Completed Trades: {len(self.trades)}")
                    return
            except Exception as e:
                print(f"[!] Warning reading state file: {e}")

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.position = None
        self.trades = []
        self.save_state()

    def save_state(self):
        data = {
            "initial_balance": self.initial_balance,
            "balance": round(self.balance, 2),
            "peak_balance": round(self.peak_balance, 2),
            "net_pnl": round(self.balance - self.initial_balance, 2),
            "net_return_pct": round((self.balance - self.initial_balance) / self.initial_balance * 100, 2),
            "total_trades": len(self.trades),
            "position": self.position,
            "trades": self.trades,
            "last_update": get_utc_str()
        }
        with open(STATE_FILE, "w") as f:
            json.dump(data, f, indent=4)

    def fetch_market_data(self) -> pd.DataFrame:
        """Fetch 500 latest 15m candles from Binance for indicators"""
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Indicators
        df['ema_macro_fast'] = df['close'].ewm(span=480, adjust=False).mean()
        df['ema_macro_slow'] = df['close'].ewm(span=1920, adjust=False).mean()
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()

        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        df['swing_low'] = df['low'].shift(1).rolling(16).min()
        df['swing_high'] = df['high'].shift(1).rolling(16).max()
        return df

    def check_position_exits(self, current_price: float):
        if not self.position:
            return

        side = self.position['side']
        entry_price = self.position['entry_price']
        sl = self.position['sl']
        tp = self.position['tp']
        size = self.position['size']
        entry_time = self.position['entry_time']

        hit_tp = False
        hit_sl = False

        if side == "LONG":
            if current_price >= tp:
                hit_tp = True
            elif current_price <= sl:
                hit_sl = True
        elif side == "SHORT":
            if current_price <= tp:
                hit_tp = True
            elif current_price >= sl:
                hit_sl = True

        if hit_tp or hit_sl:
            exit_price = tp if hit_tp else sl
            pnl = (exit_price - entry_price) * size if side == "LONG" else (entry_price - exit_price) * size
            pnl_pct = (pnl / self.balance) * 100.0

            self.balance += pnl
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance

            reason = "TARGET_PROFIT_HIT 🎯" if hit_tp else "STOP_LOSS_HIT 🛡️"
            trade_record = {
                "side": side,
                "entry_time": entry_time,
                "exit_time": get_utc_str(),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "reason": reason,
                "final_balance": round(self.balance, 2)
            }
            self.trades.append(trade_record)
            self.position = None
            self.save_state()

            # Alerts
            emoji = "🟢" if pnl >= 0 else "🔴"
            msg = (
                f"{emoji} <b>GOLD BOT: {reason}</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"<b>Side:</b> {side} Closed @ ${exit_price:,.2f}\n"
                f"<b>Trade PnL:</b> ${pnl:+,.2f} ({pnl_pct:+.2f}%)\n"
                f"<b>New Balance:</b> ${self.balance:,.2f}\n"
                f"<b>Total Closed Trades:</b> {len(self.trades)}"
            )
            print(f"\n[!] {msg}\n")
            self.send_telegram(msg)

    def scan_for_entry(self, df: pd.DataFrame, current_price: float):
        if self.position is not None:
            return  # Already in trade

        # Check hard drawdown cap
        dd = (self.peak_balance - self.balance) / self.peak_balance * 100.0
        if dd >= self.max_dd_limit:
            print(f"[!] 🛡️ Max Drawdown Cap reached ({dd:.2f}% >= {self.max_dd_limit}%). Bot paused for capital preservation.")
            return

        now = get_utc_now()
        current_hour = now.hour

        # Session Filter (London & NY: 07:00 - 20:00 UTC)
        in_session = (self.session_start_utc <= current_hour < self.session_end_utc)
        if not in_session:
            return

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        price = current_price
        op = last_row['open']
        ema_fast = last_row['ema_macro_fast']
        ema_slow = last_row['ema_macro_slow']
        ema9 = last_row['ema9']
        prev_rsi = prev_row['rsi']
        atr = last_row['atr']
        swing_low = last_row['swing_low']
        swing_high = last_row['swing_high']

        macro_bull = (ema_fast > ema_slow) and (price > ema_fast)
        macro_bear = (ema_fast < ema_slow) and (price < ema_fast)

        # 1. Long Setup: Macro Bull + RSI Dip (<38) + Candle turns green + above 9 EMA
        if macro_bull and prev_rsi < 38.0 and price > op and price > ema9:
            sl = min(swing_low, price - (self.sl_atr_mult * atr))
            risk_dist = price - sl
            risk_pct = risk_dist / price
            if 0.001 <= risk_pct <= 0.025:
                tp = price + (self.rr_ratio * risk_dist)
                dollar_risk = self.balance * (self.risk_pct / 100.0)
                qty = dollar_risk / risk_dist

                self.position = {
                    "side": "LONG",
                    "entry_time": get_utc_str(),
                    "entry_price": round(price, 2),
                    "sl": round(sl, 2),
                    "tp": round(tp, 2),
                    "size": round(qty, 4),
                    "risk_usd": round(dollar_risk, 2)
                }
                self.save_state()
                msg = (
                    f"🚀 <b>GOLD BOT: NEW LONG POSITION</b>\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"<b>Asset:</b> {self.symbol} (Gold)\n"
                    f"<b>Entry Price:</b> ${price:,.2f}\n"
                    f"<b>Stop Loss:</b> ${sl:,.2f}\n"
                    f"<b>Take Profit:</b> ${tp:,.2f} (1:2.2 RR 🎯)\n"
                    f"<b>Risk Amount:</b> ${dollar_risk:,.2f} (1.5% Equity)\n"
                    f"<b>Position Qty:</b> {qty:.4f} oz"
                )
                print(f"\n[+] {msg}\n")
                self.send_telegram(msg)

        # 2. Short Setup: Macro Bear + RSI Relief (>62) + Candle turns red + below 9 EMA
        elif macro_bear and prev_rsi > 62.0 and price < op and price < ema9:
            sl = max(swing_high, price + (self.sl_atr_mult * atr))
            risk_dist = sl - price
            risk_pct = risk_dist / price
            if 0.001 <= risk_pct <= 0.025:
                tp = price - (self.rr_ratio * risk_dist)
                dollar_risk = self.balance * (self.risk_pct / 100.0)
                qty = dollar_risk / risk_dist

                self.position = {
                    "side": "SHORT",
                    "entry_time": get_utc_str(),
                    "entry_price": round(price, 2),
                    "sl": round(sl, 2),
                    "tp": round(tp, 2),
                    "size": round(qty, 4),
                    "risk_usd": round(dollar_risk, 2)
                }
                self.save_state()
                msg = (
                    f"🔻 <b>GOLD BOT: NEW SHORT POSITION</b>\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"<b>Asset:</b> {self.symbol} (Gold)\n"
                    f"<b>Entry Price:</b> ${price:,.2f}\n"
                    f"<b>Stop Loss:</b> ${sl:,.2f}\n"
                    f"<b>Take Profit:</b> ${tp:,.2f} (1:2.2 RR 🎯)\n"
                    f"<b>Risk Amount:</b> ${dollar_risk:,.2f} (1.5% Equity)\n"
                    f"<b>Position Qty:</b> {qty:.4f} oz"
                )
                print(f"\n[+] {msg}\n")
                self.send_telegram(msg)

    def run(self):
        print("=" * 75)
        print("      🏆 GOLD QUANT SCALPER — 24/7 PAPER TESTING ENGINE ACTIVE")
        print("      Tracking Real-Time Gold (PAXG/USDT) via Binance Live Stream")
        print("=" * 75)
        print(f"  Initial Equity:   ${self.initial_balance:,.2f}")
        print(f"  Current Equity:   ${self.balance:,.2f}")
        print(f"  Risk / Trade:     {self.risk_pct}% (Max DD Hard Cap: {self.max_dd_limit}%)")
        print(f"  Risk-to-Reward:   1:{self.rr_ratio} Asymmetric Profit")
        print(f"  Active Session:   07:00 - 20:00 UTC (London & NY)")
        print(f"  Telegram Alerts:  {'CONNECTED ✅' if self.tg_token else 'DISABLED ❌'}")
        print(f"  State File:       {STATE_FILE}")
        print("=" * 75 + "\n")

        # Send Telegram startup notification
        tg_success = self.send_telegram(
            f"🏆 <b>GOLD QUANT BOT ACTIVATED!</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"<b>Mode:</b> 1-Week Live Forward Test\n"
            f"<b>Starting Balance:</b> ${self.balance:,.2f}\n"
            f"<b>Risk Per Trade:</b> 1.5% (Max DD Cap: 11.5%)\n"
            f"<b>Risk / Reward:</b> 1:2.2 RR Asymmetric\n"
            f"<b>Server:</b> AWS EC2 (eu-north-1) 24/7 Live\n"
            f"<i>You will receive real-time alerts for every entry & exit!</i>"
        )
        if tg_success:
            print("[+] Telegram notification sent successfully to your phone!")
        else:
            if self.tg_token:
                print("[!] Note: Make sure you have opened @MyGoldTradingBotf_bot on Telegram and pressed START.")

        last_status_print = 0
        last_heartbeat_time = time.time()

        while True:
            try:
                # 1. Fetch current price
                ticker = self.exchange.fetch_ticker(self.symbol)
                current_price = float(ticker['last'])

                # 2. Check open position exits
                self.check_position_exits(current_price)

                # 3. Fetch indicators and scan entry
                df = self.fetch_market_data()
                self.scan_for_entry(df, current_price)

                # Periodic terminal status (every 60 seconds)
                now_ts = time.time()
                if now_ts - last_status_print >= 60:
                    last_status_print = now_ts
                    now_str = get_utc_now().strftime("%H:%M:%S UTC")
                    last_rsi = df['rsi'].iloc[-1]
                    ema_f = df['ema_macro_fast'].iloc[-1]
                    ema_s = df['ema_macro_slow'].iloc[-1]
                    regime = "BULLISH 📈" if (ema_f > ema_s and current_price > ema_f) else ("BEARISH 📉" if (ema_f < ema_s and current_price < ema_f) else "NEUTRAL / CHOP ⚖️")
                    
                    pos_info = "NO OPEN TRADE (Scanning setups...)"
                    if self.position:
                        p = self.position
                        side = p['side']
                        unrealized = (current_price - p['entry_price']) * p['size'] if side == "LONG" else (p['entry_price'] - current_price) * p['size']
                        pos_info = f"ACTIVE {side} @ ${p['entry_price']:.2f} | uPnL: ${unrealized:+.2f} | SL: ${p['sl']:.2f} | TP: ${p['tp']:.2f}"

                    net_pnl = self.balance - self.initial_balance
                    ret_pct = (net_pnl / self.initial_balance) * 100.0
                    dd = (self.peak_balance - self.balance) / self.peak_balance * 100.0

                    print(f"[{now_str}] PAXG: ${current_price:,.2f} | Regime: {regime} | RSI: {last_rsi:.1f}")
                    print(f"          Equity: ${self.balance:,.2f} (PnL: ${net_pnl:+.2f} / {ret_pct:+.2f}%) | DD: {dd:.1f}% | Trades: {len(self.trades)}")
                    print(f"          Status: {pos_info}\n")

                # Heartbeat to Telegram every 6 hours
                if now_ts - last_heartbeat_time >= 21600:
                    last_heartbeat_time = now_ts
                    net_pnl = self.balance - self.initial_balance
                    ret_pct = (net_pnl / self.initial_balance) * 100.0
                    self.send_telegram(
                        f"📊 <b>GOLD BOT 6-HOUR STATUS REPORT</b>\n"
                        f"━━━━━━━━━━━━━━━━━━\n"
                        f"<b>Balance:</b> ${self.balance:,.2f}\n"
                        f"<b>Net PnL:</b> ${net_pnl:+,.2f} ({ret_pct:+.2f}%)\n"
                        f"<b>Completed Trades:</b> {len(self.trades)}\n"
                        f"<b>Bot Status:</b> 24/7 Healthy & Scanning"
                    )

                time.sleep(15)  # Poll every 15 seconds

            except Exception as e:
                print(f"[!] Engine cycle notice: {e}")
                time.sleep(10)

if __name__ == "__main__":
    bot = GoldLiveTester(initial_balance=1000.0)
    bot.run()
