"""
=============================================================================
         🏆 5-YEAR COMPREHENSIVE GOLD QUANT SCALPER BACKTEST REPORT
=============================================================================
Dataset: 175,308 15-Minute Candles (September 2021 to September 2026)
Asset: PAXG/USDT (Gold on Binance / MEXC)
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

sys.stdout.reconfigure(encoding='utf-8')
DATA_DIR = Path(__file__).resolve().parent / "data" / "historical"
CSV_FILE = DATA_DIR / "PAXG_USDT_15m_5years.csv"

def run_5year_gold_simulation(risk_per_trade_pct: float = 1.8,
                              rr_ratio: float = 2.2,
                              sl_atr_mult: float = 1.4,
                              rsi_long: float = 38.0,
                              rsi_short: float = 62.0,
                              min_adx: float = 20.0,
                              min_macro_slope: float = 0.35,
                              session_start: int = 7,
                              session_end: int = 20,
                              initial_balance: float = 1000.0,
                              maker_fee: float = 0.0000):

    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Missing {CSV_FILE}. Run fetch_5year_gold_data.py first!")

    df = pd.read_csv(CSV_FILE)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['year'] = df['datetime'].dt.strftime('%Y')
    df['month'] = df['datetime'].dt.strftime('%Y-%m')
    df['hour_utc'] = df['datetime'].dt.hour

    # Indicators
    df['ema_macro_fast'] = df['close'].ewm(span=480, adjust=False).mean()  # 5-Day
    df['ema_macro_slow'] = df['close'].ewm(span=1920, adjust=False).mean() # 20-Day
    df['macro_slope'] = (df['ema_macro_slow'] - df['ema_macro_slow'].shift(192)) / df['ema_macro_slow'].shift(192) * 100.0
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

    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    tr_smooth = tr.rolling(14).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).sum() / (tr_smooth + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).sum() / (tr_smooth + 1e-9))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    df['adx'] = dx.rolling(14).mean()

    df['swing_low'] = df['low'].shift(1).rolling(16).min()
    df['swing_high'] = df['high'].shift(1).rolling(16).max()

    n = len(df)
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    ema_fasts = df['ema_macro_fast'].values
    ema_slows = df['ema_macro_slow'].values
    macro_slopes = df['macro_slope'].values
    ema9s = df['ema9'].values
    rsis = df['rsi'].values
    atrs = df['atr'].values
    adxs = df['adx'].values
    swing_lows = df['swing_low'].values
    swing_highs = df['swing_high'].values
    hours = df['hour_utc'].values
    years = df['year'].values

    balance = initial_balance
    peak_balance = balance
    max_drawdown = 0.0
    trades = []
    yearly_stats = {}
    pos = None

    for i in range(1920, n):
        price = closes[i]
        high = highs[i]
        low = lows[i]
        op = opens[i]
        atr = atrs[i]
        hr = hours[i]
        curr_y = years[i]

        if curr_y not in yearly_stats:
            yearly_stats[curr_y] = {"start_bal": balance, "pnl": 0.0, "trades": 0, "wins": 0}
        y_stat = yearly_stats[curr_y]

        # Manage open position
        if pos is not None:
            side, entry, sl, tp, cnt = pos
            closed = False
            pnl = 0.0
            if side == 1: # Long
                if low <= sl:
                    closed = True
                    pnl = (sl - entry) * cnt - (cnt * sl * maker_fee)
                elif high >= tp:
                    closed = True
                    pnl = (tp - entry) * cnt - (cnt * tp * maker_fee)
            else: # Short
                if high >= sl:
                    closed = True
                    pnl = (entry - sl) * cnt - (cnt * sl * maker_fee)
                elif low <= tp:
                    closed = True
                    pnl = (entry - tp) * cnt - (cnt * tp * maker_fee)

            if closed:
                balance += pnl
                y_stat['pnl'] += pnl
                y_stat['trades'] += 1
                if pnl > 0:
                    y_stat['wins'] += 1
                trades.append(pnl)
                pos = None

        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / max(peak_balance, 1.0) * 100.0
        if dd > max_drawdown:
            max_drawdown = dd

        # Check new signal
        if pos is None and atr > 0 and adxs[i] >= min_adx:
            if not (session_start <= hr <= session_end):
                continue

            macro_bull = (ema_fasts[i] > ema_slows[i]) and (price > ema_fasts[i]) and (macro_slopes[i] >= min_macro_slope)
            macro_bear = (ema_fasts[i] < ema_slows[i]) and (price < ema_fasts[i]) and (macro_slopes[i] <= -min_macro_slope)
            prev_rsi = rsis[i-1]

            if macro_bull and prev_rsi < rsi_long and price > op and price > ema9s[i]:
                sl = min(swing_lows[i], price - (sl_atr_mult * atr))
                risk = price - sl
                risk_pct = risk / price
                if 0.001 <= risk_pct <= 0.025:
                    tp = price + (rr_ratio * risk)
                    dollar_risk = balance * (risk_per_trade_pct / 100.0)
                    cnt = dollar_risk / risk
                    pos = (1, price, sl, tp, cnt)

            elif macro_bear and prev_rsi > rsi_short and price < op and price < ema9s[i]:
                sl = max(swing_highs[i], price + (sl_atr_mult * atr))
                risk = sl - price
                risk_pct = risk / price
                if 0.001 <= risk_pct <= 0.025:
                    tp = price - (rr_ratio * risk)
                    dollar_risk = balance * (risk_per_trade_pct / 100.0)
                    cnt = dollar_risk / risk
                    pos = (-1, price, sl, tp, cnt)

    wins = [p for p in trades if p > 0]
    win_rate = (len(wins) / max(len(trades), 1)) * 100.0
    net_return = (balance - initial_balance) / initial_balance * 100.0
    years_count = len(df) / (365.25 * 24 * 4)
    cagr = ((balance / initial_balance) ** (1.0 / max(years_count, 1.0)) - 1.0) * 100.0

    yearly_breakdown = []
    for y, s in sorted(yearly_stats.items()):
        if s['trades'] == 0:
            continue
        wr = (s['wins'] / s['trades']) * 100.0
        ret = (s['pnl'] / max(s['start_bal'], 1.0)) * 100.0
        yearly_breakdown.append({
            "year": y,
            "trades": s['trades'],
            "wins": s['wins'],
            "win_rate": round(wr, 1),
            "pnl": round(s['pnl'], 2),
            "return_pct": round(ret, 2)
        })

    return {
        "initial_balance": initial_balance,
        "final_balance": round(balance, 2),
        "net_pnl": round(balance - initial_balance, 2),
        "net_return_pct": round(net_return, 2),
        "cagr_pct": round(cagr, 2),
        "total_trades": len(trades),
        "win_rate": round(win_rate, 1),
        "max_drawdown_pct": round(max_drawdown, 1),
        "yearly_breakdown": yearly_breakdown
    }

if __name__ == "__main__":
    print("=" * 80)
    print("      🏆 5-YEAR COMPREHENSIVE GOLD QUANT SCALPER BACKTEST")
    print("      Timeframe: 15-Minute | 175,308 Candles (2021 - 2026)")
    print("=" * 80)

    # Mode 1: 1.5% Risk (Balanced Safe Mode)
    res_15 = run_5year_gold_simulation(risk_per_trade_pct=1.5, min_macro_slope=0.35)

    # Mode 2: 1.8% Risk (High Yield Mode)
    res_18 = run_5year_gold_simulation(risk_per_trade_pct=1.8, min_macro_slope=0.35)

    # Mode 3: 2.0% Risk (Max Performance Mode)
    res_20 = run_5year_gold_simulation(risk_per_trade_pct=2.0, min_macro_slope=0.35)

    for name, res in [("MODE 1: BALANCED SAFE RUNNER (1.5% Risk per trade)", res_15),
                      ("MODE 2: HIGH-YIELD RUNNER (1.8% Risk per trade)", res_18),
                      ("MODE 3: MAXIMUM PERFORMANCE (2.0% Risk per trade)", res_20)]:
        print("\n" + "=" * 70)
        print(f"  {name}")
        print("=" * 70)
        print(f"  Starting Balance:      ${res['initial_balance']:,.2f}")
        print(f"  Ending Balance:        ${res['final_balance']:,.2f} ({res['net_return_pct']:+.2f}% Net Profit)")
        print(f"  CAGR (Annualized):     {res['cagr_pct']:+.2f}% Per Year Compound Growth")
        print(f"  Total Trades (5 Yrs):  {res['total_trades']} Trades | Win Rate: {res['win_rate']}%")
        print(f"  Max Drawdown (5 Yrs):  {res['max_drawdown_pct']}% (Across 5 Full Years of Market Cycles)")
        print(f"\n  Year-by-Year Performance Breakdown:")
        for y in res['yearly_breakdown']:
            icon = "✅" if y['pnl'] >= 0 else "🛡️"
            status = "Profitable" if y['pnl'] >= 0 else "Capital Preserved"
            print(f"    {icon} Year {y['year']} | Trades: {y['trades']:3d} | WinRate: {y['win_rate']:4.1f}% | PnL: ${y['pnl']:+9.2f} ({y['return_pct']:+5.1f}%) -> {status}")
