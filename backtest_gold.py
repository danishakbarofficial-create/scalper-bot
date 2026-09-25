import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
DATA_DIR = Path(__file__).resolve().parent / "data" / "historical"

def load_data():
    csv_path = DATA_DIR / "PAXG_USDT_USDT_15m_1year.csv"
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.strftime('%Y-%m')

    # 1. Macro Trend (5-Day vs 20-Day EMA)
    df['ema_macro_fast'] = df['close'].ewm(span=480, adjust=False).mean()
    df['ema_macro_slow'] = df['close'].ewm(span=1920, adjust=False).mean()
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()

    # 2. RSI (14)
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # 3. ATR (14)
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # 4. ADX (14)
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    tr_smooth = tr.rolling(14).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).sum() / (tr_smooth + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).sum() / (tr_smooth + 1e-9))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    df['adx'] = dx.rolling(14).mean()

    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['swing_low'] = df['low'].shift(1).rolling(16).min()
    df['swing_high'] = df['high'].shift(1).rolling(16).max()
    return df

def run_simulation(df, session_filter=False, min_risk=0.001, max_risk=0.035, sl_atr=1.4, rr=2.0, rsi_l=38, rsi_s=62):
    n = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values
    vol_mas = df['vol_ma'].values
    ema_fasts = df['ema_macro_fast'].values
    ema_slows = df['ema_macro_slow'].values
    ema9s = df['ema9'].values
    rsis = df['rsi'].values
    atrs = df['atr'].values
    adxs = df['adx'].values
    swing_lows = df['swing_low'].values
    swing_highs = df['swing_high'].values
    hours = df['hour'].values
    months = df['month'].values

    initial_balance = 1000.0
    balance = initial_balance
    peak_balance = balance
    max_drawdown = 0.0
    trades = []
    monthly_stats = {}
    pos = None
    taker_fee = 0.0002
    leverage = 3
    position_size_pct = 0.40

    for i in range(1920, n):
        curr_month = months[i]
        price = closes[i]
        high = highs[i]
        low = lows[i]
        op = opens[i]
        atr = atrs[i]
        hour = hours[i]

        if curr_month not in monthly_stats:
            monthly_stats[curr_month] = {"start_bal": balance, "pnl": 0.0, "trades": 0, "wins": 0}
        m_stat = monthly_stats[curr_month]

        # Manage existing position
        if pos is not None:
            side, entry, pos_sl, pos_tp, cnt, margin = pos
            closed = False
            pnl = 0.0

            if side == 1: # Long
                if low <= pos_sl:
                    closed = True
                    pnl = (pos_sl - entry) * cnt - (cnt * pos_sl * taker_fee)
                elif high >= pos_tp:
                    closed = True
                    pnl = (pos_tp - entry) * cnt - (cnt * pos_tp * taker_fee)
            else: # Short
                if high >= pos_sl:
                    closed = True
                    pnl = (entry - pos_sl) * cnt - (cnt * pos_sl * taker_fee)
                elif low <= pos_tp:
                    closed = True
                    pnl = (entry - pos_tp) * cnt - (cnt * pos_tp * taker_fee)

            if closed:
                balance += margin + pnl
                m_stat['pnl'] += pnl
                m_stat['trades'] += 1
                if pnl > 0:
                    m_stat['wins'] += 1
                trades.append(pnl)
                pos = None

        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / max(peak_balance, 1.0) * 100.0
        if dd > max_drawdown:
            max_drawdown = dd

        # Signal Check
        if pos is None and atr > 0 and adxs[i] >= 22.0:
            if session_filter and not (7 <= hour <= 20):
                continue

            macro_bull = (ema_fasts[i] > ema_slows[i]) and (price > ema_fasts[i])
            macro_bear = (ema_fasts[i] < ema_slows[i]) and (price < ema_fasts[i])
            vol_ok = volumes[i] >= (vol_mas[i] * 0.75)

            prev_rsi = rsis[i-1]

            if macro_bull and prev_rsi < rsi_l and price > op and price > ema9s[i] and vol_ok:
                sl = min(swing_lows[i], price - (sl_atr * atr))
                risk = price - sl
                risk_pct = risk / price
                if min_risk <= risk_pct <= max_risk:
                    tp = price + (rr * risk)
                    notional = balance * position_size_pct
                    margin = notional / leverage
                    cnt = notional / price
                    balance -= (margin + notional * taker_fee)
                    pos = (1, price, sl, tp, cnt, margin)

            elif macro_bear and prev_rsi > rsi_s and price < op and price < ema9s[i] and vol_ok:
                sl = max(swing_highs[i], price + (sl_atr * atr))
                risk = sl - price
                risk_pct = risk / price
                if min_risk <= risk_pct <= max_risk:
                    tp = price - (rr * risk)
                    notional = balance * position_size_pct
                    margin = notional / leverage
                    cnt = notional / price
                    balance -= (margin + notional * taker_fee)
                    pos = (-1, price, sl, tp, cnt, margin)

    wins = [p for p in trades if p > 0]
    win_rate = (len(wins) / max(len(trades), 1)) * 100.0
    net_return = (balance - initial_balance) / initial_balance * 100.0

    return {
        "final_balance": round(balance, 2),
        "net_pnl": round(balance - initial_balance, 2),
        "net_return_pct": round(net_return, 2),
        "total_trades": len(trades),
        "win_rate": round(win_rate, 1),
        "max_drawdown_pct": round(max_drawdown, 1),
        "monthly_stats": monthly_stats
    }

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"Data loaded: {len(df)} candles.")

    # 1. Config A: Direct Copy of ETH config (min_risk=0.003, no session filter)
    res_a = run_simulation(df, session_filter=False, min_risk=0.003, max_risk=0.035, sl_atr=1.4, rr=2.0)
    print("\n--- 1. DIRECT ETH CONFIG ON GOLD (No Session Filter, Min Risk 0.3%) ---")
    print(f"Trades: {res_a['total_trades']} | Win Rate: {res_a['win_rate']}% | Return: {res_a['net_return_pct']:+.2f}% | Max DD: {res_a['max_drawdown_pct']}%")

    # 2. Config B: Gold Calibrated Risk (min_risk=0.001)
    res_b = run_simulation(df, session_filter=False, min_risk=0.001, max_risk=0.025, sl_atr=1.4, rr=2.0)
    print("\n--- 2. GOLD CALIBRATED RISK THRESHOLD (0.1% - 2.5%) ---")
    print(f"Trades: {res_b['total_trades']} | Win Rate: {res_b['win_rate']}% | Return: {res_b['net_return_pct']:+.2f}% | Max DD: {res_b['max_drawdown_pct']}%")

    # 3. Config C: Gold + Session Filter (London & NY: 07:00 - 20:00 UTC)
    res_c = run_simulation(df, session_filter=True, min_risk=0.001, max_risk=0.025, sl_atr=1.4, rr=2.0)
    print("\n--- 3. GOLD + LONDON/NY SESSION FILTER (07:00 - 20:00 UTC) ---")
    print(f"Trades: {res_c['total_trades']} | Win Rate: {res_c['win_rate']}% | Return: {res_c['net_return_pct']:+.2f}% | Max DD: {res_c['max_drawdown_pct']}%")

    # 4. Optimization grid
    print("\n--- 4. GRID OPTIMIZATION (R:R & SL MULTIPLIERS with Session Filter) ---")
    best_res = None
    best_ret = -999
    best_params = None

    for rr in [1.8, 2.0, 2.2, 2.5]:
        for sl in [1.2, 1.4, 1.6]:
            for rsi_thresh in [(36, 64), (38, 62), (40, 60)]:
                res = run_simulation(df, session_filter=True, min_risk=0.001, max_risk=0.025, sl_atr=sl, rr=rr, rsi_l=rsi_thresh[0], rsi_s=rsi_thresh[1])
                print(f"SL: {sl}x | RR: {rr}x | RSI: {rsi_thresh} -> Trades: {res['total_trades']:<2} | WR: {res['win_rate']:4.1f}% | Net Ret: {res['net_return_pct']:+6.2f}% | Max DD: {res['max_drawdown_pct']:4.1f}%")
                if res['net_return_pct'] > best_ret:
                    best_ret = res['net_return_pct']
                    best_res = res
                    best_params = (sl, rr, rsi_thresh)

    print("\n" + "="*60)
    print("5. VOLATILITY-ADJUSTED RISK MODEL (1.0% Capital Risked Per Trade):")
    print("="*60)
    
    # Run 1.0% risk model
    n = len(df)
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    ema_fasts = df['ema_macro_fast'].values
    ema_slows = df['ema_macro_slow'].values
    ema9s = df['ema9'].values
    rsis = df['rsi'].values
    atrs = df['atr'].values
    adxs = df['adx'].values
    swing_lows = df['swing_low'].values
    swing_highs = df['swing_high'].values
    hours = df['hour'].values
    months = df['month'].values

    balance = 1000.0
    peak = balance
    max_dd = 0.0
    monthly_stats = {}
    pos = None
    taker_fee = 0.0002
    trades = []

    for i in range(1920, n):
        price = closes[i]
        high = highs[i]
        low = lows[i]
        op = opens[i]
        atr = atrs[i]
        hour = hours[i]
        curr_m = months[i]

        if curr_m not in monthly_stats:
            monthly_stats[curr_m] = {'start': balance, 'pnl': 0.0, 'trades': 0, 'wins': 0}

        if pos is not None:
            side, entry, sl, tp, cnt = pos
            closed = False
            pnl = 0.0
            if side == 1:
                if low <= sl:
                    pnl = (sl - entry)*cnt - (cnt*sl*taker_fee)
                    closed = True
                elif high >= tp:
                    pnl = (tp - entry)*cnt - (cnt*tp*taker_fee)
                    closed = True
            else:
                if high >= sl:
                    pnl = (entry - sl)*cnt - (cnt*sl*taker_fee)
                    closed = True
                elif low <= tp:
                    pnl = (entry - tp)*cnt - (cnt*tp*taker_fee)
                    closed = True
            if closed:
                balance += pnl
                monthly_stats[curr_m]['pnl'] += pnl
                monthly_stats[curr_m]['trades'] += 1
                if pnl > 0:
                    monthly_stats[curr_m]['wins'] += 1
                trades.append(pnl)
                pos = None

        if balance > peak:
            peak = balance
        dd = (peak - balance) / max(peak, 1.0) * 100.0
        if dd > max_dd:
            max_dd = dd

        if pos is None and atr > 0 and adxs[i] >= 22.0 and (7 <= hour <= 20):
            bull = (ema_fasts[i] > ema_slows[i]) and (price > ema_fasts[i])
            bear = (ema_fasts[i] < ema_slows[i]) and (price < ema_fasts[i])
            prev_rsi = rsis[i-1]

            if bull and prev_rsi < 40 and price > op and price > ema9s[i]:
                sl = min(swing_lows[i], price - (1.4 * atr))
                risk = price - sl
                risk_pct = risk / price
                if 0.001 <= risk_pct <= 0.025:
                    tp = price + (2.2 * risk)
                    dollar_risk = balance * 0.01
                    cnt = dollar_risk / risk
                    balance -= (cnt * price * taker_fee)
                    pos = (1, price, sl, tp, cnt)

            elif bear and prev_rsi > 60 and price < op and price < ema9s[i]:
                sl = max(swing_highs[i], price + (1.4 * atr))
                risk = sl - price
                risk_pct = risk / price
                if 0.001 <= risk_pct <= 0.025:
                    tp = price - (2.2 * risk)
                    dollar_risk = balance * 0.01
                    cnt = dollar_risk / risk
                    balance -= (cnt * price * taker_fee)
                    pos = (-1, price, sl, tp, cnt)

    wins = [p for p in trades if p > 0]
    wr = len(wins) / max(len(trades), 1) * 100.0
    net_ret = (balance - 1000.0) / 1000.0 * 100.0
    print(f"Final Capital: ${balance:.2f} ({net_ret:+.2f}%) | Trades: {len(trades)} | Win Rate: {wr:.1f}% | Max Drawdown: {max_dd:.1f}%")
    print("Month-by-Month Breakdown:")
    for m, s in sorted(monthly_stats.items()):
        if s['trades'] > 0:
            m_wr = (s['wins'] / s['trades']) * 100.0
            m_ret = (s['pnl'] / s['start']) * 100.0
            icon = "✅" if s['pnl'] >= 0 else "❌"
            print(f"  {icon} {m} | Trades: {s['trades']:2d} | WinRate: {m_wr:4.1f}% | PnL: ${s['pnl']:+6.2f} ({m_ret:+5.1f}%)")

