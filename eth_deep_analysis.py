import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/historical")

def test_realistic_compounding():
    csv_file = DATA_DIR / "ETH_USDT_USDT_15m_1year.csv"
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['hour_utc'] = df['datetime'].dt.hour
    df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
    df['month'] = df['datetime'].dt.strftime('%Y-%m')

    # Macro Trend (5-Day vs 20-Day)
    df['ema_macro_fast'] = df['close'].ewm(span=480, adjust=False).mean()  # 5-Day
    df['ema_macro_slow'] = df['close'].ewm(span=1920, adjust=False).mean() # 20-Day
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # ATR
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # ADX
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

    print("=" * 80)
    print("      ETH/USDT REALISTIC AUTO-COMPOUNDING & UPGRADE PERFORMANCE")
    print("      (Risk per Trade: 1.5% of Rolling Capital | 1-Year Historical)")
    print("=" * 80)

    # Scenarios to compare:
    scenarios = [
        {"name": "1. Non-Compounded (Fixed $1,000 base) + Taker Fee (0.02%)", "compound": False, "fee": 0.0002, "session": False},
        {"name": "2. Auto-Compounded (1.5% Risk) + Taker Fee (0.02%)", "compound": True, "fee": 0.0002, "session": False},
        {"name": "3. Auto-Compounded (1.5% Risk) + 0% Maker Fee", "compound": True, "fee": 0.0000, "session": False},
        {"name": "4. Auto-Compounded + 0% Maker + London/NY Session (08-22 UTC)", "compound": True, "fee": 0.0000, "session": True},
    ]

    for sc in scenarios:
        balance = 1000.0
        peak = balance
        max_dd = 0.0
        trades = []
        monthly_stats = {}
        pos = None
        leverage = 3
        fee_rate = sc['fee']

        for i in range(1920, len(df)):
            r = df.iloc[i]
            p = df.iloc[i-1]
            price = r['close']
            high = r['high']
            low = r['low']
            atr = r['atr']
            curr_month = r['month']

            if curr_month not in monthly_stats:
                monthly_stats[curr_month] = {"start_bal": balance, "pnl": 0.0, "trades": 0, "wins": 0}
            m_stat = monthly_stats[curr_month]

            # Manage Position
            if pos:
                closed = False
                pnl = 0.0
                side = pos['side']
                entry = pos['entry']
                cnt = pos['cnt']

                if side == 'long':
                    if low <= pos['sl']:
                        closed = True
                        pnl = (pos['sl'] - entry) * cnt - (cnt * pos['sl'] * fee_rate)
                    elif high >= pos['tp']:
                        closed = True
                        pnl = (pos['tp'] - entry) * cnt - (cnt * pos['tp'] * fee_rate)
                elif side == 'short':
                    if high >= pos['sl']:
                        closed = True
                        pnl = (entry - pos['sl']) * cnt - (cnt * pos['sl'] * fee_rate)
                    elif low <= pos['tp']:
                        closed = True
                        pnl = (entry - pos['tp']) * cnt - (cnt * pos['tp'] * fee_rate)

                if closed:
                    balance += pos['margin'] + pnl
                    m_stat['pnl'] += pnl
                    m_stat['trades'] += 1
                    if pnl > 0: m_stat['wins'] += 1
                    trades.append(pnl)
                    pos = None

            if balance > peak: peak = balance
            dd = (peak - balance) / max(peak, 1.0) * 100
            if dd > max_dd: max_dd = dd

            # Signal Check
            if pos is None and not np.isnan(atr) and atr > 0 and r['adx'] >= 22.0:
                if sc['session'] and not (8 <= r['hour_utc'] <= 22):
                    continue

                macro_bull = r['ema_macro_fast'] > r['ema_macro_slow'] and price > r['ema_macro_fast']
                macro_bear = r['ema_macro_fast'] < r['ema_macro_slow'] and price < r['ema_macro_fast']
                vol_ok = r['volume'] >= (r['vol_ma'] * 0.75)

                if macro_bull and p['rsi'] < 38 and r['close'] > r['open'] and r['close'] > r['ema9'] and vol_ok:
                    sl = min(r['swing_low'], price - (1.4 * atr))
                    risk = price - sl
                    risk_pct = risk / price
                    if 0.003 <= risk_pct <= 0.035:
                        tp = price + (2.0 * risk)
                        # Risk Sizing: Risk 1.5% of balance
                        base_cap = balance if sc['compound'] else 1000.0
                        risk_dollars = base_cap * 0.015
                        notional = min(risk_dollars / risk_pct, base_cap * leverage * 0.40)
                        margin = notional / leverage
                        cnt = notional / price
                        balance -= (margin + notional * fee_rate)
                        pos = {'side': 'long', 'entry': price, 'sl': sl, 'tp': tp, 'cnt': cnt, 'margin': margin}

                elif macro_bear and p['rsi'] > 62 and r['close'] < r['open'] and r['close'] < r['ema9'] and vol_ok:
                    sl = max(r['swing_high'], price + (1.4 * atr))
                    risk = sl - price
                    risk_pct = risk / price
                    if 0.003 <= risk_pct <= 0.035:
                        tp = price - (2.0 * risk)
                        base_cap = balance if sc['compound'] else 1000.0
                        risk_dollars = base_cap * 0.015
                        notional = min(risk_dollars / risk_pct, base_cap * leverage * 0.40)
                        margin = notional / leverage
                        cnt = notional / price
                        balance -= (margin + notional * fee_rate)
                        pos = {'side': 'short', 'entry': price, 'sl': sl, 'tp': tp, 'cnt': cnt, 'margin': margin}

        wins = [x for x in trades if x > 0]
        wr = len(wins) / max(len(trades), 1) * 100
        net_ret = (balance - 1000.0) / 1000.0 * 100
        green_m = sum(1 for m, s in monthly_stats.items() if s['pnl'] > 0)

        print(f"\n[{sc['name']}]")
        print(f"  Final Capital:    ${balance:,.2f} ({net_ret:+.2f}% Net Gain)")
        print(f"  Total Trades:     {len(trades)} Trades | Win Rate: {wr:.1f}%")
        print(f"  Max Drawdown:     {max_dd:.1f}%")
        print(f"  Green Months:     {green_m} / {len(monthly_stats)} Months Profitable")
        if sc['name'] == scenarios[-1]['name']:
            print("  Month-by-Month Breakdown (Scenario 4):")
            for m, s in sorted(monthly_stats.items()):
                if s['trades'] == 0: continue
                m_wr = s['wins'] / max(s['trades'], 1) * 100
                status_icon = "+" if s['pnl'] >= 0 else "-"
                print(f"    {m} | Trades: {s['trades']:<2} | WinRate: {m_wr:4.1f}% | PnL: ${s['pnl']:+6.2f}")

if __name__ == "__main__":
    test_realistic_compounding()
