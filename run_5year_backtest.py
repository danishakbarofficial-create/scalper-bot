import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = Path(__file__).resolve().parent / "data" / "historical"
CSV_FILE = DATA_DIR / "ETH_USDT_15m_5years.csv"

class FiveYearBacktester:
    """
    5-Year Institutional Backtester for ETH/USDT:
    - 2021 Bull Peak ($4,800) -> 2022 Bear Crash ($880) -> 2023 Chop -> 2024 Bull Run -> 2025/2026
    - Tests Mode 1 (Max Profit Runner) vs Mode 2 (Break-Even Free-Roll)
    """

    def __init__(self,
                 initial_balance: float = 1000.0,
                 leverage: int = 3,
                 base_risk_pct: float = 1.5,
                 fee_rate: float = 0.0000, # 0% Maker fee
                 daily_target_pct: float = 2.0,
                 daily_loss_pct: float = 3.0,
                 enable_freeroll_be: bool = False,
                 enable_shock_absorber: bool = False,
                 rr_ratio: float = 2.0,
                 sl_atr_mult: float = 1.4):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.base_risk_pct = base_risk_pct
        self.fee_rate = fee_rate
        self.daily_target_pct = daily_target_pct
        self.daily_loss_pct = daily_loss_pct
        self.enable_freeroll_be = enable_freeroll_be
        self.enable_shock_absorber = enable_shock_absorber
        self.rr_ratio = rr_ratio
        self.sl_atr_mult = sl_atr_mult

    def run(self) -> Dict[str, Any]:
        if not CSV_FILE.exists():
            raise FileNotFoundError(f"Missing {CSV_FILE}. Run fetch_5year_data.py first!")

        print(f"Loading 5-Year dataset from {CSV_FILE.name}...")
        df = pd.read_csv(CSV_FILE)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['year'] = df['datetime'].dt.strftime('%Y')
        df['month'] = df['datetime'].dt.strftime('%Y-%m')
        df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
        df['hour_utc'] = df['datetime'].dt.hour

        print(f"Total candles loaded: {len(df):,} ({df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]})")

        # 1. Macro Trend (5-Day vs 20-Day EMA on 15M candles)
        print("Calculating Macro EMAs, RSI, ATR, and ADX...")
        df['ema_macro_fast'] = df['close'].ewm(span=480, adjust=False).mean()  # 5-Day
        df['ema_macro_slow'] = df['close'].ewm(span=1920, adjust=False).mean() # 20-Day
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

        balance = self.initial_balance
        peak_balance = balance
        max_drawdown = 0.0

        consecutive_losses = 0
        shock_absorber_triggers = 0
        freeroll_activations = 0

        trades = []
        yearly_stats = {}
        daily_tracker = {}
        pos = None

        print(f"Executing 5-year simulation (Mode: {'Mode 2 (BE Lock)' if self.enable_freeroll_be else 'Mode 1 (Full Runner)'})...")

        for i in range(1920, len(df)):
            r = df.iloc[i]
            p = df.iloc[i-1]
            price = r['close']
            high = r['high']
            low = r['low']
            atr = r['atr']
            curr_date = r['date']
            curr_year = r['year']

            if curr_date not in daily_tracker:
                daily_tracker[curr_date] = {"pnl": 0.0, "locked": False, "killed": False, "start_bal": balance}
            day = daily_tracker[curr_date]

            if curr_year not in yearly_stats:
                yearly_stats[curr_year] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "start_bal": balance}

            # Daily circuit breakers
            day_pnl_pct = (day["pnl"] / max(day["start_bal"], 1.0)) * 100.0
            if day_pnl_pct >= self.daily_target_pct:
                day["locked"] = True
            elif day_pnl_pct <= -self.daily_loss_pct:
                day["killed"] = True

            # --- MANAGE OPEN POSITION ---
            if pos is not None:
                side = pos['side']
                entry = pos['entry_price']
                sl = pos['sl']
                tp = pos['tp']
                risk_usdt = pos['risk_usdt']

                exit_price = None
                exit_reason = None
                pnl = 0.0

                if side == 'long':
                    # Optional Free-Roll Break-Even Lock at +1.2R
                    if self.enable_freeroll_be and not pos['be_locked']:
                        dist_to_tp = tp - entry
                        if high >= entry + (0.60 * dist_to_tp):
                            pos['sl'] = entry + (entry * 0.0005) # covers fee
                            pos['be_locked'] = True
                            freeroll_activations += 1

                    # Check SL
                    if low <= pos['sl']:
                        exit_price = pos['sl']
                        exit_reason = 'STOP_LOSS' if not pos['be_locked'] else 'BREAK_EVEN'
                        contracts = pos['contracts']
                        raw_loss = (exit_price - entry) * contracts
                        fee = (exit_price * contracts) * self.fee_rate
                        pnl = raw_loss - fee
                    # Check TP
                    elif high >= tp:
                        exit_price = tp
                        exit_reason = 'TAKE_PROFIT'
                        contracts = pos['contracts']
                        raw_win = (tp - entry) * contracts
                        fee = (tp * contracts) * self.fee_rate
                        pnl = raw_win - fee

                elif side == 'short':
                    # Optional Free-Roll Break-Even Lock at +1.2R
                    if self.enable_freeroll_be and not pos['be_locked']:
                        dist_to_tp = entry - tp
                        if low <= entry - (0.60 * dist_to_tp):
                            pos['sl'] = entry - (entry * 0.0005)
                            pos['be_locked'] = True
                            freeroll_activations += 1

                    # Check SL
                    if high >= pos['sl']:
                        exit_price = pos['sl']
                        exit_reason = 'STOP_LOSS' if not pos['be_locked'] else 'BREAK_EVEN'
                        contracts = pos['contracts']
                        raw_loss = (entry - exit_price) * contracts
                        fee = (exit_price * contracts) * self.fee_rate
                        pnl = raw_loss - fee
                    # Check TP
                    elif low <= tp:
                        exit_price = tp
                        exit_reason = 'TAKE_PROFIT'
                        contracts = pos['contracts']
                        raw_win = (entry - tp) * contracts
                        fee = (tp * contracts) * self.fee_rate
                        pnl = raw_win - fee

                if exit_price is not None:
                    balance += pnl
                    day["pnl"] += pnl
                    yearly_stats[curr_year]["pnl"] += pnl
                    yearly_stats[curr_year]["trades"] += 1

                    if pnl > 0:
                        yearly_stats[curr_year]["wins"] += 1
                        consecutive_losses = 0
                    elif pnl < 0 and exit_reason != 'BREAK_EVEN':
                        yearly_stats[curr_year]["losses"] += 1
                        consecutive_losses += 1
                        if consecutive_losses >= 2 and self.enable_shock_absorber:
                            shock_absorber_triggers += 1

                    trades.append({
                        "entry_time": pos["entry_time"],
                        "exit_time": str(r["datetime"]),
                        "side": side,
                        "entry": round(entry, 2),
                        "exit": round(exit_price, 2),
                        "pnl": round(pnl, 2),
                        "reason": exit_reason,
                        "year": curr_year,
                        "balance_after": round(balance, 2)
                    })

                    if balance > peak_balance:
                        peak_balance = balance
                    dd = ((peak_balance - balance) / peak_balance) * 100.0
                    if dd > max_drawdown:
                        max_drawdown = dd

                    pos = None

            # --- ENTRY SIGNAL CHECK ---
            if pos is None:
                if day["locked"] or day["killed"]:
                    continue

                # Session filter: London & NY (08:00 - 22:00 UTC)
                hour = r['hour_utc']
                if not (8 <= hour < 22):
                    continue

                adx = r['adx']
                if np.isnan(adx) or adx < 22.0:
                    continue

                vol = r['volume']
                vol_ma = r['vol_ma']
                if np.isnan(vol_ma) or vol < (vol_ma * 0.75):
                    continue

                # Macro Trend
                fast_macro = r['ema_macro_fast']
                slow_macro = r['ema_macro_slow']
                macro_bull = price > slow_macro and fast_macro > slow_macro
                macro_bear = price < slow_macro and fast_macro < slow_macro

                rsi_prev = p['rsi']
                rsi_curr = r['rsi']
                ema9 = r['ema9']

                is_long = macro_bull and rsi_prev < 38.0 and r['close'] > r['open'] and price > ema9
                is_short = macro_bear and rsi_prev > 62.0 and r['close'] < r['open'] and price < ema9

                if is_long:
                    swing_l = r['swing_low']
                    sl = min(swing_l, price - (self.sl_atr_mult * atr))
                    risk = price - sl
                    if 0.003 <= (risk / price) <= 0.035:
                        tp = price + (self.rr_ratio * risk)

                        risk_pct = self.base_risk_pct / 100.0
                        if self.enable_shock_absorber and consecutive_losses >= 2:
                            risk_pct *= 0.5

                        risk_usdt = balance * risk_pct
                        pos_notional = risk_usdt / max(risk / price, 0.001)
                        pos_notional = min(pos_notional, balance * self.leverage * 0.35)
                        contracts = pos_notional / price

                        pos = {
                            "side": "long",
                            "entry_price": price,
                            "entry_time": str(r["datetime"]),
                            "sl": sl,
                            "tp": tp,
                            "contracts": contracts,
                            "risk_usdt": risk_usdt,
                            "be_locked": False
                        }

                elif is_short:
                    swing_h = r['swing_high']
                    sl = max(swing_h, price + (self.sl_atr_mult * atr))
                    risk = sl - price
                    if 0.003 <= (risk / price) <= 0.035:
                        tp = price - (self.rr_ratio * risk)

                        risk_pct = self.base_risk_pct / 100.0
                        if self.enable_shock_absorber and consecutive_losses >= 2:
                            risk_pct *= 0.5

                        risk_usdt = balance * risk_pct
                        pos_notional = risk_usdt / max(risk / price, 0.001)
                        pos_notional = min(pos_notional, balance * self.leverage * 0.35)
                        contracts = pos_notional / price

                        pos = {
                            "side": "short",
                            "entry_price": price,
                            "entry_time": str(r["datetime"]),
                            "sl": sl,
                            "tp": tp,
                            "contracts": contracts,
                            "risk_usdt": risk_usdt,
                            "be_locked": False
                        }

        # Calculate final stats
        net_return = ((balance - self.initial_balance) / self.initial_balance) * 100.0
        wins = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = (wins / max(len(trades), 1)) * 100.0

        yearly_list = []
        for y, s in sorted(yearly_stats.items()):
            y_wr = (s['wins'] / max(s['trades'], 1)) * 100.0
            y_ret = (s['pnl'] / max(s['start_bal'], 1.0)) * 100.0
            yearly_list.append({
                "year": y,
                "trades": s['trades'],
                "win_rate": round(y_wr, 1),
                "pnl": round(s['pnl'], 2),
                "return_pct": round(y_ret, 2)
            })

        return {
            "initial_balance": self.initial_balance,
            "final_balance": round(balance, 2),
            "net_pnl": round(balance - self.initial_balance, 2),
            "net_return_pct": round(net_return, 2),
            "total_trades": len(trades),
            "win_rate": round(win_rate, 1),
            "max_drawdown_pct": round(max_drawdown, 1),
            "yearly_stats": yearly_list,
            "trades": trades
        }

if __name__ == "__main__":
    print("=" * 80)
    print("     5-YEAR ETH/USDT QUANT SCALPER RIGOROUS BACKTEST (2021 - 2026)")
    print("=" * 80)

    # 1. Mode 1: Max Profit Runner
    tester1 = FiveYearBacktester(enable_freeroll_be=False, enable_shock_absorber=False)
    res1 = tester1.run()

    # 2. Mode 2: Free-Roll BE Protected
    tester2 = FiveYearBacktester(enable_freeroll_be=True, enable_shock_absorber=True)
    res2 = tester2.run()

    print("\n" + "=" * 65)
    print("  🏆 5-YEAR OVERALL PERFORMANCE (MODE 1: MAX PROFIT RUNNER)")
    print("=" * 65)
    print(f"  Starting Balance:        ${res1['initial_balance']:,.2f}")
    print(f"  Ending Balance:          ${res1['final_balance']:,.2f} ({res1['net_return_pct']:+.2f}% Net Return)")
    print(f"  Total Trades:            {res1['total_trades']} Trades")
    print(f"  Overall Win Rate:        {res1['win_rate']}% (1:2.0 RR)")
    print(f"  5-Year Max Drawdown:     {res1['max_drawdown_pct']}%")
    print("\n  📅 YEAR-BY-YEAR DETAILED BREAKDOWN (MODE 1):")
    print("  " + "-" * 60)
    for y in res1['yearly_stats']:
        icon = "✅" if y['pnl'] >= 0 else "❌"
        print(f"   {icon} {y['year']} | Trades: {y['trades']:<3} | WinRate: {y['win_rate']:4.1f}% | PnL: ${y['pnl']:+8.2f} ({y['return_pct']:+6.1f}%)")
    print("  " + "-" * 60)

    print("\n" + "=" * 65)
    print("  🛡️ 5-YEAR OVERALL PERFORMANCE (MODE 2: FREE-ROLL PROTECTED)")
    print("=" * 65)
    print(f"  Starting Balance:        ${res2['initial_balance']:,.2f}")
    print(f"  Ending Balance:          ${res2['final_balance']:,.2f} ({res2['net_return_pct']:+.2f}% Net Return)")
    print(f"  Total Trades:            {res2['total_trades']} Trades")
    print(f"  Overall Win Rate:        {res2['win_rate']}%")
    print(f"  5-Year Max Drawdown:     {res2['max_drawdown_pct']}%")
    print("\n  📅 YEAR-BY-YEAR DETAILED BREAKDOWN (MODE 2):")
    print("  " + "-" * 60)
    for y in res2['yearly_stats']:
        icon = "✅" if y['pnl'] >= 0 else "❌"
        print(f"   {icon} {y['year']} | Trades: {y['trades']:<3} | WinRate: {y['win_rate']:4.1f}% | PnL: ${y['pnl']:+8.2f} ({y['return_pct']:+6.1f}%)")
    print("  " + "-" * 60)
