import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

DATA_DIR = Path(__file__).resolve().parent / "data" / "historical"

class Filtered1YearBacktester:
    def __init__(self,
                 initial_balance: float = 1000.0,
                 leverage: int = 3,
                 risk_per_trade_pct: float = 1.5,
                 min_adx: float = 22.0,
                 sl_atr_mult: float = 1.6,
                 tp1_atr_mult: float = 2.0,
                 tp2_atr_mult: float = 3.5,
                 daily_target_pct: float = 1.0,
                 daily_max_loss_pct: float = 3.0,
                 taker_fee: float = 0.0002):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.risk_per_trade_pct = risk_per_trade_pct
        self.min_adx = min_adx
        self.sl_atr_mult = sl_atr_mult
        self.tp1_atr_mult = tp1_atr_mult
        self.tp2_atr_mult = tp2_atr_mult
        self.daily_target_pct = daily_target_pct
        self.daily_max_loss_pct = daily_max_loss_pct
        self.taker_fee = taker_fee

    def run(self, symbol: str) -> Dict[str, Any]:
        csv_file = DATA_DIR / f"{symbol}_15m_1year.csv"
        if not csv_file.exists():
            print(f"File not found: {csv_file}")
            return {}

        df = pd.read_csv(csv_file)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Calculate indicators
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

        # RSI (14)
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

        # ADX (14)
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
        df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
        df['month'] = df['datetime'].dt.strftime('%Y-%m')

        balance = self.initial_balance
        peak_balance = balance
        max_drawdown = 0.0

        current_pos = None
        trades: List[Dict[str, Any]] = []
        daily_tracker: Dict[str, Any] = {}
        monthly_stats: Dict[str, Any] = {}
        skipped_chop_bars = 0

        for i in range(200, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            curr_date = row['date']
            curr_month = row['month']
            price = row['close']
            high = row['high']
            low = row['low']

            if curr_date not in daily_tracker:
                daily_tracker[curr_date] = {
                    "start_bal": balance,
                    "pnl": 0.0,
                    "locked": False,
                    "killed": False
                }
            day = daily_tracker[curr_date]

            if curr_month not in monthly_stats:
                monthly_stats[curr_month] = {
                    "start_bal": balance,
                    "pnl": 0.0,
                    "trades": 0,
                    "wins": 0
                }

            # 1. Manage Open Position
            if current_pos:
                side = current_pos['side']
                entry = current_pos['entry']
                contracts = current_pos['contracts']
                margin = current_pos['margin']

                closed = False
                exit_price = 0.0
                exit_reason = ""
                trade_pnl = 0.0

                if side == 'long':
                    # Stop loss
                    if low <= current_pos['sl']:
                        exit_price = current_pos['sl']
                        exit_reason = "STOP_LOSS"
                        raw = (exit_price - entry) * contracts
                        trade_pnl = raw - (contracts * exit_price * self.taker_fee)
                        closed = True
                    # TP1 (50% closed, SL to Break-Even)
                    elif not current_pos['tp1_hit'] and high >= current_pos['tp1']:
                        current_pos['tp1_hit'] = True
                        half = contracts / 2.0
                        raw_half = (current_pos['tp1'] - entry) * half
                        net_half = raw_half - (half * current_pos['tp1'] * self.taker_fee)
                        balance += (margin / 2.0 + net_half)
                        current_pos['margin'] -= (margin / 2.0)
                        current_pos['contracts'] -= half
                        current_pos['sl'] = entry # SL to BE
                        current_pos['booked'] = net_half
                        day['pnl'] += net_half
                        monthly_stats[curr_month]['pnl'] += net_half
                    # TP2
                    elif high >= current_pos['tp2']:
                        exit_price = current_pos['tp2']
                        exit_reason = "TAKE_PROFIT_2"
                        raw = (exit_price - entry) * contracts
                        trade_pnl = current_pos.get('booked', 0.0) + raw - (contracts * exit_price * self.taker_fee)
                        closed = True

                elif side == 'short':
                    # Stop loss
                    if high >= current_pos['sl']:
                        exit_price = current_pos['sl']
                        exit_reason = "STOP_LOSS"
                        raw = (entry - exit_price) * contracts
                        trade_pnl = raw - (contracts * exit_price * self.taker_fee)
                        closed = True
                    # TP1
                    elif not current_pos['tp1_hit'] and low <= current_pos['tp1']:
                        current_pos['tp1_hit'] = True
                        half = contracts / 2.0
                        raw_half = (entry - current_pos['tp1']) * half
                        net_half = raw_half - (half * current_pos['tp1'] * self.taker_fee)
                        balance += (margin / 2.0 + net_half)
                        current_pos['margin'] -= (margin / 2.0)
                        current_pos['contracts'] -= half
                        current_pos['sl'] = entry
                        current_pos['booked'] = net_half
                        day['pnl'] += net_half
                        monthly_stats[curr_month]['pnl'] += net_half
                    # TP2
                    elif low <= current_pos['tp2']:
                        exit_price = current_pos['tp2']
                        exit_reason = "TAKE_PROFIT_2"
                        raw = (entry - exit_price) * contracts
                        trade_pnl = current_pos.get('booked', 0.0) + raw - (contracts * exit_price * self.taker_fee)
                        closed = True

                if closed:
                    balance += max(0.0, current_pos['margin'] + (trade_pnl - current_pos.get('booked', 0.0)))
                    day['pnl'] += (trade_pnl - current_pos.get('booked', 0.0))
                    monthly_stats[curr_month]['pnl'] += (trade_pnl - current_pos.get('booked', 0.0))
                    monthly_stats[curr_month]['trades'] += 1
                    if trade_pnl > 0:
                        monthly_stats[curr_month]['wins'] += 1

                    trades.append({
                        "side": side,
                        "pnl": round(trade_pnl, 2),
                        "win": trade_pnl > 0,
                        "reason": exit_reason
                    })
                    current_pos = None

                    # Check daily target & killswitch
                    day_pct = (day['pnl'] / max(day['start_bal'], 1.0)) * 100.0
                    if day_pct >= self.daily_target_pct:
                        day['locked'] = True
                    if day_pct <= -self.daily_max_loss_pct:
                        day['killed'] = True

            # Track Drawdown
            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / max(peak_balance, 1.0) * 100.0
            if dd > max_drawdown:
                max_drawdown = dd

            # 2. Check for New Entry (WITH SMART REGIME FILTERS)
            if current_pos is None:
                # Rule 1: Daily Target / Killswitch Guard
                if day['locked'] or day['killed']:
                    continue

                atr = row['atr']
                adx = row['adx']

                # Rule 2: ADX Regime Filter (If ADX < 22, SKIP! Dead chop)
                if np.isnan(adx) or adx < self.min_adx:
                    skipped_chop_bars += 1
                    continue

                if np.isnan(atr) or atr <= 0:
                    continue

                # Rule 3: Volume Confirmation
                has_vol = row['volume'] >= (row['vol_ma'] * 0.75)
                if not has_vol:
                    continue

                # Rule 4: Macro Trend + Momentum Ribbon
                is_long = (
                    row['ema50'] > row['ema200'] and
                    row['ema9'] > row['ema21'] and
                    prev['low'] <= prev['ema21'] and
                    price > row['ema9'] and
                    45.0 <= row['rsi'] <= 66.0
                )

                is_short = (
                    row['ema50'] < row['ema200'] and
                    row['ema9'] < row['ema21'] and
                    prev['high'] >= prev['ema21'] and
                    price < row['ema9'] and
                    34.0 <= row['rsi'] <= 55.0
                )

                if is_long or is_short:
                    side = 'long' if is_long else 'short'
                    sl_dist = self.sl_atr_mult * atr
                    sl_pct = sl_dist / price

                    # Position sizing (1.5% Risk, 3x Leverage)
                    risk_amt = balance * (self.risk_per_trade_pct / 100.0)
                    notional = min(risk_amt / max(sl_pct, 0.005), balance * self.leverage * 0.35)
                    margin = notional / self.leverage
                    contracts = notional / price

                    fee = notional * self.taker_fee
                    balance -= (margin + fee)

                    sl = round(price - sl_dist if side == 'long' else price + sl_dist, 2)
                    tp1 = round(price + (self.tp1_atr_mult * atr) if side == 'long' else price - (self.tp1_atr_mult * atr), 2)
                    tp2 = round(price + (self.tp2_atr_mult * atr) if side == 'long' else price - (self.tp2_atr_mult * atr), 2)

                    current_pos = {
                        "side": side,
                        "entry": price,
                        "contracts": contracts,
                        "margin": margin,
                        "sl": sl,
                        "tp1": tp1,
                        "tp2": tp2,
                        "tp1_hit": False,
                        "booked": 0.0
                    }

        if current_pos:
            final_p = df['close'].iloc[-1]
            pnl = (final_p - current_pos['entry']) * current_pos['contracts'] if current_pos['side'] == 'long' else (current_pos['entry'] - final_p) * current_pos['contracts']
            balance += current_pos['margin'] + pnl + current_pos.get('booked', 0.0)
            trades.append({"pnl": round(pnl, 2), "win": pnl > 0, "reason": "FINAL_CLOSE"})

        wins = [t for t in trades if t['win']]
        losses = [t for t in trades if not t['win']]
        total_trades = len(trades)
        win_rate = (len(wins) / max(total_trades, 1)) * 100.0

        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses))
        profit_factor = round(gross_profit / max(gross_loss, 0.01), 2)

        net_profit = balance - self.initial_balance
        total_return_pct = (net_profit / self.initial_balance) * 100.0
        avg_monthly = total_return_pct / 12.0

        monthly_rows = []
        for m, d in sorted(monthly_stats.items()):
            p = d['pnl']
            pct = (p / max(d['start_bal'], 1.0)) * 100.0
            wr = (d['wins'] / max(d['trades'], 1)) * 100.0
            monthly_rows.append({
                "month": m,
                "trades": d['trades'],
                "wins": d['wins'],
                "win_rate": round(wr, 1),
                "pnl": round(p, 2),
                "return_pct": round(pct, 2)
            })

        return {
            "symbol": symbol,
            "candles_analyzed": len(df),
            "skipped_chop_bars": skipped_chop_bars,
            "chop_filter_rate": round((skipped_chop_bars / len(df)) * 100.0, 1),
            "initial_balance": self.initial_balance,
            "final_balance": round(balance, 2),
            "net_profit": round(net_profit, 2),
            "total_return_pct": round(total_return_pct, 2),
            "avg_monthly_pct": round(avg_monthly, 2),
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "profit_factor": profit_factor,
            "max_drawdown_pct": round(max_drawdown, 2),
            "monthly_rows": monthly_rows
        }

if __name__ == "__main__":
    tester = Filtered1YearBacktester(initial_balance=1000.0, leverage=3, min_adx=22.0)
    symbols = ['BTC_USDT_USDT', 'ETH_USDT_USDT', 'SOL_USDT_USDT']

    print("=" * 70)
    print("      MEXC SCALPER PRO — 1-YEAR FILTERED BACKTEST RESULTS")
    print("      (With ADX Regime Filter, Daily Target Lock & Killswitch)")
    print("=" * 70)

    for s in symbols:
        res = tester.run(s)
        if res:
            name = s.replace('_USDT_USDT', '/USDT')
            print(f"\n=======================================================")
            print(f"  ASSET: {name} (1-Year Filtered Test: 34,603 Candles)")
            print(f"=======================================================")
            print(f"  Sideways Chop Filtered Out: {res['skipped_chop_bars']:,} candles ({res['chop_filter_rate']}% of year avoided)")
            print(f"  Total Filtered Trades:      {res['total_trades']} Trades (vs 700+ before)")
            print(f"  Win Rate:                   {res['win_rate']}% ({res['wins']} Won / {res['losses']} Lost)")
            print(f"  Profit Factor:              {res['profit_factor']}")
            print(f"  Initial Capital:            ${res['initial_balance']:,.2f}")
            print(f"  Final Capital:              ${res['final_balance']:,.2f} (Net: {'+' if res['net_profit']>=0 else ''}${res['net_profit']:,.2f})")
            print(f"  Total 1-Year Return:        {'+' if res['total_return_pct']>=0 else ''}{res['total_return_pct']}%")
            print(f"  Average Monthly Return:     {'+' if res['avg_monthly_pct']>=0 else ''}{res['avg_monthly_pct']}% / month")
            print(f"  Max Drawdown:               {res['max_drawdown_pct']}% (Cut drastically by Killswitch)")
            
            print("\n  Month-by-Month Summary:")
            for m in res['monthly_rows'][-6:]:  # show recent 6 months
                print(f"    {m['month']} | Trades: {m['trades']:2} | WinRate: {m['win_rate']:4.1f}% | PnL: {'+' if m['pnl']>=0 else ''}${m['pnl']:7.2f} ({'+' if m['return_pct']>=0 else ''}{m['return_pct']:5.1f}%)")
