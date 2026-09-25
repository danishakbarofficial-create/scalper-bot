import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

DATA_DIR = Path(__file__).resolve().parent / "data" / "historical"

class MonthlyRegimeBacktester:
    """
    Comprehensive Backtester comparing:
    1. Static Monthly Prediction (Hardcoded regime for 30 days based on prior month forecast)
    2. Dynamic Real-Time Regime Switching (Reacts dynamically to real-time market regimes)
    """

    def __init__(self,
                 initial_balance: float = 1000.0,
                 leverage: int = 3,
                 risk_per_trade_pct: float = 1.5,
                 taker_fee: float = 0.0002,
                 daily_target_pct: float = 1.0,
                 daily_max_loss_pct: float = 3.0):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.risk_per_trade_pct = risk_per_trade_pct
        self.taker_fee = taker_fee
        self.daily_target_pct = daily_target_pct
        self.daily_max_loss_pct = daily_max_loss_pct

    def prepare_data(self, symbol: str) -> pd.DataFrame:
        csv_file = DATA_DIR / f"{symbol}_15m_1year.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"File not found: {csv_file}")

        df = pd.read_csv(csv_file)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
        df['month'] = df['datetime'].dt.strftime('%Y-%m')
        df['hour_utc'] = df['datetime'].dt.hour

        # EMAs
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

        # RSI (14)
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

        # ATR (14)
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

        # Bollinger Bands (20, 2.0)
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + (2.0 * bb_std)
        df['bb_lower'] = df['bb_mid'] - (2.0 * bb_std)

        # Volume MA
        df['vol_ma'] = df['volume'].rolling(20).mean()
        return df

    def run_static_monthly_prediction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Idea A: Predict next month's regime based on previous 30-day market state,
        hardcode the bot for that month, and repeat.
        """
        months = sorted(df['month'].unique())
        # Filter full months
        months = [m for m in months if len(df[df['month'] == m]) >= 1000]

        balance = self.initial_balance
        peak_balance = balance
        max_drawdown = 0.0
        monthly_stats = []
        all_trades = []

        for idx in range(1, len(months)):
            prior_m = months[idx-1]
            target_m = months[idx]

            prior_df = df[df['month'] == prior_m]
            target_df = df[df['month'] == target_m]

            # 1. Prediction Step: Prior 30-Day Analysis
            prior_ret = (prior_df['close'].iloc[-1] - prior_df['close'].iloc[0]) / prior_df['close'].iloc[0] * 100.0
            pct_above_200 = (prior_df['close'] > prior_df['ema200']).mean() * 100.0
            avg_adx = prior_df['adx'].mean()

            # Predict Regime & Bot Profile
            if pct_above_200 >= 55.0 and prior_ret > 0:
                predicted_regime = "BULL_TREND"
                allowed_sides = ["long"]
                sl_atr = 1.3
                tp1_atr = 1.6
                tp2_atr = 2.8
            elif pct_above_200 <= 45.0 and prior_ret < 0:
                predicted_regime = "BEAR_TREND"
                allowed_sides = ["short"]
                sl_atr = 1.3
                tp1_atr = 1.6
                tp2_atr = 2.8
            else:
                predicted_regime = "SIDEWAYS_RANGE"
                allowed_sides = ["long", "short"]
                sl_atr = 1.2
                tp1_atr = 1.4
                tp2_atr = 2.4

            # 2. Run Target Month with Hardcoded Monthly Regime
            m_start_bal = balance
            m_trades = []
            daily_tracker = {}
            pos = None

            for i in range(1, len(target_df)):
                r = target_df.iloc[i]
                p = target_df.iloc[i-1]
                curr_date = r['date']
                price = r['close']
                high = r['high']
                low = r['low']

                if curr_date not in daily_tracker:
                    daily_tracker[curr_date] = {"pnl": 0.0, "locked": False, "killed": False, "start_bal": balance}
                day = daily_tracker[curr_date]

                # Position Management
                if pos:
                    side = pos['side']
                    entry = pos['entry']
                    cnt = pos['contracts']
                    closed = False
                    pnl = 0.0

                    if side == 'long':
                        if low <= pos['sl']:
                            closed = True
                            pnl = (pos['sl'] - entry) * cnt - (cnt * pos['sl'] * self.taker_fee)
                        elif not pos['tp1_hit'] and high >= pos['tp1']:
                            pos['tp1_hit'] = True
                            half = cnt / 2.0
                            booked = (pos['tp1'] - entry) * half - (half * pos['tp1'] * self.taker_fee)
                            balance += (pos['margin'] / 2.0 + booked)
                            pos['margin'] /= 2.0
                            pos['contracts'] -= half
                            pos['sl'] = entry # Break-even
                            pos['booked'] = booked
                            day['pnl'] += booked
                        elif high >= pos['tp2']:
                            closed = True
                            pnl = pos.get('booked', 0.0) + (pos['tp2'] - entry) * cnt - (cnt * pos['tp2'] * self.taker_fee)

                    elif side == 'short':
                        if high >= pos['sl']:
                            closed = True
                            pnl = (entry - pos['sl']) * cnt - (cnt * pos['sl'] * self.taker_fee)
                        elif not pos['tp1_hit'] and low <= pos['tp1']:
                            pos['tp1_hit'] = True
                            half = cnt / 2.0
                            booked = (entry - pos['tp1']) * half - (half * pos['tp1'] * self.taker_fee)
                            balance += (pos['margin'] / 2.0 + booked)
                            pos['margin'] /= 2.0
                            pos['contracts'] -= half
                            pos['sl'] = entry
                            pos['booked'] = booked
                            day['pnl'] += booked
                        elif low <= pos['tp2']:
                            closed = True
                            pnl = pos.get('booked', 0.0) + (entry - pos['tp2']) * cnt - (cnt * pos['tp2'] * self.taker_fee)

                    if closed:
                        net_pnl = pnl - pos.get('booked', 0.0)
                        balance += max(0.0, pos['margin'] + net_pnl)
                        day['pnl'] += net_pnl
                        trade_record = {"pnl": pnl, "win": pnl > 0}
                        m_trades.append(trade_record)
                        all_trades.append(trade_record)
                        pos = None

                        # Check Daily Limits
                        d_ret = day['pnl'] / max(day['start_bal'], 1.0) * 100.0
                        if d_ret >= self.daily_target_pct:
                            day['locked'] = True
                        elif d_ret <= -self.daily_max_loss_pct:
                            day['killed'] = True

                # Drawdown tracking
                if balance > peak_balance:
                    peak_balance = balance
                dd = (peak_balance - balance) / max(peak_balance, 1.0) * 100.0
                if dd > max_drawdown:
                    max_drawdown = dd

                # Entry Logic for Hardcoded Monthly Mode
                if pos is None:
                    if day['locked'] or day['killed']:
                        continue
                    if not (12 <= r['hour_utc'] <= 21):
                        continue
                    atr = r['atr']
                    if np.isnan(atr) or atr <= 0:
                        continue

                    entry_side = None

                    if predicted_regime == "BULL_TREND":
                        if r['ema21'] > r['ema50'] and p['low'] <= p['ema21'] and r['close'] > r['open'] and r['close'] > r['ema9']:
                            if 42 <= r['rsi'] <= 68 and r['volume'] >= r['vol_ma'] * 0.75:
                                entry_side = "long"

                    elif predicted_regime == "BEAR_TREND":
                        if r['ema21'] < r['ema50'] and p['high'] >= p['ema21'] and r['close'] < r['open'] and r['close'] < r['ema9']:
                            if 32 <= r['rsi'] <= 58 and r['volume'] >= r['vol_ma'] * 0.75:
                                entry_side = "short"

                    elif predicted_regime == "SIDEWAYS_RANGE":
                        if p['low'] <= p['bb_lower'] and r['close'] > r['open'] and r['rsi'] <= 38:
                            entry_side = "long"
                        elif p['high'] >= p['bb_upper'] and r['close'] < r['open'] and r['rsi'] >= 62:
                            entry_side = "short"

                    if entry_side and entry_side in allowed_sides:
                        dist = sl_atr * atr
                        sl_pct = dist / price
                        risk_val = balance * (self.risk_per_trade_pct / 100.0)
                        notional = min(risk_val / max(sl_pct, 0.005), balance * self.leverage * 0.35)
                        margin = notional / self.leverage
                        cnt = notional / price
                        fee = notional * self.taker_fee

                        balance -= (margin + fee)
                        sl = price - dist if entry_side == 'long' else price + dist
                        tp1 = price + (tp1_atr * atr) if entry_side == 'long' else price - (tp1_atr * atr)
                        tp2 = price + (tp2_atr * atr) if entry_side == 'long' else price - (tp2_atr * atr)

                        pos = {
                            "side": entry_side,
                            "entry": price,
                            "contracts": cnt,
                            "margin": margin,
                            "sl": sl,
                            "tp1": tp1,
                            "tp2": tp2,
                            "tp1_hit": False,
                            "booked": 0.0
                        }

            m_wins = [t for t in m_trades if t['win']]
            m_wr = (len(m_wins) / max(len(m_trades), 1)) * 100.0
            m_pnl = balance - m_start_bal
            m_ret = (m_pnl / m_start_bal) * 100.0

            monthly_stats.append({
                "month": target_m,
                "prediction": predicted_regime,
                "trades": len(m_trades),
                "win_rate": round(m_wr, 1),
                "pnl": round(m_pnl, 2),
                "return_pct": round(m_ret, 2),
                "end_balance": round(balance, 2)
            })

        all_wins = [t for t in all_trades if t['win']]
        return {
            "name": "Static Monthly Prediction (Hardcoded 30-Day Regime)",
            "final_balance": round(balance, 2),
            "net_pnl": round(balance - self.initial_balance, 2),
            "total_return_pct": round((balance - self.initial_balance) / self.initial_balance * 100.0, 2),
            "total_trades": len(all_trades),
            "overall_win_rate": round((len(all_wins) / max(len(all_trades), 1)) * 100.0, 1),
            "max_drawdown_pct": round(max_drawdown, 2),
            "monthly_stats": monthly_stats
        }

    def run_dynamic_regime_switching(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Idea B (Recommended): Dynamic Real-Time Regime Detection.
        Bot dynamically detects the current trend & volatility regime at each bar,
        aligns with Higher-Timeframe trend, and protects capital during chop.
        """
        balance = self.initial_balance
        peak_balance = balance
        max_drawdown = 0.0
        all_trades = []
        monthly_stats_dict = {}

        sl_atr = 1.2
        tp1_atr = 1.5
        tp2_atr = 2.5

        pos = None
        daily_tracker = {}

        for i in range(200, len(df)):
            r = df.iloc[i]
            p = df.iloc[i-1]
            curr_date = r['date']
            curr_month = r['month']
            price = r['close']
            high = r['high']
            low = r['low']

            if curr_date not in daily_tracker:
                daily_tracker[curr_date] = {"pnl": 0.0, "locked": False, "killed": False, "start_bal": balance}
            day = daily_tracker[curr_date]

            if curr_month not in monthly_stats_dict:
                monthly_stats_dict[curr_month] = {"start_bal": balance, "pnl": 0.0, "trades": 0, "wins": 0}
            m_stat = monthly_stats_dict[curr_month]

            # Position Management
            if pos:
                side = pos['side']
                entry = pos['entry']
                cnt = pos['contracts']
                closed = False
                pnl = 0.0

                if side == 'long':
                    if low <= pos['sl']:
                        closed = True
                        pnl = (pos['sl'] - entry) * cnt - (cnt * pos['sl'] * self.taker_fee)
                    elif not pos['tp1_hit'] and high >= pos['tp1']:
                        pos['tp1_hit'] = True
                        half = cnt / 2.0
                        booked = (pos['tp1'] - entry) * half - (half * pos['tp1'] * self.taker_fee)
                        balance += (pos['margin'] / 2.0 + booked)
                        pos['margin'] /= 2.0
                        pos['contracts'] -= half
                        pos['sl'] = entry # Break-even
                        pos['booked'] = booked
                        day['pnl'] += booked
                        m_stat['pnl'] += booked
                    elif high >= pos['tp2']:
                        closed = True
                        pnl = pos.get('booked', 0.0) + (pos['tp2'] - entry) * cnt - (cnt * pos['tp2'] * self.taker_fee)

                elif side == 'short':
                    if high >= pos['sl']:
                        closed = True
                        pnl = (entry - pos['sl']) * cnt - (cnt * pos['sl'] * self.taker_fee)
                    elif not pos['tp1_hit'] and low <= pos['tp1']:
                        pos['tp1_hit'] = True
                        half = cnt / 2.0
                        booked = (entry - pos['tp1']) * half - (half * pos['tp1'] * self.taker_fee)
                        balance += (pos['margin'] / 2.0 + booked)
                        pos['margin'] /= 2.0
                        pos['contracts'] -= half
                        pos['sl'] = entry
                        pos['booked'] = booked
                        day['pnl'] += booked
                        m_stat['pnl'] += booked
                    elif low <= pos['tp2']:
                        closed = True
                        pnl = pos.get('booked', 0.0) + (entry - pos['tp2']) * cnt - (cnt * pos['tp2'] * self.taker_fee)

                if closed:
                    net_pnl = pnl - pos.get('booked', 0.0)
                    balance += max(0.0, pos['margin'] + net_pnl)
                    day['pnl'] += net_pnl
                    m_stat['pnl'] += net_pnl
                    m_stat['trades'] += 1
                    if pnl > 0:
                        m_stat['wins'] += 1
                    all_trades.append({"pnl": pnl, "win": pnl > 0})
                    pos = None

                    d_ret = day['pnl'] / max(day['start_bal'], 1.0) * 100.0
                    if d_ret >= self.daily_target_pct:
                        day['locked'] = True
                    elif d_ret <= -self.daily_max_loss_pct:
                        day['killed'] = True

            # Drawdown
            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / max(peak_balance, 1.0) * 100.0
            if dd > max_drawdown:
                max_drawdown = dd

            # Dynamic Real-time Entry Logic
            if pos is None:
                if day['locked'] or day['killed']:
                    continue
                if not (12 <= r['hour_utc'] <= 21):
                    continue

                atr = r['atr']
                adx = r['adx']
                if np.isnan(atr) or np.isnan(adx) or atr <= 0 or adx < 24.0:
                    continue

                # DYNAMIC REGIME DETECTION IN REAL-TIME:
                # Is Macro Bullish?
                is_bull_regime = (price > r['ema200'] and r['ema50'] > r['ema200'] and r['ema21'] > r['ema50'])
                # Is Macro Bearish?
                is_bear_regime = (price < r['ema200'] and r['ema50'] < r['ema200'] and r['ema21'] < r['ema50'])

                entry_side = None

                # Setup A: Bull Trend Pullback Bounce
                if is_bull_regime and p['low'] <= p['ema21'] and r['close'] > r['open'] and r['close'] > r['ema9']:
                    if 45 <= r['rsi'] <= 68 and r['volume'] >= r['vol_ma'] * 0.75:
                        entry_side = "long"

                # Setup B: Bear Trend Relief Rejection
                elif is_bear_regime and p['high'] >= p['ema21'] and r['close'] < r['open'] and r['close'] < r['ema9']:
                    if 32 <= r['rsi'] <= 55 and r['volume'] >= r['vol_ma'] * 0.75:
                        entry_side = "short"

                if entry_side:
                    dist = sl_atr * atr
                    sl_pct = dist / price
                    risk_val = balance * (self.risk_per_trade_pct / 100.0)
                    notional = min(risk_val / max(sl_pct, 0.005), balance * self.leverage * 0.35)
                    margin = notional / self.leverage
                    cnt = notional / price
                    fee = notional * self.taker_fee

                    balance -= (margin + fee)
                    sl = price - dist if entry_side == 'long' else price + dist
                    tp1 = price + (tp1_atr * atr) if entry_side == 'long' else price - (tp1_atr * atr)
                    tp2 = price + (tp2_atr * atr) if entry_side == 'long' else price - (tp2_atr * atr)

                    pos = {
                        "side": entry_side,
                        "entry": price,
                        "contracts": cnt,
                        "margin": margin,
                        "sl": sl,
                        "tp1": tp1,
                        "tp2": tp2,
                        "tp1_hit": False,
                        "booked": 0.0
                    }

        # Format monthly stats
        monthly_summary = []
        for m, s in sorted(monthly_stats_dict.items()):
            if s['trades'] == 0:
                continue
            wr = (s['wins'] / max(s['trades'], 1)) * 100.0
            ret = (s['pnl'] / max(s['start_bal'], 1.0)) * 100.0
            monthly_summary.append({
                "month": m,
                "trades": s['trades'],
                "win_rate": round(wr, 1),
                "pnl": round(s['pnl'], 2),
                "return_pct": round(ret, 2)
            })

        all_wins = [t for t in all_trades if t['win']]
        return {
            "name": "Dynamic Real-Time Regime Switching (Recommended)",
            "final_balance": round(balance, 2),
            "net_pnl": round(balance - self.initial_balance, 2),
            "total_return_pct": round((balance - self.initial_balance) / self.initial_balance * 100.0, 2),
            "total_trades": len(all_trades),
            "overall_win_rate": round((len(all_wins) / max(len(all_trades), 1)) * 100.0, 1),
            "max_drawdown_pct": round(max_drawdown, 2),
            "monthly_stats": monthly_summary
        }

def run_comparison():
    tester = MonthlyRegimeBacktester()
    symbols = ["BTC_USDT_USDT", "ETH_USDT_USDT", "SOL_USDT_USDT"]

    print("=" * 80)
    print("      RESEARCH COMPARISON: STATIC MONTHLY PREDICTION VS DYNAMIC REGIMES")
    print("      Testing on 1-Year Historical 15M Data (MEXC Futures Fees Included)")
    print("=" * 80)

    for sym in symbols:
        df = tester.prepare_data(sym)
        res_static = tester.run_static_monthly_prediction(df)
        res_dynamic = tester.run_dynamic_regime_switching(df)

        asset_name = sym.replace("_USDT_USDT", "/USDT")
        print(f"\n=======================================================")
        print(f"  ASSET: {asset_name} (1-Year Historical Test)")
        print(f"=======================================================")
        
        print(f"\n  [APPROACH 1] {res_static['name']}:")
        print(f"    Total Trades:    {res_static['total_trades']}")
        print(f"    Win Rate:        {res_static['overall_win_rate']}%")
        print(f"    Final Capital:   ${res_static['final_balance']:,.2f} ({res_static['total_return_pct']:+6.2f}%)")
        print(f"    Max Drawdown:    {res_static['max_drawdown_pct']}%")
        print(f"    Month-by-Month Breakdown:")
        for m in res_static['monthly_stats']:
            print(f"      {m['month']} | Mode: {m['prediction']:<15} | Trades: {m['trades']:<2} | WinRate: {m['win_rate']:5.1f}% | PnL: ${m['pnl']:+7.2f} ({m['return_pct']:+5.1f}%)")

        print(f"\n  [APPROACH 2] {res_dynamic['name']}:")
        print(f"    Total Trades:    {res_dynamic['total_trades']}")
        print(f"    Win Rate:        {res_dynamic['overall_win_rate']}%")
        print(f"    Final Capital:   ${res_dynamic['final_balance']:,.2f} ({res_dynamic['total_return_pct']:+6.2f}%)")
        print(f"    Max Drawdown:    {res_dynamic['max_drawdown_pct']}%")
        print(f"    Month-by-Month Breakdown:")
        for m in res_dynamic['monthly_stats']:
            print(f"      {m['month']} | Trades: {m['trades']:<2} | WinRate: {m['win_rate']:5.1f}% | PnL: ${m['pnl']:+7.2f} ({m['return_pct']:+5.1f}%)")

if __name__ == "__main__":
    run_comparison()
