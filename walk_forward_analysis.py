import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

DATA_DIR = Path(__file__).resolve().parent / "data" / "historical"

class WalkForwardAnalyzer:
    """
    Institutional Standard: Walk-Forward Analysis (WFA)
    
    Divides historical data into rolling windows:
    - Train Window (In-Sample, e.g. 60 Days): Finds optimal parameters (e.g. ADX threshold, ATR multipliers).
    - Test Window (Out-of-Sample, e.g. 30 Days): Tests the best parameters on UNSEEN future data.
    - Rolls forward across the entire year.
    
    If a strategy performs well out-of-sample, it proves it has REAL predictive edge and is NOT curve-fitted!
    """
    def __init__(self, symbol: str = 'SOL_USDT_USDT'):
        self.symbol = symbol
        csv_file = DATA_DIR / f"{symbol}_15m_1year.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"Historical file not found: {csv_file}")
        self.df = pd.read_csv(csv_file)
        self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        self._prepare_indicators()

    def _prepare_indicators(self):
        df = self.df
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

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

    def simulate_window(self, sub_df: pd.DataFrame, min_adx: float, sl_atr: float, tp_atr: float, initial_bal: float = 1000.0) -> Dict[str, Any]:
        bal = initial_bal
        trades = []
        pos = None

        for i in range(1, len(sub_df)):
            r = sub_df.iloc[i]
            p = sub_df.iloc[i-1]

            if pos:
                if pos['side'] == 'long':
                    if r['low'] <= pos['sl']:
                        pnl = (pos['sl'] - pos['entry']) * pos['contracts']
                        bal += pos['margin'] + pnl
                        trades.append(pnl)
                        pos = None
                    elif r['high'] >= pos['tp']:
                        pnl = (pos['tp'] - pos['entry']) * pos['contracts']
                        bal += pos['margin'] + pnl
                        trades.append(pnl)
                        pos = None
                elif pos['side'] == 'short':
                    if r['high'] >= pos['sl']:
                        pnl = (pos['entry'] - pos['sl']) * pos['contracts']
                        bal += pos['margin'] + pnl
                        trades.append(pnl)
                        pos = None
                    elif r['low'] <= pos['tp']:
                        pnl = (pos['entry'] - pos['tp']) * pos['contracts']
                        bal += pos['margin'] + pnl
                        trades.append(pnl)
                        pos = None

            if pos is None:
                atr = r['atr']
                adx = r['adx']
                if np.isnan(atr) or atr <= 0 or np.isnan(adx) or adx < min_adx:
                    continue
                if r['volume'] < r['vol_ma'] * 0.8:
                    continue

                # Long: Trend confirmation
                if r['ema50'] > r['ema200'] and r['ema9'] > r['ema21'] and p['low'] <= p['ema21'] and r['close'] > r['ema9']:
                    entry = r['close']
                    sl = entry - (sl_atr * atr)
                    tp = entry + (tp_atr * atr)
                    notional = bal * 1.0
                    contracts = notional / entry
                    margin = notional / 3.0
                    bal -= margin
                    pos = {'side': 'long', 'entry': entry, 'sl': sl, 'tp': tp, 'contracts': contracts, 'margin': margin}
                # Short: Trend confirmation
                elif r['ema50'] < r['ema200'] and r['ema9'] < r['ema21'] and p['high'] >= p['ema21'] and r['close'] < r['ema9']:
                    entry = r['close']
                    sl = entry + (sl_atr * atr)
                    tp = entry - (tp_atr * atr)
                    notional = bal * 1.0
                    contracts = notional / entry
                    margin = notional / 3.0
                    bal -= margin
                    pos = {'side': 'short', 'entry': entry, 'sl': sl, 'tp': tp, 'contracts': contracts, 'margin': margin}

        wins = [t for t in trades if t > 0]
        pnl = bal - initial_bal
        wr = (len(wins) / max(len(trades), 1)) * 100.0
        return {
            "pnl": round(pnl, 2),
            "return_pct": round((pnl / initial_bal) * 100.0, 2),
            "trades": len(trades),
            "wins": len(wins),
            "win_rate": round(wr, 1),
            "final_bal": round(bal, 2)
        }

    def run_walk_forward(self, train_candles: int = 5760, test_candles: int = 2880) -> List[Dict[str, Any]]:
        """
        train_candles: 60 Days (60 * 96 = 5760)
        test_candles:  30 Days (30 * 96 = 2880)
        """
        total = len(self.df)
        step = test_candles
        windows = []
        start = 250

        # Param grid to optimize during training
        param_grid = [
            {"adx": 22, "sl": 1.5, "tp": 2.5},
            {"adx": 25, "sl": 1.8, "tp": 3.0},
            {"adx": 28, "sl": 1.5, "tp": 3.5},
            {"adx": 30, "sl": 2.0, "tp": 4.0},
        ]

        current_balance = 1000.0
        window_idx = 1

        while start + train_candles + test_candles <= total:
            train_df = self.df.iloc[start : start + train_candles]
            test_df = self.df.iloc[start + train_candles : start + train_candles + test_candles]

            train_start_date = str(train_df['datetime'].iloc[0])[:10]
            train_end_date = str(train_df['datetime'].iloc[-1])[:10]
            test_start_date = str(test_df['datetime'].iloc[0])[:10]
            test_end_date = str(test_df['datetime'].iloc[-1])[:10]

            # Step 1: Optimize on Train Window (In-Sample)
            best_pnl = -999999
            best_params = param_grid[0]
            for p in param_grid:
                res = self.simulate_window(train_df, p["adx"], p["sl"], p["tp"], 1000.0)
                if res["pnl"] > best_pnl:
                    best_pnl = res["pnl"]
                    best_params = p

            # Step 2: Validate on UNSEEN Test Window (Out-of-Sample)
            test_res = self.simulate_window(test_df, best_params["adx"], best_params["sl"], best_params["tp"], current_balance)
            current_balance = test_res["final_bal"]

            windows.append({
                "window": window_idx,
                "train_period": f"{train_start_date} to {train_end_date}",
                "test_period": f"{test_start_date} to {test_end_date}",
                "chosen_params": best_params,
                "in_sample_pnl": round(best_pnl, 2),
                "out_of_sample_trades": test_res["trades"],
                "out_of_sample_win_rate": test_res["win_rate"],
                "out_of_sample_pnl": test_res["pnl"],
                "out_of_sample_return": test_res["return_pct"],
                "portfolio_balance": current_balance
            })

            start += step
            window_idx += 1

        return windows

if __name__ == "__main__":
    wfa = WalkForwardAnalyzer('SOL_USDT_USDT')
    print("=" * 70)
    print("      WALK-FORWARD ANALYSIS (HONEST OUT-OF-SAMPLE VALIDATION)")
    print("      Asset: SOL/USDT | 60-Day Train / 30-Day Unseen Test Windows")
    print("=" * 70)

    results = wfa.run_walk_forward()
    for w in results:
        print(f"\n[Window {w['window']}] Out-of-Sample Test Period: {w['test_period']}")
        print(f"  Optimized On Train Data: ADX >= {w['chosen_params']['adx']}, SL {w['chosen_params']['sl']}x, TP {w['chosen_params']['tp']}x")
        print(f"  Unseen Test Trades:      {w['out_of_sample_trades']} Trades | Win Rate: {w['out_of_sample_win_rate']}%")
        print(f"  Unseen Test Net PnL:     ${w['out_of_sample_pnl']:+6.2f} ({w['out_of_sample_return']:+5.1f}%)")
        print(f"  Rolling Capital Balance: ${w['portfolio_balance']:,.2f}")
