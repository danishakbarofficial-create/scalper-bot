import time
import datetime
import pandas as pd
import numpy as np
import ccxt
from pathlib import Path
from typing import Dict, Any, List

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

class MexcBacktester:
    def __init__(self, 
                 initial_balance: float = 1000.0,
                 leverage: int = 3,
                 risk_per_trade_pct: float = 1.5,
                 sl_pct: float = 0.8,
                 tp1_pct: float = 1.2,
                 tp2_pct: float = 2.4,
                 daily_target_pct: float = 1.0,
                 daily_max_loss_pct: float = 3.0,
                 taker_fee: float = 0.0002):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.risk_per_trade_pct = risk_per_trade_pct
        self.sl_pct = sl_pct / 100.0
        self.tp1_pct = tp1_pct / 100.0
        self.tp2_pct = tp2_pct / 100.0
        self.daily_target_pct = daily_target_pct
        self.daily_max_loss_pct = daily_max_loss_pct
        self.taker_fee = taker_fee

        self.exchange = ccxt.mexc({'options': {'defaultType': 'swap'}})

    def fetch_historical_candles(self, symbol: str, timeframe: str = '5m', total_candles: int = 3000) -> pd.DataFrame:
        print(f"Fetching {total_candles} {timeframe} candles for {symbol} from MEXC...")
        all_ohlcv = []
        limit_per_req = 500
        
        # Calculate start time
        milli_per_candle = 5 * 60 * 1000 if timeframe == '5m' else (1 * 60 * 1000 if timeframe == '1m' else 15 * 60 * 1000)
        since = int(time.time() * 1000) - (total_candles * milli_per_candle)

        while len(all_ohlcv) < total_candles:
            try:
                raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit_per_req)
                if not raw or len(raw) == 0:
                    break
                all_ohlcv.extend(raw)
                since = raw[-1][0] + 1
                time.sleep(0.2)
            except Exception as e:
                print(f"Fetch warning: {e}")
                break

        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.drop_duplicates(subset=['timestamp'], inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        print(f"Loaded {len(df)} candles from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
        return df

    def run_backtest(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        if df.empty or len(df) < 60:
            return {}

        df = df.copy()
        df['ema9'] = calculate_ema(df['close'], 9)
        df['ema21'] = calculate_ema(df['close'], 21)
        df['ema50'] = calculate_ema(df['close'], 50)
        df['rsi'] = calculate_rsi(df['close'], 14)
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')

        balance = self.initial_balance
        peak_balance = balance
        max_drawdown = 0.0

        current_position = None
        trades: List[Dict[str, Any]] = []
        daily_pnl_tracker = {}

        for i in range(55, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            curr_date = row['date']
            curr_price = row['close']
            high_price = row['high']
            low_price = row['low']

            # Check new day in daily tracker
            if curr_date not in daily_pnl_tracker:
                daily_pnl_tracker[curr_date] = {
                    "start_bal": balance,
                    "pnl": 0.0,
                    "locked": False,
                    "killed": False
                }

            day_info = daily_pnl_tracker[curr_date]

            # 1. Manage Active Position (Check exits on High/Low)
            if current_position:
                side = current_position['side']
                entry = current_position['entry_price']
                contracts = current_position['contracts']
                margin = current_position['margin']

                # Update extremes
                if high_price > current_position['highest']:
                    current_position['highest'] = high_price
                if low_price < current_position['lowest']:
                    current_position['lowest'] = low_price

                closed = False
                exit_price = 0.0
                exit_reason = ""
                trade_pnl = 0.0

                if side == "long":
                    # Check Stop Loss
                    if low_price <= current_position['sl']:
                        exit_price = current_position['sl']
                        exit_reason = "STOP_LOSS"
                        raw = (exit_price - entry) * contracts
                        trade_pnl = raw - (contracts * exit_price * self.taker_fee)
                        closed = True
                    # Check TP1 (Close 50% & move SL to BE)
                    elif not current_position['tp1_hit'] and high_price >= current_position['tp1']:
                        current_position['tp1_hit'] = True
                        half = contracts / 2.0
                        raw_half = (current_position['tp1'] - entry) * half
                        net_half = raw_half - (half * current_position['tp1'] * self.taker_fee)
                        balance += (margin / 2.0 + net_half)
                        current_position['margin'] -= margin / 2.0
                        current_position['contracts'] -= half
                        current_position['sl'] = entry  # SL to Break-Even!
                        current_position['booked_pnl'] = net_half
                        day_info['pnl'] += net_half
                    # Check TP2
                    elif high_price >= current_position['tp2']:
                        exit_price = current_position['tp2']
                        exit_reason = "TAKE_PROFIT_2"
                        raw = (exit_price - entry) * contracts
                        trade_pnl = (current_position.get('booked_pnl', 0.0) + raw) - (contracts * exit_price * self.taker_fee)
                        closed = True
                    # Trailing Stop if TP1 was hit and pulls back 0.4% from high
                    elif current_position['tp1_hit']:
                        trail_price = current_position['highest'] * (1.0 - 0.004)
                        if low_price <= trail_price:
                            exit_price = trail_price
                            exit_reason = "TRAILING_STOP"
                            raw = (exit_price - entry) * contracts
                            trade_pnl = (current_position.get('booked_pnl', 0.0) + raw) - (contracts * exit_price * self.taker_fee)
                            closed = True

                elif side == "short":
                    # Check Stop Loss
                    if high_price >= current_position['sl']:
                        exit_price = current_position['sl']
                        exit_reason = "STOP_LOSS"
                        raw = (entry - exit_price) * contracts
                        trade_pnl = raw - (contracts * exit_price * self.taker_fee)
                        closed = True
                    # Check TP1
                    elif not current_position['tp1_hit'] and low_price <= current_position['tp1']:
                        current_position['tp1_hit'] = True
                        half = contracts / 2.0
                        raw_half = (entry - current_position['tp1']) * half
                        net_half = raw_half - (half * current_position['tp1'] * self.taker_fee)
                        balance += (margin / 2.0 + net_half)
                        current_position['margin'] -= margin / 2.0
                        current_position['contracts'] -= half
                        current_position['sl'] = entry  # Break-Even
                        current_position['booked_pnl'] = net_half
                        day_info['pnl'] += net_half
                    # Check TP2
                    elif low_price <= current_position['tp2']:
                        exit_price = current_position['tp2']
                        exit_reason = "TAKE_PROFIT_2"
                        raw = (entry - exit_price) * contracts
                        trade_pnl = (current_position.get('booked_pnl', 0.0) + raw) - (contracts * exit_price * self.taker_fee)
                        closed = True
                    # Trailing Stop
                    elif current_position['tp1_hit']:
                        trail_price = current_position['lowest'] * (1.0 + 0.004)
                        if high_price >= trail_price:
                            exit_price = trail_price
                            exit_reason = "TRAILING_STOP"
                            raw = (entry - exit_price) * contracts
                            trade_pnl = (current_position.get('booked_pnl', 0.0) + raw) - (contracts * exit_price * self.taker_fee)
                            closed = True

                if closed:
                    balance += max(0.0, current_position['margin'] + (trade_pnl - current_position.get('booked_pnl', 0.0)))
                    day_info['pnl'] += (trade_pnl - current_position.get('booked_pnl', 0.0))
                    
                    trades.append({
                        "entry_time": current_position['entry_time'],
                        "exit_time": row['datetime'],
                        "side": side,
                        "entry_price": entry,
                        "exit_price": round(exit_price, 2),
                        "margin": round(current_position['orig_margin'], 2),
                        "pnl": round(trade_pnl, 2),
                        "pnl_pct": round((trade_pnl / current_position['orig_margin']) * 100.0, 2),
                        "reason": exit_reason
                    })
                    current_position = None

                    # Check daily target & killswitch
                    day_pct = (day_info['pnl'] / max(day_info['start_bal'], 1.0)) * 100.0
                    if day_pct >= self.daily_target_pct:
                        day_info['locked'] = True
                    if day_pct <= -self.daily_max_loss_pct:
                        day_info['killed'] = True

            # Track Drawdown
            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / max(peak_balance, 1.0) * 100.0
            if dd > max_drawdown:
                max_drawdown = dd

            # 2. Check for New Entries if No Active Position
            if current_position is None:
                # Check if day is locked or killed
                if day_info['locked'] or day_info['killed']:
                    continue

                ema9 = row['ema9']
                ema21 = row['ema21']
                ema50 = row['ema50']
                rsi = row['rsi']
                vol = row['volume']
                vol_ma = row['vol_ma']

                has_vol = vol >= (vol_ma * 0.80)

                # Long Condition
                is_long = (
                    ema9 > ema21 and
                    curr_price >= ema50 and
                    curr_price >= ema9 and
                    48.0 <= rsi <= 68.0 and
                    has_vol and
                    ((prev['ema9'] <= prev['ema21'] and ema9 > ema21) or (prev['close'] <= prev['ema9'] and curr_price > ema9))
                )

                # Short Condition
                is_short = (
                    ema9 < ema21 and
                    curr_price <= ema50 and
                    curr_price <= ema9 and
                    32.0 <= rsi <= 52.0 and
                    has_vol and
                    ((prev['ema9'] >= prev['ema21'] and ema9 < ema21) or (prev['close'] >= prev['ema9'] and curr_price < ema9))
                )

                if is_long or is_short:
                    side = "long" if is_long else "short"
                    
                    # Position Sizing (1.5% Risk, 3x Leverage)
                    risk_amt = balance * (self.risk_per_trade_pct / 100.0)
                    notional = min(risk_amt / self.sl_pct, balance * self.leverage * 0.35)
                    margin = notional / self.leverage
                    contracts = notional / curr_price

                    fee = notional * self.taker_fee
                    balance -= (margin + fee)

                    sl = round(curr_price * (1.0 - self.sl_pct) if side == "long" else curr_price * (1.0 + self.sl_pct), 2)
                    tp1 = round(curr_price * (1.0 + self.tp1_pct) if side == "long" else curr_price * (1.0 - self.tp1_pct), 2)
                    tp2 = round(curr_price * (1.0 + self.tp2_pct) if side == "long" else curr_price * (1.0 - self.tp2_pct), 2)

                    current_position = {
                        "side": side,
                        "entry_time": row['datetime'],
                        "entry_price": curr_price,
                        "contracts": contracts,
                        "margin": margin,
                        "orig_margin": margin,
                        "sl": sl,
                        "tp1": tp1,
                        "tp2": tp2,
                        "tp1_hit": False,
                        "highest": curr_price,
                        "lowest": curr_price,
                        "booked_pnl": 0.0
                    }

        # Calculate Final Statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / max(total_trades, 1)) * 100.0

        total_gain = sum(t['pnl'] for t in winning_trades)
        total_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = round(total_gain / max(total_loss, 0.01), 2)

        net_profit = balance - self.initial_balance
        total_return_pct = (net_profit / self.initial_balance) * 100.0

        # Duration in days
        start_date = df['datetime'].iloc[0]
        end_date = df['datetime'].iloc[-1]
        days_tested = max((end_date - start_date).total_seconds() / 86400.0, 1.0)
        monthly_projected_return = (total_return_pct / days_tested) * 30.0

        return {
            "symbol": symbol,
            "days_tested": round(days_tested, 1),
            "start_date": str(start_date)[:10],
            "end_date": str(end_date)[:10],
            "initial_balance": self.initial_balance,
            "final_balance": round(balance, 2),
            "net_profit": round(net_profit, 2),
            "total_return_pct": round(total_return_pct, 2),
            "monthly_projected_return": round(monthly_projected_return, 2),
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": round(win_rate, 1),
            "profit_factor": profit_factor,
            "max_drawdown": round(max_drawdown, 2),
            "trades": trades[-20:]  # Last 20 trades sample
        }

if __name__ == "__main__":
    backtester = MexcBacktester(initial_balance=1000.0, leverage=3)
    # Test on BTC/USDT:USDT and ETH/USDT:USDT
    symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
    
    print("=" * 65)
    print("      MEXC SCALPER PRO - HISTORICAL BACKTESTING ENGINE")
    print("=" * 65)
    
    results = []
    for s in symbols:
        df = backtester.fetch_historical_candles(s, timeframe='5m', total_candles=2500)
        res = backtester.run_backtest(df, s)
        if res:
            results.append(res)
            print(f"\n--- Results for {s} ---")
            print(f"Period:             {res['start_date']} to {res['end_date']} ({res['days_tested']} Days)")
            print(f"Initial Balance:    ${res['initial_balance']}")
            print(f"Final Balance:      ${res['final_balance']} (Net: +${res['net_profit']})")
            print(f"Return for Period:  +{res['total_return_pct']}%")
            print(f"Projected Monthly:  +{res['monthly_projected_return']}%")
            print(f"Win Rate:           {res['win_rate']}% ({res['win_count']} Won / {res['loss_count']} Lost)")
            print(f"Profit Factor:      {res['profit_factor']}")
            print(f"Max Drawdown:       {res['max_drawdown']}%")
