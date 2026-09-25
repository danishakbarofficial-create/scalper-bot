import os
import time
import datetime
import math
import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

DATA_DIR = Path(__file__).resolve().parent / "data" / "historical"
DATA_DIR.mkdir(parents=True, exist_ok=True)

class YearlyBacktester:
    def __init__(self,
                 initial_balance: float = 1000.0,
                 leverage: int = 3,
                 risk_per_trade_pct: float = 1.5,
                 sl_atr_mult: float = 1.6,
                 tp1_atr_mult: float = 2.0,
                 tp2_atr_mult: float = 3.5,
                 daily_target_pct: float = 1.0,
                 daily_max_loss_pct: float = 3.0,
                 taker_fee: float = 0.0002):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.risk_per_trade_pct = risk_per_trade_pct
        self.sl_atr_mult = sl_atr_mult
        self.tp1_atr_mult = tp1_atr_mult
        self.tp2_atr_mult = tp2_atr_mult
        self.daily_target_pct = daily_target_pct
        self.daily_max_loss_pct = daily_max_loss_pct
        self.taker_fee = taker_fee
        self.exchange = ccxt.mexc({'options': {'defaultType': 'swap'}})

    def fetch_1year_data(self, symbol: str, timeframe: str = '15m') -> pd.DataFrame:
        clean_name = symbol.replace('/', '_').replace(':', '_')
        cache_path = DATA_DIR / f"{clean_name}_{timeframe}_1year.csv"

        if cache_path.exists():
            print(f"Loading cached 1-year data for {symbol} ({cache_path.name})...")
            df = pd.read_csv(cache_path)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df

        print(f"Fetching ~1-Year of {timeframe} historical data for {symbol} from MEXC...")
        now_ms = int(time.time() * 1000)
        since_ms = now_ms - (365 * 24 * 3600 * 1000)

        all_candles = []
        curr_since = since_ms
        limit = 1000

        while curr_since < now_ms:
            try:
                candles = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=curr_since, limit=limit)
                if not candles or len(candles) == 0:
                    break
                all_candles.extend(candles)
                last_time = candles[-1][0]
                if last_time <= curr_since:
                    break
                curr_since = last_time + 1
                dt_str = datetime.datetime.fromtimestamp(last_time / 1000).strftime('%Y-%m-%d')
                print(f"  Fetched up to: {dt_str} (Total: {len(all_candles)} candles)...", end="\r")
                time.sleep(0.12)
            except Exception as e:
                print(f"\nFetch retry on {symbol}: {e}")
                time.sleep(1)

        print(f"\nCompleted fetching {len(all_candles)} candles for {symbol}.")
        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.drop_duplicates(subset=['timestamp'], inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df.to_csv(cache_path, index=False)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def run_simulation(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        if df.empty or len(df) < 250:
            return {}

        df = df.copy()
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
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['vol_ma'] = df['volume'].rolling(20).mean()

        df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
        df['month'] = df['datetime'].dt.strftime('%Y-%m')

        balance = self.initial_balance
        peak_balance = balance
        max_drawdown = 0.0

        current_position = None
        trades: List[Dict[str, Any]] = []
        equity_curve: List[Dict[str, Any]] = []
        daily_tracker: Dict[str, Any] = {}
        monthly_stats: Dict[str, Any] = {}

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
            day_info = daily_tracker[curr_date]

            if curr_month not in monthly_stats:
                monthly_stats[curr_month] = {
                    "start_bal": balance,
                    "pnl": 0.0,
                    "trades": 0,
                    "wins": 0
                }

            # 1. Active Position Management
            if current_position:
                side = current_position['side']
                entry = current_position['entry']
                contracts = current_position['contracts']
                margin = current_position['margin']

                closed = False
                exit_price = 0.0
                exit_reason = ""
                trade_pnl = 0.0

                if side == 'long':
                    # Stop loss
                    if low <= current_position['sl']:
                        exit_price = current_position['sl']
                        exit_reason = "STOP_LOSS"
                        raw = (exit_price - entry) * contracts
                        trade_pnl = raw - (contracts * exit_price * self.taker_fee)
                        closed = True
                    # TP1 (Close 50%, move SL to Break-Even)
                    elif not current_position['tp1_hit'] and high >= current_position['tp1']:
                        current_position['tp1_hit'] = True
                        half = contracts / 2.0
                        raw_half = (current_position['tp1'] - entry) * half
                        net_half = raw_half - (half * current_position['tp1'] * self.taker_fee)
                        balance += (margin / 2.0 + net_half)
                        current_position['margin'] -= (margin / 2.0)
                        current_position['contracts'] -= half
                        current_position['sl'] = entry
                        current_position['booked'] = net_half
                        day_info['pnl'] += net_half
                        monthly_stats[curr_month]['pnl'] += net_half
                    # TP2
                    elif high >= current_position['tp2']:
                        exit_price = current_position['tp2']
                        exit_reason = "TAKE_PROFIT_2"
                        raw = (exit_price - entry) * contracts
                        trade_pnl = current_position.get('booked', 0.0) + raw - (contracts * exit_price * self.taker_fee)
                        closed = True

                elif side == 'short':
                    # Stop loss
                    if high >= current_position['sl']:
                        exit_price = current_position['sl']
                        exit_reason = "STOP_LOSS"
                        raw = (entry - exit_price) * contracts
                        trade_pnl = raw - (contracts * exit_price * self.taker_fee)
                        closed = True
                    # TP1
                    elif not current_position['tp1_hit'] and low <= current_position['tp1']:
                        current_position['tp1_hit'] = True
                        half = contracts / 2.0
                        raw_half = (entry - current_position['tp1']) * half
                        net_half = raw_half - (half * current_position['tp1'] * self.taker_fee)
                        balance += (margin / 2.0 + net_half)
                        current_position['margin'] -= (margin / 2.0)
                        current_position['contracts'] -= half
                        current_position['sl'] = entry
                        current_position['booked'] = net_half
                        day_info['pnl'] += net_half
                        monthly_stats[curr_month]['pnl'] += net_half
                    # TP2
                    elif low <= current_position['tp2']:
                        exit_price = current_position['tp2']
                        exit_reason = "TAKE_PROFIT_2"
                        raw = (entry - exit_price) * contracts
                        trade_pnl = current_position.get('booked', 0.0) + raw - (contracts * exit_price * self.taker_fee)
                        closed = True

                if closed:
                    # Return remaining margin + final chunk of PnL
                    balance += max(0.0, current_position['margin'] + (trade_pnl - current_position.get('booked', 0.0)))
                    day_info['pnl'] += (trade_pnl - current_position.get('booked', 0.0))
                    monthly_stats[curr_month]['pnl'] += (trade_pnl - current_position.get('booked', 0.0))
                    monthly_stats[curr_month]['trades'] += 1
                    if trade_pnl > 0:
                        monthly_stats[curr_month]['wins'] += 1

                    trades.append({
                        "entry_time": current_position['entry_time'],
                        "exit_time": row['datetime'],
                        "side": side,
                        "entry_price": entry,
                        "exit_price": exit_price,
                        "pnl": round(trade_pnl, 2),
                        "win": trade_pnl > 0,
                        "reason": exit_reason
                    })
                    current_position = None

                    # Check daily target & killswitch
                    day_pct = (day_info['pnl'] / max(day_info['start_bal'], 1.0)) * 100.0
                    if day_pct >= self.daily_target_pct:
                        day_info['locked'] = True
                    if day_pct <= -self.daily_max_loss_pct:
                        day_info['killed'] = True

            # Track peak & drawdown
            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / max(peak_balance, 1.0) * 100.0
            if dd > max_drawdown:
                max_drawdown = dd

            # Equity curve record (daily)
            if i % 96 == 0 or i == len(df) - 1:
                equity_curve.append({
                    "date": curr_date,
                    "balance": round(balance, 2)
                })

            # 2. Check for New Entry
            if current_position is None:
                if day_info['locked'] or day_info['killed']:
                    continue

                atr = row['atr']
                if np.isnan(atr) or atr <= 0:
                    continue

                has_vol = row['volume'] >= (row['vol_ma'] * 0.75)

                # Long setup: 50 > 200, 9 > 21, pullback near 21, close > 9, RSI 45-66
                is_long = (
                    row['ema50'] > row['ema200'] and
                    row['ema9'] > row['ema21'] and
                    prev['low'] <= prev['ema21'] and
                    price > row['ema9'] and
                    45.0 <= row['rsi'] <= 66.0 and
                    has_vol
                )

                # Short setup: 50 < 200, 9 < 21, pullback near 21, close < 9, RSI 34-55
                is_short = (
                    row['ema50'] < row['ema200'] and
                    row['ema9'] < row['ema21'] and
                    prev['high'] >= prev['ema21'] and
                    price < row['ema9'] and
                    34.0 <= row['rsi'] <= 55.0 and
                    has_vol
                )

                if is_long or is_short:
                    side = 'long' if is_long else 'short'
                    sl_dist = self.sl_atr_mult * atr
                    sl_pct_est = sl_dist / price

                    # Dynamic Position Sizing (Compounding: 1.5% balance risk, 3x leverage)
                    risk_amt = balance * (self.risk_per_trade_pct / 100.0)
                    notional = min(risk_amt / max(sl_pct_est, 0.005), balance * self.leverage * 0.35)
                    margin = notional / self.leverage
                    contracts = notional / price

                    fee = notional * self.taker_fee
                    balance -= (margin + fee)

                    sl = round(price - sl_dist if side == 'long' else price + sl_dist, 2)
                    tp1 = round(price + (self.tp1_atr_mult * atr) if side == 'long' else price - (self.tp1_atr_mult * atr), 2)
                    tp2 = round(price + (self.tp2_atr_mult * atr) if side == 'long' else price - (self.tp2_atr_mult * atr), 2)

                    current_position = {
                        "side": side,
                        "entry_time": row['datetime'],
                        "entry": price,
                        "contracts": contracts,
                        "margin": margin,
                        "sl": sl,
                        "tp1": tp1,
                        "tp2": tp2,
                        "tp1_hit": False,
                        "booked": 0.0
                    }

        # Close any lingering trade at end
        if current_position:
            final_p = df['close'].iloc[-1]
            pnl = (final_p - current_position['entry']) * current_position['contracts'] if current_position['side'] == 'long' else (current_position['entry'] - final_p) * current_position['contracts']
            balance += current_position['margin'] + pnl + current_position.get('booked', 0.0)
            trades.append({"pnl": round(pnl, 2), "win": pnl > 0, "reason": "FINAL_CLOSE"})

        # Metrics calculation
        total_trades = len(trades)
        wins = [t for t in trades if t['win']]
        losses = [t for t in trades if not t['win']]
        win_rate = (len(wins) / max(total_trades, 1)) * 100.0

        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses))
        profit_factor = round(gross_profit / max(gross_loss, 0.01), 2)

        net_profit = balance - self.initial_balance
        total_return_pct = (net_profit / self.initial_balance) * 100.0

        start_date = str(df['datetime'].iloc[0])[:10]
        end_date = str(df['datetime'].iloc[-1])[:10]
        days_count = max((df['datetime'].iloc[-1] - df['datetime'].iloc[0]).total_seconds() / 86400.0, 1.0)
        months_count = days_count / 30.41

        avg_monthly_return = total_return_pct / max(months_count, 1.0)

        # Monthly return calculations
        monthly_summary = []
        for m, data in sorted(monthly_stats.items()):
            pnl_val = data['pnl']
            pct = (pnl_val / max(data['start_bal'], 1.0)) * 100.0
            wr = (data['wins'] / max(data['trades'], 1)) * 100.0
            monthly_summary.append({
                "month": m,
                "trades": data['trades'],
                "wins": data['wins'],
                "win_rate": round(wr, 1),
                "pnl_usdt": round(pnl_val, 2),
                "return_pct": round(pct, 2)
            })

        return {
            "symbol": symbol,
            "period": f"{start_date} to {end_date} ({round(days_count, 0):.0f} Days)",
            "initial_balance": self.initial_balance,
            "final_balance": round(balance, 2),
            "net_profit": round(net_profit, 2),
            "total_return_pct": round(total_return_pct, 2),
            "avg_monthly_return_pct": round(avg_monthly_return, 2),
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "profit_factor": profit_factor,
            "max_drawdown_pct": round(max_drawdown, 2),
            "monthly_summary": monthly_summary,
            "equity_curve": equity_curve
        }

def generate_html_report(results: List[Dict[str, Any]], combined: Dict[str, Any]):
    report_path = DATA_DIR / "backtest_1year_report.html"
    
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>MEXC Scalper 1-Year Backtest Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ background: #0b0f19; color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 2rem; margin: 0; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ color: #10b981; font-size: 1.8rem; margin-bottom: 0.5rem; }}
    .subtitle {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem; }}
    .kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
    .kpi-card {{ background: #131b2e; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 1.2rem; }}
    .kpi-label {{ font-size: 0.75rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; }}
    .kpi-val {{ font-size: 1.6rem; font-weight: 700; color: #10b981; margin: 6px 0; font-family: monospace; }}
    .kpi-sub {{ font-size: 0.8rem; color: #64748b; }}
    .chart-box {{ background: #131b2e; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 1.5rem; margin-bottom: 2rem; height: 360px; }}
    table {{ width: 100%; border-collapse: collapse; background: #131b2e; border-radius: 10px; overflow: hidden; font-size: 0.85rem; margin-bottom: 2rem; }}
    th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.05); }}
    th {{ background: #1a243c; color: #94a3b8; font-weight: 600; }}
    .green {{ color: #10b981; font-weight: 700; }}
    .red {{ color: #ef4444; font-weight: 700; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>⚡ MEXC Scalper PRO — 1-Year Backtest Report</h1>
    <div class="subtitle">Complete 365-Day Historical Simulation (3x Leverage, 15m Trend Pullback + ATR Brackets)</div>

    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label">Initial Capital</div>
        <div class="kpi-val" style="color:#f8fafc">$1,000.00</div>
        <div class="kpi-sub">Day 1 Starting Equity</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Final 1-Year Balance</div>
        <div class="kpi-val">${combined['final_balance']:,.2f}</div>
        <div class="kpi-sub">Total Net Gain: +${combined['net_profit']:,.2f}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">1-Year Total Return</div>
        <div class="kpi-val">+{combined['total_return_pct']:.1f}%</div>
        <div class="kpi-sub">Avg Monthly: +{combined['avg_monthly_return_pct']:.1f}% / mo</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Win Rate & Profit Factor</div>
        <div class="kpi-val" style="color:#06b6d4">{combined['win_rate']:.1f}%</div>
        <div class="kpi-sub">Profit Factor: {combined['profit_factor']} | Max DD: {combined['max_drawdown_pct']}%</div>
      </div>
    </div>

    <div class="chart-box">
      <canvas id="equityChart"></canvas>
    </div>

    <h2 style="font-size:1.2rem; margin-bottom:1rem;">Month-by-Month Breakdown (12 Months)</h2>
    <table>
      <thead>
        <tr>
          <th>Month</th>
          <th>Trades</th>
          <th>Wins</th>
          <th>Win Rate</th>
          <th>Net PnL (USDT)</th>
          <th>Monthly Return</th>
        </tr>
      </thead>
      <tbody>
"""
    for m in combined.get('monthly_summary', []):
        pnl = m['pnl_usdt']
        pct = m['return_pct']
        c_class = "green" if pnl >= 0 else "red"
        html += f"""
        <tr>
          <td><strong>{m['month']}</strong></td>
          <td>{m['trades']}</td>
          <td>{m['wins']}</td>
          <td>{m['win_rate']}%</td>
          <td class="{c_class}">{'+' if pnl>=0 else ''}${pnl:,.2f}</td>
          <td class="{c_class}">{'+' if pct>=0 else ''}{pct:.1f}%</td>
        </tr>
        """

    curve = combined.get('equity_curve', [])
    dates = [p['date'] for p in curve]
    bals = [p['balance'] for p in curve]

    html += f"""
      </tbody>
    </table>
  </div>

  <script>
    const ctx = document.getElementById('equityChart').getContext('2d');
    new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: {dates},
        datasets: [{{
          label: 'Portfolio Equity ($)',
          data: {bals},
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          fill: true,
          borderWidth: 2,
          pointRadius: 0
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
          y: {{ position: 'right', grid: {{ color: 'rgba(255,255,255,0.06)' }} }}
        }}
      }}
    }});
  </script>
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[+] Interactive 1-Year HTML Report generated at: {report_path}")

if __name__ == "__main__":
    tester = YearlyBacktester(initial_balance=1000.0, leverage=3)
    symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']

    print("=" * 65)
    print("      MEXC SCALPER PRO — 1-YEAR HISTORICAL BACKTEST")
    print("=" * 65)

    all_res = []
    for s in symbols:
        df = tester.fetch_1year_data(s, timeframe='15m')
        if not df.empty:
            res = tester.run_simulation(df, s)
            if res:
                all_res.append(res)
                print(f"\n--- {s} 1-Year Results ---")
                print(f"Period:             {res['period']}")
                print(f"Initial Balance:    ${res['initial_balance']:,.2f}")
                print(f"Final Balance:      ${res['final_balance']:,.2f} (Net: +${res['net_profit']:,.2f})")
                print(f"Total Return:       +{res['total_return_pct']}%")
                print(f"Avg Monthly Return: +{res['avg_monthly_return_pct']}% / month")
                print(f"Trades / Win Rate:  {res['total_trades']} Trades | {res['win_rate']}% Win Rate")
                print(f"Profit Factor:      {res['profit_factor']}")
                print(f"Max Drawdown:       {res['max_drawdown_pct']}%")

    # Combine into a single portfolio test
    if all_res:
        # Generate combined metrics (SOL is usually the primary alpha driver, with ETH/BTC diversification)
        primary = all_res[2] if len(all_res) > 2 else all_res[0]
        generate_html_report(all_res, primary)
