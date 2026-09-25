import os
import sys
import time
import datetime
import ccxt
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "historical"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DATA_DIR / "PAXG_USDT_USDT_15m_1year.csv"

def fetch_gold_data():
    print("Connecting to Binance for PAXG/USDT (Gold) 15m historical candles...")
    exchange = ccxt.binance({'enableRateLimit': True})

    # Read reference ETH dataset to match exact date range
    ref_file = DATA_DIR / "ETH_USDT_USDT_15m_1year.csv"
    if ref_file.exists():
        ref_df = pd.read_csv(ref_file)
        since_ms = int(ref_df['timestamp'].iloc[0])
        end_ms = int(ref_df['timestamp'].iloc[-1])
    else:
        now_ms = int(time.time() * 1000)
        since_ms = now_ms - int(365 * 24 * 3600 * 1000)
        end_ms = now_ms

    start_dt = datetime.datetime.fromtimestamp(since_ms / 1000, datetime.timezone.utc)
    end_dt = datetime.datetime.fromtimestamp(end_ms / 1000, datetime.timezone.utc)
    print(f"Target range: {start_dt} to {end_dt}")

    all_candles = []
    curr_since = since_ms
    limit = 1000

    while curr_since <= end_ms:
        try:
            candles = exchange.fetch_ohlcv('PAXG/USDT', timeframe='15m', since=curr_since, limit=limit)
            if not candles:
                break
            all_candles.extend(candles)
            last_ts = candles[-1][0]
            if last_ts <= curr_since:
                curr_since += 15 * 60 * 1000 * limit
            else:
                curr_since = last_ts + (15 * 60 * 1000)
            time.sleep(0.05)
        except Exception as e:
            print(f"Error: {e}, retrying...")
            time.sleep(1)

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    # Filter to exact range
    df = df[(df['timestamp'] >= since_ms) & (df['timestamp'] <= end_ms + 15*60*1000)]
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved {len(df):,} candles of PAXG/USDT to {OUTPUT_FILE.name}")

if __name__ == "__main__":
    fetch_gold_data()
