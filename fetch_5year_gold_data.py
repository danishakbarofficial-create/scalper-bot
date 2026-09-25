import os
import sys
import time
import datetime
import ccxt
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "historical"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DATA_DIR / "PAXG_USDT_15m_5years.csv"

def fetch_5year_gold():
    if OUTPUT_FILE.exists():
        print(f"File {OUTPUT_FILE.name} already exists. Checking row count...")
        df_existing = pd.read_csv(OUTPUT_FILE)
        print(f"Loaded existing dataset with {len(df_existing):,} rows.")
        return df_existing

    print("Connecting to Binance API for 5-Year historical PAXG/USDT (Gold) 15m dataset...")
    exchange = ccxt.binance({'enableRateLimit': True})

    # End at latest timestamp available in 1-year data or now
    ref_file = DATA_DIR / "ETH_USDT_15m_5years.csv"
    if ref_file.exists():
        ref_df = pd.read_csv(ref_file)
        since_ms = int(ref_df['timestamp'].iloc[0])
        end_ms = int(ref_df['timestamp'].iloc[-1])
    else:
        now_ms = int(time.time() * 1000)
        since_ms = now_ms - int(5 * 365.25 * 24 * 3600 * 1000)
        end_ms = now_ms

    start_dt = datetime.datetime.fromtimestamp(since_ms / 1000, datetime.timezone.utc)
    end_dt = datetime.datetime.fromtimestamp(end_ms / 1000, datetime.timezone.utc)
    print(f"Date range: {start_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    all_candles = []
    curr_since = since_ms
    limit = 1000
    batch_num = 0
    start_time = time.time()

    while curr_since <= end_ms:
        try:
            candles = exchange.fetch_ohlcv('PAXG/USDT', timeframe='15m', since=curr_since, limit=limit)
            if not candles:
                print("No more candles returned.")
                break

            all_candles.extend(candles)
            last_ts = candles[-1][0]
            last_dt = datetime.datetime.fromtimestamp(last_ts / 1000, datetime.timezone.utc)

            batch_num += 1
            if batch_num % 25 == 0 or last_ts >= end_ms:
                progress = min(100.0, ((last_ts - since_ms) / max(end_ms - since_ms, 1)) * 100.0)
                print(f"[{progress:5.1f}%] Fetched {len(all_candles):,} candles (Up to {last_dt.strftime('%Y-%m-%d %H:%M')})...")

            if last_ts <= curr_since:
                curr_since += 15 * 60 * 1000 * limit
            else:
                curr_since = last_ts + (15 * 60 * 1000)

            time.sleep(0.05)
        except Exception as e:
            print(f"Fetch error: {e}. Retrying in 2 seconds...")
            time.sleep(2)

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df = df[(df['timestamp'] >= since_ms) & (df['timestamp'] <= end_ms + 15*60*1000)]
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFetch complete in {round(time.time() - start_time, 1)}s.")
    print(f"Saved {len(df):,} unique 15m candles of Gold to {OUTPUT_FILE.name} ({round(OUTPUT_FILE.stat().st_size / (1024*1024), 2)} MB)")
    return df

if __name__ == "__main__":
    fetch_5year_gold()
