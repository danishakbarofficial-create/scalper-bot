import os
import sys
import time
import datetime
import ccxt
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "historical"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "ETH_USDT_15m_5years.csv"

def fetch_5year_eth_data():
    if OUTPUT_FILE.exists():
        print(f"File {OUTPUT_FILE.name} already exists. Checking row count...")
        df_existing = pd.read_csv(OUTPUT_FILE)
        print(f"Loaded existing dataset with {len(df_existing)} rows.")
        return df_existing

    print("Initializing Binance API client for 5-Year historical ETH/USDT 15m dataset...")
    exchange = ccxt.binance({'enableRateLimit': True})

    now_ms = int(time.time() * 1000)
    # Exactly 5 years ago (5 * 365.25 days)
    since_ms = now_ms - int(5 * 365.25 * 24 * 3600 * 1000)

    start_dt = datetime.datetime.fromtimestamp(since_ms / 1000, datetime.timezone.utc)
    end_dt = datetime.datetime.fromtimestamp(now_ms / 1000, datetime.timezone.utc)
    print(f"Date range: {start_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    all_candles = []
    curr_since = since_ms
    limit = 1000
    batch_num = 0

    start_fetch_time = time.time()

    while curr_since < now_ms:
        try:
            candles = exchange.fetch_ohlcv('ETH/USDT', timeframe='15m', since=curr_since, limit=limit)
            if not candles:
                print("No more candles returned.")
                break

            all_candles.extend(candles)
            last_timestamp = candles[-1][0]
            last_dt = datetime.datetime.fromtimestamp(last_timestamp / 1000, datetime.timezone.utc)

            batch_num += 1
            if batch_num % 15 == 0 or last_timestamp >= now_ms:
                progress = min(100.0, ((last_timestamp - since_ms) / max(now_ms - since_ms, 1)) * 100.0)
                print(f"[{progress:5.1f}%] Fetched {len(all_candles):,} candles (Up to {last_dt.strftime('%Y-%m-%d %H:%M')})...")

            # Check if reached end
            if last_timestamp <= curr_since:
                # To prevent infinite loop if exchange returns same timestamp
                curr_since += 15 * 60 * 1000 * limit
            else:
                curr_since = last_timestamp + (15 * 60 * 1000)

            time.sleep(0.05)

        except Exception as e:
            print(f"Fetch error: {e}. Retrying in 2 seconds...")
            time.sleep(2)

    print(f"\nFetch complete in {round(time.time() - start_fetch_time, 1)}s. Total raw candles: {len(all_candles):,}")

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df):,} unique 15m candles to {OUTPUT_FILE} ({round(OUTPUT_FILE.stat().st_size / (1024*1024), 2)} MB)")
    return df

if __name__ == "__main__":
    fetch_5year_eth_data()
