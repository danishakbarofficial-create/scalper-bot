import os
import json
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config.json"
ENV_FILE = BASE_DIR / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

DEFAULT_CONFIG = {
    "trading_mode": "PAPER",  # "PAPER" or "LIVE"
    "mexc_api_key": os.getenv("MEXC_API_KEY", ""),
    "mexc_api_secret": os.getenv("MEXC_API_SECRET", ""),
    "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT"],
    "timeframe": "1m",
    "leverage": 3,
    "risk_per_trade_percent": 1.5,
    "max_open_positions": 2,
    "daily_profit_target_percent": 1.0,
    "daily_max_loss_percent": 3.0,
    "stop_loss_percent": 0.8,
    "take_profit_1_percent": 1.2,
    "take_profit_2_percent": 2.4,
    "use_trailing_stop": True,
    "trailing_stop_callback_percent": 0.4,
    "cooldown_seconds": 180,
    "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    "paper_initial_balance": 1000.0,
    "bot_active": False,
    "enable_session_filter": True,
    "session_start_utc": 12,
    "session_end_utc": 21
}

def load_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                config.update(saved)
        except Exception as e:
            print(f"Error loading config.json: {e}")
    return config

def save_config(new_config: dict) -> dict:
    current = load_config()
    current.update(new_config)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=4)
    return current
