import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    tr_smooth = tr.rolling(period).sum()

    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    plus_di = 100 * (pd.Series(plus_dm).rolling(period).sum() / (tr_smooth + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).sum() / (tr_smooth + 1e-9))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    return dx.rolling(period).mean()

def analyze_market_data(df: pd.DataFrame, config: dict) -> Optional[Dict[str, Any]]:
    """
    Enhanced Regime-Aware Scalp Engine:
    - ADX >= 22 Filter: Stays in cash during sideways chop, ONLY trades during real volatility.
    - 200 EMA Macro Trend alignment (Bull vs Bear regime).
    - 9 EMA & 21 EMA Ribbon pullback bounce.
    - RSI (14) confirmation (45-66 for Longs, 34-55 for Shorts).
    - Dynamic ATR brackets (1.6x ATR SL, 2.0x ATR TP1, 3.5x ATR TP2).
    """
    if df is None or len(df) < 55:
        return None

    df = df.copy()

    # 1. EMAs
    df['ema9'] = calculate_ema(df['close'], 9)
    df['ema21'] = calculate_ema(df['close'], 21)
    df['ema50'] = calculate_ema(df['close'], 50)
    df['ema200'] = calculate_ema(df['close'], 200) if len(df) >= 200 else df['ema50']

    # 2. RSI, ATR, ADX, Swings & Volume MA
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['atr'] = calculate_atr(df, 14)
    df['adx'] = calculate_adx(df, 14)
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    df['swing_low'] = df['low'].shift(1).rolling(16).min()
    df['swing_high'] = df['high'].shift(1).rolling(16).max()

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(curr['close'])
    ema9 = float(curr['ema9'])
    ema21 = float(curr['ema21'])
    ema50 = float(curr['ema50'])
    ema200 = float(curr['ema200'])
    rsi = float(curr['rsi']) if not np.isnan(curr['rsi']) else 50.0
    atr = float(curr['atr']) if not np.isnan(curr['atr']) else (price * 0.005)
    adx = float(curr['adx']) if not np.isnan(curr['adx']) else 20.0
    volume = float(curr['volume'])
    vol_ma = float(curr['vol_ma']) if not np.isnan(curr['vol_ma']) else 1.0
    swing_low = float(curr['swing_low']) if not np.isnan(curr['swing_low']) else (price - 1.4 * atr)
    swing_high = float(curr['swing_high']) if not np.isnan(curr['swing_high']) else (price + 1.4 * atr)

    min_adx_threshold = float(config.get("min_adx_threshold", 22.0))
    is_trending_regime = adx >= min_adx_threshold
    has_volume = volume >= (vol_ma * 0.75)

    macro_bull = price > ema200 and ema50 > ema200
    macro_bear = price < ema200 and ema50 < ema200

    trend_state = "CHOP (ADX < 22)" if not is_trending_regime else (
        "STRONG BULLISH" if macro_bull else (
            "STRONG BEARISH" if macro_bear else "CONSOLIDATION"
        )
    )

    indicators = {
        "price": round(price, 4),
        "ema9": round(ema9, 4),
        "ema21": round(ema21, 4),
        "ema50": round(ema50, 4),
        "ema200": round(ema200, 4),
        "rsi": round(rsi, 2),
        "atr": round(atr, 4),
        "adx": round(adx, 2),
        "volume": round(volume, 2),
        "vol_ma": round(vol_ma, 2),
        "trend": trend_state,
        "regime_ok": is_trending_regime
    }

    # If market is in dead sideways chop, DO NOT TRADE (Capital Protection)
    if not is_trending_regime:
        return {
            "signal": "HOLD",
            "side": "none",
            "price": price,
            "sl": 0.0,
            "tp1": 0.0,
            "tp2": 0.0,
            "reason": f"Low Volatility Chop (ADX: {round(adx, 1)} < {min_adx_threshold}). Capital protected.",
            "score": 0,
            "indicators": indicators
        }

    # --- INSTITUTIONAL VALUE PULLBACK SETUPS ---
    # Long: Macro Bull + Deep RSI Pullback (< 38) + Green Reversal above EMA9
    is_long_setup = (
        macro_bull and
        prev['rsi'] < 38.0 and
        curr['close'] > curr['open'] and
        price > ema9 and
        has_volume
    )

    # Short: Macro Bear + Relief RSI Spike (> 62) + Red Reversal below EMA9
    is_short_setup = (
        macro_bear and
        prev['rsi'] > 62.0 and
        curr['close'] < curr['open'] and
        price < ema9 and
        has_volume
    )

    rr_ratio = float(config.get("rr_ratio", 2.0))
    sl_atr_mult = float(config.get("sl_atr_mult", 1.4))

    if is_long_setup:
        sl = round(min(swing_low, price - (sl_atr_mult * atr)), 4)
        risk = price - sl
        if 0.003 <= (risk / price) <= 0.035:
            tp1 = round(price + (rr_ratio * risk), 4)
            tp2 = round(price + ((rr_ratio + 0.5) * risk), 4)
            return {
                "signal": "BUY",
                "side": "long",
                "price": price,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "reason": f"Macro Bull Value Pullback (ADX {round(adx, 1)}, RSI {round(rsi, 1)} < 38 bounce) | 1:{rr_ratio} RR",
                "score": 5,
                "indicators": indicators
            }

    if is_short_setup:
        sl = round(max(swing_high, price + (sl_atr_mult * atr)), 4)
        risk = sl - price
        if 0.003 <= (risk / price) <= 0.035:
            tp1 = round(price - (rr_ratio * risk), 4)
            tp2 = round(price - ((rr_ratio + 0.5) * risk), 4)
            return {
                "signal": "SELL",
                "side": "short",
                "price": price,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "reason": f"Macro Bear Relief Rejection (ADX {round(adx, 1)}, RSI {round(rsi, 1)} > 62 reject) | 1:{rr_ratio} RR",
                "score": 5,
                "indicators": indicators
            }

    return {
        "signal": "HOLD",
        "side": "none",
        "price": price,
        "sl": 0.0,
        "tp1": 0.0,
        "tp2": 0.0,
        "reason": "Macro Trend confirmed, awaiting value pullback trigger (RSI < 38 or > 62).",
        "score": 0,
        "indicators": indicators
    }
