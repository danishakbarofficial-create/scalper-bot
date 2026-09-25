import logging
import ccxt
import pandas as pd
from typing import Dict, Any, List, Optional

logger = logging.getLogger("mexc_exchange")

class MexcExchange:
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = ccxt.mexc({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
            }
        })
        self.markets_loaded = False

    def load_markets_safe(self):
        if not self.markets_loaded:
            try:
                self.exchange.load_markets()
                self.markets_loaded = True
                logger.info("MEXC swap markets loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load MEXC markets: {e}")

    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                "symbol": symbol,
                "last": float(ticker.get('last') or 0.0),
                "bid": float(ticker.get('bid') or ticker.get('last') or 0.0),
                "ask": float(ticker.get('ask') or ticker.get('last') or 0.0),
                "high": float(ticker.get('high') or 0.0),
                "low": float(ticker.get('low') or 0.0),
                "volume": float(ticker.get('baseVolume') or 0.0),
                "change_24h": float(ticker.get('percentage') or 0.0)
            }
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None

    def fetch_ohlcv_df(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> Optional[pd.DataFrame]:
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not raw or len(raw) == 0:
                return None
            df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return None

    def fetch_balance(self) -> Dict[str, float]:
        if not self.api_key or not self.api_secret:
            return {"free": 0.0, "used": 0.0, "total": 0.0}
        try:
            bal = self.exchange.fetch_balance()
            usdt = bal.get('USDT', {})
            return {
                "free": float(usdt.get('free') or 0.0),
                "used": float(usdt.get('used') or 0.0),
                "total": float(usdt.get('total') or 0.0)
            }
        except Exception as e:
            logger.error(f"Error fetching live balance: {e}")
            return {"free": 0.0, "used": 0.0, "total": 0.0}

    def set_leverage(self, leverage: int, symbol: str) -> bool:
        if not self.api_key or not self.api_secret:
            return False
        try:
            self.exchange.set_leverage(leverage, symbol)
            logger.info(f"Leverage set to {leverage}x for {symbol}")
            return True
        except Exception as e:
            logger.warning(f"Could not set leverage on MEXC for {symbol}: {e}")
            return False

    def create_order(self, symbol: str, side: str, order_type: str, amount: float, price: Optional[float] = None, params: Optional[dict] = None) -> Optional[dict]:
        """
        side: 'buy' or 'sell'
        order_type: 'market' or 'limit'
        """
        if not self.api_key or not self.api_secret:
            logger.error("Live order rejected: Missing MEXC API Key / Secret")
            return None
        try:
            params = params or {}
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params
            )
            logger.info(f"Live order created: {side} {amount} {symbol} ({order_type})")
            return order
        except Exception as e:
            logger.error(f"Order placement failed on {symbol}: {e}")
            return None

    def create_post_only_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Optional[dict]:
        """
        Guarantees 0.00% Maker Fee execution on MEXC Futures via postOnly flag.
        """
        params = {"postOnly": True}
        return self.create_order(symbol=symbol, side=side, order_type="limit", amount=amount, price=price, params=params)

    def create_native_stop_loss(self, symbol: str, side: str, amount: float, stop_price: float) -> Optional[dict]:
        """
        Places native exchange-level stop loss trigger directly on MEXC servers.
        Protects capital even if the bot or server loses internet connection!
        """
        if not self.api_key or not self.api_secret:
            return None
        try:
            params = {
                "stopPrice": stop_price,
                "reduceOnly": True
            }
            order = self.exchange.create_order(
                symbol=symbol,
                type="stop_market",
                side=side,
                amount=amount,
                params=params
            )
            logger.info(f"🛡️ Native MEXC Stop-Loss placed @ ${stop_price} for {symbol} ({side})")
            return order
        except Exception as e:
            logger.warning(f"Could not place native stop-loss on MEXC for {symbol}: {e}")
            return None

    def cancel_all_symbol_orders(self, symbol: str):
        if not self.api_key or not self.api_secret:
            return
        try:
            self.exchange.cancel_all_orders(symbol)
            logger.info(f"Cancelled all open trigger orders for {symbol}")
        except Exception as e:
            logger.warning(f"Error cancelling orders on {symbol}: {e}")

    def fetch_positions(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if not self.api_key or not self.api_secret:
            return []
        try:
            positions = self.exchange.fetch_positions(symbols)
            active = []
            for p in positions:
                contracts = float(p.get('contracts') or 0.0)
                if contracts > 0:
                    active.append({
                        "id": p.get('id', p.get('symbol')),
                        "symbol": p.get('symbol'),
                        "side": p.get('side', '').lower(),
                        "contracts": contracts,
                        "entry_price": float(p.get('entryPrice') or 0.0),
                        "mark_price": float(p.get('markPrice') or 0.0),
                        "unrealized_pnl": float(p.get('unrealizedPnl') or 0.0),
                        "leverage": int(p.get('leverage') or 1),
                        "liquidation_price": float(p.get('liquidationPrice') or 0.0),
                    })
            return active
        except Exception as e:
            logger.error(f"Error fetching live positions: {e}")
            return []
