import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import load_config
from backend.exchange import MexcExchange
from backend.strategy import analyze_market_data, calculate_ema, calculate_rsi
from backend.risk_manager import RiskManager
from backend.paper_trader import PaperTrader

class TestMexcScalper(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_indicators(self):
        # Create synthetic OHLCV data
        prices = [100.0 + i * 0.5 + np.sin(i / 5.0) * 2.0 for i in range(100)]
        df = pd.DataFrame({
            "timestamp": range(100),
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1000.0 + i * 10 for i in range(100)]
        })

        ema9 = calculate_ema(df['close'], 9)
        self.assertEqual(len(ema9), 100)
        self.assertFalse(np.isnan(ema9.iloc[-1]))

        rsi = calculate_rsi(df['close'], 14)
        self.assertEqual(len(rsi), 100)
        self.assertTrue(0 <= rsi.iloc[-1] <= 100)

        signal_res = analyze_market_data(df, self.config)
        self.assertIsNotNone(signal_res)
        self.assertIn(signal_res["signal"], ["BUY", "SELL", "HOLD"])

    def test_risk_manager(self):
        rm = RiskManager(self.config)
        can_trade, reason = rm.can_open_position(0, 1000.0)
        self.assertFalse(can_trade)  # Bot inactive initially

        # Activate bot
        cfg_active = self.config.copy()
        cfg_active["bot_active"] = True
        cfg_active["enable_session_filter"] = False
        rm.config = cfg_active
        can_trade, reason = rm.can_open_position(0, 1000.0)
        self.assertTrue(can_trade)

        # Test daily target reached
        rm.add_realized_pnl(25.0, 1000.0)  # +2.5% profit
        self.assertTrue(rm.target_locked)
        can_trade, reason = rm.can_open_position(0, 1025.0)
        self.assertFalse(can_trade)

    def test_paper_trader_flow(self):
        pt = PaperTrader(initial_balance=1000.0)
        pt.reset_account(1000.0)
        self.assertEqual(pt.balance, 1000.0)

        # Open Long
        pos = pt.open_position(
            symbol="BTC/USDT:USDT",
            side="long",
            price=85000.0,
            contracts=0.01,
            notional=850.0,
            margin=283.33,
            leverage=3,
            sl=84320.0,
            tp1=86020.0,
            tp2=87040.0
        )
        self.assertIsNotNone(pos)
        self.assertIn("BTC/USDT:USDT", pt.positions)
        self.assertLess(pt.balance, 1000.0)

        # Price hits TP1
        events = pt.update_price_and_check_exits("BTC/USDT:USDT", 86050.0, self.config)
        self.assertTrue(len(events) > 0)
        event_types = [e["type"] for e in events]
        self.assertTrue("TP1_HIT" in event_types or "BREAKEVEN_LOCKED" in event_types)

        # Price hits TP2 -> closes remainder
        events = pt.update_price_and_check_exits("BTC/USDT:USDT", 87100.0, self.config)
        self.assertTrue(len(events) > 0)
        self.assertNotIn("BTC/USDT:USDT", pt.positions)
        self.assertGreater(pt.balance, 1000.0)  # Profit booked!

if __name__ == "__main__":
    unittest.main()
