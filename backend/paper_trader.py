import os
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("paper_trader")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE = DATA_DIR / "paper_state.json"

class PaperTrader:
    def __init__(self, initial_balance: float = 1000.0, maker_fee: float = 0.0, taker_fee: float = 0.0002):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []
        self.load_state()

    def load_state(self):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.balance = float(data.get("balance", self.initial_balance))
                    self.positions = data.get("positions", {})
                    self.history = data.get("history", [])
                logger.info(f"Loaded paper trading state. Balance: ${round(self.balance, 2)}")
            except Exception as e:
                logger.error(f"Failed to load paper state: {e}")

    def save_state(self):
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "balance": self.balance,
                    "positions": self.positions,
                    "history": self.history[-200:]  # Keep last 200 trades
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save paper state: {e}")

    def get_summary(self) -> Dict[str, Any]:
        unrealized_pnl = sum(p.get("unrealized_pnl", 0.0) for p in self.positions.values())
        equity = self.balance + unrealized_pnl
        total_trades = len(self.history)
        winning_trades = len([t for t in self.history if t.get("pnl", 0.0) > 0])
        win_rate = (winning_trades / max(total_trades, 1)) * 100.0
        total_pnl = sum(t.get("pnl", 0.0) for t in self.history)

        return {
            "balance": round(self.balance, 2),
            "equity": round(equity, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_realized_pnl": round(total_pnl, 2),
            "total_trades": total_trades,
            "win_rate": round(win_rate, 1),
            "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades,
            "open_positions_count": len(self.positions)
        }

    def open_position(self, symbol: str, side: str, price: float, contracts: float, 
                      notional: float, margin: float, leverage: int, sl: float, tp1: float, tp2: float) -> Optional[Dict[str, Any]]:
        # Check if already open on this symbol
        if symbol in self.positions:
            return None

        # Check balance
        if self.balance < margin:
            logger.warning(f"Paper trade failed: Insufficient balance. Required: ${margin}, Available: ${self.balance}")
            return None

        fee = notional * self.taker_fee
        self.balance -= (margin + fee)

        pos_id = str(uuid.uuid4())[:8]
        pos = {
            "id": pos_id,
            "symbol": symbol,
            "side": side,  # "long" or "short"
            "entry_price": price,
            "current_price": price,
            "contracts": contracts,
            "notional": notional,
            "margin": margin,
            "leverage": leverage,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp1_hit": False,
            "original_margin": margin,
            "original_contracts": contracts,
            "highest_price": price,
            "lowest_price": price,
            "open_time": time.time(),
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
            "fees_paid": fee
        }

        self.positions[symbol] = pos
        self.save_state()
        logger.info(f"[PAPER] OPEN {side.upper()} on {symbol} @ {price}, Margin: ${margin}, SL: {sl}, TP1: {tp1}")
        return pos

    def update_price_and_check_exits(self, symbol: str, current_price: float, config: dict) -> List[Dict[str, Any]]:
        """
        Updates positions with live price, checks SL/TP/Trailing Stop triggers.
        Returns list of closed trades/events.
        """
        if symbol not in self.positions:
            return []

        pos = self.positions[symbol]
        pos["current_price"] = current_price
        side = pos["side"]
        entry = pos["entry_price"]
        contracts = pos["contracts"]
        leverage = pos["leverage"]

        # Track extremes for trailing stop
        if current_price > pos["highest_price"]:
            pos["highest_price"] = current_price
        if current_price < pos["lowest_price"]:
            pos["lowest_price"] = current_price

        # Calculate Unrealized PnL
        if side == "long":
            raw_pnl = (current_price - entry) * contracts
        else:
            raw_pnl = (entry - current_price) * contracts

        pos["unrealized_pnl"] = round(raw_pnl, 2)
        pos["unrealized_pnl_pct"] = round((raw_pnl / max(pos["margin"], 1e-4)) * 100.0, 2)

        events = []
        trailing_enabled = config.get("use_trailing_stop", True)
        callback_pct = float(config.get("trailing_stop_callback_percent", 0.4)) / 100.0

        # --- LONG EXIT LOGIC ---
        if side == "long":
            # 0. Free-Roll Break-Even Lock (Only if enable_freeroll_be is True in config)
            dist_to_tp = pos["tp1"] - entry
            if config.get("enable_freeroll_be", False) and dist_to_tp > 0 and not pos.get("be_locked", False):
                if current_price >= entry + (0.60 * dist_to_tp):
                    pos["sl"] = entry + (entry * 0.0005)  # Covers fees
                    pos["be_locked"] = True
                    events.append({
                        "type": "BREAKEVEN_LOCKED",
                        "symbol": symbol,
                        "side": side,
                        "price": current_price,
                        "reason": "🛡️ Free-Roll Active: Trade reached +1.2R. Stop Loss secured at Break-Even."
                    })
                    logger.info(f"[PAPER] Free-Roll Active on {symbol}: SL moved to Break-Even (${round(pos['sl'], 2)})")

            # 1. Stop Loss Hit
            if current_price <= pos["sl"]:
                closed = self._close_full(symbol, current_price, "STOP_LOSS")
                events.append(closed)
                return events

            # 2. TP1 Hit (Close 50%, set Stop Loss to Break-Even)
            if not pos["tp1_hit"] and current_price >= pos["tp1"]:
                pos["tp1_hit"] = True
                half_contracts = contracts / 2.0
                realized = (current_price - entry) * half_contracts
                fee = (half_contracts * current_price) * self.taker_fee
                net_pnl = realized - fee
                
                # Return half margin + profit to balance
                margin_returned = pos["margin"] / 2.0
                self.balance += (margin_returned + net_pnl)
                pos["margin"] -= margin_returned
                pos["contracts"] -= half_contracts
                pos["sl"] = entry + (entry * 0.0005)  # Move SL to Break-Even!
                pos["be_locked"] = True
                
                events.append({
                    "type": "TP1_HIT",
                    "symbol": symbol,
                    "side": side,
                    "exit_price": current_price,
                    "pnl": round(net_pnl, 2),
                    "reason": "TP1 reached. 50% profit booked, SL moved to Break-Even."
                })
                logger.info(f"[PAPER] TP1 HIT on {symbol} @ {current_price}! +${round(net_pnl, 2)}")

            # 3. TP2 Hit (Close remainder)
            if current_price >= pos["tp2"]:
                closed = self._close_full(symbol, current_price, "TAKE_PROFIT_2")
                events.append(closed)
                return events

            # 4. Trailing Stop (If TP1 was hit and price pulls back from high)
            if trailing_enabled and pos["tp1_hit"]:
                trail_trigger_price = pos["highest_price"] * (1.0 - callback_pct)
                if current_price <= trail_trigger_price:
                    closed = self._close_full(symbol, current_price, "TRAILING_STOP")
                    events.append(closed)
                    return events

        # --- SHORT EXIT LOGIC ---
        elif side == "short":
            # 0. Free-Roll Break-Even Lock (Only if enable_freeroll_be is True in config)
            dist_to_tp = entry - pos["tp1"]
            if config.get("enable_freeroll_be", False) and dist_to_tp > 0 and not pos.get("be_locked", False):
                if current_price <= entry - (0.60 * dist_to_tp):
                    pos["sl"] = entry - (entry * 0.0005)  # Covers fees
                    pos["be_locked"] = True
                    events.append({
                        "type": "BREAKEVEN_LOCKED",
                        "symbol": symbol,
                        "side": side,
                        "price": current_price,
                        "reason": "🛡️ Free-Roll Active: Trade reached +1.2R. Stop Loss secured at Break-Even."
                    })
                    logger.info(f"[PAPER] Free-Roll Active on {symbol}: SL moved to Break-Even (${round(pos['sl'], 2)})")

            # 1. Stop Loss Hit
            if current_price >= pos["sl"]:
                closed = self._close_full(symbol, current_price, "STOP_LOSS")
                events.append(closed)
                return events

            # 2. TP1 Hit (Close 50%, move SL to Break-Even)
            if not pos["tp1_hit"] and current_price <= pos["tp1"]:
                pos["tp1_hit"] = True
                half_contracts = contracts / 2.0
                realized = (entry - current_price) * half_contracts
                fee = (half_contracts * current_price) * self.taker_fee
                net_pnl = realized - fee

                margin_returned = pos["margin"] / 2.0
                self.balance += (margin_returned + net_pnl)
                pos["margin"] -= margin_returned
                pos["contracts"] -= half_contracts
                pos["sl"] = entry - (entry * 0.0005)
                pos["be_locked"] = True  # Move SL to Break-Even!

                events.append({
                    "type": "TP1_HIT",
                    "symbol": symbol,
                    "side": side,
                    "exit_price": current_price,
                    "pnl": round(net_pnl, 2),
                    "reason": "TP1 reached (+1.2%). 50% profit booked, SL moved to Break-Even."
                })
                logger.info(f"[PAPER] TP1 HIT on {symbol} @ {current_price}! +${round(net_pnl, 2)}")

            # 3. TP2 Hit (Close remainder)
            if current_price <= pos["tp2"]:
                closed = self._close_full(symbol, current_price, "TAKE_PROFIT_2")
                events.append(closed)
                return events

            # 4. Trailing Stop
            if trailing_enabled and pos["tp1_hit"]:
                trail_trigger_price = pos["lowest_price"] * (1.0 + callback_pct)
                if current_price >= trail_trigger_price:
                    closed = self._close_full(symbol, current_price, "TRAILING_STOP")
                    events.append(closed)
                    return events

        self.save_state()
        return events

    def _close_full(self, symbol: str, exit_price: float, reason: str) -> Dict[str, Any]:
        pos = self.positions.pop(symbol)
        side = pos["side"]
        entry = pos["entry_price"]
        contracts = pos["contracts"]
        margin = pos["margin"]

        if side == "long":
            raw_pnl = (exit_price - entry) * contracts
        else:
            raw_pnl = (entry - exit_price) * contracts

        closing_fee = (contracts * exit_price) * self.taker_fee
        net_pnl = raw_pnl - closing_fee
        total_pnl = net_pnl

        # Return remaining margin + net PnL to balance
        self.balance += max(0.0, margin + net_pnl)

        trade_record = {
            "id": pos["id"],
            "symbol": symbol,
            "side": side,
            "entry_price": entry,
            "exit_price": exit_price,
            "margin": pos["original_margin"],
            "leverage": pos["leverage"],
            "pnl": round(total_pnl, 2),
            "pnl_percent": round((total_pnl / max(pos["original_margin"], 1e-4)) * 100.0, 2),
            "open_time": pos["open_time"],
            "close_time": time.time(),
            "reason": reason
        }

        self.history.append(trade_record)
        self.save_state()
        logger.info(f"[PAPER] CLOSED {side.upper()} {symbol} @ {exit_price} ({reason}) | PnL: ${round(total_pnl, 2)}")
        return trade_record

    def manual_close_position(self, symbol: str, current_price: float) -> Optional[Dict[str, Any]]:
        if symbol in self.positions:
            return self._close_full(symbol, current_price, "MANUAL_CLOSE")
        return None

    def reset_account(self, new_balance: Optional[float] = None):
        self.balance = new_balance or self.initial_balance
        self.positions = {}
        self.history = []
        self.save_state()
        logger.info(f"Paper trading account reset to ${self.balance}")
