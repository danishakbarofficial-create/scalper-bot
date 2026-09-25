import time
from datetime import datetime, timezone
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger("risk_manager")

class RiskManager:
    def __init__(self, config: dict):
        self.config = config
        self.today_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.start_day_balance = float(config.get("paper_initial_balance", 1000.0))
        self.realized_pnl_today = 0.0
        self.last_trade_time = 0.0
        self.target_locked = False
        self.killswitch_triggered = False
        self.consecutive_losses = 0

    def check_new_day(self, current_balance: float):
        current_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if current_date_str != self.today_date_str:
            logger.info(f"New day detected: {current_date_str}. Resetting daily metrics.")
            self.today_date_str = current_date_str
            self.start_day_balance = current_balance
            self.realized_pnl_today = 0.0
            self.target_locked = False
            self.killswitch_triggered = False

    def add_realized_pnl(self, pnl: float, current_balance: float):
        self.check_new_day(current_balance)
        self.realized_pnl_today += pnl
        self.last_trade_time = time.time()

        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 2:
                logger.warning(f"📉 Shock Absorber Triggered: {self.consecutive_losses} consecutive losses. Scaling down risk by 50% until next win.")
        elif pnl > 0:
            if self.consecutive_losses >= 2:
                logger.info(f"🎉 Winning trade after loss streak! Restoring full position risk sizing.")
            self.consecutive_losses = 0

        daily_target_pct = float(self.config.get("daily_profit_target_percent", 1.0))
        daily_loss_pct = float(self.config.get("daily_max_loss_percent", 3.0))

        daily_profit_pct = (self.realized_pnl_today / max(self.start_day_balance, 1.0)) * 100.0

        # Check profit target
        if daily_profit_pct >= daily_target_pct:
            self.target_locked = True
            logger.info(f"DAILY TARGET REACHED! +{round(daily_profit_pct, 2)}% today. Locking gains.")

        # Check max loss killswitch
        if daily_profit_pct <= -daily_loss_pct:
            self.killswitch_triggered = True
            logger.warning(f"DAILY LOSS KILLSWITCH ACTIVATED! {round(daily_profit_pct, 2)}% today. Halting trading.")

    def is_in_active_session(self) -> Tuple[bool, str]:
        if not self.config.get("enable_session_filter", True):
            return True, "Session filter disabled."

        current_hour_utc = datetime.now(timezone.utc).hour
        start = int(self.config.get("session_start_utc", 12))
        end = int(self.config.get("session_end_utc", 21))

        if start <= end:
            is_active = (start <= current_hour_utc < end)
        else:  # overnight session crossing midnight
            is_active = (current_hour_utc >= start or current_hour_utc < end)

        if is_active:
            return True, f"Active High-Volatility Session ({start}:00 - {end}:00 UTC)."
        else:
            return False, f"Outside Active Session ({start}:00 - {end}:00 UTC). Current: {current_hour_utc}:00 UTC. Capital protected from off-hours chop."

    def can_open_position(self, current_open_count: int, current_balance: float) -> Tuple[bool, str]:
        self.check_new_day(current_balance)

        if not self.config.get("bot_active", False):
            return False, "Bot is currently paused/stopped."

        # Check Active Session Filter
        in_session, session_msg = self.is_in_active_session()
        if not in_session:
            return False, session_msg

        if self.killswitch_triggered:
            return False, "Daily Max Loss Killswitch is active. Trading paused to protect capital."

        if self.target_locked:
            return False, f"Daily Profit Target ({self.config.get('daily_profit_target_percent')}%) reached! Profits safely locked."

        max_positions = int(self.config.get("max_open_positions", 2))
        if current_open_count >= max_positions:
            return False, f"Max simultaneous positions limit reached ({current_open_count}/{max_positions})."

        cooldown = int(self.config.get("cooldown_seconds", 180))
        elapsed = time.time() - self.last_trade_time
        if elapsed < cooldown and self.last_trade_time > 0:
            remaining = int(cooldown - elapsed)
            return False, f"Cooldown in effect ({remaining}s remaining)."

        return True, "OK"

    def calculate_position_size(self, balance: float, entry_price: float, leverage: int) -> Dict[str, float]:
        """
        Calculates position contracts / USDT notional value based on risk parameters.
        Default risk per trade: 1.5% of total account balance.
        If Shock Absorber is active (>= 2 consecutive losses), scales risk down to 0.75%
        to protect capital and prevent drawdown spikes.
        """
        risk_pct = float(self.config.get("risk_per_trade_percent", 1.5)) / 100.0
        
        # Shock Absorber Dynamic Scaling
        shock_absorber_active = False
        if self.config.get("enable_shock_absorber", False) and self.consecutive_losses >= 2:
            risk_pct *= 0.5  # Cut risk in half
            shock_absorber_active = True

        sl_pct = float(self.config.get("stop_loss_percent", 0.8)) / 100.0

        # Max loss amount allowed for this trade
        risk_amount_usdt = balance * risk_pct
        
        # Position notional value (Position Size in USDT)
        position_notional = risk_amount_usdt / max(sl_pct, 0.001)

        # Cap single position size to maximum 35% of total balance with leverage
        max_notional = balance * leverage * 0.35
        position_notional = min(position_notional, max_notional)

        # Margin required from account
        margin_required = position_notional / max(leverage, 1)

        # Asset quantity (amount of BTC / ETH / etc.)
        contracts = position_notional / max(entry_price, 1e-6)

        return {
            "notional_usdt": round(position_notional, 2),
            "margin_usdt": round(margin_required, 2),
            "contracts": round(contracts, 5),
            "risk_amount_usdt": round(risk_amount_usdt, 2),
            "shock_absorber_active": shock_absorber_active
        }

    def get_status(self, current_balance: float) -> Dict[str, Any]:
        self.check_new_day(current_balance)
        pnl_pct = (self.realized_pnl_today / max(self.start_day_balance, 1.0)) * 100.0
        target_pct = float(self.config.get("daily_profit_target_percent", 1.0))
        target_progress = min(max((pnl_pct / max(target_pct, 0.01)) * 100.0, 0.0), 100.0)

        in_session, session_desc = self.is_in_active_session()
        return {
            "today_date": self.today_date_str,
            "start_balance": round(self.start_day_balance, 2),
            "realized_pnl_today": round(self.realized_pnl_today, 2),
            "pnl_percentage_today": round(pnl_pct, 2),
            "daily_target_percent": target_pct,
            "target_progress_percent": round(target_progress, 1),
            "target_locked": self.target_locked,
            "killswitch_triggered": self.killswitch_triggered,
            "in_active_session": in_session,
            "session_desc": session_desc,
            "consecutive_losses": self.consecutive_losses,
            "shock_absorber_active": self.consecutive_losses >= 2
        }
