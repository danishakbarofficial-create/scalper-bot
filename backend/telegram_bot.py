import logging
import requests
from typing import Dict, Any

logger = logging.getLogger("telegram_bot")

class TelegramNotifier:
    def __init__(self, token: str = "", chat_id: str = ""):
        self.token = token.strip()
        self.chat_id = chat_id.strip()

    def update_credentials(self, token: str, chat_id: str):
        self.token = token.strip()
        self.chat_id = chat_id.strip()

    def send_message(self, text: str) -> bool:
        if not self.token or not self.chat_id:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        try:
            res = requests.post(url, json=payload, timeout=5)
            if res.status_code == 200:
                return True
            else:
                logger.warning(f"Telegram API response: {res.text}")
                return False
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
            return False

    def notify_trade_opened(self, trade: Dict[str, Any], mode: str):
        icon = "🟢" if trade.get("side") == "long" else "🔴"
        side_text = trade.get("side", "").upper()
        symbol = trade.get("symbol", "").split(":")[0]
        
        msg = (
            f"<b>{icon} MEXC SCALPER: POSITION OPENED ({mode})</b>\n\n"
            f"<b>Pair:</b> {symbol}\n"
            f"<b>Side:</b> {side_text} ({trade.get('leverage')}x)\n"
            f"<b>Entry Price:</b> ${trade.get('entry_price')}\n"
            f"<b>Margin:</b> ${trade.get('margin')} USDT\n"
            f"<b>Take Profit 1:</b> ${trade.get('tp1')}\n"
            f"<b>Take Profit 2:</b> ${trade.get('tp2')}\n"
            f"<b>Stop Loss:</b> ${trade.get('sl')}\n"
            f"<b>Time:</b> Just now"
        )
        self.send_message(msg)

    def notify_trade_closed(self, trade: Dict[str, Any], mode: str):
        pnl = trade.get("pnl", 0.0)
        pnl_pct = trade.get("pnl_percent", 0.0)
        icon = "🎉 <b>PROFIT!</b>" if pnl >= 0 else "🛑 <b>STOP LOSS</b>"
        symbol = trade.get("symbol", "").split(":")[0]

        msg = (
            f"<b>{icon} MEXC SCALPER: TRADE CLOSED ({mode})</b>\n\n"
            f"<b>Pair:</b> {symbol}\n"
            f"<b>Side:</b> {trade.get('side', '').upper()}\n"
            f"<b>Exit Price:</b> ${trade.get('exit_price')}\n"
            f"<b>Net PnL:</b> {'+' if pnl >= 0 else ''}${pnl} ({'+' if pnl_pct >= 0 else ''}{pnl_pct}%)\n"
            f"<b>Reason:</b> {trade.get('reason')}"
        )
        self.send_message(msg)

    def notify_daily_target(self, pnl: float, pct: float):
        msg = (
            f"🏆 <b>DAILY TARGET ACHIEVED!</b>\n\n"
            f"Today's Realized Profit: <b>+${round(pnl, 2)} (+{round(pct, 2)}%)</b>\n"
            f"Trading is safely locked to protect today's monthly goal pacing. "
            f"Capital is safe!"
        )
        self.send_message(msg)

    def notify_breakeven(self, symbol: str, entry_price: float):
        sym = symbol.split(":")[0]
        msg = (
            f"🛡️ <b>FREE-ROLL ACTIVE: {sym}</b>\n\n"
            f"Trade reached +1.2R profit target!\n"
            f"Stop-Loss has been automatically moved to Break-Even (${round(entry_price, 2)}).\n"
            f"<b>Zero Risk Trade!</b> If market reverses, capital is 100% protected."
        )
        self.send_message(msg)

    def notify_heartbeat(self, status: Dict[str, Any], balance: float, open_positions: int, mode: str):
        pnl_today = status.get("realized_pnl_today", 0.0)
        pnl_pct = status.get("pnl_percentage_today", 0.0)
        in_sess = "🟢 Active" if status.get("in_active_session") else "⚪ Resting (Off-hours)"
        shock = "⚠️ Active (0.75% Risk)" if status.get("shock_absorber_active") else "✅ Normal (1.5% Risk)"

        msg = (
            f"💓 <b>MEXC SCALPER: SYSTEM HEARTBEAT ({mode})</b>\n\n"
            f"<b>Status:</b> Healthy & Scanning\n"
            f"<b>Current Balance:</b> ${round(balance, 2)} USDT\n"
            f"<b>Open Positions:</b> {open_positions}\n"
            f"<b>Today's Realized PnL:</b> {'+' if pnl_today >= 0 else ''}${pnl_today} ({pnl_pct}%)\n"
            f"<b>Session Status:</b> {in_sess}\n"
            f"<b>Shock Absorber:</b> {shock}\n"
            f"<b>Server Time:</b> Just now"
        )
        self.send_message(msg)
