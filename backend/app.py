import os
import sys
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import load_config, save_config, BASE_DIR
from .exchange import MexcExchange
from .strategy import analyze_market_data
from .risk_manager import RiskManager
from .paper_trader import PaperTrader
from .telegram_bot import TelegramNotifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("mexc_bot")

config = load_config()
exchange = MexcExchange(config.get("mexc_api_key", ""), config.get("mexc_api_secret", ""))
risk_manager = RiskManager(config)
paper_trader = PaperTrader(float(config.get("paper_initial_balance", 1000.0)))
telegram = TelegramNotifier(config.get("telegram_bot_token", ""), config.get("telegram_chat_id", ""))

# Connected WebSockets
active_connections: List[WebSocket] = []
# Recent log buffer for UI terminal
recent_logs: List[Dict[str, Any]] = []

def add_log(message: str, level: str = "info"):
    entry = {
        "time": time.strftime("%H:%M:%S"),
        "level": level,
        "message": message
    }
    recent_logs.append(entry)
    if len(recent_logs) > 100:
        recent_logs.pop(0)

# Background Scalping Engine
async def scalper_loop():
    logger.info("MEXC Scalping Engine background loop started.")
    add_log("Trading Engine initialized and waiting for command.")
    
    # Preload markets in background
    try:
        exchange.load_markets_safe()
    except Exception as e:
        logger.error(f"Error loading exchange markets: {e}")

    last_heartbeat_time = time.time() - 40000  # Will send initial heartbeat after 1 hour

    while True:
        try:
            current_config = load_config()
            is_active = current_config.get("bot_active", False)
            mode = current_config.get("trading_mode", "PAPER")
            symbols = current_config.get("symbols", ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"])
            timeframe = current_config.get("timeframe", "1m")
            leverage = int(current_config.get("leverage", 3))

            # Determine Current Balance
            if mode == "PAPER":
                summary = paper_trader.get_summary()
                curr_balance = summary["balance"]
                curr_open_count = summary["open_positions_count"]
            else:
                live_bal = exchange.fetch_balance()
                curr_balance = live_bal["total"]
                curr_positions = exchange.fetch_positions(symbols)
                curr_open_count = len(curr_positions)

            # Check new day / sync risk stats
            risk_manager.check_new_day(curr_balance)

            # Periodic 12-Hour System Heartbeat
            if time.time() - last_heartbeat_time >= 43200 and current_config.get("telegram_bot_token"):
                status_obj = risk_manager.get_status(curr_balance)
                telegram.notify_heartbeat(status_obj, curr_balance, curr_open_count, mode)
                last_heartbeat_time = time.time()

            # Process each symbol
            for symbol in symbols:
                try:
                    # 1. Fetch Ticker
                    ticker = exchange.fetch_ticker(symbol)
                    if not ticker or ticker.get("last", 0) <= 0:
                        continue
                    current_price = ticker["last"]

                    # 2. If Paper Mode, update existing positions and check for SL/TP triggers
                    if mode == "PAPER":
                        events = paper_trader.update_price_and_check_exits(symbol, current_price, current_config)
                        for ev in events:
                            if ev.get("type") == "BREAKEVEN_LOCKED":
                                add_log(f"🛡️ Free-Roll Active on {symbol}: SL secured at Break-Even!", "success")
                                telegram.notify_breakeven(symbol, current_price)
                            elif ev.get("type") == "TP1_HIT":
                                add_log(f"🎯 TP1 HIT on {symbol} @ ${current_price}! +${ev.get('pnl')} booked. SL -> Break-Even.", "success")
                                risk_manager.add_realized_pnl(ev.get("pnl", 0.0), paper_trader.balance)
                            elif "pnl" in ev:
                                pnl = ev.get("pnl", 0.0)
                                reason = ev.get("reason", "CLOSED")
                                status_type = "success" if pnl >= 0 else "warning"
                                add_log(f"🏁 {reason}: {symbol} closed @ ${current_price} | PnL: ${pnl}", status_type)
                                risk_manager.add_realized_pnl(pnl, paper_trader.balance)
                                telegram.notify_trade_closed(ev, mode)
                                
                                # Check if daily target was newly reached
                                if risk_manager.target_locked:
                                    status = risk_manager.get_status(paper_trader.balance)
                                    telegram.notify_daily_target(status["realized_pnl_today"], status["pnl_percentage_today"])
                                    add_log("🏆 Daily Target Hit! Profits locked for today.", "success")

                    # If bot is not active, continue scanning prices only
                    if not is_active:
                        continue

                    # 3. Analyze Market Data for Scalp Entry (200 Candles for proper EMA convergence)
                    df = exchange.fetch_ohlcv_df(symbol, timeframe=timeframe, limit=200)
                    if df is not None:
                        signal_data = analyze_market_data(df, current_config)
                        if signal_data and signal_data["signal"] in ["BUY", "SELL"]:
                            side = signal_data["side"]
                            can_trade, reason = risk_manager.can_open_position(curr_open_count, curr_balance)
                            
                            if can_trade:
                                pos_calc = risk_manager.calculate_position_size(curr_balance, current_price, leverage)
                                
                                if mode == "PAPER":
                                    # Execute Paper Order
                                    res = paper_trader.open_position(
                                        symbol=symbol,
                                        side=side,
                                        price=current_price,
                                        contracts=pos_calc["contracts"],
                                        notional=pos_calc["notional_usdt"],
                                        margin=pos_calc["margin_usdt"],
                                        leverage=leverage,
                                        sl=signal_data["sl"],
                                        tp1=signal_data["tp1"],
                                        tp2=signal_data["tp2"]
                                    )
                                    if res:
                                        shock_text = " [Shock Absorber 0.75% Risk]" if pos_calc.get("shock_absorber_active") else ""
                                        add_log(f"🚀 [PAPER] Opened {side.upper()} on {symbol} @ ${current_price} (Margin: ${pos_calc['margin_usdt']}){shock_text}", "info")
                                        telegram.notify_trade_opened(res, mode)
                                        curr_open_count += 1
                                else:
                                    # Execute Live Order on MEXC (Post-Only Limit Order for 0.00% Maker Fee)
                                    exchange.set_leverage(leverage, symbol)
                                    order_side = "buy" if side == "long" else "sell"
                                    
                                    use_maker = current_config.get("use_limit_orders", True)
                                    if use_maker:
                                        live_order = exchange.create_post_only_limit_order(
                                            symbol=symbol,
                                            side=order_side,
                                            amount=pos_calc["contracts"],
                                            price=current_price
                                        )
                                    else:
                                        live_order = exchange.create_order(
                                            symbol=symbol,
                                            side=order_side,
                                            order_type="market",
                                            amount=pos_calc["contracts"]
                                        )

                                    if live_order:
                                        # Immediately place Native Exchange Stop-Loss on MEXC for connection safety!
                                        sl_side = "sell" if side == "long" else "buy"
                                        exchange.create_native_stop_loss(
                                            symbol=symbol,
                                            side=sl_side,
                                            amount=pos_calc["contracts"],
                                            stop_price=signal_data["sl"]
                                        )

                                        add_log(f"🔥 [LIVE] Executed {side.upper()} order for {symbol} on MEXC! Native Stop-Loss set @ ${signal_data['sl']}", "success")
                                        trade_payload = {
                                            "symbol": symbol,
                                            "side": side,
                                            "leverage": leverage,
                                            "entry_price": current_price,
                                            "margin": pos_calc["margin_usdt"],
                                            "tp1": signal_data["tp1"],
                                            "tp2": signal_data["tp2"],
                                            "sl": signal_data["sl"]
                                        }
                                        telegram.notify_trade_opened(trade_payload, mode)
                                        curr_open_count += 1

                except Exception as sym_err:
                    logger.error(f"Error scanning {symbol}: {sym_err}")

            # Send real-time state to connected WebSocket clients
            await broadcast_state()

        except Exception as loop_err:
            logger.error(f"Error in scalper loop iteration: {loop_err}")

        await asyncio.sleep(4)

async def broadcast_state():
    if not active_connections:
        return

    try:
        curr_cfg = load_config()
        mode = curr_cfg.get("trading_mode", "PAPER")
        
        if mode == "PAPER":
            summary = paper_trader.get_summary()
            balance = summary["balance"]
            equity = summary["equity"]
            positions = list(paper_trader.positions.values())
            history = paper_trader.history[-15:]
        else:
            live_bal = exchange.fetch_balance()
            balance = live_bal["total"]
            positions = exchange.fetch_positions(curr_cfg.get("symbols", []))
            unrealized = sum(p.get("unrealized_pnl", 0.0) for p in positions)
            equity = balance + unrealized
            summary = {
                "balance": balance,
                "equity": equity,
                "unrealized_pnl": unrealized,
                "total_trades": len(positions),
                "win_rate": 0.0
            }
            history = []

        risk_status = risk_manager.get_status(balance)

        payload = {
            "type": "STATE_UPDATE",
            "bot_active": curr_cfg.get("bot_active", False),
            "trading_mode": mode,
            "summary": summary,
            "risk_status": risk_status,
            "positions": positions,
            "recent_trades": history,
            "recent_logs": recent_logs[-25:],
            "timestamp": time.time()
        }

        dead_connections = []
        for ws in active_connections:
            try:
                await ws.send_json(payload)
            except Exception:
                dead_connections.append(ws)
        
        for dead in dead_connections:
            if dead in active_connections:
                active_connections.remove(dead)

    except Exception as e:
        logger.error(f"Error broadcasting state: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start background task
    task = asyncio.create_task(scalper_loop())
    yield
    # Shutdown
    task.cancel()

app = FastAPI(title="MEXC Scalper Bot", lifespan=lifespan)

# REST Endpoints
@app.get("/api/status")
async def get_status():
    cfg = load_config()
    mode = cfg.get("trading_mode", "PAPER")
    if mode == "PAPER":
        summary = paper_trader.get_summary()
    else:
        live_bal = exchange.fetch_balance()
        summary = {
            "balance": live_bal["total"],
            "equity": live_bal["total"],
            "unrealized_pnl": 0.0,
            "win_rate": 0.0,
            "total_trades": 0
        }
    return {
        "bot_active": cfg.get("bot_active", False),
        "trading_mode": mode,
        "summary": summary,
        "risk_status": risk_manager.get_status(summary["balance"])
    }

@app.post("/api/bot/toggle")
async def toggle_bot():
    cfg = load_config()
    current = cfg.get("bot_active", False)
    new_state = not current
    save_config({"bot_active": new_state})
    status_str = "STARTED" if new_state else "PAUSED"
    add_log(f"Bot execution {status_str} by user.", "info")
    return {"bot_active": new_state}

@app.post("/api/bot/mode")
async def toggle_mode():
    cfg = load_config()
    current_mode = cfg.get("trading_mode", "PAPER")
    new_mode = "LIVE" if current_mode == "PAPER" else "PAPER"
    save_config({"trading_mode": new_mode})
    add_log(f"Trading mode switched to {new_mode}.", "warning")
    return {"trading_mode": new_mode}

@app.get("/api/settings")
async def get_settings():
    return load_config()

class SettingsPayload(BaseModel):
    trading_mode: str = "PAPER"
    mexc_api_key: str = ""
    mexc_api_secret: str = ""
    symbols: List[str] = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    timeframe: str = "1m"
    leverage: int = 3
    risk_per_trade_percent: float = 1.5
    max_open_positions: int = 2
    daily_profit_target_percent: float = 1.0
    daily_max_loss_percent: float = 3.0
    stop_loss_percent: float = 0.8
    take_profit_1_percent: float = 1.2
    take_profit_2_percent: float = 2.4
    use_trailing_stop: bool = True
    enable_session_filter: bool = True
    session_start_utc: int = 12
    session_end_utc: int = 21
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

@app.post("/api/settings")
async def update_settings(payload: SettingsPayload):
    new_data = payload.model_dump()
    save_config(new_data)
    # Refresh exchange credentials
    exchange.api_key = new_data.get("mexc_api_key", "")
    exchange.api_secret = new_data.get("mexc_api_secret", "")
    exchange.exchange.apiKey = exchange.api_key
    exchange.exchange.secret = exchange.api_secret
    telegram.update_credentials(new_data.get("telegram_bot_token", ""), new_data.get("telegram_chat_id", ""))
    add_log("Settings updated successfully.", "info")
    return {"status": "saved"}

@app.post("/api/paper/reset")
async def reset_paper():
    cfg = load_config()
    init_bal = float(cfg.get("paper_initial_balance", 1000.0))
    paper_trader.reset_account(init_bal)
    risk_manager.start_day_balance = init_bal
    risk_manager.realized_pnl_today = 0.0
    risk_manager.target_locked = False
    risk_manager.killswitch_triggered = False
    add_log("Paper trading account reset to $1,000.00.", "info")
    return {"status": "reset", "balance": init_bal}

class ClosePayload(BaseModel):
    symbol: str

@app.post("/api/positions/close")
async def close_position(payload: ClosePayload):
    symbol = payload.symbol
    cfg = load_config()
    mode = cfg.get("trading_mode", "PAPER")

    ticker = exchange.fetch_ticker(symbol)
    price = ticker.get("last", 0.0) if ticker else 0.0

    if mode == "PAPER":
        closed = paper_trader.manual_close_position(symbol, price)
        if closed:
            add_log(f"Position on {symbol} manually closed @ ${price}. PnL: ${closed.get('pnl')}", "info")
            risk_manager.add_realized_pnl(closed.get("pnl", 0.0), paper_trader.balance)
            return {"status": "closed", "trade": closed}
        raise HTTPException(status_code=404, detail="Position not found.")
    else:
        # Live close: submit opposite market order
        positions = exchange.fetch_positions([symbol])
        if not positions:
            raise HTTPException(status_code=404, detail="No live position on this symbol.")
        pos = positions[0]
        side = "sell" if pos["side"] == "long" else "buy"
        order = exchange.create_order(symbol, side, "market", pos["contracts"], params={"reduceOnly": True})
        add_log(f"Live position on {symbol} closed via reduceOnly market order.", "info")
        return {"status": "closed", "order": order}

@app.get("/api/chart/{symbol}")
async def get_chart_data(symbol: str):
    # Decode URL safe symbol
    symbol = symbol.replace("-", "/")
    df = exchange.fetch_ohlcv_df(symbol, timeframe="1m", limit=60)
    if df is None:
        return {"candles": []}
    
    # Calculate indicators for chart
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

    candles = []
    for _, row in df.iterrows():
        candles.append({
            "time": int(row['timestamp'] / 1000),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row['volume'],
            "ema9": round(float(row['ema9']), 2),
            "ema21": round(float(row['ema21']), 2),
            "ema50": round(float(row['ema50']), 2)
        })
    return {"candles": candles}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep-alive ping
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception:
        if websocket in active_connections:
            active_connections.remove(websocket)

# Mount Frontend static files
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")
