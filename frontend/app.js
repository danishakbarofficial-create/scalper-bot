// MEXC Scalper PRO 25% - Dashboard Client Logic
let socket = null;
let scalperChart = null;
let isBotActive = false;
let currentTradingMode = "PAPER";

// Audio alert sound synthesizer (Web Audio API)
function playNotificationSound(type = 'success') {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    
    if (type === 'success') {
      osc.frequency.setValueAtTime(587.33, ctx.currentTime); // D5
      osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.15); // A5
    } else {
      osc.frequency.setValueAtTime(440, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(220, ctx.currentTime + 0.2);
    }
    gain.gain.setValueAtTime(0.1, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.2);
    osc.start();
    osc.stop(ctx.currentTime + 0.25);
  } catch (e) {
    // Audio context may require interaction first
  }
}

// Initialize WebSocket Connection
function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    document.getElementById("ws-pulse").className = "pulse-dot";
    document.getElementById("ws-status-text").innerText = "Live Connected";
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "STATE_UPDATE") {
        updateDashboard(data);
      }
    } catch (e) {
      console.error("WS Parse error", e);
    }
  };

  socket.onclose = () => {
    document.getElementById("ws-pulse").className = "pulse-dot dot-yellow";
    document.getElementById("ws-status-text").innerText = "Reconnecting...";
    setTimeout(connectWebSocket, 3000);
  };

  socket.onerror = () => {
    socket.close();
  };
}

// Update UI with real-time payload
function updateDashboard(state) {
  isBotActive = state.bot_active;
  currentTradingMode = state.trading_mode;

  // Bot Toggle Button
  const btnBot = document.getElementById("btn-toggle-bot");
  const iconBot = document.getElementById("bot-toggle-icon");
  const txtBot = document.getElementById("bot-toggle-text");

  if (isBotActive) {
    btnBot.className = "btn btn-primary";
    btnBot.style.background = "linear-gradient(135deg, #10b981, #059669)";
    iconBot.innerText = "⏸";
    txtBot.innerText = "BOT ACTIVE (PAUSE)";
  } else {
    btnBot.className = "btn btn-secondary";
    btnBot.style.background = "rgba(255, 255, 255, 0.08)";
    iconBot.innerText = "▶";
    txtBot.innerText = "START BOT";
  }

  // Trading Mode Pill
  const modeTag = document.getElementById("mode-tag");
  const modeDesc = document.getElementById("mode-desc");
  const modeLabel = document.getElementById("trading-mode-label");
  if (currentTradingMode === "LIVE") {
    modeTag.innerText = "LIVE TRADING";
    modeTag.style.color = "#ef4444";
    modeDesc.innerText = "Real MEXC Account";
    modeLabel.innerText = "LIVE (MEXC)";
    modeLabel.style.color = "#ef4444";
  } else {
    modeTag.innerText = "PAPER MODE";
    modeTag.style.color = "#10b981";
    modeDesc.innerText = "Virtual Simulation ($1,000)";
    modeLabel.innerText = "PAPER SIM";
    modeLabel.style.color = "#10b981";
  }

  // KPIs
  const summary = state.summary || {};
  document.getElementById("kpi-equity").innerText = `$${(summary.equity || 1000).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
  document.getElementById("kpi-balance").innerText = `$${(summary.balance || 1000).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;

  const unPnl = summary.unrealized_pnl || 0.0;
  const pnlBadge = document.getElementById("kpi-unrealized-badge");
  pnlBadge.innerText = `${unPnl >= 0 ? '+' : ''}$${unPnl.toFixed(2)} PnL`;
  pnlBadge.className = unPnl > 0 ? "badge-pro" : (unPnl < 0 ? "btn-danger-outline" : "badge-neutral");

  // Daily Target KPI
  const risk = state.risk_status || {};
  const todayPnl = risk.realized_pnl_today || 0.0;
  const todayPct = risk.pnl_percentage_today || 0.0;
  const elemTodayPnl = document.getElementById("kpi-today-pnl");
  elemTodayPnl.innerHTML = `${todayPnl >= 0 ? '+' : ''}$${todayPnl.toFixed(2)} <span class="kpi-unit" id="kpi-today-pct">(${todayPct >= 0 ? '+' : ''}${todayPct.toFixed(2)}%)</span>`;
  elemTodayPnl.style.color = todayPnl >= 0 ? "#10b981" : "#ef4444";

  const targetProgress = risk.target_progress_percent || 0.0;
  document.getElementById("target-progress-bar").style.width = `${Math.min(targetProgress, 100)}%`;
  document.getElementById("target-progress-num").innerText = `${targetProgress}% Reached`;

  if (risk.target_locked) {
    document.getElementById("target-status-label").innerHTML = "<strong style='color:#10b981'>🏆 Target Hit! Profits Locked.</strong>";
  } else if (risk.killswitch_triggered) {
    document.getElementById("target-status-label").innerHTML = "<strong style='color:#ef4444'>🛑 Killswitch Halting Trading</strong>";
  } else {
    document.getElementById("target-status-label").innerText = "Daily Goal Pacing (25% Mo.)";
  }

  // Active Session Badge Update
  const sessBadge = document.getElementById("session-badge");
  const sessText = document.getElementById("session-status-text");
  const sessIcon = document.getElementById("session-icon");
  if (risk.in_active_session) {
    sessBadge.style.borderColor = "rgba(16, 185, 129, 0.4)";
    sessIcon.innerText = "🟢";
    sessText.innerText = "Active Session (High Vol)";
    sessText.style.color = "#10b981";
  } else {
    sessBadge.style.borderColor = "rgba(245, 158, 11, 0.4)";
    sessIcon.innerText = "🌙";
    sessText.innerText = "Session Sleep (Off-Hours)";
    sessText.style.color = "#f59e0b";
  }

  // Win Rate
  document.getElementById("kpi-winrate").innerText = `${summary.win_rate || 0.0}%`;
  document.getElementById("kpi-trades-count").innerText = `${summary.total_trades || 0} Trades`;
  document.getElementById("kpi-wins").innerText = summary.winning_trades || 0;
  document.getElementById("kpi-losses").innerText = summary.losing_trades || 0;

  // Open Positions Table
  renderPositions(state.positions || []);

  // History Table
  renderHistory(state.recent_trades || []);

  // Terminal Logs
  if (state.recent_logs) {
    renderLogs(state.recent_logs);
  }
}

// Render Positions Table
function renderPositions(positions) {
  const tbody = document.getElementById("positions-tbody");
  document.getElementById("positions-count-badge").innerText = `${positions.length} Active`;

  if (!positions || positions.length === 0) {
    tbody.innerHTML = `<tr class="empty-row"><td colspan="9">No active positions. Bot is scanning the orderbook for high-probability setups...</td></tr>`;
    return;
  }

  tbody.innerHTML = positions.map(p => {
    const isLong = p.side === "long";
    const sideClass = isLong ? "side-long" : "side-short";
    const pnl = p.unrealized_pnl || 0.0;
    const pnlPct = p.unrealized_pnl_pct || 0.0;
    const pnlColorClass = pnl >= 0 ? "text-green" : "text-red";
    const cleanSym = (p.symbol || "").split(":")[0];

    return `
      <tr>
        <td><strong>${cleanSym}</strong></td>
        <td><span class="side-badge ${sideClass}">${p.side.toUpperCase()} ${p.leverage}x</span></td>
        <td>$${p.entry_price}</td>
        <td>$${p.current_price || p.mark_price || p.entry_price}</td>
        <td>$${p.margin}</td>
        <td><span class="text-green">$${p.tp1}</span> / $${p.tp2}</td>
        <td><span class="text-red">$${p.sl}</span></td>
        <td class="${pnlColorClass}"><strong>${pnl >= 0 ? '+' : ''}$${pnl} (${pnl >= 0 ? '+' : ''}${pnlPct}%)</strong></td>
        <td>
          <button class="btn-table-close" onclick="closePosition('${p.symbol}')">Market Close</button>
        </td>
      </tr>
    `;
  }).join("");
}

// Render History Table
function renderHistory(history) {
  const tbody = document.getElementById("history-table");
  document.getElementById("history-count-badge").innerText = `${history.length} Completed`;

  if (!history || history.length === 0) {
    return;
  }

  const rows = [...history].reverse();
  tbody.innerHTML = rows.map(t => {
    const isLong = t.side === "long";
    const sideClass = isLong ? "side-long" : "side-short";
    const pnl = t.pnl || 0.0;
    const pnlPct = t.pnl_percent || 0.0;
    const pnlColor = pnl >= 0 ? "text-green" : "text-red";
    const timeStr = t.close_time ? new Date(t.close_time * 1000).toLocaleTimeString() : "--:--";
    const cleanSym = (t.symbol || "").split(":")[0];

    return `
      <tr>
        <td style="color:var(--text-sub)">${timeStr}</td>
        <td><strong>${cleanSym}</strong></td>
        <td><span class="side-badge ${sideClass}">${t.side.toUpperCase()}</span></td>
        <td>$${t.entry_price}</td>
        <td>$${t.exit_price}</td>
        <td class="${pnlColor}"><strong>${pnl >= 0 ? '+' : ''}$${pnl}</strong></td>
        <td class="${pnlColor}">${pnl >= 0 ? '+' : ''}${pnlPct}%</td>
        <td><span style="font-size:0.75rem; color:var(--text-muted)">${t.reason}</span></td>
      </tr>
    `;
  }).join("");
}

// Render Terminal Logs
let lastLogCount = 0;
function renderLogs(logs) {
  const container = document.getElementById("terminal-logs");
  if (!logs || logs.length === 0) return;

  container.innerHTML = logs.map(l => {
    let lvlClass = "log-info";
    if (l.level === "success") lvlClass = "log-success";
    if (l.level === "warning") lvlClass = "log-warning";
    if (l.level === "error") lvlClass = "log-error";
    return `<div class="log-line ${lvlClass}"><span class="log-time">[${l.time}]</span> ${l.message}</div>`;
  }).join("");

  if (logs.length !== lastLogCount) {
    container.scrollTop = container.scrollHeight;
    lastLogCount = logs.length;
  }
}

function clearLogs() {
  document.getElementById("terminal-logs").innerHTML = `<div class="log-line log-info"><span class="log-time">[--:--:--]</span> Terminal logs cleared.</div>`;
}

// API Actions
async function toggleBot() {
  try {
    const res = await fetch("/api/bot/toggle", { method: "POST" });
    const data = await res.json();
    playNotificationSound('success');
  } catch (e) {
    console.error("Toggle bot failed", e);
  }
}

async function toggleTradingMode() {
  const nextMode = currentTradingMode === "PAPER" ? "LIVE (Real MEXC Funds)" : "PAPER SIMULATION";
  if (currentTradingMode === "PAPER") {
    if (!confirm("⚠️ Switch to LIVE trading mode?\n\nReal orders will be placed on your MEXC Futures account. Ensure you have added your API Key & Secret in Settings.")) {
      return;
    }
  }
  try {
    const res = await fetch("/api/bot/mode", { method: "POST" });
    const data = await res.json();
    playNotificationSound('success');
  } catch (e) {
    console.error("Toggle mode failed", e);
  }
}

async function closePosition(symbol) {
  if (!confirm(`Confirm closing position on ${symbol}?`)) return;
  try {
    const res = await fetch("/api/positions/close", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol })
    });
    if (res.ok) {
      playNotificationSound('success');
    }
  } catch (e) {
    console.error("Close position error", e);
  }
}

async function resetPaperAccount() {
  if (!confirm("Are you sure you want to reset the Paper Simulation balance back to $1,000.00?")) return;
  try {
    await fetch("/api/paper/reset", { method: "POST" });
    playNotificationSound('success');
  } catch (e) {
    console.error("Reset paper error", e);
  }
}

// Chart Handling
async function loadChartData() {
  const symbol = document.getElementById("chart-symbol-select").value;
  try {
    const res = await fetch(`/api/chart/${symbol}`);
    const data = await res.json();
    if (!data.candles || data.candles.length === 0) return;

    const labels = data.candles.map(c => new Date(c.time * 1000).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}));
    const closePrices = data.candles.map(c => c.close);
    const ema9 = data.candles.map(c => c.ema9);
    const ema21 = data.candles.map(c => c.ema21);
    const ema50 = data.candles.map(c => c.ema50);

    // Update watchlist item
    const lastPrice = closePrices[closePrices.length - 1];
    if (symbol.includes("BTC")) document.getElementById("watch-price-btc").innerText = `$${lastPrice.toLocaleString()}`;
    if (symbol.includes("ETH")) document.getElementById("watch-price-eth").innerText = `$${lastPrice.toLocaleString()}`;
    if (symbol.includes("SOL")) document.getElementById("watch-price-sol").innerText = `$${lastPrice.toLocaleString()}`;

    if (!scalperChart) {
      const ctx = document.getElementById("scalperChart").getContext("2d");
      scalperChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Price',
              data: closePrices,
              borderColor: '#10b981',
              backgroundColor: 'rgba(16, 185, 129, 0.05)',
              fill: true,
              borderWidth: 2,
              tension: 0.1,
              pointRadius: 0
            },
            {
              label: 'EMA 9',
              data: ema9,
              borderColor: '#f59e0b',
              borderWidth: 1.5,
              pointRadius: 0,
              borderDash: [2, 2]
            },
            {
              label: 'EMA 21',
              data: ema21,
              borderColor: '#06b6d4',
              borderWidth: 1.5,
              pointRadius: 0
            },
            {
              label: 'EMA 50',
              data: ema50,
              borderColor: '#a855f7',
              borderWidth: 1.5,
              pointRadius: 0
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              mode: 'index',
              intersect: false,
              backgroundColor: 'rgba(15, 23, 42, 0.95)',
              titleColor: '#94a3b8',
              bodyFont: { family: 'JetBrains Mono' }
            }
          },
          scales: {
            x: {
              grid: { color: 'rgba(255, 255, 255, 0.03)' },
              ticks: { color: '#64748b', maxTicksLimit: 8 }
            },
            y: {
              position: 'right',
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } }
            }
          }
        }
      });
    } else {
      scalperChart.data.labels = labels;
      scalperChart.data.datasets[0].data = closePrices;
      scalperChart.data.datasets[1].data = ema9;
      scalperChart.data.datasets[2].data = ema21;
      scalperChart.data.datasets[3].data = ema50;
      scalperChart.update();
    }
  } catch (e) {
    console.error("Chart load error", e);
  }
}

// Modal handling
function openSettingsModal() {
  fetch("/api/settings")
    .then(r => r.json())
    .then(cfg => {
      document.getElementById("set-mode").value = cfg.trading_mode || "PAPER";
      document.getElementById("set-leverage").value = cfg.leverage || 3;
      document.getElementById("set-mexc-key").value = cfg.mexc_api_key || "";
      document.getElementById("set-mexc-secret").value = cfg.mexc_api_secret || "";
      document.getElementById("set-daily-target").value = cfg.daily_profit_target_percent || 1.0;
      document.getElementById("set-daily-loss").value = cfg.daily_max_loss_percent || 3.0;
      document.getElementById("set-risk-trade").value = cfg.risk_per_trade_percent || 1.5;
      document.getElementById("set-max-pos").value = cfg.max_open_positions || 2;
      document.getElementById("set-sl").value = cfg.stop_loss_percent || 0.8;
      document.getElementById("set-tp1").value = cfg.take_profit_1_percent || 1.2;
      document.getElementById("set-tp2").value = cfg.take_profit_2_percent || 2.4;
      document.getElementById("set-session-filter").value = (cfg.enable_session_filter !== false).toString();
      document.getElementById("set-session-start").value = cfg.session_start_utc ?? 12;
      document.getElementById("set-session-end").value = cfg.session_end_utc ?? 21;
      document.getElementById("set-tg-token").value = cfg.telegram_bot_token || "";
      document.getElementById("set-tg-chat").value = cfg.telegram_chat_id || "";
      document.getElementById("settings-modal").style.display = "flex";
    });
}

function closeSettingsModal() {
  document.getElementById("settings-modal").style.display = "none";
}

async function saveSettings(event) {
  event.preventDefault();
  const payload = {
    trading_mode: document.getElementById("set-mode").value,
    leverage: parseInt(document.getElementById("set-leverage").value),
    mexc_api_key: document.getElementById("set-mexc-key").value,
    mexc_api_secret: document.getElementById("set-mexc-secret").value,
    daily_profit_target_percent: parseFloat(document.getElementById("set-daily-target").value),
    daily_max_loss_percent: parseFloat(document.getElementById("set-daily-loss").value),
    risk_per_trade_percent: parseFloat(document.getElementById("set-risk-trade").value),
    max_open_positions: parseInt(document.getElementById("set-max-pos").value),
    stop_loss_percent: parseFloat(document.getElementById("set-sl").value),
    take_profit_1_percent: parseFloat(document.getElementById("set-tp1").value),
    take_profit_2_percent: parseFloat(document.getElementById("set-tp2").value),
    enable_session_filter: document.getElementById("set-session-filter").value === "true",
    session_start_utc: parseInt(document.getElementById("set-session-start").value),
    session_end_utc: parseInt(document.getElementById("set-session-end").value),
    telegram_bot_token: document.getElementById("set-tg-token").value,
    telegram_chat_id: document.getElementById("set-tg-chat").value,
    symbols: ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT"],
    timeframe: "15m",
    use_trailing_stop: true
  };

  try {
    const res = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      playNotificationSound('success');
      closeSettingsModal();
    }
  } catch (e) {
    console.error("Save settings error", e);
  }
}

// On page load
window.addEventListener("DOMContentLoaded", () => {
  connectWebSocket();
  loadChartData();
  // Refresh chart every 10 seconds
  setInterval(loadChartData, 10000);
});
