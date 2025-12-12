"""Dashboard HTML template with full trading functionality."""


def get_dashboard_html() -> str:
    """Generate dashboard HTML with full trading functionality."""
    return '''
<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crypto Futures Trading Bot</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-main: #f8fafc;
            --bg-card: #ffffff;
            --bg-hover: #f1f5f9;
            --border: #e2e8f0;
            --text: #1e293b;
            --text-muted: #64748b;
            --accent: #2563eb;
            --green: #10b981;
            --red: #ef4444;
            --yellow: #f59e0b;
            --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
            --shadow-lg: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.06);
        }
        /* Dark theme */
        [data-theme="dark"] {
            --bg-main: #0f172a;
            --bg-card: #1e293b;
            --bg-hover: #334155;
            --border: #475569;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --accent: #3b82f6;
            --green: #22c55e;
            --red: #f87171;
            --yellow: #fbbf24;
            --shadow: 0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2);
            --shadow-lg: 0 4px 6px rgba(0,0,0,0.4), 0 2px 4px rgba(0,0,0,0.3);
        }
        /* Theme toggle */
        .theme-toggle { width: 36px; height: 36px; border-radius: 8px; background: var(--bg-hover); border: none;
                        cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 16px;
                        transition: all 0.2s; }
        .theme-toggle:hover { background: var(--border); }
        * { font-family: 'Inter', sans-serif; box-sizing: border-box; margin: 0; padding: 0; }
        body { background: var(--bg-main); color: var(--text); min-height: 100vh; font-size: 13px; }

        /* Cards */
        .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; box-shadow: var(--shadow); }

        /* Header */
        .header { background: var(--bg-card); border-bottom: 1px solid var(--border); padding: 10px 16px; box-shadow: var(--shadow); position: sticky; top: 0; z-index: 50; }

        /* Mode Pills */
        .mode-pill { padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 500; cursor: pointer;
                     transition: all 0.2s; background: transparent; color: var(--text-muted); border: none; }
        .mode-pill:hover { color: var(--text); background: var(--bg-hover); }
        .mode-pill.active { color: white; }
        .mode-paper.active { background: var(--accent); }
        .mode-live.active { background: var(--red); }
        .mode-backtest.active { background: #7c3aed; }

        /* Exchange Chips */
        .exchange-chip { padding: 5px 12px; border-radius: 6px; font-size: 11px; font-weight: 500;
                         cursor: pointer; transition: all 0.2s; border: 1px solid var(--border);
                         background: var(--bg-card); color: var(--text-muted); display: flex; align-items: center; gap: 5px; }
        .exchange-chip:hover { border-color: var(--accent); color: var(--accent); }
        .exchange-chip.active { border-color: var(--accent); background: rgba(37,99,235,0.1); color: var(--accent); }

        /* Coin Selection */
        .coin-item { transition: all 0.2s; cursor: pointer; padding: 8px 10px; border-radius: 8px; }
        .coin-item:hover { background: var(--bg-hover); }
        .coin-item.selected { background: rgba(37,99,235,0.1); border-left: 3px solid var(--accent); }

        /* P&L Colors */
        .pnl-positive, .text-green { color: var(--green) !important; }
        .pnl-negative, .text-red { color: var(--red) !important; }

        /* Buttons */
        .btn { padding: 8px 14px; border-radius: 6px; font-size: 11px; font-weight: 600; transition: all 0.2s;
               border: none; cursor: pointer; text-transform: uppercase; letter-spacing: 0.5px; }
        .btn-success { background: var(--green); color: white; }
        .btn-danger { background: var(--red); color: white; }
        .btn-warning { background: var(--yellow); color: white; }
        .btn:hover { opacity: 0.9; transform: translateY(-1px); box-shadow: var(--shadow); }
        .btn:active { transform: translateY(0); }

        /* Status */
        .status-running { background: rgba(16,185,129,0.15); color: var(--green); }
        .status-stopped { background: rgba(239,68,68,0.15); color: var(--red); }
        .status-paused { background: rgba(245,158,11,0.15); color: var(--yellow); }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

        /* Live pulse */
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
        .live-pulse { animation: pulse 1.5s infinite; }

        /* Toast */
        .toast { position: fixed; bottom: 20px; right: 20px; padding: 12px 24px; border-radius: 8px;
                 font-size: 13px; font-weight: 500; z-index: 1000; animation: slideIn 0.3s ease; box-shadow: var(--shadow-lg); }
        @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }

        /* Modal */
        .modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.4); backdrop-filter: blur(4px);
                         display: flex; align-items: center; justify-content: center; z-index: 100; }
        .modal-content { background: var(--bg-card); border-radius: 12px; max-width: 400px; width: 90%; box-shadow: var(--shadow-lg); }

        /* Input */
        .input { background: var(--bg-main); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px;
                 width: 100%; color: var(--text); font-size: 13px; transition: all 0.2s; }
        .input:focus { border-color: var(--accent); outline: none; box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }
        .input::placeholder { color: var(--text-muted); }

        /* Select */
        .select { background: var(--bg-main); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px;
                  color: var(--text); font-size: 12px; cursor: pointer; }

        /* Tabs */
        .tab-btn { padding: 10px 18px; font-size: 13px; font-weight: 500; color: var(--text-muted);
                   border-bottom: 2px solid transparent; transition: all 0.2s; background: transparent;
                   border-top: none; border-left: none; border-right: none; cursor: pointer; }
        .tab-btn:hover { color: var(--text); background: var(--bg-hover); }
        .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }

        /* Strategy Item */
        .strategy-item { display: flex; align-items: center; justify-content: space-between; padding: 8px 12px;
                         border-radius: 8px; cursor: pointer; transition: all 0.2s; font-size: 13px; }
        .strategy-item:hover { background: var(--bg-hover); }
        .strategy-item.active { background: rgba(37,99,235,0.1); }
        .strategy-item .status-dot { width: 8px; height: 8px; border-radius: 50%; }
        .strategy-item .status-dot.on { background: var(--green); box-shadow: 0 0 8px var(--green); }
        .strategy-item .status-dot.off { background: var(--border); }

        /* Strategy Group */
        .strategy-group { margin-bottom: 16px; }
        .strategy-group-title { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 8px; font-weight: 600; }

        /* Position Row */
        .position-row { padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 12px; transition: background 0.15s; }
        .position-row:last-child { border-bottom: none; }
        .position-row:hover { background: var(--bg-hover); }

        /* Trade Row */
        .trade-row { padding: 8px 12px; border-bottom: 1px solid var(--border); font-size: 12px; transition: background 0.15s; }
        .trade-row:last-child { border-bottom: none; }
        .trade-row:hover { background: var(--bg-hover); }

        /* Summary Badge */
        .summary-badge { display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px;
                         background: var(--bg-hover); border-radius: 6px; font-size: 11px; }

        /* Stats */
        .stat-value { font-size: 20px; font-weight: 700; font-feature-settings: 'tnum'; }
        .stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); font-weight: 500; }

        /* Progress bar */
        .progress-bar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
        .progress-fill { height: 100%; background: var(--accent); transition: width 0.3s; }

        /* Chart Period Buttons */
        .chart-period-btn { padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: 500;
                           background: var(--bg-hover); border: none; cursor: pointer; color: var(--text-muted);
                           transition: all 0.2s; }
        .chart-period-btn:hover { background: var(--border); color: var(--text); }
        .chart-period-btn.active { background: var(--accent); color: white; }

        /* Calculator Modal */
        .calc-input { background: var(--bg-main); border: 1px solid var(--border); border-radius: 6px;
                      padding: 8px 12px; width: 100%; color: var(--text); font-size: 13px; }
        .calc-result { background: var(--bg-hover); padding: 12px; border-radius: 8px; }
        .calc-result .value { font-size: 18px; font-weight: 700; }

        /* Draggable Cards */
        .draggable-card { transition: transform 0.2s, box-shadow 0.2s; }
        .draggable-card.dragging { opacity: 0.5; transform: scale(1.02); box-shadow: var(--shadow-lg); }
        .draggable-card.drag-over { border: 2px dashed var(--accent); }

        /* Export Button */
        .btn-export { padding: 5px 10px; border-radius: 4px; font-size: 11px; background: var(--bg-hover);
                      border: 1px solid var(--border); cursor: pointer; color: var(--text-muted); transition: all 0.2s; }
        .btn-export:hover { background: var(--border); color: var(--text); }

        /* Sound toggle */
        .sound-toggle { padding: 4px 8px; border-radius: 4px; font-size: 11px; background: transparent;
                        border: none; cursor: pointer; color: var(--text-muted); }
        .sound-toggle.active { color: var(--accent); }

        /* Desktop Grid layout */
        .main-grid { display: grid; grid-template-columns: 200px 1fr 260px; gap: 16px; padding: 16px; min-height: calc(100vh - 60px); }
        .panel { display: flex; flex-direction: column; overflow: hidden; }

        /* Mobile Header */
        .mobile-menu-btn { display: none; }
        .header-desktop { display: flex; }
        .header-mobile { display: none; }

        /* Mobile Responsive */
        @media (max-width: 1024px) {
            .main-grid { grid-template-columns: 180px 1fr; }
            .main-grid > .panel:last-child { display: none; }
        }

        @media (max-width: 768px) {
            .header { padding: 8px 12px; }
            .header-desktop { display: none; }
            .header-mobile { display: flex; flex-direction: column; gap: 10px; }
            .mobile-menu-btn { display: flex; align-items: center; justify-content: center; width: 36px; height: 36px;
                               border-radius: 8px; background: var(--bg-hover); border: none; cursor: pointer; }

            .main-grid { display: flex; flex-direction: column; gap: 12px; padding: 12px; height: auto; min-height: auto; }
            .panel { width: 100%; max-height: none; }

            /* Mobile cards */
            .card { border-radius: 10px; }

            /* Mobile stats */
            .stats-row { display: grid !important; grid-template-columns: repeat(2, 1fr) !important; gap: 8px !important; }
            .stats-row > div { text-align: center; padding: 12px !important; background: var(--bg-hover); border-radius: 8px; }
            .stat-value { font-size: 16px; }

            /* Mobile positions/trades */
            .position-row, .trade-row { flex-direction: column; align-items: flex-start !important; gap: 6px; }
            .position-row > div, .trade-row > div { width: 100%; justify-content: space-between; }

            /* Mobile tabs */
            .tab-btn { padding: 10px 14px; font-size: 12px; flex: 1; text-align: center; }

            /* Mobile buttons */
            .btn { padding: 10px 16px; font-size: 12px; }

            /* Mobile strategies - horizontal scroll */
            .strategies-mobile { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 8px; -webkit-overflow-scrolling: touch; }
            .strategies-mobile::-webkit-scrollbar { display: none; }
            .strategy-chip { flex-shrink: 0; padding: 8px 14px; background: var(--bg-hover); border-radius: 20px; font-size: 12px;
                            display: flex; align-items: center; gap: 6px; cursor: pointer; transition: all 0.2s; }
            .strategy-chip.active { background: rgba(37,99,235,0.15); color: var(--accent); }
            .strategy-chip .status-dot { width: 6px; height: 6px; border-radius: 50%; }

            /* Hide sidebar on mobile */
            .sidebar-panel { display: none; }

            /* Mobile Quick Trade */
            .quick-trade-mobile { position: fixed; bottom: 0; left: 0; right: 0; background: var(--bg-card);
                                  border-top: 1px solid var(--border); padding: 12px 16px; z-index: 40; box-shadow: 0 -4px 12px rgba(0,0,0,0.1); }
            .quick-trade-mobile .btn { flex: 1; }
        }

        @media (max-width: 480px) {
            .stats-row { grid-template-columns: repeat(2, 1fr) !important; }
            .stats-row > div:nth-child(5) { grid-column: span 2; }
            .exchange-chip span:last-child { display: none; }
            .mode-pill { padding: 6px 10px; font-size: 11px; }
        }

        /* Touch-friendly spacing */
        @media (hover: none) and (pointer: coarse) {
            .coin-item, .strategy-item, .position-row, .trade-row { padding: 12px; }
            .btn { min-height: 44px; }
            .input, .select { min-height: 44px; }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <!-- Desktop Header -->
        <div class="header-desktop items-center justify-between">
            <!-- Logo & Exchange -->
            <div class="flex items-center gap-4">
                <div class="flex items-center gap-2">
                    <div class="w-9 h-9 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center text-white text-sm font-bold shadow-md">F</div>
                    <span class="font-semibold">Futures Bot</span>
                </div>
                <div class="flex items-center gap-2" id="exchangeChips"></div>
            </div>

            <!-- Mode Selector -->
            <div class="flex items-center gap-1 bg-[var(--bg-main)] p-1 rounded-lg">
                <button class="mode-pill mode-paper active" data-mode="paper" onclick="setMode('paper')">Paper</button>
                <button class="mode-pill mode-live" data-mode="live" onclick="setMode('live')">Live</button>
                <button class="mode-pill mode-backtest" data-mode="backtest" onclick="setMode('backtest')">Backtest</button>
            </div>

            <!-- Balance & Status & Controls -->
            <div class="flex items-center gap-5">
                <div class="text-right">
                    <div class="stat-label">Balance</div>
                    <div class="stat-value" id="balanceDisplay">$1,000.00</div>
                </div>
                <div class="flex items-center gap-2">
                    <div class="w-2.5 h-2.5 rounded-full bg-[var(--text-muted)]" id="statusDot"></div>
                    <span id="statusText" class="px-3 py-1.5 rounded-lg text-xs font-medium status-stopped">Stopped</span>
                </div>
                <div class="flex gap-2">
                    <button onclick="sendCommand('start')" class="btn btn-success">Start</button>
                    <button onclick="sendCommand('pause')" class="btn btn-warning">Pause</button>
                    <button onclick="sendCommand('stop')" class="btn btn-danger">Stop</button>
                </div>
                <div class="flex items-center gap-1 ml-2">
                    <button onclick="showCalculator()" class="theme-toggle" title="Position Calculator">🧮</button>
                    <button onclick="toggleSound()" class="sound-toggle" id="soundToggleBtn" title="Sound Notifications">🔔</button>
                    <button onclick="toggleTheme()" class="theme-toggle" id="themeToggleBtn" title="Toggle Dark Mode">🌙</button>
                </div>
            </div>
        </div>

        <!-- Mobile Header -->
        <div class="header-mobile">
            <div class="flex items-center justify-between">
                <div class="flex items-center gap-3">
                    <div class="w-9 h-9 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center text-white text-sm font-bold shadow-md">F</div>
                    <div>
                        <div class="font-semibold text-sm">Futures Bot</div>
                        <div class="flex items-center gap-1">
                            <div class="w-2 h-2 rounded-full bg-[var(--text-muted)]" id="statusDotMobile"></div>
                            <span class="text-xs text-[var(--text-muted)]" id="statusTextMobile">Stopped</span>
                        </div>
                    </div>
                </div>
                <div class="flex items-center gap-3">
                    <div class="text-right">
                        <div class="stat-label">Balance</div>
                        <div class="font-bold text-lg" id="balanceDisplayMobile">$1,000</div>
                    </div>
                </div>
            </div>
            <div class="flex items-center justify-between gap-2">
                <div class="flex items-center gap-1 bg-[var(--bg-main)] p-1 rounded-lg flex-1">
                    <button class="mode-pill mode-paper active flex-1" data-mode="paper" onclick="setMode('paper')">Paper</button>
                    <button class="mode-pill mode-live flex-1" data-mode="live" onclick="setMode('live')">Live</button>
                    <button class="mode-pill mode-backtest flex-1" data-mode="backtest" onclick="setMode('backtest')">Test</button>
                </div>
                <div class="flex gap-1">
                    <button onclick="sendCommand('start')" class="btn btn-success">▶</button>
                    <button onclick="sendCommand('stop')" class="btn btn-danger">■</button>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="main-grid">
        <!-- Left Panel: Symbols -->
        <div class="panel sidebar-panel">
            <div class="card p-4 h-full">
                <div class="flex items-center justify-between mb-3">
                    <span class="font-semibold">Symbols</span>
                    <span class="text-xs text-[var(--text-muted)] bg-[var(--bg-hover)] px-2 py-0.5 rounded-full" id="selectedCount">1</span>
                </div>
                <input type="text" class="input mb-3" placeholder="Search symbols..." id="symbolSearch" oninput="filterSymbols()">
                <div id="symbolList" class="flex-1 overflow-y-auto space-y-1" style="max-height: calc(100vh - 220px);">
                </div>
                <button onclick="showAddSymbolModal()" class="w-full mt-3 py-2.5 border-2 border-dashed border-[var(--border)] rounded-lg text-[var(--text-muted)] hover:border-[var(--accent)] hover:text-[var(--accent)] text-sm transition-all hover:bg-[var(--bg-hover)]">
                    + Add Symbol
                </button>
            </div>
        </div>

        <!-- Center Panel: Stats + Positions/History -->
        <div class="panel space-y-4">
            <!-- Stats Row -->
            <div class="card p-4">
                <div class="stats-row flex items-center gap-6">
                    <div>
                        <div class="stat-label">Total P&L</div>
                        <div class="stat-value pnl-positive" id="totalPnl">$0.00</div>
                    </div>
                    <div>
                        <div class="stat-label">Daily</div>
                        <div class="stat-value pnl-positive" id="dailyPnl">$0.00</div>
                    </div>
                    <div>
                        <div class="stat-label">Win Rate</div>
                        <div class="stat-value" id="winRate">0%</div>
                    </div>
                    <div>
                        <div class="stat-label">Trades</div>
                        <div class="stat-value" id="totalTrades">0</div>
                    </div>
                    <div>
                        <div class="stat-label">Drawdown</div>
                        <div class="stat-value text-red" id="maxDrawdown">0%</div>
                    </div>
                </div>
            </div>

            <!-- Tabbed Content -->
            <div class="card flex-1 flex flex-col overflow-hidden">
                <!-- Tab Headers -->
                <div class="flex border-b border-[var(--border)]">
                    <button class="tab-btn active" data-tab="positions" onclick="switchTab('positions')">
                        Positions <span id="positionsCount" class="ml-1 px-1.5 py-0.5 bg-[var(--bg-hover)] rounded text-xs">0</span>
                    </button>
                    <button class="tab-btn" data-tab="history" onclick="switchTab('history')">
                        History <span id="historyCount" class="ml-1 px-1.5 py-0.5 bg-[var(--bg-hover)] rounded text-xs">0</span>
                    </button>
                    <button class="tab-btn" data-tab="charts" onclick="switchTab('charts')">
                        📊 Charts
                    </button>
                </div>

                <!-- Tab Content: Positions -->
                <div id="tabPositions" class="tab-content flex-1 overflow-hidden flex flex-col">
                    <div class="flex items-center justify-between p-2 border-b border-[var(--border)]">
                        <span class="text-xs text-[var(--text-muted)]">Open Positions</span>
                        <button onclick="closeAllPositions()" class="text-xs text-red hover:bg-[var(--bg-hover)] px-2 py-1 rounded transition-colors">Close All</button>
                    </div>
                    <div id="positionsList" class="flex-1 overflow-y-auto">
                        <p class="text-[var(--text-muted)] text-xs text-center py-4">No positions</p>
                    </div>
                </div>

                <!-- Tab Content: History -->
                <div id="tabHistory" class="tab-content flex-1 overflow-hidden flex flex-col hidden">
                    <div class="flex items-center justify-between p-2 border-b border-[var(--border)]">
                        <div class="flex items-center gap-2">
                            <select id="historyFilter" class="select" onchange="filterTradeHistory()">
                                <option value="all">All</option>
                            </select>
                            <select id="historyPeriod" class="select" onchange="filterTradeHistory()">
                                <option value="all">All Time</option>
                                <option value="today">Today</option>
                                <option value="week">This Week</option>
                                <option value="month">This Month</option>
                                <option value="custom">Custom</option>
                            </select>
                        </div>
                        <div class="flex items-center gap-2">
                            <span id="historyStats" class="text-xs text-[var(--text-muted)]">0 trades</span>
                            <button onclick="exportHistoryCSV()" class="btn-export" title="Export CSV">📥 CSV</button>
                        </div>
                    </div>
                    <!-- Custom Date Range (hidden by default) -->
                    <div id="customDateRange" class="hidden p-2 border-b border-[var(--border)] bg-[var(--bg-hover)]">
                        <div class="flex items-center gap-2">
                            <input type="date" id="historyStartDate" class="input text-xs" onchange="filterTradeHistory()">
                            <span class="text-[var(--text-muted)]">-</span>
                            <input type="date" id="historyEndDate" class="input text-xs" onchange="filterTradeHistory()">
                            <button onclick="loadFromDB()" class="btn btn-sm">Load from DB</button>
                        </div>
                    </div>
                    <div id="strategySummary" class="p-2 border-b border-[var(--border)]"></div>
                    <div id="tradesList" class="flex-1 overflow-y-auto">
                        <p class="text-[var(--text-muted)] text-xs text-center py-4">No trades</p>
                    </div>
                </div>

                <!-- Tab Content: Charts -->
                <div id="tabCharts" class="tab-content flex-1 overflow-hidden flex flex-col hidden">
                    <div class="flex items-center justify-between p-2 border-b border-[var(--border)]">
                        <div class="flex items-center gap-2">
                            <span class="text-xs font-medium">Period:</span>
                            <button class="chart-period-btn active" data-period="7" onclick="setChartPeriod(7)">7D</button>
                            <button class="chart-period-btn" data-period="30" onclick="setChartPeriod(30)">30D</button>
                            <button class="chart-period-btn" data-period="0" onclick="setChartPeriod(0)">All</button>
                        </div>
                        <div class="flex items-center gap-3">
                            <span id="chartTotalPnl" class="text-xs font-bold text-green">+$0.00</span>
                            <span id="chartWinRate" class="text-xs text-[var(--text-muted)]">WR: 0%</span>
                        </div>
                    </div>
                    <div class="flex-1 p-3 overflow-y-auto">
                        <!-- Equity Curve -->
                        <div class="mb-4">
                            <div class="text-xs font-medium text-[var(--text-muted)] mb-2">Equity Curve</div>
                            <div style="height: 180px;">
                                <canvas id="equityChart"></canvas>
                            </div>
                        </div>
                        <!-- Daily PnL -->
                        <div>
                            <div class="text-xs font-medium text-[var(--text-muted)] mb-2">Daily P&L</div>
                            <div style="height: 140px;">
                                <canvas id="dailyPnlChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Backtest Panel -->
            <div id="backtestPanel" class="card p-3 hidden">
                <div class="flex items-center justify-between mb-2">
                    <span class="font-semibold text-sm">Backtest</span>
                </div>
                <div class="grid grid-cols-4 gap-2 mb-2">
                    <div>
                        <label class="stat-label">Start</label>
                        <input type="date" id="btStartDate" class="input" value="2024-01-01">
                    </div>
                    <div>
                        <label class="stat-label">End</label>
                        <input type="date" id="btEndDate" class="input" value="2024-12-01">
                    </div>
                    <div>
                        <label class="stat-label">Balance</label>
                        <input type="number" id="btBalance" class="input" value="1000">
                    </div>
                    <div class="flex items-end">
                        <button onclick="runBacktest()" class="btn w-full" style="background: var(--accent); color: white;" id="btRunBtn">Run</button>
                    </div>
                </div>
                <div class="progress-bar mb-2">
                    <div class="progress-fill" id="btProgress" style="width: 0%"></div>
                </div>
                <div id="btResults" class="hidden grid grid-cols-4 gap-2 pt-2 border-t border-[var(--border)]">
                    <div class="text-center">
                        <div class="stat-label">P&L</div>
                        <div class="font-bold" id="btTotalPnl">$0</div>
                    </div>
                    <div class="text-center">
                        <div class="stat-label">Win%</div>
                        <div class="font-bold" id="btWinRate">0%</div>
                    </div>
                    <div class="text-center">
                        <div class="stat-label">Trades</div>
                        <div class="font-bold" id="btTotalTrades">0</div>
                    </div>
                    <div class="text-center">
                        <div class="stat-label">Final</div>
                        <div class="font-bold" id="btFinalBalance">$0</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Right Panel: Strategies -->
        <div class="panel">
            <div class="card p-4 h-full flex flex-col overflow-hidden">
                <div class="flex items-center justify-between mb-3">
                    <span class="font-semibold">Strategies</span>
                    <span class="text-xs text-[var(--text-muted)] bg-[var(--bg-hover)] px-2 py-0.5 rounded-full" id="activeStrategies">0 active</span>
                </div>
                <div id="strategiesGrouped" class="flex-1 overflow-y-auto space-y-4">
                    <div class="strategy-group">
                        <div class="strategy-group-title">⚡ Scalping</div>
                        <div id="strategyGroupScalping" class="space-y-1"></div>
                    </div>
                    <div class="strategy-group">
                        <div class="strategy-group-title">📊 Order Flow</div>
                        <div id="strategyGroupAnalysis" class="space-y-1"></div>
                    </div>
                    <div class="strategy-group">
                        <div class="strategy-group-title">💰 DCA & Grid</div>
                        <div id="strategyGroupDCA" class="space-y-1"></div>
                    </div>
                    <div class="strategy-group">
                        <div class="strategy-group-title">🔄 Other</div>
                        <div id="strategyGroupOther" class="space-y-1"></div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Toast Container -->
    <div id="toastContainer"></div>

    <!-- Add Symbol Modal -->
    <div id="addSymbolModal" class="modal-overlay hidden" onclick="if(event.target===this)closeAddSymbolModal()">
        <div class="modal-content p-4">
            <h3 class="font-semibold mb-3">Add Symbol</h3>
            <div class="space-y-3">
                <div class="relative">
                    <label class="stat-label">Symbol</label>
                    <input type="text" id="newSymbolInput" class="input" placeholder="BTCUSDT..."
                           oninput="this.value=this.value.toUpperCase();filterSuggestions(this.value)"
                           onkeydown="handleSuggestionKeydown(event)"
                           autocomplete="off">
                    <div id="symbolSuggestions" class="absolute top-full left-0 right-0 bg-[var(--bg-card)] border border-[var(--border)] rounded max-h-48 overflow-y-auto hidden z-50">
                    </div>
                </div>
                <div>
                    <label class="stat-label">Name</label>
                    <input type="text" id="newSymbolName" class="input" placeholder="Bitcoin">
                </div>
                <div class="flex gap-2">
                    <button onclick="addNewSymbol()" class="btn flex-1" style="background: var(--accent); color: white;">Add</button>
                    <button onclick="closeAddSymbolModal()" class="btn flex-1" style="background: var(--bg-hover);">Cancel</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Position Calculator Modal -->
    <div id="calculatorModal" class="modal-overlay hidden" onclick="if(event.target===this)closeCalculator()">
        <div class="modal-content p-4" style="max-width: 420px;">
            <div class="flex items-center justify-between mb-4">
                <h3 class="font-semibold">🧮 Position Calculator</h3>
                <button onclick="closeCalculator()" class="text-[var(--text-muted)] hover:text-[var(--text)]">✕</button>
            </div>
            <div class="space-y-3">
                <div class="grid grid-cols-2 gap-2">
                    <div>
                        <label class="stat-label">Account Balance ($)</label>
                        <input type="number" id="calcBalance" class="calc-input" value="1000" oninput="calculatePosition()">
                    </div>
                    <div>
                        <label class="stat-label">Risk per Trade (%)</label>
                        <input type="number" id="calcRisk" class="calc-input" value="1" step="0.1" oninput="calculatePosition()">
                    </div>
                </div>
                <div class="grid grid-cols-2 gap-2">
                    <div>
                        <label class="stat-label">Entry Price ($)</label>
                        <input type="number" id="calcEntry" class="calc-input" value="100000" oninput="calculatePosition()">
                    </div>
                    <div>
                        <label class="stat-label">Stop Loss ($)</label>
                        <input type="number" id="calcStopLoss" class="calc-input" value="99000" oninput="calculatePosition()">
                    </div>
                </div>
                <div class="grid grid-cols-2 gap-2">
                    <div>
                        <label class="stat-label">Take Profit ($)</label>
                        <input type="number" id="calcTakeProfit" class="calc-input" value="102000" oninput="calculatePosition()">
                    </div>
                    <div>
                        <label class="stat-label">Leverage (x)</label>
                        <input type="number" id="calcLeverage" class="calc-input" value="10" oninput="calculatePosition()">
                    </div>
                </div>

                <div class="calc-result mt-4">
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <div class="stat-label">Position Size</div>
                            <div class="value text-[var(--accent)]" id="calcPositionSize">$0.00</div>
                        </div>
                        <div>
                            <div class="stat-label">Quantity</div>
                            <div class="value" id="calcQuantity">0.0000</div>
                        </div>
                    </div>
                    <div class="grid grid-cols-3 gap-4 mt-3 pt-3 border-t border-[var(--border)]">
                        <div>
                            <div class="stat-label">Risk Amount</div>
                            <div class="text-red font-bold" id="calcRiskAmount">-$10.00</div>
                        </div>
                        <div>
                            <div class="stat-label">Potential Profit</div>
                            <div class="text-green font-bold" id="calcProfit">+$20.00</div>
                        </div>
                        <div>
                            <div class="stat-label">R:R Ratio</div>
                            <div class="font-bold" id="calcRR">1:2.0</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Audio for notifications -->
    <audio id="soundTrade" preload="auto">
        <source src="data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU" type="audio/wav">
    </audio>

    <script>
        // State
        let ws;
        let currentMode = 'paper';
        let selectedSymbols = ['BTCUSDT'];
        let availableSymbols = {};
        let strategies = {};
        let suggestionIndex = -1;
        let availableExchanges = {};
        let selectedExchanges = ['binance'];

        // Settings (persisted to localStorage)
        let settings = {
            darkMode: false,
            soundEnabled: true,
            notificationsEnabled: false,
        };

        // =========================================================================
        // Settings & LocalStorage
        // =========================================================================
        function loadSettings() {
            try {
                const saved = localStorage.getItem('tradingBotSettings');
                if (saved) {
                    settings = { ...settings, ...JSON.parse(saved) };
                }
                // Apply settings
                if (settings.darkMode) {
                    document.documentElement.setAttribute('data-theme', 'dark');
                    document.getElementById('themeToggleBtn').textContent = '☀️';
                }
                if (settings.soundEnabled) {
                    document.getElementById('soundToggleBtn').classList.add('active');
                }
                // Request notification permission if was enabled
                if (settings.notificationsEnabled && 'Notification' in window) {
                    Notification.requestPermission();
                }
            } catch (e) {
                console.warn('Failed to load settings:', e);
            }
        }

        function saveSettings() {
            try {
                localStorage.setItem('tradingBotSettings', JSON.stringify(settings));
            } catch (e) {
                console.warn('Failed to save settings:', e);
            }
        }

        // =========================================================================
        // Dark Mode
        // =========================================================================
        function toggleTheme() {
            settings.darkMode = !settings.darkMode;
            const btn = document.getElementById('themeToggleBtn');
            if (settings.darkMode) {
                document.documentElement.setAttribute('data-theme', 'dark');
                btn.textContent = '☀️';
            } else {
                document.documentElement.removeAttribute('data-theme');
                btn.textContent = '🌙';
            }
            saveSettings();
            // Re-render charts with new theme
            if (document.getElementById('tabCharts') && !document.getElementById('tabCharts').classList.contains('hidden')) {
                renderCharts();
            }
        }

        // =========================================================================
        // Sound Notifications
        // =========================================================================
        function toggleSound() {
            settings.soundEnabled = !settings.soundEnabled;
            const btn = document.getElementById('soundToggleBtn');
            btn.classList.toggle('active', settings.soundEnabled);
            saveSettings();
            showToast(`Sound ${settings.soundEnabled ? 'ON' : 'OFF'}`);
        }

        function playSound(type = 'trade') {
            if (!settings.soundEnabled) return;
            try {
                // Create oscillator for different sounds
                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                const oscillator = audioCtx.createOscillator();
                const gainNode = audioCtx.createGain();

                oscillator.connect(gainNode);
                gainNode.connect(audioCtx.destination);

                if (type === 'win') {
                    oscillator.frequency.value = 880; // High pitch for win
                    oscillator.type = 'sine';
                } else if (type === 'loss') {
                    oscillator.frequency.value = 330; // Low pitch for loss
                    oscillator.type = 'triangle';
                } else {
                    oscillator.frequency.value = 660; // Medium for trade
                    oscillator.type = 'sine';
                }

                gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);

                oscillator.start(audioCtx.currentTime);
                oscillator.stop(audioCtx.currentTime + 0.3);
            } catch (e) {
                console.warn('Sound playback failed:', e);
            }
        }

        // =========================================================================
        // Push Notifications
        // =========================================================================
        async function requestNotificationPermission() {
            if (!('Notification' in window)) {
                showToast('Notifications not supported', 'error');
                return false;
            }

            const permission = await Notification.requestPermission();
            settings.notificationsEnabled = permission === 'granted';
            saveSettings();
            return settings.notificationsEnabled;
        }

        function sendPushNotification(title, body, icon = '📊') {
            if (!settings.notificationsEnabled || Notification.permission !== 'granted') return;

            try {
                new Notification(title, {
                    body,
                    icon: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">' + icon + '</text></svg>',
                    tag: 'trading-bot',
                    renotify: true,
                });
            } catch (e) {
                console.warn('Notification failed:', e);
            }
        }

        // =========================================================================
        // Position Calculator
        // =========================================================================
        function showCalculator() {
            document.getElementById('calculatorModal').classList.remove('hidden');
            calculatePosition();
        }

        function closeCalculator() {
            document.getElementById('calculatorModal').classList.add('hidden');
        }

        function calculatePosition() {
            const balance = parseFloat(document.getElementById('calcBalance').value) || 0;
            const riskPercent = parseFloat(document.getElementById('calcRisk').value) || 0;
            const entry = parseFloat(document.getElementById('calcEntry').value) || 0;
            const stopLoss = parseFloat(document.getElementById('calcStopLoss').value) || 0;
            const takeProfit = parseFloat(document.getElementById('calcTakeProfit').value) || 0;
            const leverage = parseFloat(document.getElementById('calcLeverage').value) || 1;

            // Calculate risk amount
            const riskAmount = balance * (riskPercent / 100);

            // Calculate stop loss distance percentage
            const slDistance = Math.abs(entry - stopLoss) / entry;

            // Calculate position size (with leverage)
            let positionSize = 0;
            let quantity = 0;

            if (slDistance > 0 && entry > 0) {
                // Position size = Risk Amount / (SL% / Leverage)
                positionSize = riskAmount / slDistance;
                quantity = positionSize / entry;
            }

            // Calculate potential profit
            const tpDistance = Math.abs(takeProfit - entry) / entry;
            const potentialProfit = positionSize * tpDistance;

            // Calculate R:R ratio
            const rrRatio = slDistance > 0 ? (tpDistance / slDistance) : 0;

            // Update display
            document.getElementById('calcPositionSize').textContent = `$${positionSize.toFixed(2)}`;
            document.getElementById('calcQuantity').textContent = quantity.toFixed(6);
            document.getElementById('calcRiskAmount').textContent = `-$${riskAmount.toFixed(2)}`;
            document.getElementById('calcProfit').textContent = `+$${potentialProfit.toFixed(2)}`;
            document.getElementById('calcRR').textContent = `1:${rrRatio.toFixed(1)}`;
        }

        // =========================================================================
        // Export CSV
        // =========================================================================
        function exportHistoryCSV() {
            if (tradeHistory.length === 0) {
                showToast('No trades to export', 'error');
                return;
            }

            // CSV header
            const headers = ['Symbol', 'Side', 'Entry Price', 'Exit Price', 'P&L', 'P&L %', 'Strategy', 'Entry Time', 'Exit Time'];
            const rows = [headers.join(',')];

            // Add data rows
            tradeHistory.forEach(t => {
                const row = [
                    t.symbol || '',
                    t.side || '',
                    t.entry_price || 0,
                    t.exit_price || 0,
                    (t.pnl || 0).toFixed(2),
                    (t.pnl_pct || 0).toFixed(2),
                    t.strategy || 'manual',
                    t.entry_time || '',
                    t.closed_at || t.exit_time || '',
                ].map(v => `"${v}"`).join(',');
                rows.push(row);
            });

            // Create and download file
            const csv = rows.join('\\n');
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = `trade_history_${new Date().toISOString().split('T')[0]}.csv`;
            link.click();

            showToast(`Exported ${tradeHistory.length} trades`);
        }

        // =========================================================================
        // Real Exchange Data
        // =========================================================================
        async function fetchRealPrices() {
            try {
                const symbols = Object.keys(availableSymbols);
                if (symbols.length === 0) return;

                // Fetch from Binance public API
                const symbolsParam = symbols.map(s => `"${s}"`).join(',');
                const res = await fetch(`https://api.binance.com/api/v3/ticker/24hr?symbols=[${symbolsParam}]`);

                if (res.ok) {
                    const data = await res.json();
                    data.forEach(ticker => {
                        if (availableSymbols[ticker.symbol]) {
                            availableSymbols[ticker.symbol].price = parseFloat(ticker.lastPrice);
                            availableSymbols[ticker.symbol].change_24h = parseFloat(ticker.priceChangePercent);
                            availableSymbols[ticker.symbol].volume_24h = parseFloat(ticker.volume);
                        }
                    });
                    renderSymbols();
                }
            } catch (e) {
                console.warn('Failed to fetch real prices:', e);
            }
        }

        // All available crypto symbols for autocomplete
        const allCryptoSymbols = [
            // Top coins by market cap
            { symbol: 'BTCUSDT', name: 'Bitcoin' },
            { symbol: 'ETHUSDT', name: 'Ethereum' },
            { symbol: 'BNBUSDT', name: 'BNB' },
            { symbol: 'XRPUSDT', name: 'XRP' },
            { symbol: 'SOLUSDT', name: 'Solana' },
            { symbol: 'ADAUSDT', name: 'Cardano' },
            { symbol: 'DOGEUSDT', name: 'Dogecoin' },
            { symbol: 'AVAXUSDT', name: 'Avalanche' },
            { symbol: 'DOTUSDT', name: 'Polkadot' },
            { symbol: 'TRXUSDT', name: 'TRON' },
            { symbol: 'LINKUSDT', name: 'Chainlink' },
            { symbol: 'MATICUSDT', name: 'Polygon' },
            { symbol: 'SHIBUSDT', name: 'Shiba Inu' },
            { symbol: 'LTCUSDT', name: 'Litecoin' },
            { symbol: 'BCHUSDT', name: 'Bitcoin Cash' },
            { symbol: 'ATOMUSDT', name: 'Cosmos' },
            { symbol: 'UNIUSDT', name: 'Uniswap' },
            { symbol: 'XLMUSDT', name: 'Stellar' },
            { symbol: 'ETCUSDT', name: 'Ethereum Classic' },
            { symbol: 'NEARUSDT', name: 'NEAR Protocol' },
            { symbol: 'APTUSDT', name: 'Aptos' },
            { symbol: 'ARBUSDT', name: 'Arbitrum' },
            { symbol: 'OPUSDT', name: 'Optimism' },
            { symbol: 'INJUSDT', name: 'Injective' },
            { symbol: 'FILUSDT', name: 'Filecoin' },
            { symbol: 'ICPUSDT', name: 'Internet Computer' },
            { symbol: 'HBARUSDT', name: 'Hedera' },
            { symbol: 'VETUSDT', name: 'VeChain' },
            { symbol: 'ALGOUSDT', name: 'Algorand' },
            { symbol: 'FTMUSDT', name: 'Fantom' },
            // DeFi & Layer 2
            { symbol: 'AAVEUSDT', name: 'Aave' },
            { symbol: 'MKRUSDT', name: 'Maker' },
            { symbol: 'CRVUSDT', name: 'Curve DAO' },
            { symbol: 'COMPUSDT', name: 'Compound' },
            { symbol: 'SNXUSDT', name: 'Synthetix' },
            { symbol: 'LDOUSDT', name: 'Lido DAO' },
            { symbol: 'GMXUSDT', name: 'GMX' },
            { symbol: 'DYDXUSDT', name: 'dYdX' },
            { symbol: '1INCHUSDT', name: '1inch' },
            { symbol: 'SUSHIUSDT', name: 'SushiSwap' },
            // Gaming & Metaverse
            { symbol: 'SANDUSDT', name: 'The Sandbox' },
            { symbol: 'MANAUSDT', name: 'Decentraland' },
            { symbol: 'AXSUSDT', name: 'Axie Infinity' },
            { symbol: 'APEUSDT', name: 'ApeCoin' },
            { symbol: 'GALAUSDT', name: 'Gala' },
            { symbol: 'ENJUSDT', name: 'Enjin' },
            { symbol: 'IMXUSDT', name: 'Immutable X' },
            // Meme coins
            { symbol: 'PEPEUSDT', name: 'Pepe' },
            { symbol: 'FLOKIUSDT', name: 'Floki' },
            { symbol: 'BONKUSDT', name: 'Bonk' },
            { symbol: 'WIFUSDT', name: 'dogwifhat' },
            // Infrastructure
            { symbol: 'GRTUSDT', name: 'The Graph' },
            { symbol: 'RNDRUSDT', name: 'Render' },
            { symbol: 'FETUSDT', name: 'Fetch.ai' },
            { symbol: 'OCEANUSDT', name: 'Ocean Protocol' },
            { symbol: 'AGIXUSDT', name: 'SingularityNET' },
            // Others
            { symbol: 'RUNEUSDT', name: 'THORChain' },
            { symbol: 'EGLDUSDT', name: 'MultiversX' },
            { symbol: 'XMRUSDT', name: 'Monero' },
            { symbol: 'EOSUSDT', name: 'EOS' },
            { symbol: 'FLOWUSDT', name: 'Flow' },
            { symbol: 'QNTUSDT', name: 'Quant' },
            { symbol: 'CHZUSDT', name: 'Chiliz' },
            { symbol: 'STXUSDT', name: 'Stacks' },
            { symbol: 'KAVAUSDT', name: 'Kava' },
            { symbol: 'ZECUSDT', name: 'Zcash' },
            { symbol: 'NEOUSDT', name: 'NEO' },
            { symbol: 'XTZUSDT', name: 'Tezos' },
            { symbol: 'THETAUSDT', name: 'Theta' },
            { symbol: 'ZILUSDT', name: 'Zilliqa' },
            { symbol: 'IOSTUSDT', name: 'IOST' },
            { symbol: 'ONTUSDT', name: 'Ontology' },
            { symbol: 'WAVESUSDT', name: 'Waves' },
            { symbol: 'DASHUSDT', name: 'Dash' },
            { symbol: 'SUIUSDT', name: 'Sui' },
            { symbol: 'SEIUSDT', name: 'Sei' },
            { symbol: 'TIAUSDT', name: 'Celestia' },
            { symbol: 'JUPUSDT', name: 'Jupiter' },
            { symbol: 'PYTHUSDT', name: 'Pyth' },
            { symbol: 'WLDUSDT', name: 'Worldcoin' },
            { symbol: 'BLURUSDT', name: 'Blur' },
            { symbol: 'PENDLEUSDT', name: 'Pendle' },
            { symbol: 'ORDIUSDT', name: 'ORDI' },
            { symbol: 'KASUSDT', name: 'Kaspa' },
        ];

        // Demo data for testing UI
        const demoPositions = [
            {
                symbol: 'BTCUSDT', side: 'BUY', size: 0.05, size_usdt: 4894.62,
                entry_price: 97245.50, mark_price: 97892.30, unrealized_pnl: 32.34,
                leverage: 20, stop_loss: 96500.00, take_profit: 99000.00,
                opened_at: '2024-12-11 14:15:22'
            },
            {
                symbol: 'ETHUSDT', side: 'SELL', size: 1.2, size_usdt: 4416.54,
                entry_price: 3650.20, mark_price: 3680.45, unrealized_pnl: -36.30,
                leverage: 15, stop_loss: 3720.00, take_profit: 3550.00,
                opened_at: '2024-12-11 12:30:45'
            },
            {
                symbol: 'SOLUSDT', side: 'BUY', size: 25, size_usdt: 5650.00,
                entry_price: 220.50, mark_price: 226.00, unrealized_pnl: 137.50,
                leverage: 10, stop_loss: 215.00, take_profit: 240.00,
                opened_at: '2024-12-11 10:05:18'
            }
        ];

        const demoTrades = [
            { symbol: 'BTCUSDT', side: 'BUY', entry_price: 96850.00, exit_price: 97120.00, pnl: 27.00, pnl_pct: 0.28, strategy: 'hybrid_scalping', entry_time: '2024-12-11 14:15:22', closed_at: '2024-12-11 14:32:15' },
            { symbol: 'ETHUSDT', side: 'SELL', entry_price: 3620.50, exit_price: 3595.20, pnl: 25.30, pnl_pct: 0.70, strategy: 'advanced_orderbook', entry_time: '2024-12-11 13:20:45', closed_at: '2024-12-11 13:45:22' },
            { symbol: 'SOLUSDT', side: 'BUY', entry_price: 225.80, exit_price: 223.50, pnl: -11.50, pnl_pct: -1.02, strategy: 'dca', entry_time: '2024-12-11 11:45:30', closed_at: '2024-12-11 12:18:44' },
            { symbol: 'BTCUSDT', side: 'SELL', entry_price: 97500.00, exit_price: 97280.00, pnl: 22.00, pnl_pct: 0.23, strategy: 'impulse_scalping', entry_time: '2024-12-11 10:58:12', closed_at: '2024-12-11 11:05:33' },
            { symbol: 'XRPUSDT', side: 'BUY', entry_price: 2.35, exit_price: 2.42, pnl: 35.00, pnl_pct: 2.98, strategy: 'hybrid_scalping', entry_time: '2024-12-11 09:45:33', closed_at: '2024-12-11 10:22:11' },
            { symbol: 'ETHUSDT', side: 'BUY', entry_price: 3580.00, exit_price: 3610.00, pnl: 16.75, pnl_pct: 0.84, strategy: 'grid_trading', entry_time: '2024-12-11 08:30:00', closed_at: '2024-12-11 09:45:00' },
            { symbol: 'DOGEUSDT', side: 'BUY', entry_price: 0.42, exit_price: 0.415, pnl: -12.50, pnl_pct: -1.19, strategy: 'mean_reversion', entry_time: '2024-12-11 07:55:20', closed_at: '2024-12-11 08:30:15' },
            { symbol: 'BTCUSDT', side: 'BUY', entry_price: 96200.00, exit_price: 96850.00, pnl: 65.00, pnl_pct: 0.68, strategy: 'advanced_orderbook', entry_time: '2024-12-10 21:30:22', closed_at: '2024-12-10 23:15:44' },
            { symbol: 'SOLUSDT', side: 'SELL', entry_price: 228.00, exit_price: 225.50, pnl: 27.50, pnl_pct: 1.10, strategy: 'cluster_analysis', entry_time: '2024-12-10 20:15:45', closed_at: '2024-12-10 22:00:22' },
            { symbol: 'AVAXUSDT', side: 'BUY', entry_price: 52.30, exit_price: 53.10, pnl: 15.30, pnl_pct: 1.53, strategy: 'smart_dca', entry_time: '2024-12-10 18:30:00', closed_at: '2024-12-10 20:45:11' }
        ];

        let useDemoData = true; // Toggle for demo mode

        // Initialize
        document.addEventListener('DOMContentLoaded', async () => {
            // Load saved settings first
            loadSettings();

            await loadInitialState();
            await loadTradeHistory();
            connectWebSocket();

            // Fetch real prices
            fetchRealPrices();
            // Update prices every 30 seconds
            setInterval(fetchRealPrices, 30000);

            // Request notification permission
            if ('Notification' in window && Notification.permission === 'default') {
                // Ask for permission on first trade or user interaction
                document.body.addEventListener('click', () => {
                    if (Notification.permission === 'default') {
                        requestNotificationPermission();
                    }
                }, { once: true });
            }

            // Load demo data
            if (useDemoData) {
                setTimeout(() => {
                    // Always load demo positions
                    updatePositions(demoPositions);
                    // Load demo trades if empty
                    if (tradeHistory.length === 0) {
                        tradeHistory = demoTrades;
                        renderTradeHistory();
                        updateHistoryCount();
                    }
                }, 500);
            }
        });

        // Load initial state
        async function loadInitialState() {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();

                // Update exchanges
                availableExchanges = data.exchanges.available;
                selectedExchanges = data.exchanges.selected;
                renderExchanges();

                // Update mode
                currentMode = data.mode;
                updateModeUI(data.mode);

                // Update balance
                updateBalance(data.balance);

                // Update symbols
                availableSymbols = data.symbols.available;
                selectedSymbols = data.symbols.selected;
                renderSymbols();

                // Update strategies
                strategies = data.strategies;
                renderStrategies();

                // Update metrics
                updateMetrics(data.metrics);

                // Update positions
                updatePositions(data.positions);

                // Update status
                updateStatus(data.status);

                // Update quick trade symbol
                updateQuickTradeSymbols();

            } catch (err) {
                console.error('Failed to load state:', err);
            }
        }

        // WebSocket
        function connectWebSocket() {
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${proto}//${location.host}/ws`);

            ws.onopen = () => {
                ws.send(JSON.stringify({ type: 'subscribe', channels: ['all'] }));
            };

            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                handleWSMessage(msg);
            };

            ws.onclose = () => setTimeout(connectWebSocket, 3000);
        }

        function handleWSMessage(msg) {
            switch(msg.type) {
                case 'status_change': updateStatus(msg.status); break;
                case 'metrics_update': updateMetrics(msg.data); break;
                case 'position_update': updatePositions(msg.data); break;
                case 'trade': addTrade(msg.data); break;
                case 'backtest_progress': updateBacktestProgress(msg.data.progress); break;
                case 'backtest_complete': showBacktestResults(msg.data); break;
                case 'mode_change': updateModeUI(msg.data.mode); break;
            }
        }

        // Mode
        function setMode(mode) {
            if (mode === 'live' && currentMode !== 'live') {
                if (!confirm('WARNING: You are switching to REAL trading. Real money will be used. Continue?')) return;
            }

            fetch('/api/mode', {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode})
            }).then(() => {
                currentMode = mode;
                updateModeUI(mode);
                showToast(`Mode: ${mode.toUpperCase()}`, mode === 'live' ? 'error' : 'success');
            });
        }

        function updateModeUI(mode) {
            document.querySelectorAll('.mode-pill').forEach(p => p.classList.remove('active'));
            document.querySelector(`.mode-${mode}`).classList.add('active');

            // Show/hide backtest panel
            document.getElementById('backtestPanel').classList.toggle('hidden', mode !== 'backtest');
        }

        // Exchanges
        function renderExchanges() {
            const container = document.getElementById('exchangeChips');
            container.innerHTML = Object.entries(availableExchanges).map(([id, ex]) => `
                <button class="exchange-chip ${selectedExchanges.includes(id) ? 'active' : ''}"
                        data-exchange="${id}" onclick="toggleExchange('${id}')">
                    <span>${ex.logo}</span>
                    <span>${ex.name}</span>
                </button>
            `).join('');
        }

        async function toggleExchange(exchange) {
            try {
                const res = await fetch(`/api/exchanges/${exchange}/toggle`, { method: 'POST' });
                if (res.ok) {
                    const data = await res.json();
                    selectedExchanges = data.selected;
                    renderExchanges();
                    showToast(`${availableExchanges[exchange].name}: ${data.enabled ? 'ON' : 'OFF'}`);
                } else {
                    const err = await res.json();
                    showToast(err.detail || 'Error', 'error');
                }
            } catch (e) {
                showToast('Failed to toggle exchange', 'error');
            }
        }

        // Balance
        function updateBalance(balance) {
            const total = balance.total + balance.unrealized_pnl;
            const formatted = `$${total.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            document.getElementById('balanceDisplay').textContent = formatted;
            const mobileEl = document.getElementById('balanceDisplayMobile');
            if (mobileEl) mobileEl.textContent = `$${Math.round(total).toLocaleString()}`;
        }

        // Status
        function updateStatus(status) {
            const dot = document.getElementById('statusDot');
            const text = document.getElementById('statusText');
            const dotMobile = document.getElementById('statusDotMobile');
            const textMobile = document.getElementById('statusTextMobile');

            const statusName = status.charAt(0).toUpperCase() + status.slice(1);
            text.textContent = statusName;
            text.className = `px-3 py-1.5 rounded-lg text-xs font-medium status-${status}`;

            const isRunning = status === 'running';
            dot.className = 'w-2.5 h-2.5 rounded-full ' + (isRunning ? 'bg-[var(--green)] live-pulse' : 'bg-[var(--text-muted)]');

            // Update mobile elements
            if (dotMobile) dotMobile.className = 'w-2 h-2 rounded-full ' + (isRunning ? 'bg-[var(--green)] live-pulse' : 'bg-[var(--text-muted)]');
            if (textMobile) textMobile.textContent = statusName;
        }

        // Symbols
        function renderSymbols() {
            const container = document.getElementById('symbolList');
            container.innerHTML = Object.entries(availableSymbols).map(([symbol, data]) => `
                <div class="coin-item ${selectedSymbols.includes(symbol) ? 'selected' : ''}"
                     data-symbol="${symbol}">
                    <div class="flex items-center justify-between">
                        <div class="flex-1 cursor-pointer" onclick="toggleSymbol('${symbol}')">
                            <div class="font-medium text-xs">${symbol}</div>
                            <div class="text-xs text-[var(--text-muted)]">${data.name}</div>
                        </div>
                        <div class="flex items-center gap-1">
                            <div class="text-right cursor-pointer" onclick="toggleSymbol('${symbol}')">
                                <div class="text-xs font-medium">$${formatPrice(data.price)}</div>
                                <div class="text-xs ${data.change_24h >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                                    ${data.change_24h >= 0 ? '+' : ''}${data.change_24h.toFixed(2)}%
                                </div>
                            </div>
                            <button onclick="event.stopPropagation();removeSymbol('${symbol}')"
                                    class="text-[var(--text-muted)] hover:text-[var(--red)] p-1 rounded transition-all" title="Remove">
                                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
            `).join('');

            document.getElementById('selectedCount').textContent = selectedSymbols.length;
        }

        function toggleSymbol(symbol) {
            if (selectedSymbols.includes(symbol)) {
                if (selectedSymbols.length === 1) return; // Keep at least one
                selectedSymbols = selectedSymbols.filter(s => s !== symbol);
            } else {
                selectedSymbols.push(symbol);
            }

            fetch('/api/symbols/select', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(selectedSymbols)
            });

            renderSymbols();
            updateQuickTradeSymbols();
        }

        function filterSymbols() {
            const search = document.getElementById('symbolSearch').value.toLowerCase();
            document.querySelectorAll('.coin-item').forEach(item => {
                const symbol = item.dataset.symbol.toLowerCase();
                item.style.display = symbol.includes(search) ? '' : 'none';
            });
        }

        function updateQuickTradeSymbols() {
            const select = document.getElementById('qtSymbol');
            select.innerHTML = selectedSymbols.map(s => `<option>${s}</option>`).join('');
        }

        // Add/Remove Symbol Modal Functions
        function showAddSymbolModal() {
            document.getElementById('addSymbolModal').classList.remove('hidden');
            document.getElementById('newSymbolInput').value = '';
            document.getElementById('newSymbolName').value = '';
            document.getElementById('symbolSuggestions').classList.add('hidden');
            suggestionIndex = -1;
            document.getElementById('newSymbolInput').focus();
        }

        function closeAddSymbolModal() {
            document.getElementById('addSymbolModal').classList.add('hidden');
            document.getElementById('symbolSuggestions').classList.add('hidden');
        }

        // Autocomplete functions
        function filterSuggestions(query) {
            const container = document.getElementById('symbolSuggestions');
            suggestionIndex = -1;

            if (!query || query.length < 1) {
                container.classList.add('hidden');
                return;
            }

            // Filter symbols that match query (in symbol or name)
            const matches = allCryptoSymbols.filter(item => {
                const symbolMatch = item.symbol.startsWith(query);
                const nameMatch = item.name.toUpperCase().includes(query);
                // Exclude already added symbols
                return (symbolMatch || nameMatch) && !availableSymbols[item.symbol];
            }).slice(0, 10); // Limit to 10 suggestions

            if (matches.length === 0) {
                container.classList.add('hidden');
                return;
            }

            container.innerHTML = matches.map((item, idx) => `
                <div class="suggestion-item p-2 hover:bg-[var(--bg-hover)] cursor-pointer flex justify-between items-center border-b border-[var(--border)] last:border-0"
                     data-index="${idx}"
                     data-symbol="${item.symbol}"
                     data-name="${item.name}"
                     onclick="selectSuggestion('${item.symbol}', '${item.name}')"
                     onmouseenter="highlightSuggestion(${idx})">
                    <div>
                        <span class="font-semibold">${highlightMatch(item.symbol, query)}</span>
                        <span class="text-[var(--text-muted)] ml-2 text-xs">${item.name}</span>
                    </div>
                </div>
            `).join('');

            container.classList.remove('hidden');
        }

        function highlightMatch(text, query) {
            const idx = text.toUpperCase().indexOf(query.toUpperCase());
            if (idx === -1) return text;
            return text.slice(0, idx) + '<span class="text-blue-500">' + text.slice(idx, idx + query.length) + '</span>' + text.slice(idx + query.length);
        }

        function selectSuggestion(symbol, name) {
            document.getElementById('newSymbolInput').value = symbol;
            document.getElementById('newSymbolName').value = name;
            document.getElementById('symbolSuggestions').classList.add('hidden');
            suggestionIndex = -1;
        }

        function highlightSuggestion(idx) {
            suggestionIndex = idx;
            updateSuggestionHighlight();
        }

        function updateSuggestionHighlight() {
            const items = document.querySelectorAll('.suggestion-item');
            items.forEach((item, i) => {
                item.classList.toggle('bg-blue-50', i === suggestionIndex);
            });
        }

        function handleSuggestionKeydown(e) {
            const container = document.getElementById('symbolSuggestions');
            const items = container.querySelectorAll('.suggestion-item');

            if (container.classList.contains('hidden') || items.length === 0) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    addNewSymbol();
                }
                return;
            }

            switch(e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    suggestionIndex = Math.min(suggestionIndex + 1, items.length - 1);
                    updateSuggestionHighlight();
                    items[suggestionIndex]?.scrollIntoView({ block: 'nearest' });
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    suggestionIndex = Math.max(suggestionIndex - 1, 0);
                    updateSuggestionHighlight();
                    items[suggestionIndex]?.scrollIntoView({ block: 'nearest' });
                    break;
                case 'Enter':
                    e.preventDefault();
                    if (suggestionIndex >= 0 && items[suggestionIndex]) {
                        const item = items[suggestionIndex];
                        selectSuggestion(item.dataset.symbol, item.dataset.name);
                    } else {
                        addNewSymbol();
                    }
                    break;
                case 'Escape':
                    container.classList.add('hidden');
                    suggestionIndex = -1;
                    break;
            }
        }

        // Close suggestions when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('#addSymbolModal .relative')) {
                document.getElementById('symbolSuggestions')?.classList.add('hidden');
            }
        });

        async function addNewSymbol() {
            const symbol = document.getElementById('newSymbolInput').value.trim().toUpperCase();
            const name = document.getElementById('newSymbolName').value.trim() || symbol.replace('USDT', '');

            if (!symbol) {
                showToast('Please enter a symbol', 'error');
                return;
            }

            if (!symbol.endsWith('USDT')) {
                showToast('Symbol must end with USDT', 'error');
                return;
            }

            if (availableSymbols[symbol]) {
                showToast(`${symbol} already exists`, 'error');
                return;
            }

            try {
                const res = await fetch(`/api/symbols/add?symbol=${symbol}&name=${encodeURIComponent(name)}`, {
                    method: 'POST'
                });

                if (res.ok) {
                    availableSymbols[symbol] = { name, price: 0, change_24h: 0, volume_24h: 0 };
                    renderSymbols();
                    closeAddSymbolModal();
                    showToast(`${symbol} added successfully`);
                } else {
                    const err = await res.json();
                    showToast(err.detail || 'Failed to add symbol', 'error');
                }
            } catch (e) {
                showToast('Failed to add symbol', 'error');
            }
        }

        async function removeSymbol(symbol) {
            if (Object.keys(availableSymbols).length <= 1) {
                showToast('Cannot remove last symbol', 'error');
                return;
            }

            if (!confirm(`Remove ${symbol}?`)) return;

            try {
                const res = await fetch(`/api/symbols/${symbol}`, {
                    method: 'DELETE'
                });

                if (res.ok) {
                    const data = await res.json();
                    delete availableSymbols[symbol];
                    selectedSymbols = data.selected || selectedSymbols.filter(s => s !== symbol);
                    renderSymbols();
                    updateQuickTradeSymbols();
                    showToast(`${symbol} removed`);
                } else {
                    const err = await res.json();
                    showToast(err.detail || 'Failed to remove symbol', 'error');
                }
            } catch (e) {
                showToast('Failed to remove symbol', 'error');
            }
        }

        // Tab switching
        function switchTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelector(`.tab-btn[data-tab="${tab}"]`).classList.add('active');
            document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));

            if (tab === 'positions') {
                document.getElementById('tabPositions').classList.remove('hidden');
            } else if (tab === 'history') {
                document.getElementById('tabHistory').classList.remove('hidden');
            } else if (tab === 'charts') {
                document.getElementById('tabCharts').classList.remove('hidden');
                renderCharts();
            }
        }

        // Strategies - Grouped by category
        const strategyGroups = {
            scalping: ['hybrid_scalping', 'impulse_scalping', 'volume_spike'],
            analysis: ['advanced_orderbook', 'orderbook_imbalance', 'print_tape', 'cluster_analysis'],
            dca: ['dca', 'grid_trading', 'smart_dca'],
            other: ['mean_reversion']
        };

        const strategyIcons = {
            'hybrid_scalping': '🎯', 'advanced_orderbook': '📊', 'print_tape': '📜',
            'cluster_analysis': '🔬', 'impulse_scalping': '⚡', 'dca': '💰',
            'grid_trading': '📐', 'mean_reversion': '🔄', 'orderbook_imbalance': '📈',
            'volume_spike': '📢', 'smart_dca': '🤖'
        };

        function renderStrategies() {
            // Group strategies
            const groups = { scalping: [], analysis: [], dca: [], other: [] };

            Object.entries(strategies).forEach(([name, config]) => {
                let placed = false;
                for (const [group, list] of Object.entries(strategyGroups)) {
                    if (list.includes(name)) {
                        groups[group].push({ name, config });
                        placed = true;
                        break;
                    }
                }
                if (!placed) groups.other.push({ name, config });
            });

            // Render each group
            const renderGroup = (items, containerId) => {
                const container = document.getElementById(containerId);
                if (!container) return;
                container.innerHTML = items.map(({ name, config }) => `
                    <div class="strategy-item ${config.enabled ? 'active' : ''}" onclick="toggleStrategy('${name}')">
                        <div class="flex items-center gap-2">
                            <span class="text-xs">${strategyIcons[name] || '📌'}</span>
                            <span class="text-xs">${formatName(name)}</span>
                        </div>
                        <div class="status-dot ${config.enabled ? 'on' : 'off'}"></div>
                    </div>
                `).join('');
            };

            renderGroup(groups.scalping, 'strategyGroupScalping');
            renderGroup(groups.analysis, 'strategyGroupAnalysis');
            renderGroup(groups.dca, 'strategyGroupDCA');
            renderGroup(groups.other, 'strategyGroupOther');

            // Update active count
            const active = Object.values(strategies).filter(s => s.enabled).length;
            document.getElementById('activeStrategies').textContent = `${active} active`;

            // Update filter dropdown
            updateStrategyFilter();
        }

        function updateStrategyFilter() {
            const select = document.getElementById('historyFilter');
            if (!select) return;
            const strategyNames = Object.keys(strategies);
            select.innerHTML = '<option value="all">All Strategies</option>' +
                strategyNames.map(name => `<option value="${name}">${formatName(name)}</option>`).join('');
        }

        async function toggleStrategy(name) {
            await fetch(`/api/strategies/${name}/toggle`, { method: 'POST' });
            strategies[name].enabled = !strategies[name].enabled;
            renderStrategies();
            showToast(`${formatName(name)} ${strategies[name].enabled ? 'enabled' : 'disabled'}`);
        }

        // Metrics
        function updateMetrics(m) {
            document.getElementById('totalPnl').textContent = `${m.total_pnl >= 0 ? '+' : ''}$${m.total_pnl.toFixed(2)}`;
            document.getElementById('totalPnl').className = `text-xl font-bold ${m.total_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`;

            document.getElementById('dailyPnl').textContent = `${m.daily_pnl >= 0 ? '+' : ''}$${(m.daily_pnl || 0).toFixed(2)}`;
            document.getElementById('dailyPnl').className = `text-xl font-bold ${(m.daily_pnl || 0) >= 0 ? 'pnl-positive' : 'pnl-negative'}`;

            document.getElementById('winRate').textContent = `${m.win_rate.toFixed(1)}%`;
            document.getElementById('totalTrades').textContent = m.total_trades;
            document.getElementById('maxDrawdown').textContent = `${m.max_drawdown_pct.toFixed(1)}%`;
        }

        // Positions - Single Row
        function updatePositions(positions) {
            const container = document.getElementById('positionsList');
            const countEl = document.getElementById('positionsCount');

            if (!positions || positions.length === 0) {
                container.innerHTML = '<p class="text-[var(--text-muted)] text-xs text-center py-4">No positions</p>';
                if (countEl) countEl.textContent = '0';
                return;
            }

            if (countEl) countEl.textContent = positions.length;

            container.innerHTML = positions.map(p => {
                const isLong = p.side === 'BUY';
                const pnlPct = p.entry_price > 0 ? ((p.mark_price - p.entry_price) / p.entry_price * 100 * (isLong ? 1 : -1)) : 0;
                const isProfit = p.unrealized_pnl >= 0;
                const openedTime = formatDateTime(p.opened_at || p.entry_time);
                return `
                <div class="position-row flex items-center justify-between">
                    <div class="flex items-center gap-2">
                        <span class="font-semibold">${p.symbol}</span>
                        <span class="${isLong ? 'text-green' : 'text-red'}">${isLong ? 'L' : 'S'}${p.leverage || ''}x</span>
                        <span class="text-[var(--text-muted)]">$${formatPrice(p.entry_price)}→${formatPrice(p.mark_price)}</span>
                        <span class="text-[var(--border)]">|</span>
                        <span class="text-red">SL:${p.stop_loss ? formatPrice(p.stop_loss) : '—'}</span>
                        <span class="text-green">TP:${p.take_profit ? formatPrice(p.take_profit) : '—'}</span>
                        <span class="text-[var(--border)]">|</span>
                        <span class="text-[var(--text-muted)] text-xs" title="Opened at">⏱ ${openedTime}</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="font-bold ${isProfit ? 'text-green' : 'text-red'}">${isProfit ? '+' : ''}$${p.unrealized_pnl.toFixed(2)} (${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(1)}%)</span>
                        <button onclick="closePosition('${p.symbol}')" class="text-[var(--text-muted)] hover:text-[var(--red)]">✕</button>
                    </div>
                </div>
            `}).join('');
        }

        // Trade history storage
        let tradeHistory = [];

        // Trades - Detailed card view
        function addTrade(trade) {
            tradeHistory.unshift(trade);
            if (tradeHistory.length > 100) tradeHistory.pop();
            renderTradeHistory();
            updateHistoryCount();

            // Play sound notification
            if (trade.pnl >= 0) {
                playSound('win');
            } else {
                playSound('loss');
            }

            // Send push notification
            const pnlSign = trade.pnl >= 0 ? '+' : '';
            const icon = trade.pnl >= 0 ? '💰' : '📉';
            sendPushNotification(
                `${trade.symbol} ${trade.side}`,
                `P&L: ${pnlSign}$${trade.pnl.toFixed(2)} (${pnlSign}${(trade.pnl_pct || 0).toFixed(1)}%)`,
                icon
            );
        }

        function renderTradeHistory() {
            const container = document.getElementById('tradesList');
            const filterValue = document.getElementById('historyFilter')?.value || 'all';
            const periodValue = document.getElementById('historyPeriod')?.value || 'all';
            const customDateRange = document.getElementById('customDateRange');

            // Show/hide custom date range
            if (customDateRange) {
                customDateRange.classList.toggle('hidden', periodValue !== 'custom');
            }

            let filteredTrades = tradeHistory;

            // Filter by strategy
            if (filterValue !== 'all') {
                filteredTrades = filteredTrades.filter(t => t.strategy === filterValue);
            }

            // Filter by time period
            if (periodValue !== 'all') {
                const now = new Date();
                let startDate;

                if (periodValue === 'today') {
                    startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate());
                } else if (periodValue === 'week') {
                    startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
                } else if (periodValue === 'month') {
                    startDate = new Date(now.getFullYear(), now.getMonth(), 1);
                } else if (periodValue === 'custom') {
                    const startInput = document.getElementById('historyStartDate')?.value;
                    const endInput = document.getElementById('historyEndDate')?.value;
                    if (startInput) startDate = new Date(startInput);
                    if (endInput) {
                        const endDate = new Date(endInput);
                        endDate.setHours(23, 59, 59, 999);
                        filteredTrades = filteredTrades.filter(t => {
                            const tradeDate = new Date(t.closed_at || t.exit_time || '');
                            return !isNaN(tradeDate) && tradeDate <= endDate;
                        });
                    }
                }

                if (startDate) {
                    filteredTrades = filteredTrades.filter(t => {
                        const tradeDate = new Date(t.closed_at || t.exit_time || '');
                        return !isNaN(tradeDate) && tradeDate >= startDate;
                    });
                }
            }

            // Update stats
            updateHistoryStats(filteredTrades);

            if (filteredTrades.length === 0) {
                container.innerHTML = '<p class="text-[var(--text-muted)] text-xs text-center py-4">No trades</p>';
                updateStrategySummary([]);
                return;
            }

            container.innerHTML = filteredTrades.map(trade => {
                const isProfit = trade.pnl >= 0;
                const isLong = trade.side === 'BUY';
                const pnlPct = trade.pnl_pct || 0;
                const entryTime = formatDateTime(trade.entry_time);
                const exitTime = formatDateTime(trade.closed_at || trade.exit_time);
                return `
                <div class="trade-row flex items-center justify-between">
                    <div class="flex items-center gap-2">
                        <span class="font-medium">${trade.symbol}</span>
                        <span class="${isLong ? 'text-green' : 'text-red'}">${isLong ? 'L' : 'S'}</span>
                        <span class="text-[var(--border)]">|</span>
                        <span class="text-[var(--text-muted)]">${strategyIcons[trade.strategy] || ''} ${formatName(trade.strategy || 'Manual')}</span>
                        <span class="text-[var(--border)]">|</span>
                        <span class="text-[var(--text-muted)]">$${formatPrice(trade.entry_price || 0)}→${formatPrice(trade.exit_price || 0)}</span>
                    </div>
                    <div class="flex items-center gap-3">
                        <span class="text-[var(--text-muted)] text-xs" title="Entry → Exit">⏱ ${entryTime} → ${exitTime}</span>
                        <span class="font-bold ${isProfit ? 'text-green' : 'text-red'}">${isProfit ? '+' : ''}$${trade.pnl.toFixed(2)} (${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(1)}%)</span>
                    </div>
                </div>
                `;
            }).join('');

            updateStrategySummary(filteredTrades);
        }

        function updateStrategySummary(trades) {
            const container = document.getElementById('strategySummary');
            if (!container) return;

            // Group by strategy
            const byStrategy = {};
            trades.forEach(t => {
                const s = t.strategy || 'Manual';
                if (!byStrategy[s]) byStrategy[s] = { trades: 0, pnl: 0, wins: 0 };
                byStrategy[s].trades++;
                byStrategy[s].pnl += t.pnl || 0;
                if (t.pnl >= 0) byStrategy[s].wins++;
            });

            const entries = Object.entries(byStrategy);
            if (entries.length === 0) {
                container.innerHTML = '';
                return;
            }

            // Sort by PNL descending
            entries.sort((a, b) => b[1].pnl - a[1].pnl);

            container.innerHTML = '<div class="flex flex-wrap gap-2">' + entries.slice(0, 6).map(([name, data]) => {
                const winRate = data.trades > 0 ? (data.wins / data.trades * 100) : 0;
                const isProfit = data.pnl >= 0;
                return `
                <span class="summary-badge">
                    <span class="text-xs">${strategyIcons[name] || ''} ${formatName(name)}:</span>
                    <span class="font-bold text-xs ${isProfit ? 'text-green' : 'text-red'}">${isProfit ? '+' : ''}$${data.pnl.toFixed(0)}</span>
                    <span class="text-[var(--text-muted)] text-xs">${data.trades}t/${winRate.toFixed(0)}%</span>
                </span>`;
            }).join('') + '</div>';
        }

        function updateHistoryCount() {
            const countEl = document.getElementById('historyCount');
            if (countEl) countEl.textContent = tradeHistory.length;
        }

        function updateHistoryStats(trades) {
            const el = document.getElementById('historyStats');
            if (!el) return;

            const count = trades.length;
            const totalPnl = trades.reduce((sum, t) => sum + (t.pnl || 0), 0);
            const wins = trades.filter(t => t.pnl >= 0).length;
            const winRate = count > 0 ? (wins / count * 100).toFixed(0) : 0;

            const pnlClass = totalPnl >= 0 ? 'text-green' : 'text-red';
            const pnlSign = totalPnl >= 0 ? '+' : '';

            el.innerHTML = `${count} trades | <span class="${pnlClass}">${pnlSign}$${totalPnl.toFixed(2)}</span> | WR: ${winRate}%`;
        }

        function filterTradeHistory() {
            renderTradeHistory();
        }

        // Load trades from database with filters
        async function loadFromDB() {
            const startDate = document.getElementById('historyStartDate')?.value;
            const endDate = document.getElementById('historyEndDate')?.value;
            const strategy = document.getElementById('historyFilter')?.value;

            let url = '/api/trades/db?limit=500';
            if (startDate) url += `&start_date=${startDate}`;
            if (endDate) url += `&end_date=${endDate}`;
            if (strategy && strategy !== 'all') url += `&strategy=${strategy}`;

            try {
                const res = await fetch(url);
                if (res.ok) {
                    const data = await res.json();
                    tradeHistory = data.trades.map(t => ({
                        symbol: t.symbol,
                        side: t.side,
                        entry_price: t.entry_price,
                        exit_price: t.exit_price,
                        pnl: t.pnl,
                        pnl_pct: t.pnl_pct,
                        strategy: t.strategy,
                        entry_time: t.entry_time,
                        closed_at: t.exit_time,
                    }));
                    renderTradeHistory();
                    updateHistoryCount();
                    showToast(`Loaded ${data.trades.length} trades from DB`);
                }
            } catch (e) {
                console.error('Failed to load from DB:', e);
                showToast('Failed to load from DB', 'error');
            }
        }

        // Load trade history on startup
        async function loadTradeHistory() {
            try {
                const res = await fetch('/api/trades/history');
                if (res.ok) {
                    tradeHistory = await res.json();
                    renderTradeHistory();
                    updateHistoryCount();
                }
            } catch (e) {
                console.error('Failed to load trade history:', e);
            }
        }

        // Commands
        async function sendCommand(cmd) {
            const res = await fetch('/api/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: cmd})
            });
            const data = await res.json();
            showToast(data.message);
        }

        async function closePosition(symbol) {
            if (!confirm(`Close ${symbol} position?`)) return;
            try {
                const res = await fetch('/api/position/close', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({symbol})
                });
                const data = await res.json();
                if (res.ok) {
                    showToast(`Closed ${symbol}: P&L $${(data.pnl || 0).toFixed(2)}`);
                    await fetchState(); // Refresh UI
                } else {
                    showToast(data.detail || 'Failed to close position', 'error');
                }
            } catch (e) {
                showToast('Error closing position: ' + e.message, 'error');
            }
        }

        async function closeAllPositions() {
            if (!confirm('Close ALL positions?')) return;
            try {
                const res = await fetch('/api/positions');
                const positions = await res.json();
                if (positions.length === 0) {
                    showToast('No positions to close', 'error');
                    return;
                }
                let closed = 0;
                for (const p of positions) {
                    const closeRes = await fetch('/api/position/close', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({symbol: p.symbol})
                    });
                    if (closeRes.ok) closed++;
                }
                showToast(`Closed ${closed} positions`);
                await fetchState(); // Refresh UI
            } catch (e) {
                showToast('Error closing positions: ' + e.message, 'error');
            }
        }

        async function quickTrade(side) {
            const symbol = document.getElementById('qtSymbol').value;
            const amount = parseFloat(document.getElementById('qtAmount').value);
            if (!symbol || !amount || amount <= 0) {
                showToast('Enter valid symbol and amount', 'error');
                return;
            }
            const price = availableSymbols[symbol]?.price || 0;
            if (price <= 0) {
                showToast('Invalid price for ' + symbol, 'error');
                return;
            }
            const quantity = amount / price;

            try {
                const res = await fetch('/api/order', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ symbol, side, quantity, order_type: 'MARKET' })
                });
                const data = await res.json();
                if (res.ok) {
                    showToast(`${side} ${symbol} $${amount.toFixed(2)} @ $${price.toFixed(2)}`);
                    await fetchState(); // Refresh UI
                } else {
                    showToast(data.detail || 'Order failed', 'error');
                }
            } catch (e) {
                showToast('Error placing order: ' + e.message, 'error');
            }
        }

        // Backtest
        async function runBacktest() {
            const startDate = document.getElementById('btStartDate').value;
            const endDate = document.getElementById('btEndDate').value;
            const balance = parseFloat(document.getElementById('btBalance').value);

            await fetch('/api/backtest/config', {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ start_date: startDate, end_date: endDate, initial_balance: balance })
            });

            await fetch('/api/backtest/run', { method: 'POST' });
            document.getElementById('btRunBtn').disabled = true;
            showToast('Backtest started...');
        }

        function updateBacktestProgress(progress) {
            document.getElementById('btProgress').style.width = `${progress}%`;
            document.getElementById('btProgressText').textContent = `${progress}%`;
        }

        function showBacktestResults(results) {
            document.getElementById('btRunBtn').disabled = false;
            document.getElementById('btResults').classList.remove('hidden');

            document.getElementById('btTotalPnl').textContent = `$${results.total_pnl.toFixed(2)}`;
            document.getElementById('btTotalPnl').className = `text-lg font-bold ${results.total_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`;
            document.getElementById('btWinRate').textContent = `${results.win_rate.toFixed(1)}%`;
            document.getElementById('btTotalTrades').textContent = results.total_trades;
            document.getElementById('btFinalBalance').textContent = `$${results.final_balance.toFixed(2)}`;

            showToast('Backtest complete!');
        }

        // =========================================================================
        // Charts
        // =========================================================================
        let equityChart = null;
        let dailyPnlChart = null;
        let chartPeriod = 7; // days

        function setChartPeriod(days) {
            chartPeriod = days;
            document.querySelectorAll('.chart-period-btn').forEach(btn => {
                btn.classList.toggle('active', parseInt(btn.dataset.period) === days);
            });
            renderCharts();
        }

        function renderCharts() {
            const trades = getFilteredTradesForChart();
            renderEquityCurve(trades);
            renderDailyPnl(trades);
            updateChartStats(trades);
        }

        function getFilteredTradesForChart() {
            if (chartPeriod === 0) return [...tradeHistory].reverse();

            const cutoff = new Date();
            cutoff.setDate(cutoff.getDate() - chartPeriod);

            return tradeHistory.filter(t => {
                const date = new Date((t.closed_at || t.exit_time || '').replace(' ', 'T'));
                return !isNaN(date) && date >= cutoff;
            }).reverse();
        }

        function renderEquityCurve(trades) {
            const ctx = document.getElementById('equityChart');
            if (!ctx) return;

            // Calculate cumulative equity
            const initialBalance = 1000;
            let balance = initialBalance;
            const data = [{ x: 'Start', y: balance }];

            trades.forEach((t, i) => {
                balance += t.pnl || 0;
                const label = formatChartDate(t.closed_at || t.exit_time);
                data.push({ x: label, y: balance });
            });

            if (equityChart) equityChart.destroy();

            equityChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.map(d => d.x),
                    datasets: [{
                        label: 'Balance',
                        data: data.map(d => d.y),
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37, 99, 235, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: data.length > 20 ? 0 : 3,
                        pointHoverRadius: 5,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (ctx) => `$${ctx.raw.toFixed(2)}`
                            }
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            grid: { display: false },
                            ticks: { maxTicksLimit: 6, font: { size: 10 }, color: '#94a3b8' }
                        },
                        y: {
                            display: true,
                            grid: { color: 'rgba(148, 163, 184, 0.1)' },
                            ticks: {
                                font: { size: 10 },
                                color: '#94a3b8',
                                callback: (v) => '$' + v
                            }
                        }
                    }
                }
            });
        }

        function renderDailyPnl(trades) {
            const ctx = document.getElementById('dailyPnlChart');
            if (!ctx) return;

            // Group by date
            const byDate = {};
            trades.forEach(t => {
                const dateStr = (t.closed_at || t.exit_time || '').split(' ')[0] || 'Unknown';
                if (!byDate[dateStr]) byDate[dateStr] = 0;
                byDate[dateStr] += t.pnl || 0;
            });

            const dates = Object.keys(byDate).slice(-14); // Last 14 days
            const values = dates.map(d => byDate[d]);
            const colors = values.map(v => v >= 0 ? '#10b981' : '#ef4444');

            if (dailyPnlChart) dailyPnlChart.destroy();

            dailyPnlChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: dates.map(d => formatChartDateShort(d)),
                    datasets: [{
                        label: 'P&L',
                        data: values,
                        backgroundColor: colors,
                        borderRadius: 4,
                        barThickness: dates.length > 10 ? 'flex' : 20,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (ctx) => {
                                    const val = ctx.raw;
                                    return `${val >= 0 ? '+' : ''}$${val.toFixed(2)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            grid: { display: false },
                            ticks: { font: { size: 10 }, color: '#94a3b8' }
                        },
                        y: {
                            display: true,
                            grid: { color: 'rgba(148, 163, 184, 0.1)' },
                            ticks: {
                                font: { size: 10 },
                                color: '#94a3b8',
                                callback: (v) => '$' + v
                            }
                        }
                    }
                }
            });
        }

        function updateChartStats(trades) {
            const totalPnl = trades.reduce((sum, t) => sum + (t.pnl || 0), 0);
            const wins = trades.filter(t => t.pnl >= 0).length;
            const winRate = trades.length > 0 ? (wins / trades.length * 100).toFixed(0) : 0;

            const pnlEl = document.getElementById('chartTotalPnl');
            const wrEl = document.getElementById('chartWinRate');

            if (pnlEl) {
                pnlEl.textContent = `${totalPnl >= 0 ? '+' : ''}$${totalPnl.toFixed(2)}`;
                pnlEl.className = `text-xs font-bold ${totalPnl >= 0 ? 'text-green' : 'text-red'}`;
            }
            if (wrEl) {
                wrEl.textContent = `WR: ${winRate}% (${trades.length} trades)`;
            }
        }

        function formatChartDate(dateStr) {
            if (!dateStr) return '';
            const date = new Date(dateStr.replace(' ', 'T'));
            if (isNaN(date)) return dateStr;
            return date.toLocaleDateString('uk-UA', { day: '2-digit', month: '2-digit' });
        }

        function formatChartDateShort(dateStr) {
            if (!dateStr) return '';
            const parts = dateStr.split('-');
            if (parts.length === 3) return `${parts[2]}.${parts[1]}`;
            return dateStr;
        }

        // =========================================================================
        // Utils
        // =========================================================================
        function formatPrice(price) {
            return price >= 1000 ? price.toLocaleString() : price.toFixed(price < 1 ? 4 : 2);
        }

        function formatName(name) {
            return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        }

        function formatDateTime(dateStr) {
            if (!dateStr) return '—';
            // Handle ISO format or "YYYY-MM-DD HH:mm:ss" format
            const date = new Date(dateStr.replace(' ', 'T'));
            if (isNaN(date)) return dateStr;

            const now = new Date();
            const isToday = date.toDateString() === now.toDateString();
            const yesterday = new Date(now);
            yesterday.setDate(yesterday.getDate() - 1);
            const isYesterday = date.toDateString() === yesterday.toDateString();

            const time = date.toLocaleTimeString('uk-UA', { hour: '2-digit', minute: '2-digit' });

            if (isToday) return time;
            if (isYesterday) return `Вч ${time}`;
            return date.toLocaleDateString('uk-UA', { day: '2-digit', month: '2-digit' }) + ' ' + time;
        }

        function showToast(msg, type = 'success') {
            const toast = document.createElement('div');
            toast.className = `toast ${type === 'error' ? 'bg-red-500' : 'bg-green-500'}`;
            toast.textContent = msg;
            document.getElementById('toastContainer').appendChild(toast);
            setTimeout(() => toast.remove(), 3000);
        }
    </script>
</body>
</html>
'''
