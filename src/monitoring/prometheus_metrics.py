"""
Prometheus metrics exporter for the trading bot.

Exposes metrics for monitoring trading performance, latency, and system health.
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Any, Optional
from aiohttp import web
from loguru import logger


# =============================================================================
# Metrics Storage
# =============================================================================

@dataclass
class TradingMetrics:
    """Container for all trading metrics."""

    # P&L metrics
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    equity: float = 100.0
    drawdown_percent: float = 0.0
    max_drawdown_percent: float = 0.0

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Position metrics
    position_size: float = 0.0
    position_side: str = ""  # LONG, SHORT, or empty
    unrealized_pnl: float = 0.0

    # Latency metrics (in ms)
    order_latency_p50: float = 0.0
    order_latency_p95: float = 0.0
    order_latency_p99: float = 0.0
    websocket_latency: float = 0.0

    # Activity counters
    signals_generated: int = 0
    orders_executed: int = 0
    orders_rejected: int = 0
    errors_total: int = 0

    # Connection status
    websocket_connected: int = 0  # 1 = connected, 0 = disconnected
    api_connected: int = 0

    # Market data
    spread_bps: float = 0.0
    funding_rate: float = 0.0

    # System metrics
    database_size_bytes: int = 0
    uptime_seconds: float = 0.0

    # Per-symbol metrics
    symbol_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


# Global metrics instance
trading_metrics = TradingMetrics()
_start_time = time.time()


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """
    Collect and update trading metrics.

    Usage:
        collector = MetricsCollector()

        # Update metrics
        collector.record_trade(pnl=10.5, is_winner=True)
        collector.record_latency("order", 45.2)
        collector.set_position(size=0.01, side="LONG")

        # Get Prometheus format
        metrics_text = collector.export_prometheus()
    """

    def __init__(self):
        self._latency_samples: Dict[str, list] = {
            "order": [],
            "websocket": [],
        }
        self._max_samples = 1000

    def record_trade(
        self,
        pnl: float,
        is_winner: bool,
        symbol: str = "BTCUSDT",
    ) -> None:
        """Record a completed trade."""
        trading_metrics.total_trades += 1
        trading_metrics.total_pnl += pnl
        trading_metrics.daily_pnl += pnl
        trading_metrics.equity += pnl

        if is_winner:
            trading_metrics.winning_trades += 1
        else:
            trading_metrics.losing_trades += 1

        # Update win rate
        if trading_metrics.total_trades > 0:
            trading_metrics.win_rate = (
                trading_metrics.winning_trades / trading_metrics.total_trades * 100
            )

        # Update symbol metrics
        if symbol not in trading_metrics.symbol_metrics:
            trading_metrics.symbol_metrics[symbol] = {
                "trades": 0,
                "pnl": 0.0,
                "wins": 0,
            }

        trading_metrics.symbol_metrics[symbol]["trades"] += 1
        trading_metrics.symbol_metrics[symbol]["pnl"] += pnl
        if is_winner:
            trading_metrics.symbol_metrics[symbol]["wins"] += 1

    def record_latency(self, metric_type: str, latency_ms: float) -> None:
        """Record a latency sample."""
        if metric_type not in self._latency_samples:
            self._latency_samples[metric_type] = []

        samples = self._latency_samples[metric_type]
        samples.append(latency_ms)

        # Keep only recent samples
        if len(samples) > self._max_samples:
            samples.pop(0)

        # Update percentiles
        if samples:
            sorted_samples = sorted(samples)
            n = len(sorted_samples)

            if metric_type == "order":
                trading_metrics.order_latency_p50 = sorted_samples[int(n * 0.5)]
                trading_metrics.order_latency_p95 = sorted_samples[int(n * 0.95)]
                trading_metrics.order_latency_p99 = sorted_samples[min(int(n * 0.99), n - 1)]
            elif metric_type == "websocket":
                trading_metrics.websocket_latency = sorted_samples[int(n * 0.5)]

    def set_position(
        self,
        size: float,
        side: str,
        unrealized_pnl: float = 0.0,
    ) -> None:
        """Update current position."""
        trading_metrics.position_size = size
        trading_metrics.position_side = side
        trading_metrics.unrealized_pnl = unrealized_pnl

    def set_connection_status(
        self,
        websocket: bool = None,
        api: bool = None,
    ) -> None:
        """Update connection status."""
        if websocket is not None:
            trading_metrics.websocket_connected = 1 if websocket else 0
        if api is not None:
            trading_metrics.api_connected = 1 if api else 0

    def set_market_data(
        self,
        spread_bps: float = None,
        funding_rate: float = None,
    ) -> None:
        """Update market data metrics."""
        if spread_bps is not None:
            trading_metrics.spread_bps = spread_bps
        if funding_rate is not None:
            trading_metrics.funding_rate = funding_rate

    def record_signal(self) -> None:
        """Record signal generation."""
        trading_metrics.signals_generated += 1

    def record_order(self, executed: bool = True) -> None:
        """Record order execution."""
        if executed:
            trading_metrics.orders_executed += 1
        else:
            trading_metrics.orders_rejected += 1

    def record_error(self) -> None:
        """Record an error."""
        trading_metrics.errors_total += 1

    def update_drawdown(self, current_equity: float, peak_equity: float) -> None:
        """Update drawdown metrics."""
        if peak_equity > 0:
            drawdown = (peak_equity - current_equity) / peak_equity * 100
            trading_metrics.drawdown_percent = max(0, drawdown)
            trading_metrics.max_drawdown_percent = max(
                trading_metrics.max_drawdown_percent,
                trading_metrics.drawdown_percent,
            )

    def set_profit_factor(self, gross_profit: float, gross_loss: float) -> None:
        """Update profit factor."""
        if gross_loss > 0:
            trading_metrics.profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            trading_metrics.profit_factor = float('inf')
        else:
            trading_metrics.profit_factor = 0.0

    def reset_daily(self) -> None:
        """Reset daily metrics."""
        trading_metrics.daily_pnl = 0.0

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        trading_metrics.uptime_seconds = time.time() - _start_time

        lines = [
            "# HELP trading_total_pnl_usd Total profit/loss in USD",
            "# TYPE trading_total_pnl_usd gauge",
            f"trading_total_pnl_usd {trading_metrics.total_pnl:.4f}",
            "",
            "# HELP trading_daily_pnl_usd Daily profit/loss in USD",
            "# TYPE trading_daily_pnl_usd gauge",
            f"trading_daily_pnl_usd {trading_metrics.daily_pnl:.4f}",
            "",
            "# HELP trading_equity_usd Current equity in USD",
            "# TYPE trading_equity_usd gauge",
            f"trading_equity_usd {trading_metrics.equity:.4f}",
            "",
            "# HELP trading_drawdown_percent Current drawdown percentage",
            "# TYPE trading_drawdown_percent gauge",
            f"trading_drawdown_percent {trading_metrics.drawdown_percent:.4f}",
            "",
            "# HELP trading_max_drawdown_percent Maximum drawdown percentage",
            "# TYPE trading_max_drawdown_percent gauge",
            f"trading_max_drawdown_percent {trading_metrics.max_drawdown_percent:.4f}",
            "",
            "# HELP trading_total_trades Total number of trades",
            "# TYPE trading_total_trades counter",
            f"trading_total_trades {trading_metrics.total_trades}",
            "",
            "# HELP trading_winning_trades Number of winning trades",
            "# TYPE trading_winning_trades counter",
            f"trading_winning_trades {trading_metrics.winning_trades}",
            "",
            "# HELP trading_losing_trades Number of losing trades",
            "# TYPE trading_losing_trades counter",
            f"trading_losing_trades {trading_metrics.losing_trades}",
            "",
            "# HELP trading_win_rate_percent Win rate percentage",
            "# TYPE trading_win_rate_percent gauge",
            f"trading_win_rate_percent {trading_metrics.win_rate:.2f}",
            "",
            "# HELP trading_profit_factor Profit factor (gross profit / gross loss)",
            "# TYPE trading_profit_factor gauge",
            f"trading_profit_factor {trading_metrics.profit_factor:.4f}",
            "",
            "# HELP trading_position_size_usd Current position size in USD",
            "# TYPE trading_position_size_usd gauge",
            f"trading_position_size_usd {trading_metrics.position_size:.4f}",
            "",
            "# HELP trading_unrealized_pnl_usd Unrealized P&L in USD",
            "# TYPE trading_unrealized_pnl_usd gauge",
            f"trading_unrealized_pnl_usd {trading_metrics.unrealized_pnl:.4f}",
            "",
            "# HELP trading_order_latency_ms_p50 Order latency P50 in milliseconds",
            "# TYPE trading_order_latency_ms_p50 gauge",
            f"trading_order_latency_ms_p50 {trading_metrics.order_latency_p50:.2f}",
            "",
            "# HELP trading_order_latency_ms_p95 Order latency P95 in milliseconds",
            "# TYPE trading_order_latency_ms_p95 gauge",
            f"trading_order_latency_ms_p95 {trading_metrics.order_latency_p95:.2f}",
            "",
            "# HELP trading_order_latency_ms_p99 Order latency P99 in milliseconds",
            "# TYPE trading_order_latency_ms_p99 gauge",
            f"trading_order_latency_ms_p99 {trading_metrics.order_latency_p99:.2f}",
            "",
            "# HELP trading_websocket_latency_ms WebSocket message latency in milliseconds",
            "# TYPE trading_websocket_latency_ms gauge",
            f"trading_websocket_latency_ms {trading_metrics.websocket_latency:.2f}",
            "",
            "# HELP trading_signals_generated_total Total signals generated",
            "# TYPE trading_signals_generated_total counter",
            f"trading_signals_generated_total {trading_metrics.signals_generated}",
            "",
            "# HELP trading_orders_executed_total Total orders executed",
            "# TYPE trading_orders_executed_total counter",
            f"trading_orders_executed_total {trading_metrics.orders_executed}",
            "",
            "# HELP trading_orders_rejected_total Total orders rejected",
            "# TYPE trading_orders_rejected_total counter",
            f"trading_orders_rejected_total {trading_metrics.orders_rejected}",
            "",
            "# HELP trading_errors_total Total errors",
            "# TYPE trading_errors_total counter",
            f"trading_errors_total {trading_metrics.errors_total}",
            "",
            "# HELP trading_websocket_connected WebSocket connection status (1=connected, 0=disconnected)",
            "# TYPE trading_websocket_connected gauge",
            f"trading_websocket_connected {trading_metrics.websocket_connected}",
            "",
            "# HELP trading_api_connected API connection status (1=connected, 0=disconnected)",
            "# TYPE trading_api_connected gauge",
            f"trading_api_connected {trading_metrics.api_connected}",
            "",
            "# HELP trading_spread_bps Current spread in basis points",
            "# TYPE trading_spread_bps gauge",
            f"trading_spread_bps {trading_metrics.spread_bps:.4f}",
            "",
            "# HELP trading_funding_rate Current funding rate",
            "# TYPE trading_funding_rate gauge",
            f"trading_funding_rate {trading_metrics.funding_rate:.8f}",
            "",
            "# HELP trading_uptime_seconds Bot uptime in seconds",
            "# TYPE trading_uptime_seconds counter",
            f"trading_uptime_seconds {trading_metrics.uptime_seconds:.0f}",
            "",
        ]

        # Add per-symbol metrics
        for symbol, metrics in trading_metrics.symbol_metrics.items():
            lines.extend([
                f'trading_symbol_trades{{symbol="{symbol}"}} {metrics["trades"]}',
                f'trading_symbol_pnl{{symbol="{symbol}"}} {metrics["pnl"]:.4f}',
            ])

        return "\n".join(lines)


# =============================================================================
# HTTP Server for Prometheus Scraping
# =============================================================================

collector = MetricsCollector()


async def metrics_handler(request: web.Request) -> web.Response:
    """Handle /metrics endpoint for Prometheus."""
    metrics_text = collector.export_prometheus()
    return web.Response(
        text=metrics_text,
        content_type="text/plain; charset=utf-8",
    )


async def health_handler(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.json_response({
        "status": "healthy",
        "uptime": time.time() - _start_time,
    })


async def start_metrics_server(
    host: str = "0.0.0.0",
    port: int = 9090,
) -> web.AppRunner:
    """
    Start the Prometheus metrics HTTP server.

    Args:
        host: Host to bind to
        port: Port to bind to

    Returns:
        AppRunner for cleanup
    """
    app = web.Application()
    app.router.add_get("/metrics", metrics_handler)
    app.router.add_get("/health", health_handler)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host, port)
    await site.start()

    logger.info(f"Prometheus metrics server started on http://{host}:{port}/metrics")

    return runner
