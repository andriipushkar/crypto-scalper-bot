"""
Dashboard API

FastAPI routes for the portfolio dashboard.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from loguru import logger

from .portfolio import PortfolioAnalytics
from .metrics import PerformanceMetrics


def create_dashboard_app(
    portfolio: Optional[PortfolioAnalytics] = None,
    title: str = "Trading Dashboard API",
) -> "FastAPI":
    """Create FastAPI dashboard application."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI not installed. Install with: pip install fastapi")

    app = FastAPI(
        title=title,
        version="1.0.0",
        description="Portfolio analytics and monitoring dashboard",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize services
    if portfolio is None:
        portfolio = PortfolioAnalytics()

    metrics_calculator = PerformanceMetrics()

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    @app.get("/api/v1/portfolio/summary")
    async def get_portfolio_summary() -> Dict[str, Any]:
        """Get portfolio summary."""
        return portfolio.to_dict()

    @app.get("/api/v1/portfolio/positions")
    async def get_positions() -> List[Dict[str, Any]]:
        """Get all open positions."""
        return [
            {
                "symbol": symbol,
                "side": pos.side,
                "quantity": float(pos.quantity),
                "entry_price": float(pos.entry_price),
                "current_price": float(pos.current_price),
                "unrealized_pnl": float(pos.unrealized_pnl),
                "unrealized_pnl_pct": float(pos.unrealized_pnl_pct),
                "entry_time": pos.entry_time.isoformat(),
                "leverage": pos.leverage,
            }
            for symbol, pos in portfolio.positions.items()
        ]

    @app.get("/api/v1/portfolio/allocation")
    async def get_allocation() -> Dict[str, Any]:
        """Get portfolio allocation breakdown."""
        allocation = portfolio.get_allocation()
        return {
            "by_symbol": {k: float(v) for k, v in allocation.by_symbol.items()},
            "by_side": {k: float(v) for k, v in allocation.by_side.items()},
            "cash_pct": float(allocation.cash_pct),
        }

    @app.get("/api/v1/portfolio/exposure")
    async def get_exposure() -> Dict[str, Any]:
        """Get portfolio exposure metrics."""
        return portfolio.get_exposure()

    @app.get("/api/v1/portfolio/drawdown")
    async def get_drawdown() -> Dict[str, Any]:
        """Get drawdown metrics."""
        return portfolio.get_drawdown()

    @app.get("/api/v1/trades")
    async def get_trades(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get trade history with pagination."""
        trades = portfolio.closed_trades

        if symbol:
            trades = [t for t in trades if t.get('symbol') == symbol]

        total = len(trades)
        trades = trades[offset:offset + limit]

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "trades": trades,
        }

    @app.get("/api/v1/performance/stats")
    async def get_performance_stats() -> Dict[str, Any]:
        """Get performance statistics."""
        stats = metrics_calculator.calculate_all(
            trades=portfolio.closed_trades,
            initial_balance=float(portfolio.initial_balance),
        )

        return {
            "returns": {
                "total": stats.total_return,
                "total_pct": stats.total_return_pct,
                "annualized": stats.annualized_return,
            },
            "risk": {
                "volatility": stats.volatility,
                "annualized_volatility": stats.annualized_volatility,
                "sharpe_ratio": stats.sharpe_ratio,
                "sortino_ratio": stats.sortino_ratio,
                "calmar_ratio": stats.calmar_ratio,
            },
            "drawdown": {
                "max": stats.max_drawdown,
                "max_duration": stats.max_drawdown_duration,
                "current": stats.current_drawdown,
            },
            "trades": {
                "total": stats.total_trades,
                "winning": stats.winning_trades,
                "losing": stats.losing_trades,
                "win_rate": stats.win_rate,
            },
            "pnl": {
                "gross_profit": stats.gross_profit,
                "gross_loss": stats.gross_loss,
                "net_profit": stats.net_profit,
                "profit_factor": stats.profit_factor,
            },
            "averages": {
                "avg_win": stats.avg_win,
                "avg_loss": stats.avg_loss,
                "avg_trade": stats.avg_trade,
                "largest_win": stats.largest_win,
                "largest_loss": stats.largest_loss,
            },
            "ratios": {
                "reward_risk": stats.reward_risk_ratio,
                "expectancy": stats.expectancy,
            },
            "streaks": {
                "max_win": stats.max_win_streak,
                "max_loss": stats.max_loss_streak,
                "current": stats.current_streak,
                "current_type": stats.current_streak_type,
            },
        }

    @app.get("/api/v1/performance/equity")
    async def get_equity_curve() -> Dict[str, Any]:
        """Get equity curve data."""
        snapshots = portfolio.snapshots

        return {
            "timestamps": [s.timestamp.isoformat() for s in snapshots],
            "values": [float(s.total_value) for s in snapshots],
            "daily_pnl": [float(s.daily_pnl) for s in snapshots],
            "daily_pnl_pct": [float(s.daily_pnl_pct) for s in snapshots],
        }

    @app.get("/api/v1/performance/by-symbol")
    async def get_performance_by_symbol() -> Dict[str, Any]:
        """Get performance breakdown by symbol."""
        return portfolio.get_symbol_performance()

    @app.get("/api/v1/performance/by-time")
    async def get_performance_by_time() -> Dict[str, Any]:
        """Get performance breakdown by time."""
        return portfolio.get_time_analysis()

    @app.get("/api/v1/performance/top-trades")
    async def get_top_trades(n: int = Query(5, ge=1, le=20)) -> Dict[str, Any]:
        """Get top performing trades."""
        return {
            "best": portfolio.get_top_performers(n),
            "worst": portfolio.get_worst_performers(n),
        }

    @app.get("/api/v1/performance/rolling")
    async def get_rolling_metrics(
        window: int = Query(20, ge=5, le=100),
    ) -> List[Dict[str, Any]]:
        """Get rolling performance metrics."""
        return metrics_calculator.calculate_rolling_metrics(
            trades=portfolio.closed_trades,
            window=window,
        )

    @app.get("/api/v1/analysis/risk")
    async def get_risk_analysis() -> Dict[str, Any]:
        """Get risk analysis."""
        stats = metrics_calculator.calculate_all(
            trades=portfolio.closed_trades,
            initial_balance=float(portfolio.initial_balance),
        )

        exposure = portfolio.get_exposure()
        drawdown = portfolio.get_drawdown()

        return {
            "volatility": {
                "daily": stats.volatility,
                "annualized": stats.annualized_volatility,
            },
            "risk_adjusted_returns": {
                "sharpe": stats.sharpe_ratio,
                "sortino": stats.sortino_ratio,
                "calmar": stats.calmar_ratio,
            },
            "exposure": exposure,
            "drawdown": drawdown,
            "value_at_risk": {
                "var_95": stats.volatility * 1.645 * float(portfolio.total_value),
                "var_99": stats.volatility * 2.326 * float(portfolio.total_value),
            },
        }

    @app.get("/api/v1/analysis/daily")
    async def get_daily_analysis(
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get daily analysis."""
        if date:
            target_date = datetime.fromisoformat(date).date()
        else:
            target_date = datetime.now().date()

        day_trades = [
            t for t in portfolio.closed_trades
            if t.get('exit_time', '')[:10] == str(target_date)
        ]

        pnls = [t.get('pnl', 0) for t in day_trades]
        winning = [p for p in pnls if p > 0]

        return {
            "date": str(target_date),
            "trades": len(day_trades),
            "total_pnl": sum(pnls),
            "win_rate": (len(winning) / len(pnls) * 100) if pnls else 0,
            "trades_list": day_trades,
        }

    return app


# Standalone dashboard server
async def run_dashboard_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    portfolio: Optional[PortfolioAnalytics] = None,
) -> None:
    """Run standalone dashboard server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn not installed. Install with: pip install uvicorn")

    app = create_dashboard_app(portfolio)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()
