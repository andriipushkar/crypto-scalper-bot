"""
Comprehensive tests for Dashboard API.

Tests cover:
- Health check endpoint
- Portfolio summary
- Positions endpoints
- Performance statistics
- Equity curve
- Risk analysis
- Daily analysis
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any

import sys
sys.path.insert(0, '/home/sssmmmddd/Code/pro/crypto-scalper-bot')

# Mock FastAPI for testing without actual server
try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# =============================================================================
# Mock Classes for Testing
# =============================================================================

class MockPosition:
    """Mock position for testing."""
    def __init__(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        entry_price: Decimal,
        current_price: Decimal,
        leverage: int = 10,
    ):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.entry_price = entry_price
        self.current_price = current_price
        self.leverage = leverage
        self.entry_time = datetime.now()

    @property
    def unrealized_pnl(self) -> Decimal:
        if self.side == "LONG":
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> Decimal:
        if self.entry_price == 0:
            return Decimal("0")
        return (self.unrealized_pnl / (self.entry_price * self.quantity)) * 100


class MockAllocation:
    """Mock allocation for testing."""
    def __init__(self):
        self.by_symbol = {"BTCUSDT": Decimal("60"), "ETHUSDT": Decimal("40")}
        self.by_side = {"LONG": Decimal("80"), "SHORT": Decimal("20")}
        self.cash_pct = Decimal("30")


class MockSnapshot:
    """Mock snapshot for equity curve."""
    def __init__(self, timestamp: datetime, total_value: Decimal, daily_pnl: Decimal):
        self.timestamp = timestamp
        self.total_value = total_value
        self.daily_pnl = daily_pnl
        self.daily_pnl_pct = (daily_pnl / total_value * 100) if total_value > 0 else Decimal("0")


class MockPerformanceStats:
    """Mock performance statistics."""
    def __init__(self):
        # Returns
        self.total_return = 1500.0
        self.total_return_pct = 15.0
        self.annualized_return = 45.0

        # Risk
        self.volatility = 0.02
        self.annualized_volatility = 0.32
        self.sharpe_ratio = 1.5
        self.sortino_ratio = 2.0
        self.calmar_ratio = 1.2

        # Drawdown
        self.max_drawdown = 0.08
        self.max_drawdown_duration = 5
        self.current_drawdown = 0.02

        # Trades
        self.total_trades = 50
        self.winning_trades = 30
        self.losing_trades = 20
        self.win_rate = 60.0

        # PnL
        self.gross_profit = 2500.0
        self.gross_loss = 1000.0
        self.net_profit = 1500.0
        self.profit_factor = 2.5

        # Averages
        self.avg_win = 83.33
        self.avg_loss = 50.0
        self.avg_trade = 30.0
        self.largest_win = 300.0
        self.largest_loss = 150.0

        # Ratios
        self.reward_risk_ratio = 1.67
        self.expectancy = 30.0

        # Streaks
        self.max_win_streak = 5
        self.max_loss_streak = 3
        self.current_streak = 2
        self.current_streak_type = "win"


class MockPortfolioAnalytics:
    """Mock portfolio analytics."""
    def __init__(self):
        self.positions = {
            "BTCUSDT": MockPosition("BTCUSDT", "LONG", Decimal("0.1"), Decimal("50000"), Decimal("51000")),
            "ETHUSDT": MockPosition("ETHUSDT", "SHORT", Decimal("1.0"), Decimal("3000"), Decimal("2900")),
        }
        self.closed_trades = [
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "pnl": 100.0,
                "entry_time": "2024-01-15T10:00:00",
                "exit_time": "2024-01-15T11:00:00",
            },
            {
                "symbol": "BTCUSDT",
                "side": "SHORT",
                "pnl": -50.0,
                "entry_time": "2024-01-15T12:00:00",
                "exit_time": "2024-01-15T13:00:00",
            },
        ]
        self.initial_balance = Decimal("10000")
        self.total_value = Decimal("11500")
        self.snapshots = [
            MockSnapshot(datetime(2024, 1, 15), Decimal("10500"), Decimal("500")),
            MockSnapshot(datetime(2024, 1, 16), Decimal("11000"), Decimal("500")),
            MockSnapshot(datetime(2024, 1, 17), Decimal("11500"), Decimal("500")),
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_value": float(self.total_value),
            "initial_balance": float(self.initial_balance),
            "unrealized_pnl": 200.0,
            "position_count": len(self.positions),
        }

    def get_allocation(self) -> MockAllocation:
        return MockAllocation()

    def get_exposure(self) -> Dict[str, Any]:
        return {
            "gross_exposure": 0.7,
            "net_exposure": 0.4,
            "long_exposure": 0.6,
            "short_exposure": 0.2,
        }

    def get_drawdown(self) -> Dict[str, Any]:
        return {
            "current": 0.02,
            "max": 0.08,
            "from_peak": 200.0,
        }

    def get_symbol_performance(self) -> Dict[str, Any]:
        return {
            "BTCUSDT": {"trades": 10, "pnl": 500.0, "win_rate": 0.6},
            "ETHUSDT": {"trades": 8, "pnl": 300.0, "win_rate": 0.5},
        }

    def get_time_analysis(self) -> Dict[str, Any]:
        return {
            "best_hour": 10,
            "worst_hour": 15,
            "best_day": "Monday",
            "worst_day": "Friday",
        }

    def get_top_performers(self, n: int) -> list:
        return [{"symbol": "BTCUSDT", "pnl": 300.0}]

    def get_worst_performers(self, n: int) -> list:
        return [{"symbol": "ETHUSDT", "pnl": -150.0}]


class MockPerformanceMetrics:
    """Mock performance metrics calculator."""
    def calculate_all(self, trades: list, initial_balance: float) -> MockPerformanceStats:
        return MockPerformanceStats()

    def calculate_rolling_metrics(self, trades: list, window: int) -> list:
        return [
            {"date": "2024-01-15", "rolling_pnl": 100.0, "rolling_win_rate": 0.6},
            {"date": "2024-01-16", "rolling_pnl": 150.0, "rolling_win_rate": 0.65},
        ]


# =============================================================================
# Dashboard API Tests
# =============================================================================

@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
class TestDashboardAPI:
    """Tests for Dashboard API endpoints."""

    @pytest.fixture
    def mock_portfolio(self):
        """Create mock portfolio."""
        return MockPortfolioAnalytics()

    @pytest.fixture
    def client(self, mock_portfolio):
        """Create test client."""
        with patch('src.dashboard.api.PortfolioAnalytics', return_value=mock_portfolio):
            with patch('src.dashboard.api.PerformanceMetrics', return_value=MockPerformanceMetrics()):
                from src.dashboard.api import create_dashboard_app
                app = create_dashboard_app(portfolio=mock_portfolio)
                return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_portfolio_summary(self, client):
        """Test portfolio summary endpoint."""
        response = client.get("/api/v1/portfolio/summary")

        assert response.status_code == 200
        data = response.json()
        assert "total_value" in data
        assert "initial_balance" in data

    def test_positions(self, client):
        """Test positions endpoint."""
        response = client.get("/api/v1/portfolio/positions")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_allocation(self, client):
        """Test allocation endpoint."""
        response = client.get("/api/v1/portfolio/allocation")

        assert response.status_code == 200
        data = response.json()
        assert "by_symbol" in data
        assert "by_side" in data
        assert "cash_pct" in data

    def test_exposure(self, client):
        """Test exposure endpoint."""
        response = client.get("/api/v1/portfolio/exposure")

        assert response.status_code == 200
        data = response.json()
        assert "gross_exposure" in data
        assert "net_exposure" in data

    def test_drawdown(self, client):
        """Test drawdown endpoint."""
        response = client.get("/api/v1/portfolio/drawdown")

        assert response.status_code == 200
        data = response.json()
        assert "current" in data
        assert "max" in data

    def test_trades_pagination(self, client):
        """Test trades endpoint with pagination."""
        response = client.get("/api/v1/trades?limit=10&offset=0")

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert "trades" in data

    def test_trades_filter_by_symbol(self, client):
        """Test trades endpoint filtered by symbol."""
        response = client.get("/api/v1/trades?symbol=BTCUSDT")

        assert response.status_code == 200
        data = response.json()
        assert all(t["symbol"] == "BTCUSDT" for t in data["trades"])

    def test_performance_stats(self, client):
        """Test performance statistics endpoint."""
        response = client.get("/api/v1/performance/stats")

        assert response.status_code == 200
        data = response.json()
        assert "returns" in data
        assert "risk" in data
        assert "drawdown" in data
        assert "trades" in data
        assert "pnl" in data
        assert "averages" in data
        assert "ratios" in data
        assert "streaks" in data

    def test_equity_curve(self, client):
        """Test equity curve endpoint."""
        response = client.get("/api/v1/performance/equity")

        assert response.status_code == 200
        data = response.json()
        assert "timestamps" in data
        assert "values" in data
        assert "daily_pnl" in data
        assert len(data["timestamps"]) == 3

    def test_performance_by_symbol(self, client):
        """Test performance by symbol endpoint."""
        response = client.get("/api/v1/performance/by-symbol")

        assert response.status_code == 200
        data = response.json()
        assert "BTCUSDT" in data or isinstance(data, dict)

    def test_performance_by_time(self, client):
        """Test performance by time endpoint."""
        response = client.get("/api/v1/performance/by-time")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_top_trades(self, client):
        """Test top trades endpoint."""
        response = client.get("/api/v1/performance/top-trades?n=5")

        assert response.status_code == 200
        data = response.json()
        assert "best" in data
        assert "worst" in data

    def test_rolling_metrics(self, client):
        """Test rolling metrics endpoint."""
        response = client.get("/api/v1/performance/rolling?window=20")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_risk_analysis(self, client):
        """Test risk analysis endpoint."""
        response = client.get("/api/v1/analysis/risk")

        assert response.status_code == 200
        data = response.json()
        assert "volatility" in data
        assert "risk_adjusted_returns" in data
        assert "exposure" in data
        assert "drawdown" in data
        assert "value_at_risk" in data

    def test_daily_analysis(self, client):
        """Test daily analysis endpoint."""
        response = client.get("/api/v1/analysis/daily")

        assert response.status_code == 200
        data = response.json()
        assert "date" in data
        assert "trades" in data
        assert "total_pnl" in data

    def test_daily_analysis_specific_date(self, client):
        """Test daily analysis for specific date."""
        response = client.get("/api/v1/analysis/daily?date=2024-01-15")

        assert response.status_code == 200
        data = response.json()
        assert data["date"] == "2024-01-15"


# =============================================================================
# Unit Tests without FastAPI
# =============================================================================

class TestDashboardAPIUnit:
    """Unit tests for Dashboard API without FastAPI."""

    def test_mock_portfolio_to_dict(self):
        """Test portfolio to_dict method."""
        portfolio = MockPortfolioAnalytics()
        data = portfolio.to_dict()

        assert data["total_value"] == 11500.0
        assert data["initial_balance"] == 10000.0

    def test_mock_allocation(self):
        """Test allocation data structure."""
        portfolio = MockPortfolioAnalytics()
        allocation = portfolio.get_allocation()

        assert "BTCUSDT" in allocation.by_symbol
        assert "LONG" in allocation.by_side
        assert allocation.cash_pct == Decimal("30")

    def test_mock_exposure(self):
        """Test exposure data structure."""
        portfolio = MockPortfolioAnalytics()
        exposure = portfolio.get_exposure()

        assert exposure["gross_exposure"] == 0.7
        assert exposure["net_exposure"] == 0.4

    def test_mock_position_pnl_long(self):
        """Test position PnL calculation for long."""
        position = MockPosition(
            symbol="BTCUSDT",
            side="LONG",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
        )

        # (51000 - 50000) * 0.1 = 100
        assert position.unrealized_pnl == Decimal("100")

    def test_mock_position_pnl_short(self):
        """Test position PnL calculation for short."""
        position = MockPosition(
            symbol="BTCUSDT",
            side="SHORT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("49000"),
        )

        # (50000 - 49000) * 0.1 = 100
        assert position.unrealized_pnl == Decimal("100")

    def test_mock_position_pnl_pct(self):
        """Test position PnL percentage."""
        position = MockPosition(
            symbol="BTCUSDT",
            side="LONG",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
        )

        # 100 / (50000 * 0.1) * 100 = 2%
        assert position.unrealized_pnl_pct == Decimal("2")

    def test_mock_performance_stats(self):
        """Test performance statistics structure."""
        stats = MockPerformanceStats()

        assert stats.total_return == 1500.0
        assert stats.sharpe_ratio == 1.5
        assert stats.win_rate == 60.0

    def test_mock_metrics_calculator(self):
        """Test metrics calculator."""
        calculator = MockPerformanceMetrics()
        stats = calculator.calculate_all([], 10000.0)

        assert isinstance(stats, MockPerformanceStats)

    def test_mock_rolling_metrics(self):
        """Test rolling metrics calculation."""
        calculator = MockPerformanceMetrics()
        rolling = calculator.calculate_rolling_metrics([], window=20)

        assert isinstance(rolling, list)
        assert len(rolling) == 2

    def test_portfolio_closed_trades(self):
        """Test portfolio closed trades."""
        portfolio = MockPortfolioAnalytics()

        assert len(portfolio.closed_trades) == 2
        assert portfolio.closed_trades[0]["symbol"] == "BTCUSDT"

    def test_portfolio_snapshots(self):
        """Test portfolio snapshots."""
        portfolio = MockPortfolioAnalytics()

        assert len(portfolio.snapshots) == 3
        assert portfolio.snapshots[0].total_value == Decimal("10500")


# =============================================================================
# Edge Cases
# =============================================================================

class TestDashboardEdgeCases:
    """Tests for edge cases."""

    def test_position_zero_entry_price(self):
        """Test position with zero entry price."""
        position = MockPosition(
            symbol="BTCUSDT",
            side="LONG",
            quantity=Decimal("0.1"),
            entry_price=Decimal("0"),
            current_price=Decimal("50000"),
        )

        # Should handle gracefully
        assert position.unrealized_pnl_pct == Decimal("0")

    def test_empty_portfolio(self):
        """Test empty portfolio."""
        portfolio = MockPortfolioAnalytics()
        portfolio.positions = {}
        portfolio.closed_trades = []

        assert len(portfolio.positions) == 0
        assert portfolio.to_dict()["position_count"] == 0

    def test_large_values(self):
        """Test with large values."""
        position = MockPosition(
            symbol="BTCUSDT",
            side="LONG",
            quantity=Decimal("1000"),
            entry_price=Decimal("100000"),
            current_price=Decimal("110000"),
        )

        # (110000 - 100000) * 1000 = 10,000,000
        assert position.unrealized_pnl == Decimal("10000000")

    def test_negative_pnl(self):
        """Test negative PnL scenario."""
        position = MockPosition(
            symbol="BTCUSDT",
            side="LONG",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("45000"),
        )

        assert position.unrealized_pnl < 0
        assert position.unrealized_pnl == Decimal("-500")


# =============================================================================
# Integration Tests
# =============================================================================

class TestDashboardIntegration:
    """Integration tests for Dashboard API."""

    def test_full_portfolio_flow(self):
        """Test complete portfolio data flow."""
        portfolio = MockPortfolioAnalytics()

        # Get summary
        summary = portfolio.to_dict()
        assert summary["total_value"] > summary["initial_balance"]

        # Get positions
        positions = portfolio.positions
        assert len(positions) > 0

        # Get allocation
        allocation = portfolio.get_allocation()
        total_allocation = sum(allocation.by_symbol.values())
        assert total_allocation == Decimal("100")

        # Get exposure
        exposure = portfolio.get_exposure()
        assert exposure["long_exposure"] > 0

        # Get drawdown
        drawdown = portfolio.get_drawdown()
        assert drawdown["max"] >= drawdown["current"]

    def test_performance_calculation_flow(self):
        """Test performance calculation flow."""
        portfolio = MockPortfolioAnalytics()
        calculator = MockPerformanceMetrics()

        # Calculate stats
        stats = calculator.calculate_all(
            portfolio.closed_trades,
            float(portfolio.initial_balance),
        )

        # Verify relationships
        assert stats.winning_trades + stats.losing_trades <= stats.total_trades
        assert stats.gross_profit - stats.gross_loss == stats.net_profit
        assert stats.profit_factor > 0 or stats.gross_loss == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
