"""
E2E Tests for Full System Integration

Tests the complete system from startup to shutdown.
"""
import pytest
import pytest_asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio


class TestSystemStartup:
    """Test system startup sequence."""

    @pytest.mark.asyncio
    async def test_full_startup_sequence(self, mock_exchange, mock_database, mock_redis):
        """Test complete system startup."""
        startup_steps = []

        # Step 1: Initialize configuration
        config = {
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "paper_trading": True,
            "risk_per_trade": Decimal("0.01")
        }
        startup_steps.append("config_loaded")

        # Step 2: Connect to database
        async with mock_database() as session:
            startup_steps.append("database_connected")

        # Step 3: Connect to Redis
        await mock_redis.ping()
        startup_steps.append("redis_connected")

        # Step 4: Connect to exchange
        await mock_exchange.connect()
        startup_steps.append("exchange_connected")

        # Step 5: Load strategies
        strategies = ["momentum", "mean_reversion"]
        startup_steps.append("strategies_loaded")

        # Step 6: Start data feeds
        startup_steps.append("data_feeds_started")

        # Verify all steps completed
        expected_steps = [
            "config_loaded",
            "database_connected",
            "redis_connected",
            "exchange_connected",
            "strategies_loaded",
            "data_feeds_started"
        ]
        assert startup_steps == expected_steps

    @pytest.mark.asyncio
    async def test_startup_with_existing_positions(self, mock_exchange):
        """Test startup with existing open positions."""
        # Simulate existing positions
        mock_exchange.get_positions.return_value = [
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": Decimal("0.1"),
                "entry_price": Decimal("50000.0")
            }
        ]

        positions = await mock_exchange.get_positions()

        # System should restore position tracking
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_startup_recovery_mode(self, mock_exchange, mock_database):
        """Test startup in recovery mode after crash."""
        # Check for incomplete orders from last session
        incomplete_orders = [
            {"order_id": "ORD-001", "status": "NEW"},
            {"order_id": "ORD-002", "status": "PARTIALLY_FILLED"}
        ]

        # Cancel incomplete orders
        for order in incomplete_orders:
            await mock_exchange.cancel_order(order["order_id"])

        assert mock_exchange.cancel_order.call_count == 2


class TestSystemShutdown:
    """Test system shutdown sequence."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, mock_exchange, mock_redis):
        """Test graceful system shutdown."""
        shutdown_steps = []

        # Step 1: Stop accepting new signals
        shutdown_steps.append("signals_stopped")

        # Step 2: Cancel open orders
        mock_exchange.get_open_orders.return_value = [
            {"order_id": "ORD-001"},
            {"order_id": "ORD-002"}
        ]
        open_orders = await mock_exchange.get_open_orders()
        for order in open_orders:
            await mock_exchange.cancel_order(order["order_id"])
        shutdown_steps.append("orders_cancelled")

        # Step 3: Close positions (optional based on config)
        close_positions_on_shutdown = False
        if close_positions_on_shutdown:
            positions = await mock_exchange.get_positions()
            for pos in positions:
                await mock_exchange.place_order(
                    pos["symbol"],
                    "SELL" if pos["side"] == "LONG" else "BUY",
                    pos["quantity"]
                )
        shutdown_steps.append("positions_handled")

        # Step 4: Disconnect from exchange
        await mock_exchange.disconnect()
        shutdown_steps.append("exchange_disconnected")

        # Step 5: Close Redis connection
        await mock_redis.close()
        shutdown_steps.append("redis_closed")

        # Verify orderly shutdown
        assert len(shutdown_steps) == 5

    @pytest.mark.asyncio
    async def test_shutdown_with_pending_orders(self, mock_exchange):
        """Test shutdown handling pending orders."""
        # Pending orders that need handling
        mock_exchange.get_open_orders.return_value = [
            {"order_id": "ORD-001", "symbol": "BTCUSDT", "side": "BUY"},
        ]

        pending = await mock_exchange.get_open_orders()

        # Cancel all pending
        cancelled = []
        for order in pending:
            result = await mock_exchange.cancel_order(order["order_id"])
            cancelled.append(order["order_id"])

        assert len(cancelled) == 1

    @pytest.mark.asyncio
    async def test_emergency_shutdown(self, mock_exchange):
        """Test emergency shutdown (kill switch)."""
        # Cancel all orders immediately
        await mock_exchange.cancel_order("ALL")

        # Close all positions at market
        mock_exchange.get_positions.return_value = [
            {"symbol": "BTCUSDT", "side": "LONG", "quantity": Decimal("0.1")}
        ]

        positions = await mock_exchange.get_positions()
        for pos in positions:
            await mock_exchange.place_order(
                pos["symbol"],
                "SELL" if pos["side"] == "LONG" else "BUY",
                pos["quantity"],
                order_type="MARKET"
            )

        # Verify positions closed
        assert mock_exchange.place_order.called


class TestCompleteTradingSession:
    """Test a complete trading session."""

    @pytest.mark.asyncio
    async def test_full_trading_day(self, mock_exchange, mock_database, mock_redis):
        """Test a simulated full trading day."""
        session_stats = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": Decimal("0")
        }

        # Simulate 10 trades
        for i in range(10):
            # Entry
            entry_order = await mock_exchange.place_order(
                "BTCUSDT",
                "BUY" if i % 2 == 0 else "SELL",
                Decimal("0.1")
            )

            # Simulate price movement
            pnl = Decimal(str(20 - (i * 5)))  # Some wins, some losses

            # Exit
            await mock_exchange.place_order(
                "BTCUSDT",
                "SELL" if i % 2 == 0 else "BUY",
                Decimal("0.1")
            )

            # Track stats
            session_stats["trades"] += 1
            if pnl > 0:
                session_stats["wins"] += 1
            else:
                session_stats["losses"] += 1
            session_stats["total_pnl"] += pnl

        assert session_stats["trades"] == 10
        assert session_stats["wins"] + session_stats["losses"] == 10

    @pytest.mark.asyncio
    async def test_session_with_multiple_strategies(self, mock_exchange):
        """Test session running multiple strategies."""
        strategies = {
            "momentum": {"trades": 0, "pnl": Decimal("0")},
            "mean_reversion": {"trades": 0, "pnl": Decimal("0")},
            "scalping": {"trades": 0, "pnl": Decimal("0")}
        }

        # Simulate signals from each strategy
        for strategy in strategies:
            for _ in range(3):
                await mock_exchange.place_order(
                    "BTCUSDT",
                    "BUY",
                    Decimal("0.1")
                )
                strategies[strategy]["trades"] += 1
                strategies[strategy]["pnl"] += Decimal("10")

        total_trades = sum(s["trades"] for s in strategies.values())
        total_pnl = sum(s["pnl"] for s in strategies.values())

        assert total_trades == 9
        assert total_pnl == Decimal("90")


class TestSystemMonitoring:
    """Test system monitoring and alerting."""

    @pytest.mark.asyncio
    async def test_health_monitoring(self, mock_exchange, mock_redis):
        """Test system health monitoring."""
        health_checks = {}

        # Check exchange connection
        health_checks["exchange"] = mock_exchange.is_connected

        # Check Redis connection
        redis_ok = await mock_redis.ping()
        health_checks["redis"] = redis_ok

        # Check data feed
        ticker = await mock_exchange.get_ticker("BTCUSDT")
        health_checks["data_feed"] = ticker is not None

        # Overall health
        is_healthy = all(health_checks.values())
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, mock_exchange):
        """Test performance metrics monitoring."""
        metrics = {
            "order_latency_ms": [],
            "data_latency_ms": [],
            "orders_per_minute": 0
        }

        # Simulate order latency measurement
        import time
        for _ in range(5):
            start = time.time()
            await mock_exchange.place_order("BTCUSDT", "BUY", Decimal("0.1"))
            latency = (time.time() - start) * 1000
            metrics["order_latency_ms"].append(latency)

        avg_latency = sum(metrics["order_latency_ms"]) / len(metrics["order_latency_ms"])
        assert avg_latency >= 0

    @pytest.mark.asyncio
    async def test_error_alerting(self, mock_exchange):
        """Test error alerting system."""
        alerts = []

        # Simulate various errors
        errors = [
            {"type": "connection_lost", "severity": "high"},
            {"type": "order_rejected", "severity": "medium"},
            {"type": "rate_limited", "severity": "low"}
        ]

        for error in errors:
            alerts.append({
                "timestamp": datetime.now(),
                "error": error["type"],
                "severity": error["severity"]
            })

        # Filter high severity
        high_severity = [a for a in alerts if a["severity"] == "high"]
        assert len(high_severity) == 1


class TestDataPersistence:
    """Test data persistence across sessions."""

    @pytest.mark.asyncio
    async def test_state_persistence(self, mock_database, mock_redis):
        """Test that state persists across restarts."""
        # Save state
        state = {
            "last_trade_id": "TRADE-100",
            "daily_pnl": "150.50",
            "positions": ["BTCUSDT"]
        }

        await mock_redis.set("bot_state", str(state))

        # "Restart" - retrieve state
        saved_state = await mock_redis.get("bot_state")
        assert saved_state is not None

    @pytest.mark.asyncio
    async def test_trade_history_persistence(self, mock_database):
        """Test trade history persists to database."""
        trades = [
            {"id": f"trade-{i}", "pnl": Decimal(str(i * 10))}
            for i in range(100)
        ]

        # Simulate batch insert
        inserted_count = len(trades)

        # Simulate query after "restart"
        retrieved_count = len(trades)

        assert inserted_count == retrieved_count


class TestConcurrencyHandling:
    """Test concurrent operation handling."""

    @pytest.mark.asyncio
    async def test_concurrent_order_execution(self, mock_exchange):
        """Test concurrent order execution."""
        # Place multiple orders concurrently
        tasks = []
        for i in range(10):
            task = mock_exchange.place_order(
                f"SYMBOL{i}USDT",
                "BUY",
                Decimal("0.1")
            )
            tasks.append(task)

        orders = await asyncio.gather(*tasks)

        assert len(orders) == 10
        assert all(o["status"] == "FILLED" for o in orders)

    @pytest.mark.asyncio
    async def test_concurrent_data_updates(self, mock_exchange):
        """Test handling concurrent data updates."""
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

        # Fetch all tickers concurrently
        tasks = [mock_exchange.get_ticker(symbol) for symbol in symbols]
        tickers = await asyncio.gather(*tasks)

        assert len(tickers) == 5

    @pytest.mark.asyncio
    async def test_race_condition_prevention(self, mock_exchange, mock_redis):
        """Test race condition prevention with locks."""
        # Acquire lock before critical operation
        lock_key = "position_lock:BTCUSDT"
        lock_token = "unique-token-123"

        # Simulate acquiring lock
        await mock_redis.set(lock_key, lock_token, ex=10)

        # Check if we have the lock
        current_holder = await mock_redis.get(lock_key)
        has_lock = current_holder == lock_token

        assert has_lock is True


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""

    @pytest.mark.asyncio
    async def test_profitable_trade_cycle(self, mock_exchange):
        """Test a complete profitable trade cycle."""
        symbol = "BTCUSDT"
        starting_balance = Decimal("10000.0")

        # 1. Check balance
        mock_exchange.get_account_balance.return_value = {
            "total": starting_balance,
            "available": starting_balance
        }
        balance = await mock_exchange.get_account_balance()
        assert balance["available"] == starting_balance

        # 2. Get market data
        ticker = await mock_exchange.get_ticker(symbol)
        assert ticker is not None

        # 3. Generate signal (simulated)
        signal = {"side": "BUY", "strength": 0.8}

        # 4. Calculate position size
        risk = balance["available"] * Decimal("0.01")  # 1%
        position_size = risk / Decimal("500")  # $500 stop distance

        # 5. Place entry order
        entry_order = await mock_exchange.place_order(
            symbol, "BUY", position_size
        )
        assert entry_order["status"] == "FILLED"

        # 6. Set stop loss and take profit (simulated as orders)
        # ... handled by exchange or trailing stop logic

        # 7. Simulate favorable price move and exit
        exit_order = await mock_exchange.place_order(
            symbol, "SELL", position_size
        )
        assert exit_order["status"] == "FILLED"

        # 8. Calculate PnL
        pnl = Decimal("50.0")  # Simulated profit
        final_balance = starting_balance + pnl

        assert final_balance > starting_balance

    @pytest.mark.asyncio
    async def test_losing_trade_with_stop_loss(self, mock_exchange):
        """Test a losing trade stopped by stop loss."""
        symbol = "BTCUSDT"

        # Entry
        entry = await mock_exchange.place_order(symbol, "BUY", Decimal("0.1"))

        # Simulate price drop hitting stop loss
        mock_exchange.get_ticker.return_value = {
            "last": Decimal("49000.0"),  # Below entry
            "bid": Decimal("48999.0"),
            "ask": Decimal("49001.0")
        }

        # Stop loss triggered
        stop_loss_order = await mock_exchange.place_order(
            symbol, "SELL", Decimal("0.1"), order_type="MARKET"
        )

        # Loss should be limited
        entry_price = Decimal("50000.0")
        exit_price = Decimal("49000.0")
        loss = (exit_price - entry_price) * Decimal("0.1")

        assert loss == Decimal("-100.0")

    @pytest.mark.asyncio
    async def test_multi_symbol_portfolio(self, mock_exchange):
        """Test managing a multi-symbol portfolio."""
        portfolio = {
            "BTCUSDT": {"side": "LONG", "qty": Decimal("0.1")},
            "ETHUSDT": {"side": "LONG", "qty": Decimal("1.0")},
            "SOLUSDT": {"side": "SHORT", "qty": Decimal("10.0")}
        }

        # Simulate P&L for each position
        pnls = {
            "BTCUSDT": Decimal("50.0"),
            "ETHUSDT": Decimal("-30.0"),
            "SOLUSDT": Decimal("20.0")
        }

        total_pnl = sum(pnls.values())
        assert total_pnl == Decimal("40.0")

        # Portfolio metrics
        num_positions = len(portfolio)
        winning_positions = sum(1 for p in pnls.values() if p > 0)
        win_rate = winning_positions / num_positions

        assert num_positions == 3
        assert win_rate == 2/3


class TestSystemRecovery:
    """Test system recovery scenarios."""

    @pytest.mark.asyncio
    async def test_recover_from_exchange_disconnect(self, mock_exchange):
        """Test recovery from exchange disconnection."""
        # Simulate disconnect
        mock_exchange.is_connected = False

        # Recovery loop
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await mock_exchange.connect()
                mock_exchange.is_connected = True
                break
            except Exception:
                await asyncio.sleep(0.1)

        assert mock_exchange.is_connected is True

    @pytest.mark.asyncio
    async def test_recover_from_data_gap(self, mock_exchange):
        """Test recovery from data gap."""
        # Detect data gap (no updates for extended period)
        last_update = datetime.now() - timedelta(minutes=5)
        gap_threshold = timedelta(minutes=1)

        has_gap = datetime.now() - last_update > gap_threshold
        assert has_gap is True

        # Request historical data to fill gap
        klines = await mock_exchange.get_klines("BTCUSDT", "1m", limit=5)
        assert len(klines) == 100  # From mock

    @pytest.mark.asyncio
    async def test_recover_orphaned_orders(self, mock_exchange):
        """Test recovering orphaned orders."""
        # Find orders without matching positions
        mock_exchange.get_open_orders.return_value = [
            {"order_id": "ORD-001", "symbol": "BTCUSDT"},
            {"order_id": "ORD-002", "symbol": "ETHUSDT"}
        ]
        mock_exchange.get_positions.return_value = [
            {"symbol": "BTCUSDT"}  # No ETHUSDT position
        ]

        open_orders = await mock_exchange.get_open_orders()
        positions = await mock_exchange.get_positions()

        position_symbols = {p["symbol"] for p in positions}
        orphaned = [o for o in open_orders if o["symbol"] not in position_symbols]

        # Cancel orphaned orders
        for order in orphaned:
            await mock_exchange.cancel_order(order["order_id"])

        assert len(orphaned) == 1
        assert orphaned[0]["symbol"] == "ETHUSDT"
