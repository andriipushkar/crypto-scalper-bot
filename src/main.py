#!/usr/bin/env python3
"""
Crypto Scalper Bot - Main Entry Point.

Usage:
    python -m src.main [OPTIONS]

Options:
    --mode      Mode: paper, live, collect (default: paper)
    --config    Config file path (default: config/settings.yaml)
    --symbols   Override symbols (comma-separated)
    --dry-run   Show config and exit without trading

Examples:
    # Paper trading on testnet
    python -m src.main --mode paper

    # Data collection only
    python -m src.main --mode collect

    # Live trading (requires production API keys)
    python -m src.main --mode live

    # Custom symbols
    python -m src.main --symbols BTCUSDT,ETHUSDT
"""

import argparse
import asyncio
import signal
import sys
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from src.utils.config import load_config, validate_config
from src.utils.logger import setup_logger
from src.utils.performance import setup_uvloop, optimize_process
from src.utils.security import SecretFilter
from src.core.engine import TradingEngine
from src.strategy import (
    OrderBookImbalanceStrategy,
    VolumeSpikeStrategy,
    ImpulseScalpingStrategy,
    AdvancedOrderBookStrategy,
    HybridScalpingStrategy,
)
from src.analytics import PrintTapeAnalyzer, ClusterAnalyzer
from src.risk import RiskManager
from src.execution import BinanceFuturesAPI, OrderExecutor
from src.data.websocket import create_websocket_from_config
from src.data.orderbook import OrderBookManager
from src.data.storage import MarketDataStorage


# =============================================================================
# Graceful Shutdown Handler
# =============================================================================

class GracefulShutdown:
    """
    Handle graceful shutdown with proper cleanup.

    Ensures all components are stopped properly on SIGTERM/SIGINT.
    """

    def __init__(self):
        self._shutdown_event = asyncio.Event()
        self._shutdown_callbacks = []
        self._is_shutting_down = False

    def add_callback(self, callback):
        """Add a shutdown callback."""
        self._shutdown_callbacks.append(callback)

    def trigger(self):
        """Trigger shutdown."""
        if self._is_shutting_down:
            logger.warning("Shutdown already in progress, forcing exit...")
            sys.exit(1)

        self._is_shutting_down = True
        logger.info("Graceful shutdown initiated...")
        self._shutdown_event.set()

    async def wait(self):
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    async def execute_callbacks(self):
        """Execute all shutdown callbacks."""
        for callback in self._shutdown_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Shutdown callback error: {e}")

    @property
    def is_shutting_down(self) -> bool:
        return self._is_shutting_down


shutdown_handler = GracefulShutdown()


class ScalperBot:
    """
    Main bot orchestrator.

    Handles initialization and lifecycle of all components
    based on selected mode.
    """

    def __init__(self, config: dict, mode: str = "paper"):
        """
        Initialize bot.

        Args:
            config: Configuration dictionary
            mode: Operating mode (paper, live, collect)
        """
        self.config = config
        self.mode = mode

        # Components (initialized in start())
        self.engine: TradingEngine = None
        self.api: BinanceFuturesAPI = None
        self.storage: MarketDataStorage = None

        self._running = False

    async def start(self) -> None:
        """Start the bot."""
        logger.info("=" * 60)
        logger.info(f"CRYPTO SCALPER BOT - {self.mode.upper()} MODE")
        logger.info("=" * 60)

        self._running = True

        if self.mode == "collect":
            await self._run_data_collection()
        elif self.mode in ("paper", "live"):
            await self._run_trading()
        else:
            logger.error(f"Unknown mode: {self.mode}")

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        logger.info("Stopping bot...")
        self._running = False

        if self.engine:
            await self.engine.stop()

        if self.api:
            await self.api.close()

        if self.storage:
            await self.storage.stop()

        logger.info("Bot stopped")

    async def _run_data_collection(self) -> None:
        """Run in data collection mode."""
        from src.data.models import TradeEvent, DepthUpdateEvent, MarkPriceEvent, OrderBookEvent

        logger.info("Starting data collection mode")

        exchange_config = self.config.get("exchange", {})
        data_config = self.config.get("data", {})
        trading_config = self.config.get("trading", {})

        # Initialize WebSocket
        websocket = create_websocket_from_config(self.config)

        # Initialize order book manager
        orderbook_manager = OrderBookManager(
            depth=data_config.get("orderbook_depth", 20),
            testnet=exchange_config.get("testnet", True),
        )

        for symbol in trading_config.get("symbols", ["BTCUSDT"]):
            orderbook_manager.get_book(symbol)

        # Initialize storage
        storage_config = data_config.get("storage", {})
        self.storage = MarketDataStorage(
            database_path=storage_config.get("database_path", "data/raw/market_data.db"),
            flush_interval=storage_config.get("flush_interval", 60),
        )
        await self.storage.start()

        # Stats
        stats = {"trades": 0, "depth": 0, "mark_price": 0}

        # Callbacks
        async def on_trade(event: TradeEvent):
            stats["trades"] += 1
            await self.storage.store_trade(event)
            if stats["trades"] % 100 == 0:
                logger.debug(f"Trades: {stats['trades']}")

        async def on_depth(event: DepthUpdateEvent):
            stats["depth"] += 1
            await orderbook_manager.handle_depth_update(event)

            symbol = event.update.symbol
            book = orderbook_manager.get_book(symbol)
            if book.is_synced:
                snapshot = book.get_snapshot()
                await self.storage.store_orderbook(OrderBookEvent(snapshot=snapshot))

        async def on_mark_price(event: MarkPriceEvent):
            stats["mark_price"] += 1
            await self.storage.store_mark_price(event)

        websocket.on_trade(on_trade)
        websocket.on_depth(on_depth)
        websocket.on_mark_price(on_mark_price)

        # Sync order books
        logger.info("Syncing order books...")
        await orderbook_manager.sync_all()

        # Run
        try:
            await websocket.start()
        finally:
            await websocket.stop()
            logger.info(f"Final stats: {stats}")

    async def _run_trading(self) -> None:
        """Run in trading mode (paper or live)."""
        exchange_config = self.config.get("exchange", {})
        trading_config = self.config.get("trading", {})
        strategies_config = self.config.get("strategies", {})

        # Determine if testnet
        is_testnet = exchange_config.get("testnet", True)
        if self.mode == "live":
            is_testnet = False
            logger.warning("!" * 60)
            logger.warning("LIVE TRADING MODE - REAL MONEY AT RISK")
            logger.warning("!" * 60)

        # Initialize API
        self.api = BinanceFuturesAPI(testnet=is_testnet)

        if not self.api.has_credentials:
            logger.error("API credentials not configured!")
            logger.error("Set environment variables or update .env file")
            return

        await self.api.connect()

        # Configure exchange
        leverage = trading_config.get("leverage", 10)
        margin_type = trading_config.get("margin_type", "CROSSED")

        for symbol in trading_config.get("symbols", ["BTCUSDT"]):
            try:
                await self.api.set_leverage(symbol, leverage)
                await self.api.set_margin_type(symbol, margin_type)
            except Exception as e:
                logger.warning(f"Config {symbol}: {e}")

        # Log account info
        balance = await self.api.get_balance("USDT")
        positions = await self.api.get_positions()
        logger.info(f"Balance: ${balance:.2f} USDT")
        logger.info(f"Open positions: {len(positions)}")

        # Initialize risk manager
        from src.utils.config import load_risk_config
        risk_config = load_risk_config()
        risk_manager = RiskManager(risk_config)

        # Initialize executor
        executor = OrderExecutor(
            api=self.api,
            risk_manager=risk_manager,
            use_limit_orders=trading_config.get("orders", {}).get("default_type") == "LIMIT",
        )

        # Initialize strategies
        strategies = []

        # Basic strategies
        if strategies_config.get("orderbook_imbalance", {}).get("enabled", True):
            strategies.append(OrderBookImbalanceStrategy(
                strategies_config.get("orderbook_imbalance", {})
            ))

        if strategies_config.get("volume_spike", {}).get("enabled", True):
            strategies.append(VolumeSpikeStrategy(
                strategies_config.get("volume_spike", {})
            ))

        # Advanced scalping strategies
        # Check if hybrid scalping is enabled - it combines all advanced strategies
        if strategies_config.get("hybrid_scalping", {}).get("enabled", False):
            # Initialize analyzers for hybrid strategy
            print_tape_analyzer = None
            cluster_analyzer = None
            impulse_strategy = None
            advanced_orderbook = None

            if strategies_config.get("print_tape", {}).get("enabled", True):
                print_tape_analyzer = PrintTapeAnalyzer(
                    strategies_config.get("print_tape", {})
                )
                logger.info("Initialized PrintTapeAnalyzer")

            if strategies_config.get("cluster_analysis", {}).get("enabled", True):
                cluster_analyzer = ClusterAnalyzer(
                    strategies_config.get("cluster_analysis", {})
                )
                logger.info("Initialized ClusterAnalyzer")

            if strategies_config.get("impulse_scalping", {}).get("enabled", False):
                impulse_strategy = ImpulseScalpingStrategy(
                    strategies_config.get("impulse_scalping", {})
                )
                logger.info("Initialized ImpulseScalpingStrategy")

            if strategies_config.get("advanced_orderbook", {}).get("enabled", True):
                advanced_orderbook = AdvancedOrderBookStrategy(
                    strategies_config.get("advanced_orderbook", {})
                )
                logger.info("Initialized AdvancedOrderBookStrategy")

            # Create hybrid strategy combining all enabled components
            hybrid_config = strategies_config.get("hybrid_scalping", {})
            hybrid_strategy = HybridScalpingStrategy(
                config=hybrid_config,
                orderbook_strategy=advanced_orderbook,
                impulse_strategy=impulse_strategy,
                tape_analyzer=print_tape_analyzer,
                cluster_analyzer=cluster_analyzer,
            )
            strategies.append(hybrid_strategy)
            logger.info("Initialized HybridScalpingStrategy with components")

        else:
            # Individual advanced strategies (if hybrid is disabled)
            if strategies_config.get("advanced_orderbook", {}).get("enabled", False):
                strategies.append(AdvancedOrderBookStrategy(
                    strategies_config.get("advanced_orderbook", {})
                ))

            if strategies_config.get("impulse_scalping", {}).get("enabled", False):
                strategies.append(ImpulseScalpingStrategy(
                    strategies_config.get("impulse_scalping", {})
                ))

        if not strategies:
            logger.error("No strategies enabled!")
            return

        # Initialize engine
        self.engine = TradingEngine(self.config)

        for strategy in strategies:
            self.engine.add_strategy(strategy)

        self.engine.set_risk_manager(risk_manager)
        self.engine.set_executor(executor)

        # Run
        await self.engine.start()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Crypto Scalper Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        choices=["paper", "live", "collect"],
        default="paper",
        help="Operating mode (default: paper)",
    )

    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Config file path (default: config/settings.yaml)",
    )

    parser.add_argument(
        "--symbols",
        help="Override symbols (comma-separated)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show config and exit without trading",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    # Load environment
    load_dotenv()

    # Parse arguments
    args = parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Override symbols if provided
    if args.symbols:
        config["trading"]["symbols"] = [s.strip().upper() for s in args.symbols.split(",")]

    # Setup logging with secret filtering
    log_level = "DEBUG" if args.debug else config.get("logging", {}).get("level", "INFO")
    log_config = config.get("logging", {})
    file_config = log_config.get("file", {})

    setup_logger(
        level=log_level,
        log_file=file_config.get("path") if file_config.get("enabled") else None,
    )

    # Add secret filter to prevent leaking sensitive data in logs
    logger.configure(patcher=lambda record: SecretFilter()(record))

    # Apply process optimizations
    optimizations = optimize_process()
    logger.debug(f"Process optimizations: {optimizations}")

    # Validate config
    errors = validate_config(config)
    if errors:
        for error in errors:
            logger.error(f"Config error: {error}")
        sys.exit(1)

    # Dry run - show config and exit
    if args.dry_run:
        import yaml
        logger.info("Configuration:")
        # Redact sensitive info
        safe_config = config.copy()
        if "exchange" in safe_config:
            safe_config["exchange"] = {k: v for k, v in safe_config["exchange"].items()
                                       if "key" not in k.lower() and "secret" not in k.lower()}
        print(yaml.dump(safe_config, default_flow_style=False))
        logger.info("Dry run complete - exiting")
        return

    # Safety check for live mode
    if args.mode == "live":
        logger.warning("=" * 60)
        logger.warning("LIVE TRADING MODE SELECTED")
        logger.warning("Real money will be at risk!")
        logger.warning("=" * 60)

        confirm = input("Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            logger.info("Aborted")
            return

    # Create and run bot
    bot = ScalperBot(config, mode=args.mode)

    # Register shutdown callback
    shutdown_handler.add_callback(bot.stop)

    # Signal handlers
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_handler.trigger)

    # Run with graceful shutdown support
    try:
        # Start bot and wait for shutdown signal concurrently
        bot_task = asyncio.create_task(bot.start())

        # Wait for either bot to finish or shutdown signal
        shutdown_task = asyncio.create_task(shutdown_handler.wait())

        done, pending = await asyncio.wait(
            [bot_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Execute shutdown callbacks
        await shutdown_handler.execute_callbacks()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)

    logger.info("Bot exited cleanly")


def run():
    """Entry point with uvloop optimization."""
    # Try to enable uvloop for better performance
    uvloop_enabled = setup_uvloop()

    if uvloop_enabled:
        logger.info("Running with uvloop (optimized event loop)")
    else:
        logger.info("Running with default asyncio event loop")

    # Run main
    asyncio.run(main())


if __name__ == "__main__":
    run()
