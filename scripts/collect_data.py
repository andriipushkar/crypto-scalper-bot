#!/usr/bin/env python3
"""
Data collection script.

Connects to Binance Futures WebSocket and stores market data
to SQLite for later analysis.

Usage:
    python scripts/collect_data.py

The script will run indefinitely, collecting:
- Aggregated trades
- Order book updates
- Mark prices / funding rates

Press Ctrl+C to stop gracefully.
"""

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from loguru import logger

from src.utils.logger import setup_logger
from src.data.websocket import BinanceWebSocket, create_websocket_from_config
from src.data.orderbook import OrderBookManager
from src.data.storage import MarketDataStorage
from src.data.models import TradeEvent, DepthUpdateEvent, MarkPriceEvent


class DataCollector:
    """
    Main data collection orchestrator.

    Coordinates WebSocket, order book, and storage components.
    """

    def __init__(self, config: dict):
        """
        Initialize data collector.

        Args:
            config: Configuration dictionary from settings.yaml
        """
        self.config = config
        self._running = False

        # Components
        self.websocket: BinanceWebSocket = None
        self.orderbook_manager: OrderBookManager = None
        self.storage: MarketDataStorage = None

        # Stats
        self._start_time: datetime = None
        self._trade_count = 0
        self._depth_count = 0
        self._mark_price_count = 0

    async def start(self) -> None:
        """Start data collection."""
        logger.info("Starting data collector")
        self._running = True
        self._start_time = datetime.utcnow()

        # Initialize components
        await self._init_components()

        # Register callbacks
        self._register_callbacks()

        # Sync order books first
        await self._sync_order_books()

        # Start WebSocket
        logger.info("Starting WebSocket connection...")
        await self.websocket.start()

    async def stop(self) -> None:
        """Stop data collection gracefully."""
        logger.info("Stopping data collector...")
        self._running = False

        # Stop WebSocket
        if self.websocket:
            await self.websocket.stop()

        # Stop storage (final flush)
        if self.storage:
            await self.storage.stop()

        # Print final stats
        self._print_stats()
        logger.info("Data collector stopped")

    async def _init_components(self) -> None:
        """Initialize all components."""
        exchange_config = self.config.get("exchange", {})
        data_config = self.config.get("data", {})
        trading_config = self.config.get("trading", {})

        # WebSocket
        self.websocket = create_websocket_from_config(self.config)

        # Order book manager
        self.orderbook_manager = OrderBookManager(
            depth=data_config.get("orderbook_depth", 20),
            testnet=exchange_config.get("testnet", True),
        )

        # Pre-create order books for all symbols
        for symbol in trading_config.get("symbols", ["BTCUSDT"]):
            self.orderbook_manager.get_book(symbol)

        # Storage
        storage_config = data_config.get("storage", {})
        self.storage = MarketDataStorage(
            database_path=storage_config.get("database_path", "data/raw/market_data.db"),
            flush_interval=storage_config.get("flush_interval", 60),
        )
        await self.storage.start()

    def _register_callbacks(self) -> None:
        """Register event callbacks."""
        # Trade events
        self.websocket.on_trade(self._on_trade)

        # Depth events -> order book manager -> storage
        self.websocket.on_depth(self._on_depth)

        # Mark price events
        self.websocket.on_mark_price(self._on_mark_price)

    async def _sync_order_books(self) -> None:
        """Synchronize all order books with exchange."""
        logger.info("Syncing order books...")
        await self.orderbook_manager.sync_all()
        logger.info("Order books synced")

    # =========================================================================
    # Event Handlers
    # =========================================================================

    async def _on_trade(self, event: TradeEvent) -> None:
        """Handle trade event."""
        self._trade_count += 1

        # Store trade
        await self.storage.store_trade(event)

        # Log periodically
        if self._trade_count % 100 == 0:
            trade = event.trade
            logger.debug(
                f"Trade: {trade.symbol} {trade.side.value} "
                f"{trade.quantity}@{trade.price}"
            )

    async def _on_depth(self, event: DepthUpdateEvent) -> None:
        """Handle depth update event."""
        self._depth_count += 1

        # Update order book
        await self.orderbook_manager.handle_depth_update(event)

        # Get updated snapshot and store
        symbol = event.update.symbol
        book = self.orderbook_manager.get_book(symbol)

        if book.is_synced:
            from src.data.models import OrderBookEvent
            snapshot = book.get_snapshot()
            await self.storage.store_orderbook(OrderBookEvent(snapshot=snapshot))

            # Log periodically
            if self._depth_count % 100 == 0:
                mid = snapshot.mid_price
                spread = snapshot.spread_bps
                imbalance = snapshot.imbalance(5)
                logger.debug(
                    f"OrderBook: {symbol} mid={mid:.2f} "
                    f"spread={spread:.2f}bps imbalance={imbalance:.2f}"
                )

    async def _on_mark_price(self, event: MarkPriceEvent) -> None:
        """Handle mark price event."""
        self._mark_price_count += 1

        # Store mark price
        await self.storage.store_mark_price(event)

        # Log periodically
        if self._mark_price_count % 60 == 0:  # ~1 per minute with 1s updates
            mp = event.mark_price
            logger.debug(
                f"MarkPrice: {mp.symbol} price={mp.mark_price:.2f} "
                f"funding={float(mp.funding_rate) * 100:.4f}%"
            )

    def _print_stats(self) -> None:
        """Print collection statistics."""
        if not self._start_time:
            return

        runtime = datetime.utcnow() - self._start_time
        hours = runtime.total_seconds() / 3600

        logger.info("=" * 50)
        logger.info("Data Collection Statistics")
        logger.info("=" * 50)
        logger.info(f"Runtime: {runtime}")
        logger.info(f"Trades collected: {self._trade_count:,}")
        logger.info(f"Depth updates: {self._depth_count:,}")
        logger.info(f"Mark prices: {self._mark_price_count:,}")

        if hours > 0:
            logger.info(f"Trades/hour: {self._trade_count / hours:,.0f}")
            logger.info(f"Depth updates/hour: {self._depth_count / hours:,.0f}")

        logger.info("=" * 50)

        # Storage stats
        if self.storage:
            stats = self.storage.stats
            logger.info(f"Stored to DB - Trades: {stats['trades_stored']:,}")
            logger.info(f"Stored to DB - OrderBooks: {stats['orderbooks_stored']:,}")
            logger.info(f"Stored to DB - MarkPrices: {stats['mark_prices_stored']:,}")


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        return yaml.safe_load(f)


async def main():
    """Main entry point."""
    # Load config
    config = load_config()

    # Setup logging
    log_config = config.get("logging", {})
    file_config = log_config.get("file", {})

    setup_logger(
        level=log_config.get("level", "INFO"),
        log_file=file_config.get("path") if file_config.get("enabled") else None,
        rotation=file_config.get("rotation", "1 day"),
        retention=file_config.get("retention", "30 days"),
    )

    logger.info("=" * 50)
    logger.info("Crypto Scalper Bot - Data Collector")
    logger.info("=" * 50)

    # Create collector
    collector = DataCollector(config)

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(collector.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_handler)

    # Start collection
    try:
        await collector.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        await collector.stop()


if __name__ == "__main__":
    asyncio.run(main())
