#!/usr/bin/env python3
"""
WebSocket connection test script.

Connects to Binance Futures WebSocket and displays live data
for a short period to verify everything works.

Usage:
    python scripts/test_websocket.py

Press Ctrl+C to stop.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from loguru import logger

from src.utils.logger import setup_logger
from src.data.websocket import create_websocket_from_config
from src.data.orderbook import OrderBookManager
from src.data.models import TradeEvent, DepthUpdateEvent, MarkPriceEvent


# Statistics
stats = {
    "trades": 0,
    "depth_updates": 0,
    "mark_prices": 0,
    "start_time": None,
}


async def on_trade(event: TradeEvent) -> None:
    """Handle trade event."""
    stats["trades"] += 1
    trade = event.trade

    # Show every 10th trade
    if stats["trades"] % 10 == 1:
        side_emoji = "🟢" if trade.side.value == "BUY" else "🔴"
        logger.info(
            f"{side_emoji} TRADE: {trade.symbol} "
            f"{trade.side.value:4} {float(trade.quantity):>12.4f} @ {float(trade.price):>10.2f} "
            f"| Value: ${float(trade.value):>10.2f}"
        )


async def on_depth(event: DepthUpdateEvent, orderbook_manager: OrderBookManager) -> None:
    """Handle depth update event."""
    stats["depth_updates"] += 1

    # Update order book
    await orderbook_manager.handle_depth_update(event)

    # Show order book state every 50 updates
    if stats["depth_updates"] % 50 == 1:
        symbol = event.update.symbol
        book = orderbook_manager.get_book(symbol)

        if book.is_synced:
            snapshot = book.get_snapshot()

            if snapshot.best_bid and snapshot.best_ask:
                logger.info(
                    f"📊 ORDERBOOK: {symbol} | "
                    f"Bid: {float(snapshot.best_bid.price):>10.2f} ({float(snapshot.best_bid.quantity):>8.4f}) | "
                    f"Ask: {float(snapshot.best_ask.price):>10.2f} ({float(snapshot.best_ask.quantity):>8.4f}) | "
                    f"Spread: {float(snapshot.spread_bps or 0):>5.2f}bps | "
                    f"Imbalance: {float(snapshot.imbalance(5)):>5.2f}"
                )


async def on_mark_price(event: MarkPriceEvent) -> None:
    """Handle mark price event."""
    stats["mark_prices"] += 1
    mp = event.mark_price

    # Show every update (they come every 1 second)
    if stats["mark_prices"] % 5 == 1:
        funding_pct = float(mp.funding_rate) * 100
        funding_emoji = "📈" if funding_pct > 0 else "📉" if funding_pct < 0 else "➡️"

        logger.info(
            f"{funding_emoji} MARK: {mp.symbol} | "
            f"Mark: {float(mp.mark_price):>10.2f} | "
            f"Index: {float(mp.index_price):>10.2f} | "
            f"Funding: {funding_pct:>+7.4f}% | "
            f"Next: {mp.next_funding_time.strftime('%H:%M:%S')}"
        )


def print_stats():
    """Print final statistics."""
    if not stats["start_time"]:
        return

    runtime = datetime.utcnow() - stats["start_time"]
    seconds = runtime.total_seconds()

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Runtime: {runtime}")
    print(f"Trades received: {stats['trades']:,}")
    print(f"Depth updates received: {stats['depth_updates']:,}")
    print(f"Mark prices received: {stats['mark_prices']:,}")

    if seconds > 0:
        print(f"\nRates:")
        print(f"  Trades/sec: {stats['trades'] / seconds:.1f}")
        print(f"  Depth updates/sec: {stats['depth_updates'] / seconds:.1f}")
        print(f"  Mark prices/sec: {stats['mark_prices'] / seconds:.1f}")

    print("=" * 60)

    # Verdict
    if stats["trades"] > 0 and stats["depth_updates"] > 0:
        print("✅ WebSocket connection WORKING!")
    else:
        print("❌ WebSocket connection FAILED - no data received")


async def main():
    """Main entry point."""
    # Setup logging
    setup_logger(level="INFO")

    print("=" * 60)
    print("Binance Futures WebSocket Test")
    print("=" * 60)
    print("This script will connect to Binance and show live data.")
    print("Press Ctrl+C to stop.\n")

    # Load config
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        logger.error("Config file not found: config/settings.yaml")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    exchange_config = config.get("exchange", {})
    trading_config = config.get("trading", {})
    data_config = config.get("data", {})

    symbols = trading_config.get("symbols", ["BTCUSDT"])
    testnet = exchange_config.get("testnet", True)

    print(f"Environment: {'TESTNET' if testnet else 'MAINNET'}")
    print(f"Symbols: {', '.join(symbols)}")
    print("-" * 60 + "\n")

    # Create components
    websocket = create_websocket_from_config(config)
    orderbook_manager = OrderBookManager(
        depth=data_config.get("orderbook_depth", 20),
        testnet=testnet,
    )

    # Pre-create order books
    for symbol in symbols:
        orderbook_manager.get_book(symbol)

    # Register callbacks
    websocket.on_trade(on_trade)
    websocket.on_depth(lambda e: on_depth(e, orderbook_manager))
    websocket.on_mark_price(on_mark_price)

    # Sync order books
    logger.info("Syncing order books...")
    await orderbook_manager.sync_all()
    logger.info("Order books synced!\n")

    stats["start_time"] = datetime.utcnow()

    # Run for a limited time or until interrupted
    try:
        await websocket.start()
    except KeyboardInterrupt:
        logger.info("\nStopping...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await websocket.stop()
        print_stats()


if __name__ == "__main__":
    asyncio.run(main())
