#!/usr/bin/env python3
"""
Live Monitoring Dashboard.

Displays real-time market data and bot status in the terminal.

Usage:
    python scripts/monitor.py

Features:
- Live price and spread display
- Order book imbalance visualization
- Recent trades display
- Bot status (if running)

Press Ctrl+C to exit.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from dotenv import load_dotenv


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_price(price: float, decimals: int = 2) -> str:
    """Format price with color based on change."""
    return f"${price:,.{decimals}f}"


def format_pct(value: float) -> str:
    """Format percentage with sign."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def format_imbalance_bar(imbalance: float, width: int = 20) -> str:
    """Create visual bar for imbalance."""
    # imbalance > 1 = bullish (green), < 1 = bearish (red)
    if imbalance >= 1:
        # Bullish
        fill = min(int((imbalance - 1) * width), width)
        bar = "█" * fill + "░" * (width - fill)
        return f"[{bar}] {imbalance:.2f}"
    else:
        # Bearish
        inv = 1 / imbalance if imbalance > 0 else 10
        fill = min(int((inv - 1) * width), width)
        bar = "░" * (width - fill) + "█" * fill
        return f"[{bar}] {imbalance:.2f}"


class LiveMonitor:
    """Live monitoring dashboard."""

    def __init__(self, config: dict):
        self.config = config
        self._running = False

        # State
        self._last_price = 0.0
        self._last_spread_bps = 0.0
        self._last_imbalance = 1.0
        self._trade_count = 0
        self._buy_volume = 0.0
        self._sell_volume = 0.0
        self._recent_trades = []

    async def start(self):
        """Start monitoring."""
        from src.data.websocket import create_websocket_from_config
        from src.data.orderbook import OrderBookManager
        from src.data.models import TradeEvent, DepthUpdateEvent

        self._running = True

        exchange_config = self.config.get("exchange", {})
        data_config = self.config.get("data", {})
        trading_config = self.config.get("trading", {})

        symbols = trading_config.get("symbols", ["BTCUSDT"])
        symbol = symbols[0]

        # Initialize components
        websocket = create_websocket_from_config(self.config)
        orderbook_manager = OrderBookManager(
            depth=data_config.get("orderbook_depth", 20),
            testnet=exchange_config.get("testnet", True),
        )

        for s in symbols:
            orderbook_manager.get_book(s)

        # Callbacks
        async def on_trade(event: TradeEvent):
            trade = event.trade
            self._trade_count += 1

            if str(trade.side.value) == "BUY":
                self._buy_volume += float(trade.value)
            else:
                self._sell_volume += float(trade.value)

            self._recent_trades.append({
                "time": trade.timestamp,
                "side": str(trade.side.value),
                "price": float(trade.price),
                "qty": float(trade.quantity),
                "value": float(trade.value),
            })

            # Keep last 10 trades
            self._recent_trades = self._recent_trades[-10:]

        async def on_depth(event: DepthUpdateEvent):
            await orderbook_manager.handle_depth_update(event)

            book = orderbook_manager.get_book(symbol)
            if book.is_synced:
                snapshot = book.get_snapshot()
                if snapshot.mid_price:
                    self._last_price = float(snapshot.mid_price)
                if snapshot.spread_bps:
                    self._last_spread_bps = float(snapshot.spread_bps)

                self._last_imbalance = float(snapshot.imbalance(5))

        websocket.on_trade(on_trade)
        websocket.on_depth(on_depth)

        # Sync order books
        print("Syncing order books...")
        await orderbook_manager.sync_all()

        # Start display loop
        display_task = asyncio.create_task(self._display_loop(symbol))

        # Start WebSocket
        try:
            await websocket.start()
        finally:
            display_task.cancel()
            await websocket.stop()

    async def _display_loop(self, symbol: str):
        """Update display periodically."""
        while self._running:
            self._render(symbol)
            await asyncio.sleep(0.5)

    def _render(self, symbol: str):
        """Render the dashboard."""
        clear_screen()

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        print("╔════════════════════════════════════════════════════════════╗")
        print("║              CRYPTO SCALPER BOT - MONITOR                  ║")
        print(f"║  {now:^56}  ║")
        print("╠════════════════════════════════════════════════════════════╣")

        # Price section
        print("║  PRICE                                                     ║")
        print(f"║    {symbol}: {format_price(self._last_price):>15}                          ║")
        print(f"║    Spread: {self._last_spread_bps:>6.2f} bps                                  ║")
        print("╠════════════════════════════════════════════════════════════╣")

        # Order book imbalance
        print("║  ORDER BOOK IMBALANCE (5 levels)                           ║")
        bar = format_imbalance_bar(self._last_imbalance, 30)
        direction = "BULLISH" if self._last_imbalance > 1.1 else "BEARISH" if self._last_imbalance < 0.9 else "NEUTRAL"
        print(f"║    {bar:42} {direction:>8} ║")
        print("╠════════════════════════════════════════════════════════════╣")

        # Volume
        total_volume = self._buy_volume + self._sell_volume
        buy_pct = (self._buy_volume / total_volume * 100) if total_volume > 0 else 50
        print("║  VOLUME                                                    ║")
        print(f"║    Trades: {self._trade_count:>8,}                                      ║")
        print(f"║    Buy:  ${self._buy_volume:>12,.2f} ({buy_pct:>5.1f}%)                    ║")
        print(f"║    Sell: ${self._sell_volume:>12,.2f} ({100-buy_pct:>5.1f}%)                    ║")
        print("╠════════════════════════════════════════════════════════════╣")

        # Recent trades
        print("║  RECENT TRADES                                             ║")
        if self._recent_trades:
            for trade in reversed(self._recent_trades[-5:]):
                time_str = trade["time"].strftime("%H:%M:%S")
                side = trade["side"]
                side_indicator = "▲" if side == "BUY" else "▼"
                price = trade["price"]
                qty = trade["qty"]
                print(f"║    {time_str} {side_indicator} {side:4} {qty:>10.4f} @ {price:>10.2f}        ║")
        else:
            print("║    Waiting for trades...                                   ║")

        print("╠════════════════════════════════════════════════════════════╣")
        print("║  Press Ctrl+C to exit                                      ║")
        print("╚════════════════════════════════════════════════════════════╝")


async def main():
    """Main entry point."""
    load_dotenv()

    # Load config
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("CRYPTO SCALPER BOT - LIVE MONITOR")
    print("=" * 60)

    exchange_config = config.get("exchange", {})
    trading_config = config.get("trading", {})

    print(f"Environment: {'TESTNET' if exchange_config.get('testnet', True) else 'MAINNET'}")
    print(f"Symbols: {trading_config.get('symbols', ['BTCUSDT'])}")
    print("=" * 60)
    print()

    monitor = LiveMonitor(config)

    try:
        await monitor.start()
    except KeyboardInterrupt:
        print("\nMonitoring stopped")


if __name__ == "__main__":
    asyncio.run(main())
