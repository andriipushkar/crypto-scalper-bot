"""
Live Trading Loop for Paper Mode

Connects to Binance WebSocket, runs strategies, and executes paper trades.
"""
import asyncio
import json
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

import websockets
from loguru import logger

from src.trading.paper_trading import PaperTradingEngine, OrderType


@dataclass
class OHLCV:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class LiveTradingLoop:
    """
    Live trading loop that:
    1. Connects to Binance WebSocket for real-time klines
    2. Accumulates candle history
    3. Runs strategies on each closed candle
    4. Executes paper trades
    """

    def __init__(
        self,
        symbols: List[str],
        timeframe: str = "1m",
        initial_balance: float = 1000.0,
        leverage: int = 10,
    ):
        self.symbols = [s.lower() for s in symbols]
        self.timeframe = timeframe
        self.running = False

        # Paper trading engine
        self.engine = PaperTradingEngine(
            initial_balance=Decimal(str(initial_balance)),
            commission_rate=Decimal("0.0004"),
            slippage_rate=Decimal("0.0001"),
            leverage=leverage,
        )

        # Candle history per symbol (for strategy calculations)
        self.history: Dict[str, List[OHLCV]] = {s.upper(): [] for s in self.symbols}
        self.current_candles: Dict[str, dict] = {}

        # Strategies (will be set externally)
        self.strategies: List[Callable] = []

        # Callbacks
        self.on_trade: Optional[Callable] = None
        self.on_signal: Optional[Callable] = None

        # Stats
        self.candles_processed = 0
        self.signals_generated = 0

    def add_strategy(self, strategy_func: Callable):
        """Add a strategy function."""
        self.strategies.append(strategy_func)
        logger.info(f"Added strategy: {strategy_func.__name__}")

    async def start(self):
        """Start the live trading loop."""
        self.running = True
        logger.info(f"Starting live trading loop for {len(self.symbols)} symbols")

        # Build WebSocket URL for multiple streams
        streams = [f"{s}@kline_{self.timeframe}" for s in self.symbols]
        ws_url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"

        while self.running:
            try:
                async with websockets.connect(ws_url, ping_interval=20) as ws:
                    logger.info("Connected to Binance Futures WebSocket")

                    async for message in ws:
                        if not self.running:
                            break

                        try:
                            data = json.loads(message)
                            await self._process_kline(data)
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)

    async def stop(self):
        """Stop the live trading loop."""
        self.running = False
        logger.info("Stopping live trading loop")

    async def _process_kline(self, data: dict):
        """Process incoming kline data."""
        if "data" not in data:
            return

        kline_data = data["data"]
        if kline_data.get("e") != "kline":
            return

        k = kline_data["k"]
        symbol = k["s"]  # e.g., "BTCUSDT"
        is_closed = k["x"]  # Is this kline closed?

        # Update current price in paper engine
        current_price = Decimal(k["c"])
        self.engine.update_price(symbol, current_price)

        # Only process closed candles
        if is_closed:
            candle = OHLCV(
                timestamp=datetime.fromtimestamp(k["t"] / 1000),
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
            )

            # Add to history (keep last 100 candles)
            if symbol not in self.history:
                self.history[symbol] = []
            self.history[symbol].append(candle)
            if len(self.history[symbol]) > 100:
                self.history[symbol] = self.history[symbol][-100:]

            self.candles_processed += 1

            # Run strategies
            await self._run_strategies(symbol, candle)

    async def _run_strategies(self, symbol: str, candle: OHLCV):
        """Run all strategies for a symbol."""
        history = self.history.get(symbol, [])

        if len(history) < 50:
            return  # Need enough history

        for strategy in self.strategies:
            try:
                signal = await self._execute_strategy(strategy, symbol, candle, history)
                if signal:
                    self.signals_generated += 1
                    if self.on_signal:
                        self.on_signal(signal)
            except Exception as e:
                logger.error(f"Strategy error {strategy.__name__}: {e}")

    async def _execute_strategy(
        self,
        strategy: Callable,
        symbol: str,
        candle: OHLCV,
        history: List[OHLCV]
    ) -> Optional[dict]:
        """Execute a single strategy and place orders if signaled."""

        # Create a simple interface for the strategy
        class StrategyEngine:
            def __init__(self, engine: PaperTradingEngine, symbol: str):
                self._engine = engine
                self._symbol = symbol
                self.positions = {}
                if symbol in engine.positions:
                    pos = engine.positions[symbol]
                    self.positions[symbol] = type('Position', (), {
                        'entry_price': float(pos.entry_price),
                        'side': 'BUY' if pos.side == 'long' else 'SELL',
                        'stop_loss': None,
                        'take_profit': None,
                    })()

            def get_available_balance(self):
                return float(self._engine.available_balance)

            async def place_order(self, symbol, side, qty):
                order = await self._engine.place_order(
                    symbol=symbol,
                    side="buy" if str(side) == "OrderSide.BUY" else "sell",
                    quantity=Decimal(str(qty)),
                    order_type=OrderType.MARKET,
                )
                return type('Order', (), {'filled': order.status.value == 'filled'})()

            def set_stop_loss(self, symbol, price):
                pass  # TODO: Implement SL orders

            def set_take_profit(self, symbol, price):
                pass  # TODO: Implement TP orders

        # Wrap engine for strategy compatibility
        eng = StrategyEngine(self.engine, symbol)

        # Run the strategy (synchronous strategies from backtest)
        try:
            strategy(eng, symbol, candle, history)
        except TypeError:
            # Strategy might be async
            await strategy(eng, symbol, candle, history)

        return None

    def get_status(self) -> dict:
        """Get current status."""
        return {
            "running": self.running,
            "symbols": [s.upper() for s in self.symbols],
            "candles_processed": self.candles_processed,
            "signals_generated": self.signals_generated,
            "positions": len(self.engine.positions),
            "trades": len(self.engine.trades),
            "balance": self.engine.get_balance(),
        }


# Singleton instance for web app integration
_live_trading_instance: Optional[LiveTradingLoop] = None


def get_live_trading() -> Optional[LiveTradingLoop]:
    """Get the live trading instance."""
    return _live_trading_instance


def set_live_trading(instance: LiveTradingLoop):
    """Set the live trading instance."""
    global _live_trading_instance
    _live_trading_instance = instance


async def start_live_trading(
    symbols: List[str],
    strategies: List[Callable],
    initial_balance: float = 1000.0,
) -> LiveTradingLoop:
    """Start live trading with given symbols and strategies."""
    global _live_trading_instance

    loop = LiveTradingLoop(
        symbols=symbols,
        timeframe="1m",
        initial_balance=initial_balance,
    )

    for strategy in strategies:
        loop.add_strategy(strategy)

    _live_trading_instance = loop

    # Start in background
    asyncio.create_task(loop.start())

    return loop
