"""
Binance Futures WebSocket client.

Handles connection to Binance Futures WebSocket streams:
- aggTrade: Aggregated trade stream
- depth: Order book depth updates
- markPrice: Mark price and funding rate
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, Callable, Dict, List, Set, Any
from decimal import Decimal

import websockets
from websockets.client import WebSocketClientProtocol
from loguru import logger

from src.data.models import (
    Trade,
    DepthUpdate,
    MarkPrice,
    TradeEvent,
    DepthUpdateEvent,
    MarkPriceEvent,
)


class BinanceWebSocket:
    """
    Async WebSocket client for Binance Futures.

    Manages connection lifecycle, automatic reconnection,
    and message parsing.
    """

    # Binance WebSocket endpoints
    MAINNET_WS = "wss://fstream.binance.com/ws"
    TESTNET_WS = "wss://stream.binancefuture.com/ws"

    def __init__(
        self,
        symbols: List[str],
        testnet: bool = True,
        ping_interval: int = 30,
        ping_timeout: int = 10,
        reconnect_delay: int = 5,
        max_reconnect_attempts: int = 10,
    ):
        """
        Initialize WebSocket client.

        Args:
            symbols: List of symbols to subscribe to (e.g., ["BTCUSDT"])
            testnet: Use testnet endpoint
            ping_interval: Seconds between pings
            ping_timeout: Seconds to wait for pong
            reconnect_delay: Seconds to wait before reconnecting
            max_reconnect_attempts: Max reconnection attempts before giving up
        """
        self.symbols = [s.lower() for s in symbols]
        self.testnet = testnet
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self._ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_count = 0
        self._subscribed_streams: Set[str] = set()

        # Callbacks for different event types
        self._trade_callbacks: List[Callable[[TradeEvent], Any]] = []
        self._depth_callbacks: List[Callable[[DepthUpdateEvent], Any]] = []
        self._mark_price_callbacks: List[Callable[[MarkPriceEvent], Any]] = []

        # Stats
        self._messages_received = 0
        self._last_message_time: Optional[datetime] = None

    @property
    def ws_url(self) -> str:
        """Get WebSocket URL based on environment."""
        return self.TESTNET_WS if self.testnet else self.MAINNET_WS

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._ws is not None and self._ws.open

    @property
    def stats(self) -> Dict:
        """Get connection statistics."""
        return {
            "connected": self.is_connected,
            "messages_received": self._messages_received,
            "last_message_time": self._last_message_time,
            "reconnect_count": self._reconnect_count,
            "subscribed_streams": list(self._subscribed_streams),
        }

    # =========================================================================
    # Callback Registration
    # =========================================================================

    def on_trade(self, callback: Callable[[TradeEvent], Any]) -> None:
        """Register callback for trade events."""
        self._trade_callbacks.append(callback)
        logger.debug(f"Registered trade callback: {callback.__name__}")

    def on_depth(self, callback: Callable[[DepthUpdateEvent], Any]) -> None:
        """Register callback for depth update events."""
        self._depth_callbacks.append(callback)
        logger.debug(f"Registered depth callback: {callback.__name__}")

    def on_mark_price(self, callback: Callable[[MarkPriceEvent], Any]) -> None:
        """Register callback for mark price events."""
        self._mark_price_callbacks.append(callback)
        logger.debug(f"Registered mark price callback: {callback.__name__}")

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> None:
        """
        Establish WebSocket connection.

        Connects to Binance and subscribes to configured streams.
        """
        streams = self._build_stream_list()
        url = f"{self.ws_url}/{'/'.join(streams)}"

        logger.info(f"Connecting to WebSocket: {url[:50]}...")

        try:
            self._ws = await websockets.connect(
                url,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
            )
            self._subscribed_streams = set(streams)
            self._reconnect_count = 0
            logger.info(f"Connected to Binance {'Testnet' if self.testnet else 'Mainnet'}")
            logger.info(f"Subscribed to streams: {streams}")

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    async def disconnect(self) -> None:
        """Close WebSocket connection gracefully."""
        self._running = False

        if self._ws:
            await self._ws.close()
            self._ws = None
            logger.info("Disconnected from WebSocket")

    async def start(self) -> None:
        """
        Start the WebSocket client.

        Connects and begins processing messages.
        Handles automatic reconnection on disconnection.
        """
        self._running = True
        logger.info("Starting WebSocket client")

        while self._running:
            try:
                if not self.is_connected:
                    await self.connect()

                await self._process_messages()

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")
                await self._handle_reconnect()

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await self._handle_reconnect()

        logger.info("WebSocket client stopped")

    async def stop(self) -> None:
        """Stop the WebSocket client."""
        logger.info("Stopping WebSocket client")
        self._running = False
        await self.disconnect()

    async def _handle_reconnect(self) -> None:
        """Handle reconnection logic."""
        if not self._running:
            return

        self._reconnect_count += 1

        if self._reconnect_count > self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) exceeded")
            self._running = False
            return

        delay = self.reconnect_delay * self._reconnect_count
        logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_count})")
        await asyncio.sleep(delay)

    # =========================================================================
    # Stream Building
    # =========================================================================

    def _build_stream_list(self) -> List[str]:
        """Build list of streams to subscribe to."""
        streams = []

        for symbol in self.symbols:
            # Aggregated trades (for volume analysis)
            streams.append(f"{symbol}@aggTrade")

            # Depth updates (for order book)
            # Using @depth@100ms for faster updates
            streams.append(f"{symbol}@depth@100ms")

            # Mark price (for funding rate)
            # @1s for 1-second updates
            streams.append(f"{symbol}@markPrice@1s")

        return streams

    # =========================================================================
    # Message Processing
    # =========================================================================

    async def _process_messages(self) -> None:
        """Process incoming WebSocket messages."""
        if not self._ws:
            return

        async for raw_message in self._ws:
            if not self._running:
                break

            try:
                message = json.loads(raw_message)
                self._messages_received += 1
                self._last_message_time = datetime.utcnow()

                await self._route_message(message)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse message: {e}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def _route_message(self, message: dict) -> None:
        """Route message to appropriate handler based on event type."""
        # Combined stream format has 'stream' and 'data' keys
        if "stream" in message:
            stream = message["stream"]
            data = message["data"]
        else:
            # Single stream format
            stream = None
            data = message

        event_type = data.get("e")

        if event_type == "aggTrade":
            await self._handle_trade(data)
        elif event_type == "depthUpdate":
            await self._handle_depth(data)
        elif event_type == "markPriceUpdate":
            await self._handle_mark_price(data)
        else:
            logger.debug(f"Unknown event type: {event_type}")

    async def _handle_trade(self, data: dict) -> None:
        """Handle aggTrade message."""
        try:
            trade = Trade.from_binance(data)
            event = TradeEvent(trade=trade)

            for callback in self._trade_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)

        except Exception as e:
            logger.error(f"Error handling trade: {e}")

    async def _handle_depth(self, data: dict) -> None:
        """Handle depthUpdate message."""
        try:
            update = DepthUpdate.from_binance(data)
            event = DepthUpdateEvent(update=update)

            for callback in self._depth_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)

        except Exception as e:
            logger.error(f"Error handling depth update: {e}")

    async def _handle_mark_price(self, data: dict) -> None:
        """Handle markPriceUpdate message."""
        try:
            mark_price = MarkPrice.from_binance(data)
            event = MarkPriceEvent(mark_price=mark_price)

            for callback in self._mark_price_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)

        except Exception as e:
            logger.error(f"Error handling mark price: {e}")

    # =========================================================================
    # Manual Subscription (for dynamic subscription changes)
    # =========================================================================

    async def subscribe(self, streams: List[str]) -> None:
        """
        Subscribe to additional streams.

        Args:
            streams: List of stream names (e.g., ["btcusdt@aggTrade"])
        """
        if not self._ws:
            logger.warning("Cannot subscribe: not connected")
            return

        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(datetime.utcnow().timestamp() * 1000),
        }

        await self._ws.send(json.dumps(subscribe_msg))
        self._subscribed_streams.update(streams)
        logger.info(f"Subscribed to: {streams}")

    async def unsubscribe(self, streams: List[str]) -> None:
        """
        Unsubscribe from streams.

        Args:
            streams: List of stream names to unsubscribe from
        """
        if not self._ws:
            logger.warning("Cannot unsubscribe: not connected")
            return

        unsubscribe_msg = {
            "method": "UNSUBSCRIBE",
            "params": streams,
            "id": int(datetime.utcnow().timestamp() * 1000),
        }

        await self._ws.send(json.dumps(unsubscribe_msg))
        self._subscribed_streams -= set(streams)
        logger.info(f"Unsubscribed from: {streams}")


# =============================================================================
# Factory function for creating WebSocket from config
# =============================================================================

def create_websocket_from_config(config: dict) -> BinanceWebSocket:
    """
    Create BinanceWebSocket from configuration dictionary.

    Args:
        config: Configuration dictionary (from settings.yaml)

    Returns:
        Configured BinanceWebSocket instance
    """
    exchange_config = config.get("exchange", {})
    trading_config = config.get("trading", {})
    ws_config = exchange_config.get("websocket", {})

    return BinanceWebSocket(
        symbols=trading_config.get("symbols", ["BTCUSDT"]),
        testnet=exchange_config.get("testnet", True),
        ping_interval=ws_config.get("ping_interval", 30),
        ping_timeout=ws_config.get("ping_timeout", 10),
        reconnect_delay=ws_config.get("reconnect_delay", 5),
        max_reconnect_attempts=ws_config.get("max_reconnect_attempts", 10),
    )
