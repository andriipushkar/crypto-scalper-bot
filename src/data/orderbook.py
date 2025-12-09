"""
Order Book management.

Maintains local order book state synchronized with exchange
via WebSocket depth updates.
"""

import asyncio
from collections import OrderedDict
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Callable, Any

import aiohttp
from loguru import logger

from src.data.models import (
    OrderBookLevel,
    OrderBookSnapshot,
    DepthUpdate,
    DepthUpdateEvent,
    OrderBookEvent,
)


class OrderBook:
    """
    Local order book for a single symbol.

    Maintains bid and ask levels, synchronized via WebSocket updates.
    Uses sorted dictionaries for efficient price-level management.
    """

    # Binance REST API endpoints for order book snapshots
    MAINNET_API = "https://fapi.binance.com"
    TESTNET_API = "https://testnet.binancefuture.com"

    def __init__(
        self,
        symbol: str,
        depth: int = 20,
        testnet: bool = True,
    ):
        """
        Initialize order book.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            depth: Number of levels to maintain
            testnet: Use testnet API
        """
        self.symbol = symbol.upper()
        self.depth = depth
        self.testnet = testnet

        # Price -> Quantity mappings
        # Bids: sorted descending (highest first)
        # Asks: sorted ascending (lowest first)
        self._bids: Dict[Decimal, Decimal] = {}
        self._asks: Dict[Decimal, Decimal] = {}

        # Synchronization state
        self._last_update_id: int = 0
        self._synced: bool = False
        self._pending_updates: List[DepthUpdate] = []

        # Timestamps
        self._last_update_time: Optional[datetime] = None
        self._sync_time: Optional[datetime] = None

        # Callbacks for order book updates
        self._callbacks: List[Callable[[OrderBookEvent], Any]] = []

    @property
    def api_url(self) -> str:
        """Get API URL based on environment."""
        return self.TESTNET_API if self.testnet else self.MAINNET_API

    @property
    def is_synced(self) -> bool:
        """Check if order book is synchronized with exchange."""
        return self._synced

    @property
    def last_update_id(self) -> int:
        """Get last processed update ID."""
        return self._last_update_id

    # =========================================================================
    # Public Interface
    # =========================================================================

    def on_update(self, callback: Callable[[OrderBookEvent], Any]) -> None:
        """Register callback for order book updates."""
        self._callbacks.append(callback)

    def get_snapshot(self) -> OrderBookSnapshot:
        """
        Get current order book snapshot.

        Returns:
            OrderBookSnapshot with current state
        """
        # Sort bids descending (best first)
        sorted_bids = sorted(self._bids.items(), key=lambda x: x[0], reverse=True)
        bids = [OrderBookLevel(price=p, quantity=q) for p, q in sorted_bids[:self.depth]]

        # Sort asks ascending (best first)
        sorted_asks = sorted(self._asks.items(), key=lambda x: x[0])
        asks = [OrderBookLevel(price=p, quantity=q) for p, q in sorted_asks[:self.depth]]

        return OrderBookSnapshot(
            symbol=self.symbol,
            timestamp=self._last_update_time or datetime.utcnow(),
            bids=bids,
            asks=asks,
            last_update_id=self._last_update_id,
        )

    def get_best_bid(self) -> Optional[Tuple[Decimal, Decimal]]:
        """Get best bid (price, quantity)."""
        if not self._bids:
            return None
        price = max(self._bids.keys())
        return (price, self._bids[price])

    def get_best_ask(self) -> Optional[Tuple[Decimal, Decimal]]:
        """Get best ask (price, quantity)."""
        if not self._asks:
            return None
        price = min(self._asks.keys())
        return (price, self._asks[price])

    def get_mid_price(self) -> Optional[Decimal]:
        """Get mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            return (best_bid[0] + best_ask[0]) / 2
        return None

    def get_spread(self) -> Optional[Decimal]:
        """Get spread in price units."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            return best_ask[0] - best_bid[0]
        return None

    def get_spread_bps(self) -> Optional[Decimal]:
        """Get spread in basis points."""
        spread = self.get_spread()
        mid = self.get_mid_price()

        if spread and mid:
            return (spread / mid) * 10000
        return None

    def get_imbalance(self, levels: int = 5) -> Optional[Decimal]:
        """
        Calculate order book imbalance.

        Args:
            levels: Number of levels to consider

        Returns:
            Ratio of bid volume to ask volume (>1 = more bids)
        """
        snapshot = self.get_snapshot()
        bid_volume = snapshot.bid_volume(levels)
        ask_volume = snapshot.ask_volume(levels)

        if ask_volume == 0:
            return Decimal("999.0")

        return bid_volume / ask_volume

    # =========================================================================
    # Synchronization
    # =========================================================================

    async def sync(self) -> None:
        """
        Synchronize order book with exchange.

        Fetches snapshot from REST API and applies pending updates.
        """
        logger.info(f"Syncing order book for {self.symbol}")

        try:
            snapshot = await self._fetch_snapshot()
            self._apply_snapshot(snapshot)

            # Apply any pending updates that came while we were fetching
            self._apply_pending_updates()

            self._synced = True
            self._sync_time = datetime.utcnow()
            logger.info(f"Order book synced. Last update ID: {self._last_update_id}")

        except Exception as e:
            logger.error(f"Failed to sync order book: {e}")
            raise

    async def _fetch_snapshot(self) -> dict:
        """Fetch order book snapshot from REST API."""
        url = f"{self.api_url}/fapi/v1/depth"
        params = {
            "symbol": self.symbol,
            "limit": min(self.depth * 2, 1000),  # Get extra for buffer
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"API error {response.status}: {text}")

                return await response.json()

    def _apply_snapshot(self, snapshot: dict) -> None:
        """Apply REST API snapshot to order book."""
        self._bids.clear()
        self._asks.clear()

        for bid in snapshot["bids"]:
            price = Decimal(bid[0])
            quantity = Decimal(bid[1])
            if quantity > 0:
                self._bids[price] = quantity

        for ask in snapshot["asks"]:
            price = Decimal(ask[0])
            quantity = Decimal(ask[1])
            if quantity > 0:
                self._asks[price] = quantity

        self._last_update_id = snapshot["lastUpdateId"]
        self._last_update_time = datetime.utcnow()

    def _apply_pending_updates(self) -> None:
        """Apply pending updates received before snapshot."""
        if not self._pending_updates:
            return

        # Sort by update ID
        self._pending_updates.sort(key=lambda x: x.first_update_id)

        applied = 0
        for update in self._pending_updates:
            # Drop updates that are too old
            if update.final_update_id <= self._last_update_id:
                continue

            # First valid update should overlap with snapshot
            if applied == 0:
                if update.first_update_id > self._last_update_id + 1:
                    logger.warning(
                        f"Gap in updates: snapshot={self._last_update_id}, "
                        f"first update={update.first_update_id}"
                    )

            self._apply_update(update)
            applied += 1

        logger.debug(f"Applied {applied} pending updates")
        self._pending_updates.clear()

    # =========================================================================
    # Update Processing
    # =========================================================================

    async def handle_depth_update(self, event: DepthUpdateEvent) -> None:
        """
        Handle depth update from WebSocket.

        Args:
            event: Depth update event
        """
        update = event.update

        # Filter for our symbol
        if update.symbol != self.symbol:
            return

        # If not synced, queue the update
        if not self._synced:
            self._pending_updates.append(update)
            return

        # Validate update sequence
        if update.first_update_id > self._last_update_id + 1:
            logger.warning(
                f"Gap in updates: expected {self._last_update_id + 1}, "
                f"got {update.first_update_id}. Re-syncing..."
            )
            self._synced = False
            await self.sync()
            return

        # Skip old updates
        if update.final_update_id <= self._last_update_id:
            return

        # Apply the update
        self._apply_update(update)

        # Notify callbacks
        await self._notify_callbacks()

    def _apply_update(self, update: DepthUpdate) -> None:
        """Apply a single depth update."""
        # Update bids
        for level in update.bids:
            if level.quantity == 0:
                self._bids.pop(level.price, None)
            else:
                self._bids[level.price] = level.quantity

        # Update asks
        for level in update.asks:
            if level.quantity == 0:
                self._asks.pop(level.price, None)
            else:
                self._asks[level.price] = level.quantity

        # Trim to depth limit
        self._trim_to_depth()

        self._last_update_id = update.final_update_id
        self._last_update_time = update.timestamp

    def _trim_to_depth(self) -> None:
        """Trim order book to configured depth."""
        # Keep top `depth` bids (highest prices)
        if len(self._bids) > self.depth * 2:
            sorted_bids = sorted(self._bids.keys(), reverse=True)
            for price in sorted_bids[self.depth * 2:]:
                del self._bids[price]

        # Keep top `depth` asks (lowest prices)
        if len(self._asks) > self.depth * 2:
            sorted_asks = sorted(self._asks.keys())
            for price in sorted_asks[self.depth * 2:]:
                del self._asks[price]

    async def _notify_callbacks(self) -> None:
        """Notify registered callbacks of update."""
        snapshot = self.get_snapshot()
        event = OrderBookEvent(snapshot=snapshot)

        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in order book callback: {e}")


class OrderBookManager:
    """
    Manages multiple order books for different symbols.
    """

    def __init__(self, depth: int = 20, testnet: bool = True):
        """
        Initialize order book manager.

        Args:
            depth: Depth for each order book
            testnet: Use testnet API
        """
        self.depth = depth
        self.testnet = testnet
        self._books: Dict[str, OrderBook] = {}

    def get_book(self, symbol: str) -> OrderBook:
        """
        Get or create order book for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            OrderBook instance
        """
        symbol = symbol.upper()

        if symbol not in self._books:
            self._books[symbol] = OrderBook(
                symbol=symbol,
                depth=self.depth,
                testnet=self.testnet,
            )
            logger.debug(f"Created order book for {symbol}")

        return self._books[symbol]

    async def sync_all(self) -> None:
        """Synchronize all order books."""
        tasks = [book.sync() for book in self._books.values()]
        await asyncio.gather(*tasks)

    async def handle_depth_update(self, event: DepthUpdateEvent) -> None:
        """
        Route depth update to appropriate order book.

        Args:
            event: Depth update event
        """
        symbol = event.update.symbol
        book = self.get_book(symbol)
        await book.handle_depth_update(event)

    def get_all_snapshots(self) -> Dict[str, OrderBookSnapshot]:
        """Get snapshots for all order books."""
        return {
            symbol: book.get_snapshot()
            for symbol, book in self._books.items()
        }
