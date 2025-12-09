"""
Binance Futures API wrapper.

Handles order placement, position management, and account operations.
"""

import asyncio
import hashlib
import hmac
import os
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, List, Any
from urllib.parse import urlencode

import aiohttp
from loguru import logger

from src.data.models import (
    Order,
    OrderType,
    OrderStatus,
    TimeInForce,
    Side,
    Position,
    Signal,
    SignalType,
)


class BinanceAPIError(Exception):
    """Binance API error."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"Binance API Error {code}: {message}")


class BinanceFuturesAPI:
    """
    Async client for Binance Futures API.

    Handles:
    - Order placement (market, limit)
    - Position queries
    - Account balance
    - Symbol info
    """

    MAINNET_API = "https://fapi.binance.com"
    TESTNET_API = "https://testnet.binancefuture.com"

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        testnet: bool = True,
    ):
        """
        Initialize API client.

        Args:
            api_key: API key (or from env)
            api_secret: API secret (or from env)
            testnet: Use testnet endpoint
        """
        self.testnet = testnet

        # Get credentials from env if not provided
        if testnet:
            self.api_key = api_key or os.getenv("BINANCE_TESTNET_API_KEY", "")
            self.api_secret = api_secret or os.getenv("BINANCE_TESTNET_API_SECRET", "")
        else:
            self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
            self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")

        self.base_url = self.TESTNET_API if testnet else self.MAINNET_API

        # Session
        self._session: Optional[aiohttp.ClientSession] = None

        # Symbol info cache
        self._symbol_info: Dict[str, Dict] = {}

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests

        logger.info(f"Binance API initialized (testnet={testnet})")

    @property
    def has_credentials(self) -> bool:
        """Check if API credentials are set."""
        return bool(self.api_key and self.api_secret)

    # =========================================================================
    # Session Management
    # =========================================================================

    async def connect(self) -> None:
        """Create HTTP session."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
            logger.debug("HTTP session created")

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
            logger.debug("HTTP session closed")

    # =========================================================================
    # Request Handling
    # =========================================================================

    def _sign(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature."""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        signed: bool = False,
    ) -> Dict:
        """
        Make API request.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Request parameters
            signed: Whether to sign the request

        Returns:
            Response data
        """
        await self.connect()

        # Rate limiting
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

        url = f"{self.base_url}{endpoint}"
        params = params or {}

        # Add timestamp and signature for signed requests
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._sign(params)

        headers = {"X-MBX-APIKEY": self.api_key}

        try:
            if method == "GET":
                async with self._session.get(url, params=params, headers=headers) as resp:
                    data = await resp.json()
            elif method == "POST":
                async with self._session.post(url, params=params, headers=headers) as resp:
                    data = await resp.json()
            elif method == "DELETE":
                async with self._session.delete(url, params=params, headers=headers) as resp:
                    data = await resp.json()
            else:
                raise ValueError(f"Unsupported method: {method}")

            # Check for errors
            if "code" in data and data["code"] != 200:
                raise BinanceAPIError(data["code"], data.get("msg", "Unknown error"))

            return data

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error: {e}")
            raise

    # =========================================================================
    # Market Data
    # =========================================================================

    async def get_exchange_info(self) -> Dict:
        """Get exchange information."""
        return await self._request("GET", "/fapi/v1/exchangeInfo")

    async def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol-specific information."""
        if symbol not in self._symbol_info:
            info = await self.get_exchange_info()
            for s in info.get("symbols", []):
                self._symbol_info[s["symbol"]] = s

        return self._symbol_info.get(symbol, {})

    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Get current price for symbol."""
        data = await self._request("GET", "/fapi/v1/ticker/price", {"symbol": symbol})
        return Decimal(data["price"])

    # =========================================================================
    # Account
    # =========================================================================

    async def get_account(self) -> Dict:
        """Get account information."""
        return await self._request("GET", "/fapi/v2/account", signed=True)

    async def get_balance(self, asset: str = "USDT") -> Decimal:
        """Get balance for asset."""
        account = await self.get_account()
        for balance in account.get("assets", []):
            if balance["asset"] == asset:
                return Decimal(balance["availableBalance"])
        return Decimal("0")

    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        account = await self.get_account()
        positions = []

        for pos in account.get("positions", []):
            size = Decimal(pos["positionAmt"])
            if size == 0:
                continue

            positions.append(Position(
                symbol=pos["symbol"],
                side=Side.BUY if size > 0 else Side.SELL,
                size=abs(size),
                entry_price=Decimal(pos["entryPrice"]),
                mark_price=Decimal(pos["markPrice"]),
                liquidation_price=Decimal(pos["liquidationPrice"]) if pos["liquidationPrice"] else None,
                unrealized_pnl=Decimal(pos["unrealizedProfit"]),
                realized_pnl=Decimal("0"),
                leverage=int(pos["leverage"]),
                margin_type=pos["marginType"],
                updated_at=datetime.utcnow(),
            ))

        return positions

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    # =========================================================================
    # Orders
    # =========================================================================

    async def place_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal = None,
        stop_price: Decimal = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
    ) -> Order:
        """
        Place an order.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            order_type: Order type
            quantity: Order quantity
            price: Limit price (for LIMIT orders)
            stop_price: Stop price (for STOP orders)
            time_in_force: Time in force
            reduce_only: Close position only

        Returns:
            Order object
        """
        # Get symbol precision
        info = await self.get_symbol_info(symbol)
        quantity_precision = info.get("quantityPrecision", 3)
        price_precision = info.get("pricePrecision", 2)

        # Round quantity and price
        quantity = quantity.quantize(Decimal(10) ** -quantity_precision, rounding=ROUND_DOWN)

        params = {
            "symbol": symbol,
            "side": side.value,
            "type": order_type.value,
            "quantity": str(quantity),
        }

        if order_type == OrderType.LIMIT:
            if price is None:
                raise ValueError("Price required for LIMIT orders")
            price = price.quantize(Decimal(10) ** -price_precision)
            params["price"] = str(price)
            params["timeInForce"] = time_in_force.value

        if stop_price:
            stop_price = stop_price.quantize(Decimal(10) ** -price_precision)
            params["stopPrice"] = str(stop_price)

        if reduce_only:
            params["reduceOnly"] = "true"

        logger.info(f"Placing order: {side.value} {quantity} {symbol} @ {price or 'MARKET'}")

        data = await self._request("POST", "/fapi/v1/order", params, signed=True)

        return self._parse_order(data)

    async def place_market_order(
        self,
        symbol: str,
        side: Side,
        quantity: Decimal,
        reduce_only: bool = False,
    ) -> Order:
        """Place a market order."""
        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            reduce_only=reduce_only,
        )

    async def place_limit_order(
        self,
        symbol: str,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
    ) -> Order:
        """Place a limit order."""
        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
        )

    async def cancel_order(self, symbol: str, order_id: str) -> Order:
        """Cancel an order."""
        params = {
            "symbol": symbol,
            "orderId": order_id,
        }

        data = await self._request("DELETE", "/fapi/v1/order", params, signed=True)
        return self._parse_order(data)

    async def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for symbol."""
        params = {"symbol": symbol}
        await self._request("DELETE", "/fapi/v1/allOpenOrders", params, signed=True)
        logger.info(f"Cancelled all orders for {symbol}")
        return True

    async def get_order(self, symbol: str, order_id: str) -> Order:
        """Get order by ID."""
        params = {
            "symbol": symbol,
            "orderId": order_id,
        }
        data = await self._request("GET", "/fapi/v1/order", params, signed=True)
        return self._parse_order(data)

    async def get_open_orders(self, symbol: str = None) -> List[Order]:
        """Get all open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol

        data = await self._request("GET", "/fapi/v1/openOrders", params, signed=True)
        return [self._parse_order(o) for o in data]

    def _parse_order(self, data: Dict) -> Order:
        """Parse order from API response."""
        return Order(
            order_id=str(data["orderId"]),
            client_order_id=data.get("clientOrderId", ""),
            symbol=data["symbol"],
            side=Side(data["side"]),
            order_type=OrderType(data["type"]),
            quantity=Decimal(data["origQty"]),
            price=Decimal(data["price"]) if data.get("price") else None,
            stop_price=Decimal(data["stopPrice"]) if data.get("stopPrice") else None,
            status=OrderStatus(data["status"]),
            filled_quantity=Decimal(data.get("executedQty", "0")),
            average_price=Decimal(data["avgPrice"]) if data.get("avgPrice") else None,
            commission=Decimal("0"),  # Not in order response
            created_at=datetime.fromtimestamp(data["time"] / 1000) if "time" in data else datetime.utcnow(),
            updated_at=datetime.fromtimestamp(data["updateTime"] / 1000) if "updateTime" in data else datetime.utcnow(),
            time_in_force=TimeInForce(data.get("timeInForce", "GTC")),
        )

    # =========================================================================
    # Leverage & Margin
    # =========================================================================

    async def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """Set leverage for symbol."""
        params = {
            "symbol": symbol,
            "leverage": leverage,
        }
        data = await self._request("POST", "/fapi/v1/leverage", params, signed=True)
        logger.info(f"Set leverage for {symbol}: {leverage}x")
        return data

    async def set_margin_type(self, symbol: str, margin_type: str) -> Dict:
        """
        Set margin type for symbol.

        Args:
            symbol: Trading symbol
            margin_type: CROSSED or ISOLATED
        """
        params = {
            "symbol": symbol,
            "marginType": margin_type,
        }
        try:
            data = await self._request("POST", "/fapi/v1/marginType", params, signed=True)
            logger.info(f"Set margin type for {symbol}: {margin_type}")
            return data
        except BinanceAPIError as e:
            # -4046 means margin type already set
            if e.code == -4046:
                return {"msg": "Margin type already set"}
            raise


class OrderExecutor:
    """
    High-level order executor.

    Converts signals to orders with proper risk management.
    """

    def __init__(
        self,
        api: BinanceFuturesAPI,
        risk_manager=None,
        use_limit_orders: bool = True,
        limit_offset_ticks: int = 1,
    ):
        """
        Initialize executor.

        Args:
            api: Binance API client
            risk_manager: Risk manager instance
            use_limit_orders: Use limit orders instead of market
            limit_offset_ticks: Tick offset for limit orders
        """
        self.api = api
        self.risk_manager = risk_manager
        self.use_limit_orders = use_limit_orders
        self.limit_offset_ticks = limit_offset_ticks

    async def execute_signal(self, signal: Signal) -> Optional[Order]:
        """
        Execute a trading signal.

        Args:
            signal: Trading signal

        Returns:
            Order if successful, None otherwise
        """
        if signal.signal_type == SignalType.NO_ACTION:
            return None

        # Get position size from signal metadata or calculate
        quantity = signal.metadata.get("position_size")

        if quantity is None:
            if self.risk_manager:
                quantity = self.risk_manager.calculate_position_size(signal)
            else:
                logger.warning("No position size and no risk manager")
                return None

        if quantity <= 0:
            logger.warning("Position size is zero")
            return None

        quantity = Decimal(str(quantity))

        # Determine order side
        side = signal.side
        if side is None:
            logger.warning("Signal has no side")
            return None

        # Place order
        try:
            if self.use_limit_orders:
                # Calculate limit price with offset
                tick_size = Decimal("0.1")  # TODO: get from symbol info
                offset = tick_size * self.limit_offset_ticks

                if side == Side.BUY:
                    price = signal.price + offset  # Slightly above mid for buys
                else:
                    price = signal.price - offset  # Slightly below mid for sells

                order = await self.api.place_limit_order(
                    symbol=signal.symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    time_in_force=TimeInForce.IOC,  # Immediate or cancel
                )
            else:
                order = await self.api.place_market_order(
                    symbol=signal.symbol,
                    side=side,
                    quantity=quantity,
                )

            logger.info(
                f"Order executed: {order.order_id} {side.value} "
                f"{quantity} {signal.symbol} - Status: {order.status.value}"
            )

            return order

        except BinanceAPIError as e:
            logger.error(f"Order failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return None

    async def close_position(self, position: Position) -> Optional[Order]:
        """
        Close a position with market order.

        Args:
            position: Position to close

        Returns:
            Order if successful, None otherwise
        """
        # Close side is opposite of position side
        close_side = position.side.opposite

        try:
            order = await self.api.place_market_order(
                symbol=position.symbol,
                side=close_side,
                quantity=position.size,
                reduce_only=True,
            )

            logger.info(f"Position closed: {position.symbol} {order.order_id}")
            return order

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return None
