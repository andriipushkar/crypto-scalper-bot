"""
Kraken Futures API implementation.

Supports Kraken Futures perpetual contracts.
"""

import base64
import hashlib
import hmac
import time
import urllib.parse
from decimal import Decimal
from typing import Dict, List, Optional, Any

import aiohttp
from loguru import logger

from src.data.models import Side, OrderType, OrderStatus
from src.execution.exchange_base import (
    BaseExchangeAPI,
    Exchange,
    ExchangeCredentials,
    ExchangeSymbolInfo,
    ExchangeOrder,
    ExchangePosition,
    ExchangeBalance,
    ExchangeTicker,
)


class KrakenAPIError(Exception):
    """Kraken API error."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"Kraken API Error [{code}]: {message}")


class KrakenFuturesAPI(BaseExchangeAPI):
    """
    Kraken Futures API.

    Usage:
        credentials = ExchangeCredentials(
            api_key="your_key",
            api_secret="your_secret",
            testnet=True,
        )
        api = KrakenFuturesAPI(credentials)
        await api.connect()

        ticker = await api.get_ticker("PI_XBTUSD")
        order = await api.place_market_order("PI_XBTUSD", Side.BUY, Decimal("0.001"))
    """

    # Kraken Futures URLs
    MAINNET_URL = "https://futures.kraken.com/derivatives/api/v3"
    TESTNET_URL = "https://demo-futures.kraken.com/derivatives/api/v3"
    MAINNET_WS = "wss://futures.kraken.com/ws/v1"
    TESTNET_WS = "wss://demo-futures.kraken.com/ws/v1"

    def __init__(self, credentials: ExchangeCredentials):
        super().__init__(credentials)
        self._nonce = 0

    @property
    def exchange(self) -> Exchange:
        return Exchange.KRAKEN

    @property
    def base_url(self) -> str:
        return self.TESTNET_URL if self.testnet else self.MAINNET_URL

    @property
    def ws_url(self) -> str:
        return self.TESTNET_WS if self.testnet else self.MAINNET_WS

    def _get_nonce(self) -> int:
        """Generate unique nonce."""
        self._nonce = max(self._nonce + 1, int(time.time() * 1000))
        return self._nonce

    def _sign(self, endpoint: str, post_data: str, nonce: str) -> str:
        """
        Generate signature for authenticated requests.

        Kraken uses:
        HMAC-SHA512 of (endpoint + SHA256(nonce + post_data)) using base64 decoded secret
        """
        if not self.credentials.api_secret:
            return ""

        # Create the message to sign
        sha256_hash = hashlib.sha256((nonce + post_data).encode()).digest()
        message = endpoint.encode() + sha256_hash

        # Decode the secret and create signature
        secret_decoded = base64.b64decode(self.credentials.api_secret)
        signature = hmac.new(secret_decoded, message, hashlib.sha512)

        return base64.b64encode(signature.digest()).decode()

    def _get_headers(
        self,
        endpoint: str,
        post_data: str = "",
        nonce: str = "",
    ) -> Dict[str, str]:
        """Get headers for authenticated requests."""
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        if self.credentials.api_key:
            headers["APIKey"] = self.credentials.api_key
            headers["Nonce"] = nonce
            headers["Authent"] = self._sign(endpoint, post_data, nonce)

        return headers

    async def connect(self) -> None:
        """Initialize connection."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        logger.info(f"Kraken API connected ({'testnet' if self.testnet else 'mainnet'})")

    async def close(self) -> None:
        """Close connection."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        signed: bool = False,
    ) -> Dict[str, Any]:
        """Make API request."""
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        headers = {}
        post_data = ""

        if signed:
            nonce = str(self._get_nonce())
            params["nonce"] = nonce
            post_data = urllib.parse.urlencode(params)
            headers = self._get_headers(endpoint, post_data, nonce)

        try:
            if method == "GET":
                async with self._session.get(url, params=params, headers=headers) as resp:
                    data = await resp.json()
            else:
                async with self._session.post(url, data=post_data, headers=headers) as resp:
                    data = await resp.json()

            # Check for errors
            if data.get("error"):
                errors = data.get("error", [])
                error_msg = errors[0] if errors else "Unknown error"
                raise KrakenAPIError("API_ERROR", error_msg)

            if data.get("result") == "error":
                raise KrakenAPIError(
                    data.get("serverTime", "ERROR"),
                    data.get("error", "Unknown error"),
                )

            return data.get("result", data)

        except aiohttp.ClientError as e:
            logger.error(f"Kraken request failed: {e}")
            raise KrakenAPIError("CONNECTION", str(e))

    async def ping(self) -> bool:
        """Check connection."""
        try:
            await self._request("GET", "/time")
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Market Data
    # -------------------------------------------------------------------------

    async def get_exchange_info(self) -> Dict[str, ExchangeSymbolInfo]:
        """Get trading rules."""
        data = await self._request("GET", "/instruments")

        symbols = {}
        instruments = data.get("instruments", [])

        for item in instruments:
            symbol = item.get("symbol", "")
            if not symbol or item.get("type") != "futures_inverse":
                # Focus on perpetual futures
                if item.get("type") not in ["flexible_futures", "perpetual"]:
                    continue

            # Parse contract specifications
            tick_size = Decimal(str(item.get("tickSize", "0.1")))
            contract_size = Decimal(str(item.get("contractSize", "1")))

            symbols[symbol] = ExchangeSymbolInfo(
                symbol=symbol,
                base_asset=item.get("underlying", "XBT"),
                quote_asset="USD",
                price_precision=self._count_decimals(tick_size),
                quantity_precision=self._count_decimals(contract_size),
                min_quantity=Decimal(str(item.get("minTradeSize", "1"))),
                max_quantity=Decimal(str(item.get("maxPositionSize", "1000000"))),
                min_notional=Decimal("1"),
                tick_size=tick_size,
                step_size=contract_size,
            )

        return symbols

    def _count_decimals(self, value: Decimal) -> int:
        """Count decimal places."""
        str_val = str(value)
        if "." in str_val:
            return len(str_val.split(".")[1])
        return 0

    async def get_ticker(self, symbol: str) -> ExchangeTicker:
        """Get ticker."""
        data = await self._request("GET", "/tickers")

        tickers = data.get("tickers", [])
        ticker_data = next((t for t in tickers if t.get("symbol") == symbol), {})

        if not ticker_data:
            raise KrakenAPIError("NOT_FOUND", f"Ticker not found for {symbol}")

        return ExchangeTicker(
            symbol=symbol,
            last_price=Decimal(str(ticker_data.get("last", "0"))),
            bid_price=Decimal(str(ticker_data.get("bid", "0"))),
            ask_price=Decimal(str(ticker_data.get("ask", "0"))),
            volume_24h=Decimal(str(ticker_data.get("vol24h", "0"))),
            price_change_24h=Decimal(str(ticker_data.get("change24h", "0"))),
            timestamp=int(time.time() * 1000),
        )

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get orderbook."""
        data = await self._request("GET", f"/orderbook", {"symbol": symbol})

        orderbook = data.get("orderBook", {})

        return {
            "bids": [
                [Decimal(str(b[0])), Decimal(str(b[1]))]
                for b in orderbook.get("bids", [])[:limit]
            ],
            "asks": [
                [Decimal(str(a[0])), Decimal(str(a[1]))]
                for a in orderbook.get("asks", [])[:limit]
            ],
            "timestamp": int(time.time() * 1000),
        }

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        data = await self._request("GET", "/history", {"symbol": symbol})

        history = data.get("history", [])[:limit]

        return [
            {
                "id": t.get("uid", ""),
                "price": Decimal(str(t.get("price", "0"))),
                "quantity": Decimal(str(t.get("size", "0"))),
                "side": t.get("side", "").upper(),
                "timestamp": int(t.get("time", 0)),
            }
            for t in history
        ]

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get klines/candles."""
        # Kraken uses different interval format
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
        }

        resolution = interval_map.get(interval, "1h")

        data = await self._request(
            "GET",
            "/candles",
            {"symbol": symbol, "resolution": resolution, "limit": limit},
        )

        candles = data.get("candles", [])

        return [
            {
                "timestamp": int(c.get("time", 0)),
                "open": Decimal(str(c.get("open", "0"))),
                "high": Decimal(str(c.get("high", "0"))),
                "low": Decimal(str(c.get("low", "0"))),
                "close": Decimal(str(c.get("close", "0"))),
                "volume": Decimal(str(c.get("volume", "0"))),
            }
            for c in candles
        ]

    # -------------------------------------------------------------------------
    # Account
    # -------------------------------------------------------------------------

    async def get_balance(self) -> List[ExchangeBalance]:
        """Get account balance."""
        data = await self._request("GET", "/accounts", signed=True)

        accounts = data.get("accounts", {})
        balances = []

        # Kraken has different account types
        for acc_type, acc_data in accounts.items():
            if isinstance(acc_data, dict):
                balances.append(ExchangeBalance(
                    asset=acc_data.get("currency", "USD"),
                    wallet_balance=Decimal(str(acc_data.get("balance", "0"))),
                    available_balance=Decimal(str(acc_data.get("available", "0"))),
                    unrealized_pnl=Decimal(str(acc_data.get("unrealizedPnl", "0"))),
                    margin_balance=Decimal(str(acc_data.get("marginEquity", "0"))),
                ))

        return balances

    async def get_positions(self, symbol: Optional[str] = None) -> List[ExchangePosition]:
        """Get positions."""
        data = await self._request("GET", "/openpositions", signed=True)

        positions = []
        open_positions = data.get("openPositions", [])

        for p in open_positions:
            pos_symbol = p.get("symbol", "")
            if symbol and pos_symbol != symbol:
                continue

            size = Decimal(str(p.get("size", "0")))
            if size == 0:
                continue

            side = Side.BUY if p.get("side") == "long" else Side.SELL

            positions.append(ExchangePosition(
                symbol=pos_symbol,
                side=side,
                quantity=abs(size),
                entry_price=Decimal(str(p.get("price", "0"))),
                mark_price=Decimal(str(p.get("markPrice", "0"))),
                unrealized_pnl=Decimal(str(p.get("unrealizedPnl", "0"))),
                leverage=int(p.get("leverage", 1)),
                margin_type="cross",  # Kraken uses cross margin
                liquidation_price=Decimal(str(p.get("liquidationPrice", "0")))
                if p.get("liquidationPrice")
                else None,
            ))

        return positions

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage."""
        try:
            await self._request(
                "POST",
                "/leverage",
                {"symbol": symbol, "leverage": leverage},
                signed=True,
            )
            return True
        except KrakenAPIError:
            return False

    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """Set margin type (Kraken only supports cross margin for futures)."""
        if margin_type.lower() != "cross":
            logger.warning("Kraken Futures only supports cross margin")
            return False
        return True

    # -------------------------------------------------------------------------
    # Orders
    # -------------------------------------------------------------------------

    def _parse_order(self, data: Dict[str, Any]) -> ExchangeOrder:
        """Parse order response."""
        side = Side.BUY if data.get("side") == "buy" else Side.SELL

        status_map = {
            "placed": OrderStatus.NEW,
            "partiallyFilled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELED,
            "rejected": OrderStatus.REJECTED,
        }

        type_map = {
            "mkt": OrderType.MARKET,
            "lmt": OrderType.LIMIT,
            "stp": OrderType.STOP_MARKET,
            "take_profit": OrderType.TAKE_PROFIT_MARKET,
        }

        return ExchangeOrder(
            order_id=data.get("order_id", data.get("orderId", "")),
            client_order_id=data.get("cliOrdId"),
            symbol=data.get("symbol", ""),
            side=side,
            order_type=type_map.get(data.get("orderType", ""), OrderType.MARKET),
            status=status_map.get(data.get("status", ""), OrderStatus.NEW),
            quantity=Decimal(str(data.get("size", data.get("qty", "0")))),
            filled_quantity=Decimal(str(data.get("filled", "0"))),
            price=Decimal(str(data.get("limitPrice", "0")))
            if data.get("limitPrice")
            else None,
            avg_fill_price=Decimal(str(data.get("avgFillPrice", "0")))
            if data.get("avgFillPrice")
            else None,
            commission=Decimal("0"),  # Kraken includes this in P&L
            created_at=int(data.get("receivedTime", 0)),
            updated_at=int(data.get("lastUpdateTime", time.time() * 1000)),
        )

    async def place_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
    ) -> ExchangeOrder:
        """Place order."""
        # Kraken order type mapping
        kraken_type = "mkt" if order_type == OrderType.MARKET else "lmt"

        params = {
            "orderType": kraken_type,
            "symbol": symbol,
            "side": "buy" if side == Side.BUY else "sell",
            "size": str(quantity),
        }

        if price and order_type == OrderType.LIMIT:
            params["limitPrice"] = str(price)

        if client_order_id:
            params["cliOrdId"] = client_order_id

        if reduce_only:
            params["reduceOnly"] = "true"

        if stop_price:
            params["stopPrice"] = str(stop_price)
            params["orderType"] = "stp"

        data = await self._request("POST", "/sendorder", params, signed=True)

        send_status = data.get("sendStatus", {})

        return self._parse_order({
            "order_id": send_status.get("order_id", ""),
            "symbol": symbol,
            "side": "buy" if side == Side.BUY else "sell",
            "orderType": kraken_type,
            "size": str(quantity),
            "status": send_status.get("status", "placed"),
            "receivedTime": send_status.get("receivedTime", int(time.time() * 1000)),
        })

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order."""
        try:
            await self._request(
                "POST",
                "/cancelorder",
                {"order_id": order_id},
                signed=True,
            )
            return True
        except KrakenAPIError:
            return False

    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all orders."""
        try:
            data = await self._request(
                "POST",
                "/cancelallorders",
                {"symbol": symbol},
                signed=True,
            )
            return data.get("cancelStatus", {}).get("cancelledOrders", 0)
        except KrakenAPIError:
            return 0

    async def get_order(self, symbol: str, order_id: str) -> ExchangeOrder:
        """Get order."""
        data = await self._request(
            "GET",
            "/orders",
            {"orderIds": order_id},
            signed=True,
        )

        orders = data.get("orders", [])
        if not orders:
            raise KrakenAPIError("NOT_FOUND", "Order not found")

        return self._parse_order(orders[0])

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """Get open orders."""
        data = await self._request("GET", "/openorders", signed=True)

        orders = []
        for o in data.get("openOrders", []):
            if symbol and o.get("symbol") != symbol:
                continue
            orders.append(self._parse_order(o))

        return orders

    async def get_order_history(self, symbol: str, limit: int = 50) -> List[ExchangeOrder]:
        """Get order history."""
        data = await self._request(
            "GET",
            "/orders",
            {"symbol": symbol},
            signed=True,
        )

        orders = [self._parse_order(o) for o in data.get("orders", [])[:limit]]
        return orders

    # -------------------------------------------------------------------------
    # Funding
    # -------------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> Decimal:
        """Get current funding rate."""
        ticker = await self.get_ticker(symbol)

        # Kraken includes funding in ticker data
        data = await self._request("GET", "/tickers")
        tickers = data.get("tickers", [])
        ticker_data = next((t for t in tickers if t.get("symbol") == symbol), {})

        return Decimal(str(ticker_data.get("fundingRate", "0")))

    async def get_funding_history(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get funding history."""
        data = await self._request(
            "GET",
            "/historicalfundingrates",
            {"symbol": symbol},
        )

        rates = data.get("rates", [])[:limit]

        return [
            {
                "symbol": symbol,
                "funding_rate": Decimal(str(r.get("fundingRate", "0"))),
                "timestamp": int(r.get("timestamp", 0)),
            }
            for r in rates
        ]

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to Kraken format.

        Kraken uses: PI_XBTUSD for Bitcoin perpetual
        """
        symbol = symbol.upper()

        # Common conversions
        conversions = {
            "BTCUSD": "PI_XBTUSD",
            "BTCUSDT": "PI_XBTUSD",
            "ETHUSD": "PI_ETHUSD",
            "ETHUSDT": "PI_ETHUSD",
        }

        return conversions.get(symbol, symbol)
