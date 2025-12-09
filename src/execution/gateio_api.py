"""
Gate.io Futures API implementation.

Supports Gate.io USDT-margined perpetual contracts.
"""

import hashlib
import hmac
import time
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


class GateIOAPIError(Exception):
    """Gate.io API error."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"Gate.io API Error [{code}]: {message}")


class GateIOFuturesAPI(BaseExchangeAPI):
    """
    Gate.io Futures API.

    Usage:
        credentials = ExchangeCredentials(
            api_key="your_key",
            api_secret="your_secret",
            testnet=True,
        )
        api = GateIOFuturesAPI(credentials)
        await api.connect()

        ticker = await api.get_ticker("BTC_USDT")
        order = await api.place_market_order("BTC_USDT", Side.BUY, Decimal("1"))
    """

    # Gate.io Futures URLs
    MAINNET_URL = "https://api.gateio.ws/api/v4"
    TESTNET_URL = "https://fx-api-testnet.gateio.ws/api/v4"
    MAINNET_WS = "wss://fx-ws.gateio.ws/v4/ws/usdt"
    TESTNET_WS = "wss://fx-ws-testnet.gateio.ws/v4/ws/usdt"

    def __init__(self, credentials: ExchangeCredentials):
        super().__init__(credentials)
        self._settle = "usdt"  # Settlement currency

    @property
    def exchange(self) -> Exchange:
        return Exchange.GATEIO

    @property
    def base_url(self) -> str:
        return self.TESTNET_URL if self.testnet else self.MAINNET_URL

    @property
    def ws_url(self) -> str:
        return self.TESTNET_WS if self.testnet else self.MAINNET_WS

    def _sign(
        self,
        method: str,
        url: str,
        query_string: str = "",
        body: str = "",
    ) -> Dict[str, str]:
        """Generate signature for authenticated requests."""
        timestamp = str(int(time.time()))

        # Create hash of body
        body_hash = hashlib.sha512(body.encode()).hexdigest()

        # Create signature string
        sign_string = f"{method}\n{url}\n{query_string}\n{body_hash}\n{timestamp}"

        # Generate HMAC signature
        signature = hmac.new(
            self.credentials.api_secret.encode(),
            sign_string.encode(),
            hashlib.sha512,
        ).hexdigest()

        return {
            "KEY": self.credentials.api_key,
            "Timestamp": timestamp,
            "SIGN": signature,
        }

    def _get_headers(
        self,
        method: str,
        endpoint: str,
        query_string: str = "",
        body: str = "",
    ) -> Dict[str, str]:
        """Get headers for authenticated requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.credentials.api_key:
            auth_headers = self._sign(method, endpoint, query_string, body)
            headers.update(auth_headers)

        return headers

    async def connect(self) -> None:
        """Initialize connection."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        logger.info(f"Gate.io API connected ({'testnet' if self.testnet else 'mainnet'})")

    async def close(self) -> None:
        """Close connection."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        body: Dict[str, Any] = None,
        signed: bool = False,
    ) -> Any:
        """Make API request."""
        import json

        url = f"{self.base_url}{endpoint}"
        params = params or {}
        body_str = json.dumps(body) if body else ""

        # Build query string
        query_string = "&".join(f"{k}={v}" for k, v in params.items()) if params else ""

        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        if signed:
            headers = self._get_headers(method, endpoint, query_string, body_str)

        full_url = f"{url}?{query_string}" if query_string else url

        try:
            if method == "GET":
                async with self._session.get(full_url, headers=headers) as resp:
                    data = await resp.json()
            elif method == "POST":
                async with self._session.post(full_url, data=body_str, headers=headers) as resp:
                    data = await resp.json()
            elif method == "DELETE":
                async with self._session.delete(full_url, headers=headers) as resp:
                    data = await resp.json()
            else:
                raise ValueError(f"Unsupported method: {method}")

            # Check for errors
            if isinstance(data, dict) and "label" in data:
                raise GateIOAPIError(
                    data.get("label", "UNKNOWN"),
                    data.get("message", "Unknown error"),
                )

            return data

        except aiohttp.ClientError as e:
            logger.error(f"Gate.io request failed: {e}")
            raise GateIOAPIError("CONNECTION", str(e))

    async def ping(self) -> bool:
        """Check connection."""
        try:
            await self._request("GET", f"/futures/{self._settle}/contracts")
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Market Data
    # -------------------------------------------------------------------------

    async def get_exchange_info(self) -> Dict[str, ExchangeSymbolInfo]:
        """Get trading rules."""
        data = await self._request("GET", f"/futures/{self._settle}/contracts")

        symbols = {}
        for item in (data if isinstance(data, list) else []):
            symbol = item.get("name", "")

            symbols[symbol] = ExchangeSymbolInfo(
                symbol=symbol,
                base_asset=item.get("underlying", ""),
                quote_asset=item.get("settle", "USDT"),
                price_precision=int(-Decimal(str(item.get("order_price_round", "0.1"))).as_tuple().exponent),
                quantity_precision=0,  # Gate.io uses contract count
                min_quantity=Decimal(str(item.get("order_size_min", "1"))),
                max_quantity=Decimal(str(item.get("order_size_max", "1000000"))),
                min_notional=Decimal("1"),
                tick_size=Decimal(str(item.get("order_price_round", "0.1"))),
                step_size=Decimal("1"),  # Contracts
            )

        return symbols

    async def get_ticker(self, symbol: str) -> ExchangeTicker:
        """Get ticker."""
        data = await self._request(
            "GET",
            f"/futures/{self._settle}/tickers",
            {"contract": symbol},
        )

        ticker_data = data[0] if isinstance(data, list) and data else {}

        return ExchangeTicker(
            symbol=symbol,
            last_price=Decimal(str(ticker_data.get("last", "0"))),
            bid_price=Decimal(str(ticker_data.get("highest_bid", "0"))),
            ask_price=Decimal(str(ticker_data.get("lowest_ask", "0"))),
            volume_24h=Decimal(str(ticker_data.get("volume_24h", "0"))),
            price_change_24h=Decimal(str(ticker_data.get("change_percentage", "0"))),
            timestamp=int(time.time() * 1000),
        )

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get orderbook."""
        data = await self._request(
            "GET",
            f"/futures/{self._settle}/order_book",
            {"contract": symbol, "limit": limit},
        )

        return {
            "bids": [
                [Decimal(str(b.get("p", "0"))), Decimal(str(b.get("s", "0")))]
                for b in data.get("bids", [])
            ],
            "asks": [
                [Decimal(str(a.get("p", "0"))), Decimal(str(a.get("s", "0")))]
                for a in data.get("asks", [])
            ],
            "timestamp": int(time.time() * 1000),
        }

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        data = await self._request(
            "GET",
            f"/futures/{self._settle}/trades",
            {"contract": symbol, "limit": limit},
        )

        return [
            {
                "id": str(t.get("id", "")),
                "price": Decimal(str(t.get("price", "0"))),
                "quantity": Decimal(str(abs(t.get("size", 0)))),
                "side": "BUY" if t.get("size", 0) > 0 else "SELL",
                "timestamp": int(t.get("create_time", 0)) * 1000,
            }
            for t in (data if isinstance(data, list) else [])
        ]

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get klines/candles."""
        # Gate.io interval mapping
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "8h": "8h", "1d": "1d", "1w": "7d",
        }

        data = await self._request(
            "GET",
            f"/futures/{self._settle}/candlesticks",
            {
                "contract": symbol,
                "interval": interval_map.get(interval, "1h"),
                "limit": limit,
            },
        )

        return [
            {
                "timestamp": int(c.get("t", 0)) * 1000,
                "open": Decimal(str(c.get("o", "0"))),
                "high": Decimal(str(c.get("h", "0"))),
                "low": Decimal(str(c.get("l", "0"))),
                "close": Decimal(str(c.get("c", "0"))),
                "volume": Decimal(str(c.get("v", "0"))),
            }
            for c in (data if isinstance(data, list) else [])
        ]

    # -------------------------------------------------------------------------
    # Account
    # -------------------------------------------------------------------------

    async def get_balance(self) -> List[ExchangeBalance]:
        """Get account balance."""
        data = await self._request(
            "GET",
            f"/futures/{self._settle}/accounts",
            signed=True,
        )

        return [ExchangeBalance(
            asset=data.get("currency", "USDT"),
            wallet_balance=Decimal(str(data.get("total", "0"))),
            available_balance=Decimal(str(data.get("available", "0"))),
            unrealized_pnl=Decimal(str(data.get("unrealised_pnl", "0"))),
            margin_balance=Decimal(str(data.get("margin", "0"))),
        )]

    async def get_positions(self, symbol: Optional[str] = None) -> List[ExchangePosition]:
        """Get positions."""
        params = {}
        if symbol:
            params["contract"] = symbol

        data = await self._request(
            "GET",
            f"/futures/{self._settle}/positions",
            params,
            signed=True,
        )

        positions = []
        for p in (data if isinstance(data, list) else []):
            size = int(p.get("size", 0))
            if size == 0:
                continue

            side = Side.BUY if size > 0 else Side.SELL

            positions.append(ExchangePosition(
                symbol=p.get("contract", ""),
                side=side,
                quantity=Decimal(str(abs(size))),
                entry_price=Decimal(str(p.get("entry_price", "0"))),
                mark_price=Decimal(str(p.get("mark_price", "0"))),
                unrealized_pnl=Decimal(str(p.get("unrealised_pnl", "0"))),
                leverage=int(p.get("leverage", 1)),
                margin_type="cross" if p.get("mode") == "single" else "isolated",
                liquidation_price=Decimal(str(p.get("liq_price", "0")))
                if p.get("liq_price")
                else None,
            ))

        return positions

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage."""
        try:
            await self._request(
                "POST",
                f"/futures/{self._settle}/positions/{symbol}/leverage",
                body={"leverage": str(leverage)},
                signed=True,
            )
            return True
        except GateIOAPIError:
            return False

    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """Set margin type."""
        try:
            mode = "single" if margin_type.lower() == "cross" else "dual"
            await self._request(
                "POST",
                f"/futures/{self._settle}/positions/{symbol}/margin",
                body={"mode": mode},
                signed=True,
            )
            return True
        except GateIOAPIError:
            return False

    # -------------------------------------------------------------------------
    # Orders
    # -------------------------------------------------------------------------

    def _parse_order(self, data: Dict[str, Any]) -> ExchangeOrder:
        """Parse order response."""
        size = int(data.get("size", 0))
        side = Side.BUY if size > 0 else Side.SELL

        status_map = {
            "open": OrderStatus.NEW,
            "finished": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELED,
        }

        return ExchangeOrder(
            order_id=str(data.get("id", "")),
            client_order_id=data.get("text"),
            symbol=data.get("contract", ""),
            side=side,
            order_type=OrderType.MARKET if data.get("tif") == "ioc" else OrderType.LIMIT,
            status=status_map.get(data.get("status", ""), OrderStatus.NEW),
            quantity=Decimal(str(abs(size))),
            filled_quantity=Decimal(str(abs(int(data.get("left", 0)) - size))),
            price=Decimal(str(data.get("price", "0"))) if data.get("price") else None,
            avg_fill_price=Decimal(str(data.get("fill_price", "0")))
            if data.get("fill_price")
            else None,
            commission=Decimal(str(data.get("fee", "0"))),
            created_at=int(data.get("create_time", 0)) * 1000,
            updated_at=int(data.get("finish_time", time.time())) * 1000,
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
        # Gate.io uses signed size (positive = long, negative = short)
        size = int(quantity) if side == Side.BUY else -int(quantity)

        body = {
            "contract": symbol,
            "size": size,
            "price": str(price) if price else "0",
            "tif": "gtc" if order_type == OrderType.LIMIT else "ioc",
        }

        if reduce_only:
            body["reduce_only"] = True

        if client_order_id:
            body["text"] = client_order_id

        if order_type == OrderType.MARKET:
            body["price"] = "0"
            body["tif"] = "ioc"

        data = await self._request(
            "POST",
            f"/futures/{self._settle}/orders",
            body=body,
            signed=True,
        )

        return self._parse_order(data)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order."""
        try:
            await self._request(
                "DELETE",
                f"/futures/{self._settle}/orders/{order_id}",
                signed=True,
            )
            return True
        except GateIOAPIError:
            return False

    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all orders."""
        try:
            data = await self._request(
                "DELETE",
                f"/futures/{self._settle}/orders",
                {"contract": symbol},
                signed=True,
            )
            return len(data) if isinstance(data, list) else 0
        except GateIOAPIError:
            return 0

    async def get_order(self, symbol: str, order_id: str) -> ExchangeOrder:
        """Get order."""
        data = await self._request(
            "GET",
            f"/futures/{self._settle}/orders/{order_id}",
            signed=True,
        )

        return self._parse_order(data)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """Get open orders."""
        params = {"status": "open"}
        if symbol:
            params["contract"] = symbol

        data = await self._request(
            "GET",
            f"/futures/{self._settle}/orders",
            params,
            signed=True,
        )

        return [self._parse_order(o) for o in (data if isinstance(data, list) else [])]

    async def get_order_history(self, symbol: str, limit: int = 50) -> List[ExchangeOrder]:
        """Get order history."""
        data = await self._request(
            "GET",
            f"/futures/{self._settle}/orders",
            {"contract": symbol, "status": "finished", "limit": limit},
            signed=True,
        )

        return [self._parse_order(o) for o in (data if isinstance(data, list) else [])]

    # -------------------------------------------------------------------------
    # Funding
    # -------------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> Decimal:
        """Get current funding rate."""
        data = await self._request(
            "GET",
            f"/futures/{self._settle}/contracts/{symbol}",
        )

        return Decimal(str(data.get("funding_rate", "0")))

    async def get_funding_history(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get funding history."""
        data = await self._request(
            "GET",
            f"/futures/{self._settle}/funding_rate",
            {"contract": symbol, "limit": limit},
        )

        return [
            {
                "symbol": symbol,
                "funding_rate": Decimal(str(r.get("r", "0"))),
                "timestamp": int(r.get("t", 0)) * 1000,
            }
            for r in (data if isinstance(data, list) else [])
        ]

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to Gate.io format.

        Gate.io uses: BTC_USDT for Bitcoin/USDT perpetual
        """
        symbol = symbol.upper()

        # Common conversions
        conversions = {
            "BTCUSDT": "BTC_USDT",
            "ETHUSDT": "ETH_USDT",
            "BTCUSD": "BTC_USD",
            "ETHUSD": "ETH_USD",
        }

        if symbol in conversions:
            return conversions[symbol]

        # If no underscore, add it before USD/USDT
        if "_" not in symbol:
            if symbol.endswith("USDT"):
                return symbol[:-4] + "_USDT"
            elif symbol.endswith("USD"):
                return symbol[:-3] + "_USD"

        return symbol
