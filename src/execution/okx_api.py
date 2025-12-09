"""
OKX Futures API implementation.

Supports OKX USDT Perpetual Swaps.
"""

import base64
import hashlib
import hmac
import time
from datetime import datetime, timezone
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


class OKXAPIError(Exception):
    """OKX API error."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"OKX API Error {code}: {message}")


class OKXFuturesAPI(BaseExchangeAPI):
    """
    OKX USDT Perpetual Swaps API.

    Usage:
        credentials = ExchangeCredentials(
            api_key="your_key",
            api_secret="your_secret",
            passphrase="your_passphrase",  # Required for OKX
            testnet=True,
        )
        api = OKXFuturesAPI(credentials)
        await api.connect()

        ticker = await api.get_ticker("BTC-USDT-SWAP")
        order = await api.place_market_order("BTC-USDT-SWAP", Side.BUY, Decimal("0.001"))
    """

    MAINNET_URL = "https://www.okx.com"
    TESTNET_URL = "https://www.okx.com"  # OKX uses header for testnet
    MAINNET_WS = "wss://ws.okx.com:8443/ws/v5/public"
    TESTNET_WS = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"

    def __init__(self, credentials: ExchangeCredentials):
        super().__init__(credentials)
        if not credentials.passphrase:
            raise ValueError("OKX requires passphrase in credentials")

    @property
    def exchange(self) -> Exchange:
        return Exchange.OKX

    @property
    def base_url(self) -> str:
        return self.MAINNET_URL  # OKX uses header for testnet

    @property
    def ws_url(self) -> str:
        return self.TESTNET_WS if self.testnet else self.MAINNET_WS

    def _get_timestamp(self) -> str:
        """Get ISO 8601 timestamp."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate signature."""
        message = timestamp + method + path + body
        signature = hmac.new(
            self.credentials.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(signature).decode()

    def _get_headers(
        self,
        method: str,
        path: str,
        body: str = "",
    ) -> Dict[str, str]:
        """Get headers for authenticated requests."""
        timestamp = self._get_timestamp()
        signature = self._sign(timestamp, method, path, body)

        headers = {
            "OK-ACCESS-KEY": self.credentials.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.credentials.passphrase,
            "Content-Type": "application/json",
        }

        if self.testnet:
            headers["x-simulated-trading"] = "1"

        return headers

    async def connect(self) -> None:
        """Initialize connection."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        logger.info(f"OKX API connected ({'testnet' if self.testnet else 'mainnet'})")

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
        path = f"/api/v5{endpoint}"
        url = f"{self.base_url}{path}"

        body = ""
        if method == "GET" and params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            path = f"{path}?{query}"
            url = f"{url}?{query}"
        elif method == "POST" and params:
            import json
            body = json.dumps(params)

        headers = self._get_headers(method, path, body) if signed else {}
        if self.testnet:
            headers["x-simulated-trading"] = "1"

        async with self._session.request(
            method,
            url,
            data=body if method == "POST" else None,
            headers=headers,
        ) as resp:
            data = await resp.json()

            if data.get("code") != "0":
                raise OKXAPIError(
                    data.get("code", "-1"),
                    data.get("msg", "Unknown error"),
                )

            return data.get("data", [])

    async def ping(self) -> bool:
        """Check connection."""
        try:
            await self._request("GET", "/public/time")
            return True
        except Exception:
            return False

    def normalize_symbol(self, symbol: str) -> str:
        """Convert symbol to OKX format (BTC-USDT-SWAP)."""
        symbol = symbol.upper().replace("/", "")
        if not symbol.endswith("-SWAP"):
            # Convert BTCUSDT to BTC-USDT-SWAP
            if symbol.endswith("USDT"):
                base = symbol[:-4]
                return f"{base}-USDT-SWAP"
        return symbol

    # -------------------------------------------------------------------------
    # Market Data
    # -------------------------------------------------------------------------

    async def get_exchange_info(self) -> Dict[str, ExchangeSymbolInfo]:
        """Get trading rules."""
        data = await self._request(
            "GET",
            "/public/instruments",
            {"instType": "SWAP"},
        )

        symbols = {}
        for item in data:
            symbol = item["instId"]
            symbols[symbol] = ExchangeSymbolInfo(
                symbol=symbol,
                base_asset=item.get("ctValCcy", ""),
                quote_asset=item.get("settleCcy", "USDT"),
                price_precision=len(item.get("tickSz", "0.01").split(".")[-1]),
                quantity_precision=len(item.get("lotSz", "0.001").split(".")[-1]),
                min_quantity=Decimal(item.get("minSz", "0.001")),
                max_quantity=Decimal(item.get("maxLmtSz", "10000")),
                min_notional=Decimal("5"),  # OKX default
                tick_size=Decimal(item.get("tickSz", "0.01")),
                step_size=Decimal(item.get("lotSz", "0.001")),
            )

        return symbols

    async def get_ticker(self, symbol: str) -> ExchangeTicker:
        """Get ticker."""
        symbol = self.normalize_symbol(symbol)
        data = await self._request(
            "GET",
            "/market/ticker",
            {"instId": symbol},
        )

        if not data:
            raise OKXAPIError("-1", f"Ticker not found for {symbol}")

        ticker_data = data[0]
        return ExchangeTicker(
            symbol=symbol,
            last_price=Decimal(ticker_data.get("last", "0")),
            bid_price=Decimal(ticker_data.get("bidPx", "0")),
            ask_price=Decimal(ticker_data.get("askPx", "0")),
            volume_24h=Decimal(ticker_data.get("vol24h", "0")),
            price_change_24h=Decimal(ticker_data.get("sodUtc8", "0")),
            timestamp=int(ticker_data.get("ts", int(time.time() * 1000))),
        )

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get orderbook."""
        symbol = self.normalize_symbol(symbol)
        data = await self._request(
            "GET",
            "/market/books",
            {"instId": symbol, "sz": str(limit)},
        )

        if not data:
            return {"bids": [], "asks": [], "timestamp": int(time.time() * 1000)}

        book = data[0]
        return {
            "bids": [[Decimal(p[0]), Decimal(p[1])] for p in book.get("bids", [])],
            "asks": [[Decimal(p[0]), Decimal(p[1])] for p in book.get("asks", [])],
            "timestamp": int(book.get("ts", int(time.time() * 1000))),
        }

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        symbol = self.normalize_symbol(symbol)
        data = await self._request(
            "GET",
            "/market/trades",
            {"instId": symbol, "limit": str(limit)},
        )

        return [
            {
                "id": t.get("tradeId"),
                "price": Decimal(t.get("px", "0")),
                "quantity": Decimal(t.get("sz", "0")),
                "side": t.get("side", "").upper(),
                "timestamp": int(t.get("ts", 0)),
            }
            for t in data
        ]

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get klines."""
        symbol = self.normalize_symbol(symbol)

        # Convert interval format
        interval_map = {
            "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1H", "2h": "2H", "4h": "4H", "1d": "1D", "1w": "1W",
        }

        data = await self._request(
            "GET",
            "/market/candles",
            {
                "instId": symbol,
                "bar": interval_map.get(interval, interval),
                "limit": str(limit),
            },
        )

        return [
            {
                "timestamp": int(k[0]),
                "open": Decimal(k[1]),
                "high": Decimal(k[2]),
                "low": Decimal(k[3]),
                "close": Decimal(k[4]),
                "volume": Decimal(k[5]),
            }
            for k in data
        ]

    # -------------------------------------------------------------------------
    # Account
    # -------------------------------------------------------------------------

    async def get_balance(self) -> List[ExchangeBalance]:
        """Get account balance."""
        data = await self._request("GET", "/account/balance", signed=True)

        balances = []
        for account in data:
            for detail in account.get("details", []):
                balances.append(ExchangeBalance(
                    asset=detail.get("ccy", ""),
                    wallet_balance=Decimal(detail.get("cashBal", "0")),
                    available_balance=Decimal(detail.get("availBal", "0")),
                    unrealized_pnl=Decimal(detail.get("upl", "0")),
                    margin_balance=Decimal(detail.get("eq", "0")),
                ))

        return balances

    async def get_positions(self, symbol: Optional[str] = None) -> List[ExchangePosition]:
        """Get positions."""
        params = {"instType": "SWAP"}
        if symbol:
            params["instId"] = self.normalize_symbol(symbol)

        data = await self._request("GET", "/account/positions", params, signed=True)

        positions = []
        for p in data:
            size = Decimal(p.get("pos", "0"))
            if size == 0:
                continue

            side = Side.BUY if size > 0 else Side.SELL
            positions.append(ExchangePosition(
                symbol=p.get("instId", ""),
                side=side,
                quantity=abs(size),
                entry_price=Decimal(p.get("avgPx", "0")),
                mark_price=Decimal(p.get("markPx", "0")),
                unrealized_pnl=Decimal(p.get("upl", "0")),
                leverage=int(p.get("lever", 1)),
                margin_type=p.get("mgnMode", "cross"),
                liquidation_price=Decimal(p.get("liqPx", "0")) if p.get("liqPx") else None,
            ))

        return positions

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage."""
        symbol = self.normalize_symbol(symbol)
        try:
            await self._request(
                "POST",
                "/account/set-leverage",
                {
                    "instId": symbol,
                    "lever": str(leverage),
                    "mgnMode": "cross",
                },
                signed=True,
            )
            return True
        except OKXAPIError:
            return False

    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """Set margin type."""
        symbol = self.normalize_symbol(symbol)
        try:
            await self._request(
                "POST",
                "/account/set-leverage",
                {
                    "instId": symbol,
                    "lever": "10",
                    "mgnMode": margin_type.lower(),
                },
                signed=True,
            )
            return True
        except OKXAPIError:
            return False

    # -------------------------------------------------------------------------
    # Orders
    # -------------------------------------------------------------------------

    def _parse_order(self, data: Dict[str, Any]) -> ExchangeOrder:
        """Parse order response."""
        side_map = {"buy": Side.BUY, "sell": Side.SELL}
        status_map = {
            "live": OrderStatus.NEW,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELED,
        }
        type_map = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
        }

        return ExchangeOrder(
            order_id=data.get("ordId", ""),
            client_order_id=data.get("clOrdId"),
            symbol=data.get("instId", ""),
            side=side_map.get(data.get("side", "").lower(), Side.BUY),
            order_type=type_map.get(data.get("ordType", "").lower(), OrderType.MARKET),
            status=status_map.get(data.get("state", "").lower(), OrderStatus.NEW),
            quantity=Decimal(data.get("sz", "0")),
            filled_quantity=Decimal(data.get("accFillSz", "0")),
            price=Decimal(data.get("px", "0")) if data.get("px") else None,
            avg_fill_price=Decimal(data.get("avgPx", "0")) if data.get("avgPx") else None,
            commission=Decimal(data.get("fee", "0")),
            created_at=int(data.get("cTime", 0)),
            updated_at=int(data.get("uTime", 0)),
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
        symbol = self.normalize_symbol(symbol)

        params = {
            "instId": symbol,
            "tdMode": "cross",
            "side": "buy" if side == Side.BUY else "sell",
            "ordType": "market" if order_type == OrderType.MARKET else "limit",
            "sz": str(quantity),
        }

        if reduce_only:
            params["reduceOnly"] = True

        if price and order_type == OrderType.LIMIT:
            params["px"] = str(price)

        if client_order_id:
            params["clOrdId"] = client_order_id

        data = await self._request("POST", "/trade/order", params, signed=True)

        if not data:
            raise OKXAPIError("-1", "Order placement failed")

        order_data = data[0]
        if order_data.get("sCode") != "0":
            raise OKXAPIError(order_data.get("sCode", "-1"), order_data.get("sMsg", "Error"))

        return ExchangeOrder(
            order_id=order_data.get("ordId", ""),
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            status=OrderStatus.NEW,
            quantity=quantity,
            filled_quantity=Decimal("0"),
            price=price,
            avg_fill_price=None,
            commission=Decimal("0"),
            created_at=int(time.time() * 1000),
            updated_at=int(time.time() * 1000),
        )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order."""
        symbol = self.normalize_symbol(symbol)
        try:
            await self._request(
                "POST",
                "/trade/cancel-order",
                {"instId": symbol, "ordId": order_id},
                signed=True,
            )
            return True
        except OKXAPIError:
            return False

    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all orders."""
        symbol = self.normalize_symbol(symbol)
        try:
            orders = await self.get_open_orders(symbol)
            cancelled = 0
            for order in orders:
                if await self.cancel_order(symbol, order.order_id):
                    cancelled += 1
            return cancelled
        except OKXAPIError:
            return 0

    async def get_order(self, symbol: str, order_id: str) -> ExchangeOrder:
        """Get order."""
        symbol = self.normalize_symbol(symbol)
        data = await self._request(
            "GET",
            "/trade/order",
            {"instId": symbol, "ordId": order_id},
            signed=True,
        )

        if not data:
            raise OKXAPIError("-1", "Order not found")

        return self._parse_order(data[0])

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """Get open orders."""
        params = {"instType": "SWAP"}
        if symbol:
            params["instId"] = self.normalize_symbol(symbol)

        data = await self._request("GET", "/trade/orders-pending", params, signed=True)

        return [self._parse_order(o) for o in data]

    async def get_order_history(self, symbol: str, limit: int = 50) -> List[ExchangeOrder]:
        """Get order history."""
        symbol = self.normalize_symbol(symbol)
        data = await self._request(
            "GET",
            "/trade/orders-history-archive",
            {"instType": "SWAP", "instId": symbol, "limit": str(limit)},
            signed=True,
        )

        return [self._parse_order(o) for o in data]

    # -------------------------------------------------------------------------
    # Funding
    # -------------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> Decimal:
        """Get current funding rate."""
        symbol = self.normalize_symbol(symbol)
        data = await self._request(
            "GET",
            "/public/funding-rate",
            {"instId": symbol},
        )

        if not data:
            return Decimal("0")

        return Decimal(data[0].get("fundingRate", "0"))

    async def get_funding_history(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get funding history."""
        symbol = self.normalize_symbol(symbol)
        data = await self._request(
            "GET",
            "/public/funding-rate-history",
            {"instId": symbol, "limit": str(limit)},
        )

        return [
            {
                "symbol": f.get("instId"),
                "funding_rate": Decimal(f.get("fundingRate", "0")),
                "timestamp": int(f.get("fundingTime", 0)),
            }
            for f in data
        ]
