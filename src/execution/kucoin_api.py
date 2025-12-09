"""
KuCoin Futures API implementation.

Supports KuCoin Futures perpetual contracts.
"""

import base64
import hashlib
import hmac
import time
import uuid
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


class KuCoinAPIError(Exception):
    """KuCoin API error."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"KuCoin API Error [{code}]: {message}")


class KuCoinFuturesAPI(BaseExchangeAPI):
    """
    KuCoin Futures API.

    Usage:
        credentials = ExchangeCredentials(
            api_key="your_key",
            api_secret="your_secret",
            passphrase="your_passphrase",  # Required for KuCoin
            testnet=True,
        )
        api = KuCoinFuturesAPI(credentials)
        await api.connect()

        ticker = await api.get_ticker("XBTUSDTM")
        order = await api.place_market_order("XBTUSDTM", Side.BUY, Decimal("1"))
    """

    # KuCoin Futures URLs
    MAINNET_URL = "https://api-futures.kucoin.com"
    TESTNET_URL = "https://api-sandbox-futures.kucoin.com"
    MAINNET_WS = "wss://ws-api-futures.kucoin.com"
    TESTNET_WS = "wss://ws-api-sandbox-futures.kucoin.com"

    def __init__(self, credentials: ExchangeCredentials):
        super().__init__(credentials)
        self._ws_token: Optional[str] = None
        self._ws_endpoint: Optional[str] = None

    @property
    def exchange(self) -> Exchange:
        return Exchange.KUCOIN

    @property
    def base_url(self) -> str:
        return self.TESTNET_URL if self.testnet else self.MAINNET_URL

    @property
    def ws_url(self) -> str:
        return self.TESTNET_WS if self.testnet else self.MAINNET_WS

    def _sign(self, timestamp: str, method: str, endpoint: str, body: str = "") -> str:
        """
        Generate signature for authenticated requests.

        KuCoin uses: HMAC-SHA256 of (timestamp + method + endpoint + body)
        """
        if not self.credentials.api_secret:
            return ""

        message = timestamp + method + endpoint + body
        signature = hmac.new(
            self.credentials.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        )
        return base64.b64encode(signature.digest()).decode()

    def _sign_passphrase(self) -> str:
        """Sign the passphrase for v2 API."""
        if not self.credentials.passphrase or not self.credentials.api_secret:
            return ""

        signature = hmac.new(
            self.credentials.api_secret.encode(),
            self.credentials.passphrase.encode(),
            hashlib.sha256,
        )
        return base64.b64encode(signature.digest()).decode()

    def _get_headers(
        self,
        method: str,
        endpoint: str,
        body: str = "",
    ) -> Dict[str, str]:
        """Get headers for authenticated requests."""
        timestamp = str(int(time.time() * 1000))

        headers = {
            "Content-Type": "application/json",
            "KC-API-KEY": self.credentials.api_key or "",
            "KC-API-SIGN": self._sign(timestamp, method, endpoint, body),
            "KC-API-TIMESTAMP": timestamp,
            "KC-API-PASSPHRASE": self._sign_passphrase(),
            "KC-API-KEY-VERSION": "2",
        }

        return headers

    async def connect(self) -> None:
        """Initialize connection."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        logger.info(f"KuCoin API connected ({'testnet' if self.testnet else 'mainnet'})")

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

        body = ""
        headers = {"Content-Type": "application/json"}

        if method == "GET" and params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            endpoint_with_query = f"{endpoint}?{query}"
            url = f"{self.base_url}{endpoint_with_query}"
            if signed:
                headers = self._get_headers(method, endpoint_with_query)
        elif method == "POST":
            import json
            body = json.dumps(params) if params else ""
            if signed:
                headers = self._get_headers(method, endpoint, body)

        try:
            if method == "GET":
                async with self._session.get(url, headers=headers) as resp:
                    data = await resp.json()
            else:
                async with self._session.post(url, data=body, headers=headers) as resp:
                    data = await resp.json()

            # Check for errors
            code = data.get("code", "200000")
            if code != "200000":
                raise KuCoinAPIError(code, data.get("msg", "Unknown error"))

            return data.get("data", {})

        except aiohttp.ClientError as e:
            logger.error(f"KuCoin request failed: {e}")
            raise KuCoinAPIError("CONNECTION", str(e))

    async def ping(self) -> bool:
        """Check connection."""
        try:
            await self._request("GET", "/api/v1/timestamp")
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Market Data
    # -------------------------------------------------------------------------

    async def get_exchange_info(self) -> Dict[str, ExchangeSymbolInfo]:
        """Get trading rules."""
        data = await self._request("GET", "/api/v1/contracts/active")

        symbols = {}

        for item in data if isinstance(data, list) else []:
            symbol = item.get("symbol", "")

            symbols[symbol] = ExchangeSymbolInfo(
                symbol=symbol,
                base_asset=item.get("baseCurrency", ""),
                quote_asset=item.get("quoteCurrency", "USDT"),
                price_precision=int(item.get("tickSize", "0.1").count("0")),
                quantity_precision=0,  # KuCoin uses lot size
                min_quantity=Decimal(str(item.get("lotSize", "1"))),
                max_quantity=Decimal(str(item.get("maxOrderQty", "1000000"))),
                min_notional=Decimal(str(item.get("minOrderQty", "1"))),
                tick_size=Decimal(str(item.get("tickSize", "0.1"))),
                step_size=Decimal(str(item.get("lotSize", "1"))),
            )

        return symbols

    async def get_ticker(self, symbol: str) -> ExchangeTicker:
        """Get ticker."""
        data = await self._request("GET", f"/api/v1/ticker", {"symbol": symbol})

        return ExchangeTicker(
            symbol=symbol,
            last_price=Decimal(str(data.get("price", "0"))),
            bid_price=Decimal(str(data.get("bestBidPrice", "0"))),
            ask_price=Decimal(str(data.get("bestAskPrice", "0"))),
            volume_24h=Decimal(str(data.get("size", "0"))),
            price_change_24h=Decimal(str(data.get("priceChgPct", "0"))) * 100,
            timestamp=int(data.get("ts", time.time() * 1000000) / 1000),
        )

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get orderbook."""
        # KuCoin has fixed depth levels: 20, 100
        depth = 100 if limit > 20 else 20

        data = await self._request(
            "GET",
            f"/api/v1/level2/depth{depth}",
            {"symbol": symbol},
        )

        return {
            "bids": [
                [Decimal(str(b[0])), Decimal(str(b[1]))]
                for b in data.get("bids", [])[:limit]
            ],
            "asks": [
                [Decimal(str(a[0])), Decimal(str(a[1]))]
                for a in data.get("asks", [])[:limit]
            ],
            "timestamp": int(data.get("ts", time.time() * 1000000) / 1000),
        }

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        data = await self._request(
            "GET",
            "/api/v1/trade/history",
            {"symbol": symbol},
        )

        trades = data if isinstance(data, list) else []

        return [
            {
                "id": t.get("tradeId", ""),
                "price": Decimal(str(t.get("price", "0"))),
                "quantity": Decimal(str(t.get("size", "0"))),
                "side": t.get("side", "").upper(),
                "timestamp": int(t.get("ts", 0) / 1000000),  # KuCoin uses nanoseconds
            }
            for t in trades[:limit]
        ]

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get klines/candles."""
        # KuCoin granularity in minutes
        interval_map = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "8h": 480,
            "12h": 720, "1d": 1440, "1w": 10080,
        }

        granularity = interval_map.get(interval, 60)

        # Calculate time range
        end_time = int(time.time() * 1000)
        start_time = end_time - (granularity * 60 * 1000 * limit)

        data = await self._request(
            "GET",
            "/api/v1/kline/query",
            {
                "symbol": symbol,
                "granularity": granularity,
                "from": start_time,
                "to": end_time,
            },
        )

        candles = data if isinstance(data, list) else []

        return [
            {
                "timestamp": int(c[0]),
                "open": Decimal(str(c[1])),
                "high": Decimal(str(c[2])),
                "low": Decimal(str(c[3])),
                "close": Decimal(str(c[4])),
                "volume": Decimal(str(c[5])),
            }
            for c in candles
        ]

    # -------------------------------------------------------------------------
    # Account
    # -------------------------------------------------------------------------

    async def get_balance(self) -> List[ExchangeBalance]:
        """Get account balance."""
        data = await self._request("GET", "/api/v1/account-overview", signed=True)

        return [ExchangeBalance(
            asset=data.get("currency", "USDT"),
            wallet_balance=Decimal(str(data.get("accountEquity", "0"))),
            available_balance=Decimal(str(data.get("availableBalance", "0"))),
            unrealized_pnl=Decimal(str(data.get("unrealisedPNL", "0"))),
            margin_balance=Decimal(str(data.get("marginBalance", "0"))),
        )]

    async def get_positions(self, symbol: Optional[str] = None) -> List[ExchangePosition]:
        """Get positions."""
        if symbol:
            data = await self._request(
                "GET",
                f"/api/v1/position",
                {"symbol": symbol},
                signed=True,
            )
            positions_data = [data] if data else []
        else:
            data = await self._request("GET", "/api/v1/positions", signed=True)
            positions_data = data if isinstance(data, list) else []

        positions = []
        for p in positions_data:
            qty = Decimal(str(p.get("currentQty", "0")))
            if qty == 0:
                continue

            side = Side.BUY if qty > 0 else Side.SELL

            positions.append(ExchangePosition(
                symbol=p.get("symbol", ""),
                side=side,
                quantity=abs(qty),
                entry_price=Decimal(str(p.get("avgEntryPrice", "0"))),
                mark_price=Decimal(str(p.get("markPrice", "0"))),
                unrealized_pnl=Decimal(str(p.get("unrealisedPnl", "0"))),
                leverage=int(p.get("realLeverage", 1)),
                margin_type="cross" if p.get("crossMode") else "isolated",
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
                "/api/v1/position/risk-limit-level/change",
                {"symbol": symbol, "level": leverage},
                signed=True,
            )
            return True
        except KuCoinAPIError:
            return False

    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """Set margin type."""
        try:
            is_cross = margin_type.lower() == "cross"
            await self._request(
                "POST",
                "/api/v1/position/margin/auto-deposit-status",
                {"symbol": symbol, "status": is_cross},
                signed=True,
            )
            return True
        except KuCoinAPIError:
            return False

    # -------------------------------------------------------------------------
    # Orders
    # -------------------------------------------------------------------------

    def _parse_order(self, data: Dict[str, Any]) -> ExchangeOrder:
        """Parse order response."""
        side = Side.BUY if data.get("side") == "buy" else Side.SELL

        status_map = {
            "open": OrderStatus.NEW,
            "done": OrderStatus.FILLED,
            "match": OrderStatus.PARTIALLY_FILLED,
        }

        # Determine status from dealSize and size
        deal_size = Decimal(str(data.get("dealSize", "0")))
        size = Decimal(str(data.get("size", "0")))

        if data.get("status") == "done":
            status = OrderStatus.FILLED
        elif data.get("cancelExist"):
            status = OrderStatus.CANCELED
        elif deal_size > 0:
            status = OrderStatus.PARTIALLY_FILLED
        else:
            status = OrderStatus.NEW

        type_map = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
        }

        return ExchangeOrder(
            order_id=data.get("id", data.get("orderId", "")),
            client_order_id=data.get("clientOid"),
            symbol=data.get("symbol", ""),
            side=side,
            order_type=type_map.get(data.get("type", ""), OrderType.MARKET),
            status=status,
            quantity=size,
            filled_quantity=deal_size,
            price=Decimal(str(data.get("price", "0"))) if data.get("price") else None,
            avg_fill_price=Decimal(str(data.get("dealValue", "0"))) / deal_size
            if deal_size > 0
            else None,
            commission=Decimal(str(data.get("fee", "0"))),
            created_at=int(data.get("createdAt", 0)),
            updated_at=int(data.get("updatedAt", time.time() * 1000)),
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
        params = {
            "clientOid": client_order_id or str(uuid.uuid4()),
            "symbol": symbol,
            "side": "buy" if side == Side.BUY else "sell",
            "type": "market" if order_type == OrderType.MARKET else "limit",
            "size": int(quantity),  # KuCoin uses contract count
            "leverage": "10",
        }

        if price and order_type == OrderType.LIMIT:
            params["price"] = str(price)

        if reduce_only:
            params["reduceOnly"] = True

        if stop_price:
            params["stop"] = "up" if side == Side.BUY else "down"
            params["stopPrice"] = str(stop_price)
            params["stopPriceType"] = "TP"

        data = await self._request("POST", "/api/v1/orders", params, signed=True)

        return self._parse_order({
            "id": data.get("orderId", ""),
            "clientOid": params["clientOid"],
            "symbol": symbol,
            "side": params["side"],
            "type": params["type"],
            "size": str(quantity),
            "status": "open",
            "createdAt": int(time.time() * 1000),
        })

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order."""
        try:
            await self._request(
                "DELETE",
                f"/api/v1/orders/{order_id}",
                signed=True,
            )
            return True
        except KuCoinAPIError:
            return False

    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all orders."""
        try:
            data = await self._request(
                "DELETE",
                "/api/v1/orders",
                {"symbol": symbol},
                signed=True,
            )
            cancelled = data.get("cancelledOrderIds", [])
            return len(cancelled) if isinstance(cancelled, list) else 0
        except KuCoinAPIError:
            return 0

    async def get_order(self, symbol: str, order_id: str) -> ExchangeOrder:
        """Get order."""
        data = await self._request(
            "GET",
            f"/api/v1/orders/{order_id}",
            signed=True,
        )

        if not data:
            raise KuCoinAPIError("NOT_FOUND", "Order not found")

        return self._parse_order(data)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """Get open orders."""
        params = {"status": "active"}
        if symbol:
            params["symbol"] = symbol

        data = await self._request("GET", "/api/v1/orders", params, signed=True)

        items = data.get("items", []) if isinstance(data, dict) else []
        return [self._parse_order(o) for o in items]

    async def get_order_history(self, symbol: str, limit: int = 50) -> List[ExchangeOrder]:
        """Get order history."""
        data = await self._request(
            "GET",
            "/api/v1/orders",
            {"symbol": symbol, "status": "done", "pageSize": limit},
            signed=True,
        )

        items = data.get("items", []) if isinstance(data, dict) else []
        return [self._parse_order(o) for o in items]

    # -------------------------------------------------------------------------
    # Funding
    # -------------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> Decimal:
        """Get current funding rate."""
        data = await self._request(
            "GET",
            f"/api/v1/funding-rate/{symbol}/current",
        )

        return Decimal(str(data.get("value", "0")))

    async def get_funding_history(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get funding history."""
        data = await self._request(
            "GET",
            "/api/v1/contract/funding-rates",
            {"symbol": symbol, "pageSize": limit},
        )

        items = data.get("dataList", []) if isinstance(data, dict) else []

        return [
            {
                "symbol": symbol,
                "funding_rate": Decimal(str(r.get("fundingRate", "0"))),
                "timestamp": int(r.get("timepoint", 0)),
            }
            for r in items
        ]

    # -------------------------------------------------------------------------
    # WebSocket Token
    # -------------------------------------------------------------------------

    async def get_ws_token(self, private: bool = False) -> Dict[str, str]:
        """Get WebSocket connection token."""
        endpoint = "/api/v1/bullet-private" if private else "/api/v1/bullet-public"

        data = await self._request("POST", endpoint, signed=private)

        token = data.get("token", "")
        servers = data.get("instanceServers", [])

        if servers:
            endpoint = servers[0].get("endpoint", self.ws_url)
        else:
            endpoint = self.ws_url

        self._ws_token = token
        self._ws_endpoint = endpoint

        return {
            "token": token,
            "endpoint": f"{endpoint}?token={token}",
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to KuCoin format.

        KuCoin uses: XBTUSDTM for Bitcoin/USDT perpetual
        """
        symbol = symbol.upper()

        # Common conversions
        conversions = {
            "BTCUSDT": "XBTUSDTM",
            "BTCUSD": "XBTUSDM",
            "ETHUSDT": "ETHUSDTM",
            "ETHUSD": "ETHUSDM",
        }

        return conversions.get(symbol, symbol)
