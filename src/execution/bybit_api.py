"""
Bybit Futures API implementation.

Supports Bybit USDT Perpetual contracts.
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


class BybitAPIError(Exception):
    """Bybit API error."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"Bybit API Error {code}: {message}")


class BybitFuturesAPI(BaseExchangeAPI):
    """
    Bybit USDT Perpetual Futures API.

    Usage:
        credentials = ExchangeCredentials(
            api_key="your_key",
            api_secret="your_secret",
            testnet=True,
        )
        api = BybitFuturesAPI(credentials)
        await api.connect()

        ticker = await api.get_ticker("BTCUSDT")
        order = await api.place_market_order("BTCUSDT", Side.BUY, Decimal("0.001"))
    """

    MAINNET_URL = "https://api.bybit.com"
    TESTNET_URL = "https://api-testnet.bybit.com"
    MAINNET_WS = "wss://stream.bybit.com/v5/public/linear"
    TESTNET_WS = "wss://stream-testnet.bybit.com/v5/public/linear"

    def __init__(self, credentials: ExchangeCredentials):
        super().__init__(credentials)
        self._recv_window = 5000

    @property
    def exchange(self) -> Exchange:
        return Exchange.BYBIT

    @property
    def base_url(self) -> str:
        return self.TESTNET_URL if self.testnet else self.MAINNET_URL

    @property
    def ws_url(self) -> str:
        return self.TESTNET_WS if self.testnet else self.MAINNET_WS

    def _sign(self, params: Dict[str, Any], timestamp: int) -> str:
        """Generate signature for authenticated requests."""
        param_str = str(timestamp) + self.credentials.api_key + str(self._recv_window)

        if params:
            sorted_params = sorted(params.items())
            param_str += "&".join(f"{k}={v}" for k, v in sorted_params)

        return hmac.new(
            self.credentials.api_secret.encode(),
            param_str.encode(),
            hashlib.sha256,
        ).hexdigest()

    def _get_headers(self, params: Dict[str, Any] = None) -> Dict[str, str]:
        """Get headers for authenticated requests."""
        timestamp = int(time.time() * 1000)
        signature = self._sign(params or {}, timestamp)

        return {
            "X-BAPI-API-KEY": self.credentials.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": str(self._recv_window),
            "Content-Type": "application/json",
        }

    async def connect(self) -> None:
        """Initialize connection."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        logger.info(f"Bybit API connected ({'testnet' if self.testnet else 'mainnet'})")

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
        headers = self._get_headers(params) if signed else {}

        async with self._session.request(
            method,
            url,
            params=params if method == "GET" else None,
            json=params if method == "POST" else None,
            headers=headers,
        ) as resp:
            data = await resp.json()

            if data.get("retCode", 0) != 0:
                raise BybitAPIError(
                    data.get("retCode", -1),
                    data.get("retMsg", "Unknown error"),
                )

            return data.get("result", {})

    async def ping(self) -> bool:
        """Check connection."""
        try:
            await self._request("GET", "/v5/market/time")
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Market Data
    # -------------------------------------------------------------------------

    async def get_exchange_info(self) -> Dict[str, ExchangeSymbolInfo]:
        """Get trading rules."""
        data = await self._request(
            "GET",
            "/v5/market/instruments-info",
            {"category": "linear"},
        )

        symbols = {}
        for item in data.get("list", []):
            symbol = item["symbol"]
            lot_filter = item.get("lotSizeFilter", {})
            price_filter = item.get("priceFilter", {})

            symbols[symbol] = ExchangeSymbolInfo(
                symbol=symbol,
                base_asset=item.get("baseCoin", ""),
                quote_asset=item.get("quoteCoin", "USDT"),
                price_precision=int(price_filter.get("tickSize", "0.01").count("0")),
                quantity_precision=int(lot_filter.get("qtyStep", "0.001").count("0")),
                min_quantity=Decimal(lot_filter.get("minOrderQty", "0.001")),
                max_quantity=Decimal(lot_filter.get("maxOrderQty", "100")),
                min_notional=Decimal(lot_filter.get("minNotionalValue", "5")),
                tick_size=Decimal(price_filter.get("tickSize", "0.01")),
                step_size=Decimal(lot_filter.get("qtyStep", "0.001")),
            )

        return symbols

    async def get_ticker(self, symbol: str) -> ExchangeTicker:
        """Get ticker."""
        data = await self._request(
            "GET",
            "/v5/market/tickers",
            {"category": "linear", "symbol": symbol},
        )

        ticker_data = data.get("list", [{}])[0]
        return ExchangeTicker(
            symbol=symbol,
            last_price=Decimal(ticker_data.get("lastPrice", "0")),
            bid_price=Decimal(ticker_data.get("bid1Price", "0")),
            ask_price=Decimal(ticker_data.get("ask1Price", "0")),
            volume_24h=Decimal(ticker_data.get("volume24h", "0")),
            price_change_24h=Decimal(ticker_data.get("price24hPcnt", "0")) * 100,
            timestamp=int(time.time() * 1000),
        )

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get orderbook."""
        data = await self._request(
            "GET",
            "/v5/market/orderbook",
            {"category": "linear", "symbol": symbol, "limit": limit},
        )

        return {
            "bids": [[Decimal(p), Decimal(q)] for p, q in data.get("b", [])],
            "asks": [[Decimal(p), Decimal(q)] for p, q in data.get("a", [])],
            "timestamp": data.get("ts", int(time.time() * 1000)),
        }

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        data = await self._request(
            "GET",
            "/v5/market/recent-trade",
            {"category": "linear", "symbol": symbol, "limit": limit},
        )

        return [
            {
                "id": t.get("execId"),
                "price": Decimal(t.get("price", "0")),
                "quantity": Decimal(t.get("size", "0")),
                "side": t.get("side", "").upper(),
                "timestamp": int(t.get("time", 0)),
            }
            for t in data.get("list", [])
        ]

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get klines."""
        # Convert interval format
        interval_map = {
            "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "2h": "120", "4h": "240", "1d": "D", "1w": "W",
        }

        data = await self._request(
            "GET",
            "/v5/market/kline",
            {
                "category": "linear",
                "symbol": symbol,
                "interval": interval_map.get(interval, interval),
                "limit": limit,
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
            for k in data.get("list", [])
        ]

    # -------------------------------------------------------------------------
    # Account
    # -------------------------------------------------------------------------

    async def get_balance(self) -> List[ExchangeBalance]:
        """Get account balance."""
        data = await self._request(
            "GET",
            "/v5/account/wallet-balance",
            {"accountType": "UNIFIED"},
            signed=True,
        )

        balances = []
        for account in data.get("list", []):
            for coin in account.get("coin", []):
                balances.append(ExchangeBalance(
                    asset=coin.get("coin", ""),
                    wallet_balance=Decimal(coin.get("walletBalance", "0")),
                    available_balance=Decimal(coin.get("availableToWithdraw", "0")),
                    unrealized_pnl=Decimal(coin.get("unrealisedPnl", "0")),
                    margin_balance=Decimal(coin.get("equity", "0")),
                ))

        return balances

    async def get_positions(self, symbol: Optional[str] = None) -> List[ExchangePosition]:
        """Get positions."""
        params = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol

        data = await self._request("GET", "/v5/position/list", params, signed=True)

        positions = []
        for p in data.get("list", []):
            size = Decimal(p.get("size", "0"))
            if size == 0:
                continue

            side = Side.BUY if p.get("side") == "Buy" else Side.SELL
            positions.append(ExchangePosition(
                symbol=p.get("symbol", ""),
                side=side,
                quantity=size,
                entry_price=Decimal(p.get("avgPrice", "0")),
                mark_price=Decimal(p.get("markPrice", "0")),
                unrealized_pnl=Decimal(p.get("unrealisedPnl", "0")),
                leverage=int(p.get("leverage", 1)),
                margin_type="cross" if p.get("tradeMode") == 0 else "isolated",
                liquidation_price=Decimal(p.get("liqPrice", "0")) if p.get("liqPrice") else None,
            ))

        return positions

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage."""
        try:
            await self._request(
                "POST",
                "/v5/position/set-leverage",
                {
                    "category": "linear",
                    "symbol": symbol,
                    "buyLeverage": str(leverage),
                    "sellLeverage": str(leverage),
                },
                signed=True,
            )
            return True
        except BybitAPIError:
            return False

    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """Set margin type."""
        try:
            trade_mode = 0 if margin_type.lower() == "cross" else 1
            await self._request(
                "POST",
                "/v5/position/switch-isolated",
                {
                    "category": "linear",
                    "symbol": symbol,
                    "tradeMode": trade_mode,
                    "buyLeverage": "10",
                    "sellLeverage": "10",
                },
                signed=True,
            )
            return True
        except BybitAPIError:
            return False

    # -------------------------------------------------------------------------
    # Orders
    # -------------------------------------------------------------------------

    def _parse_order(self, data: Dict[str, Any]) -> ExchangeOrder:
        """Parse order response."""
        side = Side.BUY if data.get("side") == "Buy" else Side.SELL

        status_map = {
            "Created": OrderStatus.NEW,
            "New": OrderStatus.NEW,
            "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELED,
            "Rejected": OrderStatus.REJECTED,
        }

        type_map = {
            "Market": OrderType.MARKET,
            "Limit": OrderType.LIMIT,
        }

        return ExchangeOrder(
            order_id=data.get("orderId", ""),
            client_order_id=data.get("orderLinkId"),
            symbol=data.get("symbol", ""),
            side=side,
            order_type=type_map.get(data.get("orderType", ""), OrderType.MARKET),
            status=status_map.get(data.get("orderStatus", ""), OrderStatus.NEW),
            quantity=Decimal(data.get("qty", "0")),
            filled_quantity=Decimal(data.get("cumExecQty", "0")),
            price=Decimal(data.get("price", "0")) if data.get("price") else None,
            avg_fill_price=Decimal(data.get("avgPrice", "0")) if data.get("avgPrice") else None,
            commission=Decimal(data.get("cumExecFee", "0")),
            created_at=int(data.get("createdTime", 0)),
            updated_at=int(data.get("updatedTime", 0)),
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
            "category": "linear",
            "symbol": symbol,
            "side": "Buy" if side == Side.BUY else "Sell",
            "orderType": "Market" if order_type == OrderType.MARKET else "Limit",
            "qty": str(quantity),
            "reduceOnly": reduce_only,
        }

        if price and order_type == OrderType.LIMIT:
            params["price"] = str(price)
            params["timeInForce"] = "GTC"

        if client_order_id:
            params["orderLinkId"] = client_order_id

        if stop_price:
            params["triggerPrice"] = str(stop_price)

        data = await self._request("POST", "/v5/order/create", params, signed=True)

        return self._parse_order({
            "orderId": data.get("orderId"),
            "orderLinkId": data.get("orderLinkId"),
            "symbol": symbol,
            "side": params["side"],
            "orderType": params["orderType"],
            "qty": str(quantity),
            "orderStatus": "New",
        })

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order."""
        try:
            await self._request(
                "POST",
                "/v5/order/cancel",
                {
                    "category": "linear",
                    "symbol": symbol,
                    "orderId": order_id,
                },
                signed=True,
            )
            return True
        except BybitAPIError:
            return False

    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all orders."""
        try:
            data = await self._request(
                "POST",
                "/v5/order/cancel-all",
                {"category": "linear", "symbol": symbol},
                signed=True,
            )
            return len(data.get("list", []))
        except BybitAPIError:
            return 0

    async def get_order(self, symbol: str, order_id: str) -> ExchangeOrder:
        """Get order."""
        data = await self._request(
            "GET",
            "/v5/order/realtime",
            {
                "category": "linear",
                "symbol": symbol,
                "orderId": order_id,
            },
            signed=True,
        )

        orders = data.get("list", [])
        if not orders:
            raise BybitAPIError(-1, "Order not found")

        return self._parse_order(orders[0])

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """Get open orders."""
        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol

        data = await self._request("GET", "/v5/order/realtime", params, signed=True)

        return [self._parse_order(o) for o in data.get("list", [])]

    async def get_order_history(self, symbol: str, limit: int = 50) -> List[ExchangeOrder]:
        """Get order history."""
        data = await self._request(
            "GET",
            "/v5/order/history",
            {"category": "linear", "symbol": symbol, "limit": limit},
            signed=True,
        )

        return [self._parse_order(o) for o in data.get("list", [])]

    # -------------------------------------------------------------------------
    # Funding
    # -------------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> Decimal:
        """Get current funding rate."""
        ticker = await self.get_ticker(symbol)
        data = await self._request(
            "GET",
            "/v5/market/tickers",
            {"category": "linear", "symbol": symbol},
        )

        ticker_data = data.get("list", [{}])[0]
        return Decimal(ticker_data.get("fundingRate", "0"))

    async def get_funding_history(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get funding history."""
        data = await self._request(
            "GET",
            "/v5/market/funding/history",
            {"category": "linear", "symbol": symbol, "limit": limit},
        )

        return [
            {
                "symbol": f.get("symbol"),
                "funding_rate": Decimal(f.get("fundingRate", "0")),
                "timestamp": int(f.get("fundingRateTimestamp", 0)),
            }
            for f in data.get("list", [])
        ]
