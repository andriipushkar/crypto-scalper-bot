"""
Arbitrage Strategy

Cross-exchange and triangular arbitrage detection and execution.
"""
import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import time

from loguru import logger


class ArbitrageType(Enum):
    """Types of arbitrage."""
    CROSS_EXCHANGE = "cross_exchange"  # Same pair, different exchanges
    TRIANGULAR = "triangular"  # A -> B -> C -> A on same exchange
    FUNDING_RATE = "funding_rate"  # Spot vs perpetual


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    id: str
    arb_type: ArbitrageType
    symbol: str
    exchanges: List[str]
    buy_exchange: str
    sell_exchange: str
    buy_price: Decimal
    sell_price: Decimal
    spread: Decimal
    spread_pct: Decimal
    estimated_profit: Decimal
    estimated_profit_pct: Decimal
    volume_available: Decimal
    detected_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    executed: bool = False
    execution_result: Optional[Dict] = None

    def is_profitable(self, min_profit_pct: Decimal = Decimal("0.1")) -> bool:
        """Check if opportunity is still profitable after fees."""
        return self.estimated_profit_pct >= min_profit_pct

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.arb_type.value,
            "symbol": self.symbol,
            "exchanges": self.exchanges,
            "buy_exchange": self.buy_exchange,
            "sell_exchange": self.sell_exchange,
            "buy_price": str(self.buy_price),
            "sell_price": str(self.sell_price),
            "spread_pct": str(self.spread_pct),
            "estimated_profit_pct": str(self.estimated_profit_pct),
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class TriangularPath:
    """Triangular arbitrage path."""
    base: str
    quote: str
    intermediate: str
    path: List[Tuple[str, str]]  # List of (pair, side)
    profit_pct: Decimal


class ArbitrageDetector:
    """Detect arbitrage opportunities across exchanges."""

    def __init__(
        self,
        exchanges: List[str],
        min_profit_pct: Decimal = Decimal("0.1"),
        fee_rate: Decimal = Decimal("0.001"),
        slippage: Decimal = Decimal("0.0005"),
    ):
        self.exchanges = exchanges
        self.min_profit_pct = min_profit_pct
        self.fee_rate = fee_rate
        self.slippage = slippage

        # Price data
        self.prices: Dict[str, Dict[str, Dict]] = {}  # exchange -> symbol -> {bid, ask, volume}
        self.order_books: Dict[str, Dict[str, Dict]] = {}

        # Detected opportunities
        self.opportunities: List[ArbitrageOpportunity] = []
        self._opportunity_counter = 0

        # Callbacks
        self.on_opportunity: Optional[Callable] = None

    def update_price(
        self,
        exchange: str,
        symbol: str,
        bid: Decimal,
        ask: Decimal,
        volume: Decimal = Decimal("0"),
    ):
        """Update price for exchange/symbol."""
        if exchange not in self.prices:
            self.prices[exchange] = {}

        self.prices[exchange][symbol] = {
            "bid": bid,
            "ask": ask,
            "volume": volume,
            "timestamp": datetime.now(),
        }

    def update_order_book(
        self,
        exchange: str,
        symbol: str,
        bids: List[Tuple[Decimal, Decimal]],  # price, quantity
        asks: List[Tuple[Decimal, Decimal]],
    ):
        """Update order book for exchange/symbol."""
        if exchange not in self.order_books:
            self.order_books[exchange] = {}

        self.order_books[exchange][symbol] = {
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.now(),
        }

    def scan_cross_exchange(self, symbol: str) -> List[ArbitrageOpportunity]:
        """Scan for cross-exchange arbitrage on a symbol."""
        opportunities = []

        # Get prices from all exchanges
        exchange_prices = []
        for exchange in self.exchanges:
            if exchange in self.prices and symbol in self.prices[exchange]:
                data = self.prices[exchange][symbol]
                exchange_prices.append((exchange, data))

        if len(exchange_prices) < 2:
            return []

        # Compare all pairs
        for i, (ex1, data1) in enumerate(exchange_prices):
            for ex2, data2 in exchange_prices[i+1:]:
                # Check buy on ex1, sell on ex2
                opp1 = self._check_cross_exchange_opp(
                    symbol, ex1, data1, ex2, data2
                )
                if opp1:
                    opportunities.append(opp1)

                # Check buy on ex2, sell on ex1
                opp2 = self._check_cross_exchange_opp(
                    symbol, ex2, data2, ex1, data1
                )
                if opp2:
                    opportunities.append(opp2)

        return opportunities

    def _check_cross_exchange_opp(
        self,
        symbol: str,
        buy_exchange: str,
        buy_data: Dict,
        sell_exchange: str,
        sell_data: Dict,
    ) -> Optional[ArbitrageOpportunity]:
        """Check for cross-exchange opportunity."""
        buy_price = buy_data["ask"]  # We buy at ask
        sell_price = sell_data["bid"]  # We sell at bid

        # Calculate spread
        spread = sell_price - buy_price
        spread_pct = (spread / buy_price) * 100

        # Calculate profit after fees
        total_fee_pct = self.fee_rate * 2 * 100  # Buy and sell fees
        slippage_pct = self.slippage * 2 * 100

        profit_pct = spread_pct - total_fee_pct - slippage_pct

        if profit_pct < self.min_profit_pct:
            return None

        # Estimate volume
        volume = min(buy_data["volume"], sell_data["volume"]) / 10  # Conservative

        self._opportunity_counter += 1

        return ArbitrageOpportunity(
            id=f"ce_{self._opportunity_counter}",
            arb_type=ArbitrageType.CROSS_EXCHANGE,
            symbol=symbol,
            exchanges=[buy_exchange, sell_exchange],
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            buy_price=buy_price,
            sell_price=sell_price,
            spread=spread,
            spread_pct=spread_pct,
            estimated_profit=volume * spread * (profit_pct / 100),
            estimated_profit_pct=profit_pct,
            volume_available=volume,
            expires_at=datetime.now() + timedelta(seconds=30),
        )

    def scan_triangular(self, exchange: str, base: str = "USDT") -> List[ArbitrageOpportunity]:
        """Scan for triangular arbitrage opportunities."""
        if exchange not in self.prices:
            return []

        opportunities = []
        symbols = list(self.prices[exchange].keys())

        # Find all possible triangular paths
        paths = self._find_triangular_paths(symbols, base)

        for path in paths:
            opp = self._check_triangular_opp(exchange, path, base)
            if opp:
                opportunities.append(opp)

        return opportunities

    def _find_triangular_paths(self, symbols: List[str], base: str) -> List[TriangularPath]:
        """Find all possible triangular arbitrage paths."""
        paths = []

        # Parse symbols to get currency pairs
        pairs: Dict[str, List[str]] = {}  # currency -> [quote currencies]
        for symbol in symbols:
            # Assume format like BTCUSDT, ETHBTC, etc
            for quote in ["USDT", "BTC", "ETH", "BNB"]:
                if symbol.endswith(quote):
                    base_curr = symbol[:-len(quote)]
                    if base_curr not in pairs:
                        pairs[base_curr] = []
                    pairs[base_curr].append(quote)
                    break

        # Find triangular paths: base -> intermediate -> target -> base
        for curr1, quotes1 in pairs.items():
            if base not in quotes1:
                continue

            for curr2, quotes2 in pairs.items():
                if curr2 == curr1 or curr1 not in quotes2:
                    continue

                if base in quotes2:
                    # Found a path: base -> curr1 -> curr2 -> base
                    paths.append(TriangularPath(
                        base=base,
                        quote=curr1,
                        intermediate=curr2,
                        path=[
                            (f"{curr1}{base}", "buy"),  # USDT -> BTC
                            (f"{curr2}{curr1}", "buy"),  # BTC -> ETH
                            (f"{curr2}{base}", "sell"),  # ETH -> USDT
                        ],
                        profit_pct=Decimal("0"),
                    ))

        return paths

    def _check_triangular_opp(
        self,
        exchange: str,
        path: TriangularPath,
        base: str,
    ) -> Optional[ArbitrageOpportunity]:
        """Check for triangular arbitrage opportunity."""
        prices = self.prices[exchange]

        # Calculate path profit
        value = Decimal("1")  # Start with 1 unit

        for symbol, side in path.path:
            if symbol not in prices:
                return None

            data = prices[symbol]
            if side == "buy":
                # We buy, pay ask price
                value = value / data["ask"]
            else:
                # We sell, receive bid price
                value = value * data["bid"]

            # Deduct fee
            value = value * (1 - self.fee_rate)

        # Calculate profit
        profit_pct = (value - 1) * 100

        if profit_pct < self.min_profit_pct:
            return None

        self._opportunity_counter += 1

        return ArbitrageOpportunity(
            id=f"tri_{self._opportunity_counter}",
            arb_type=ArbitrageType.TRIANGULAR,
            symbol=f"{path.quote}-{path.intermediate}-{base}",
            exchanges=[exchange],
            buy_exchange=exchange,
            sell_exchange=exchange,
            buy_price=Decimal("1"),
            sell_price=value,
            spread=value - 1,
            spread_pct=profit_pct,
            estimated_profit=Decimal("0"),  # Depends on volume
            estimated_profit_pct=profit_pct,
            volume_available=Decimal("0"),
            expires_at=datetime.now() + timedelta(seconds=10),
        )

    def scan_all(self, symbols: List[str]) -> List[ArbitrageOpportunity]:
        """Scan for all types of arbitrage."""
        all_opportunities = []

        # Cross-exchange
        for symbol in symbols:
            opps = self.scan_cross_exchange(symbol)
            all_opportunities.extend(opps)

        # Triangular (for each exchange)
        for exchange in self.exchanges:
            opps = self.scan_triangular(exchange)
            all_opportunities.extend(opps)

        # Sort by profit
        all_opportunities.sort(key=lambda x: x.estimated_profit_pct, reverse=True)

        # Notify callbacks
        for opp in all_opportunities:
            if self.on_opportunity:
                self.on_opportunity(opp)

        self.opportunities = all_opportunities
        return all_opportunities


class ArbitrageExecutor:
    """Execute arbitrage trades."""

    def __init__(
        self,
        exchange_clients: Dict[str, Any],
        max_position_size: Decimal = Decimal("1000"),
        max_concurrent: int = 3,
    ):
        self.exchange_clients = exchange_clients
        self.max_position_size = max_position_size
        self.max_concurrent = max_concurrent

        self._active_executions = 0
        self._execution_lock = asyncio.Lock()

        # Stats
        self.stats = {
            "total_attempts": 0,
            "successful": 0,
            "failed": 0,
            "total_profit": Decimal("0"),
        }

    async def execute_cross_exchange(
        self,
        opportunity: ArbitrageOpportunity,
        quantity: Decimal,
    ) -> Dict[str, Any]:
        """Execute cross-exchange arbitrage."""
        async with self._execution_lock:
            if self._active_executions >= self.max_concurrent:
                return {"success": False, "error": "Max concurrent executions reached"}

            self._active_executions += 1

        try:
            self.stats["total_attempts"] += 1

            buy_client = self.exchange_clients.get(opportunity.buy_exchange)
            sell_client = self.exchange_clients.get(opportunity.sell_exchange)

            if not buy_client or not sell_client:
                return {"success": False, "error": "Exchange client not found"}

            # Place orders simultaneously
            buy_task = buy_client.place_order(
                symbol=opportunity.symbol,
                side="buy",
                quantity=float(quantity),
                order_type="market",
            )
            sell_task = sell_client.place_order(
                symbol=opportunity.symbol,
                side="sell",
                quantity=float(quantity),
                order_type="market",
            )

            buy_result, sell_result = await asyncio.gather(
                buy_task, sell_task, return_exceptions=True
            )

            # Check results
            if isinstance(buy_result, Exception) or isinstance(sell_result, Exception):
                self.stats["failed"] += 1
                return {
                    "success": False,
                    "error": str(buy_result if isinstance(buy_result, Exception) else sell_result),
                }

            # Calculate actual profit
            buy_price = Decimal(str(buy_result.get("avg_price", opportunity.buy_price)))
            sell_price = Decimal(str(sell_result.get("avg_price", opportunity.sell_price)))
            actual_profit = (sell_price - buy_price) * quantity

            self.stats["successful"] += 1
            self.stats["total_profit"] += actual_profit

            opportunity.executed = True
            opportunity.execution_result = {
                "buy_order": buy_result,
                "sell_order": sell_result,
                "actual_profit": float(actual_profit),
            }

            logger.info(
                f"Arbitrage executed: {opportunity.symbol} "
                f"Buy @ {buy_price} on {opportunity.buy_exchange}, "
                f"Sell @ {sell_price} on {opportunity.sell_exchange}, "
                f"Profit: {actual_profit}"
            )

            return {
                "success": True,
                "profit": float(actual_profit),
                "buy_order": buy_result,
                "sell_order": sell_result,
            }

        except Exception as e:
            self.stats["failed"] += 1
            logger.error(f"Arbitrage execution failed: {e}")
            return {"success": False, "error": str(e)}

        finally:
            async with self._execution_lock:
                self._active_executions -= 1

    async def execute_triangular(
        self,
        opportunity: ArbitrageOpportunity,
        initial_amount: Decimal,
    ) -> Dict[str, Any]:
        """Execute triangular arbitrage."""
        # Implementation similar to cross-exchange but with 3 orders
        # Sequential execution is safer for triangular
        return {"success": False, "error": "Not implemented"}

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            **{k: float(v) if isinstance(v, Decimal) else v for k, v in self.stats.items()},
            "success_rate": (
                self.stats["successful"] / self.stats["total_attempts"] * 100
                if self.stats["total_attempts"] > 0 else 0
            ),
            "active_executions": self._active_executions,
        }


class ArbitrageBot:
    """Complete arbitrage trading bot."""

    def __init__(
        self,
        detector: ArbitrageDetector,
        executor: ArbitrageExecutor,
        symbols: List[str],
        scan_interval_ms: int = 100,
        auto_execute: bool = False,
        min_profit_pct: Decimal = Decimal("0.2"),
    ):
        self.detector = detector
        self.executor = executor
        self.symbols = symbols
        self.scan_interval_ms = scan_interval_ms
        self.auto_execute = auto_execute
        self.min_profit_pct = min_profit_pct

        self._running = False
        self._scan_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the arbitrage bot."""
        self._running = True
        self._scan_task = asyncio.create_task(self._scan_loop())
        logger.info("Arbitrage bot started")

    async def stop(self):
        """Stop the arbitrage bot."""
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        logger.info("Arbitrage bot stopped")

    async def _scan_loop(self):
        """Main scanning loop."""
        while self._running:
            try:
                opportunities = self.detector.scan_all(self.symbols)

                if opportunities and self.auto_execute:
                    # Execute best opportunity
                    best = opportunities[0]
                    if best.estimated_profit_pct >= self.min_profit_pct:
                        quantity = min(
                            best.volume_available,
                            self.executor.max_position_size / best.buy_price
                        )
                        await self.executor.execute_cross_exchange(best, quantity)

                await asyncio.sleep(self.scan_interval_ms / 1000)

            except Exception as e:
                logger.error(f"Scan loop error: {e}")
                await asyncio.sleep(1)

    def get_opportunities(self) -> List[Dict[str, Any]]:
        """Get current opportunities."""
        return [o.to_dict() for o in self.detector.opportunities]

    def get_stats(self) -> Dict[str, Any]:
        """Get bot statistics."""
        return {
            "running": self._running,
            "opportunities_count": len(self.detector.opportunities),
            "execution_stats": self.executor.get_stats(),
        }
