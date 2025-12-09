"""
Backtesting engine for strategy evaluation.

Simulates trading on historical data with realistic execution.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, Callable, Type

from loguru import logger

from src.data.models import (
    Trade, OrderBookSnapshot, Signal, SignalType,
    Order, OrderType, OrderStatus, Position, Side,
)
from src.strategy.base import BaseStrategy
from src.risk.manager import RiskManager, RiskConfig
from src.backtest.data_loader import HistoricalDataLoader, OHLCV
from src.utils.metrics import TradeRecord, PerformanceMetrics, calculate_metrics


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    # Capital
    initial_capital: Decimal = Decimal("100")
    leverage: int = 10

    # Execution simulation
    slippage_bps: float = 1.0  # Slippage in basis points
    commission_rate: float = 0.0004  # 0.04% taker fee

    # Timing
    latency_ms: int = 50  # Simulated latency

    # Risk
    max_position_pct: float = 0.5  # Max 50% of capital per trade
    max_daily_loss_pct: float = 0.05  # 5% daily loss limit

    # Data
    use_orderbook: bool = True
    warmup_bars: int = 100  # Warmup period for indicators


@dataclass
class BacktestResult:
    """Backtesting results."""
    # Basic info
    symbol: str
    strategy_name: str
    start_time: datetime
    end_time: datetime
    config: BacktestConfig

    # Capital
    initial_capital: Decimal
    final_capital: Decimal

    # Performance
    metrics: PerformanceMetrics
    trades: List[TradeRecord] = field(default_factory=list)

    # Equity curve
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)

    # Drawdown
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_pct: float = 0.0

    @property
    def total_return(self) -> Decimal:
        """Total return in currency."""
        return self.final_capital - self.initial_capital

    @property
    def total_return_pct(self) -> float:
        """Total return percentage."""
        if self.initial_capital == 0:
            return 0.0
        return float((self.final_capital - self.initial_capital) / self.initial_capital * 100)

    def summary(self) -> str:
        """Generate summary report."""
        return f"""
================================================================================
                          BACKTEST RESULTS
================================================================================
Symbol:         {self.symbol}
Strategy:       {self.strategy_name}
Period:         {self.start_time.strftime('%Y-%m-%d %H:%M')} to {self.end_time.strftime('%Y-%m-%d %H:%M')}

CAPITAL
-------
Initial:        ${float(self.initial_capital):,.2f}
Final:          ${float(self.final_capital):,.2f}
Return:         ${float(self.total_return):,.2f} ({self.total_return_pct:+.2f}%)
Max Drawdown:   ${float(self.max_drawdown):,.2f} ({self.max_drawdown_pct:.2f}%)

TRADES
------
Total Trades:   {self.metrics.total_trades}
Winners:        {self.metrics.winning_trades}
Losers:         {self.metrics.losing_trades}
Win Rate:       {self.metrics.win_rate:.1f}%

Gross Profit:   ${float(self.metrics.gross_profit):,.2f}
Gross Loss:     ${float(self.metrics.gross_loss):,.2f}
Profit Factor:  {self.metrics.profit_factor:.2f}

Avg Win:        ${float(self.metrics.avg_win):,.2f}
Avg Loss:       ${float(self.metrics.avg_loss):,.2f}
Avg Trade:      ${float(self.metrics.avg_trade):,.2f}

RISK METRICS
------------
Sharpe Ratio:   {self.metrics.sharpe_ratio:.2f}
Sortino Ratio:  {self.metrics.sortino_ratio:.2f}
Calmar Ratio:   {self.metrics.calmar_ratio:.2f}

Max Consec Wins:   {self.metrics.max_consecutive_wins}
Max Consec Losses: {self.metrics.max_consecutive_losses}
================================================================================
"""


class SimulatedPosition:
    """Simulated position for backtesting."""

    def __init__(
        self,
        symbol: str,
        side: Side,
        entry_price: Decimal,
        quantity: Decimal,
        entry_time: datetime,
        leverage: int = 1,
    ):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.leverage = leverage
        self.commission_paid = Decimal("0")

    @property
    def notional_value(self) -> Decimal:
        """Position notional value."""
        return self.entry_price * self.quantity

    @property
    def margin_required(self) -> Decimal:
        """Required margin."""
        return self.notional_value / self.leverage

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L."""
        if self.side == Side.BUY:
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: Decimal) -> float:
        """Unrealized P&L as percentage of margin."""
        pnl = self.unrealized_pnl(current_price)
        if self.margin_required == 0:
            return 0.0
        return float(pnl / self.margin_required * 100)


class BacktestEngine:
    """
    Backtesting engine for strategy evaluation.

    Features:
    - Realistic execution simulation (slippage, commissions, latency)
    - Position and risk management
    - Equity curve tracking
    - Detailed performance metrics
    """

    def __init__(
        self,
        config: BacktestConfig = None,
        data_loader: HistoricalDataLoader = None,
    ):
        """
        Initialize backtesting engine.

        Args:
            config: Backtesting configuration
            data_loader: Historical data loader
        """
        self.config = config or BacktestConfig()
        self.data_loader = data_loader

        # State
        self._capital = self.config.initial_capital
        self._position: Optional[SimulatedPosition] = None
        self._trades: List[TradeRecord] = []
        self._equity_curve: List[Dict[str, Any]] = []

        # Tracking
        self._peak_capital = self.config.initial_capital
        self._max_drawdown = Decimal("0")
        self._max_drawdown_pct = 0.0
        self._daily_pnl = Decimal("0")
        self._current_date: Optional[datetime] = None

        # Signals processed
        self._signals_generated = 0
        self._signals_executed = 0

    def reset(self) -> None:
        """Reset engine state."""
        self._capital = self.config.initial_capital
        self._position = None
        self._trades = []
        self._equity_curve = []
        self._peak_capital = self.config.initial_capital
        self._max_drawdown = Decimal("0")
        self._max_drawdown_pct = 0.0
        self._daily_pnl = Decimal("0")
        self._current_date = None
        self._signals_generated = 0
        self._signals_executed = 0

    # =========================================================================
    # Execution Simulation
    # =========================================================================

    def _apply_slippage(self, price: Decimal, side: Side) -> Decimal:
        """Apply slippage to execution price."""
        slippage_factor = Decimal(str(self.config.slippage_bps / 10000))

        if side == Side.BUY:
            # Buy at higher price (unfavorable)
            return price * (1 + slippage_factor)
        else:
            # Sell at lower price (unfavorable)
            return price * (1 - slippage_factor)

    def _calculate_commission(self, notional: Decimal) -> Decimal:
        """Calculate commission for trade."""
        return notional * Decimal(str(self.config.commission_rate))

    def _can_open_position(self, signal: Signal, price: Decimal) -> bool:
        """Check if position can be opened."""
        # Already have a position
        if self._position is not None:
            return False

        # Check daily loss limit
        daily_loss_limit = self.config.initial_capital * Decimal(str(self.config.max_daily_loss_pct))
        if self._daily_pnl < -daily_loss_limit:
            logger.debug("Daily loss limit reached")
            return False

        return True

    def _calculate_position_size(self, price: Decimal) -> Decimal:
        """Calculate position size based on risk parameters."""
        # Max position value based on capital and leverage
        max_position_value = self._capital * self.config.leverage * Decimal(str(self.config.max_position_pct))

        # Calculate quantity
        quantity = max_position_value / price

        # Round to reasonable precision
        return quantity.quantize(Decimal("0.001"))

    def _open_position(
        self,
        signal: Signal,
        price: Decimal,
        timestamp: datetime,
    ) -> Optional[TradeRecord]:
        """Open a new position."""
        if not self._can_open_position(signal, price):
            return None

        # Determine side
        if signal.signal_type == SignalType.LONG:
            side = Side.BUY
        elif signal.signal_type == SignalType.SHORT:
            side = Side.SELL
        else:
            return None

        # Apply slippage
        execution_price = self._apply_slippage(price, side)

        # Calculate position size
        quantity = self._calculate_position_size(execution_price)

        if quantity <= 0:
            return None

        # Calculate commission
        notional = execution_price * quantity
        commission = self._calculate_commission(notional)

        # Check if we have enough capital for margin + commission
        margin_required = notional / self.config.leverage
        if margin_required + commission > self._capital:
            logger.debug("Insufficient capital for trade")
            return None

        # Deduct commission
        self._capital -= commission

        # Create position
        self._position = SimulatedPosition(
            symbol=signal.symbol,
            side=side,
            entry_price=execution_price,
            quantity=quantity,
            entry_time=timestamp,
            leverage=self.config.leverage,
        )
        self._position.commission_paid = commission

        self._signals_executed += 1

        logger.debug(
            f"Opened {side.value} position: {quantity} @ {execution_price}"
        )

        return None  # Trade record created on close

    def _close_position(
        self,
        price: Decimal,
        timestamp: datetime,
        reason: str = "signal",
    ) -> Optional[TradeRecord]:
        """Close current position."""
        if self._position is None:
            return None

        # Apply slippage (opposite direction)
        close_side = Side.SELL if self._position.side == Side.BUY else Side.BUY
        execution_price = self._apply_slippage(price, close_side)

        # Calculate P&L
        pnl = self._position.unrealized_pnl(execution_price)

        # Calculate closing commission
        notional = execution_price * self._position.quantity
        commission = self._calculate_commission(notional)

        # Total commission
        total_commission = self._position.commission_paid + commission

        # Net P&L
        net_pnl = pnl - commission

        # Update capital
        self._capital += net_pnl

        # Update daily P&L
        self._daily_pnl += net_pnl

        # Create trade record
        trade = TradeRecord(
            symbol=self._position.symbol,
            side="LONG" if self._position.side == Side.BUY else "SHORT",
            entry_time=self._position.entry_time,
            exit_time=timestamp,
            entry_price=self._position.entry_price,
            exit_price=execution_price,
            quantity=self._position.quantity,
            pnl=pnl,
            commission=total_commission,
            strategy=reason,
        )

        self._trades.append(trade)

        logger.debug(
            f"Closed position: P&L = {float(pnl):.2f}, Net = {float(net_pnl):.2f}"
        )

        # Clear position
        self._position = None

        return trade

    def _process_signal(
        self,
        signal: Signal,
        current_price: Decimal,
        timestamp: datetime,
    ) -> Optional[TradeRecord]:
        """Process a trading signal."""
        self._signals_generated += 1

        if signal.signal_type == SignalType.NO_ACTION:
            return None

        # Close signals
        if signal.signal_type in (SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT):
            if self._position is not None:
                # Check if close signal matches position
                if signal.signal_type == SignalType.CLOSE_LONG and self._position.side == Side.BUY:
                    return self._close_position(current_price, timestamp, signal.strategy)
                elif signal.signal_type == SignalType.CLOSE_SHORT and self._position.side == Side.SELL:
                    return self._close_position(current_price, timestamp, signal.strategy)
            return None

        # Open signals - close existing position first if opposite direction
        if self._position is not None:
            if signal.signal_type == SignalType.LONG and self._position.side == Side.SELL:
                self._close_position(current_price, timestamp, "reversal")
            elif signal.signal_type == SignalType.SHORT and self._position.side == Side.BUY:
                self._close_position(current_price, timestamp, "reversal")
            else:
                # Same direction, skip
                return None

        # Open new position
        return self._open_position(signal, current_price, timestamp)

    def _update_equity(self, timestamp: datetime, price: Decimal) -> None:
        """Update equity curve and drawdown."""
        # Calculate current equity
        equity = self._capital
        if self._position:
            equity += self._position.unrealized_pnl(price)

        # Update peak and drawdown
        if equity > self._peak_capital:
            self._peak_capital = equity

        drawdown = self._peak_capital - equity
        if drawdown > self._max_drawdown:
            self._max_drawdown = drawdown
            if self._peak_capital > 0:
                self._max_drawdown_pct = float(drawdown / self._peak_capital * 100)

        # Record equity point
        self._equity_curve.append({
            "timestamp": timestamp,
            "equity": float(equity),
            "capital": float(self._capital),
            "unrealized_pnl": float(equity - self._capital),
            "drawdown": float(drawdown),
        })

    def _check_new_day(self, timestamp: datetime) -> None:
        """Reset daily tracking on new day."""
        current_day = timestamp.date()

        if self._current_date is None:
            self._current_date = current_day
        elif current_day != self._current_date:
            logger.debug(f"New day: {current_day}, previous P&L: {self._daily_pnl}")
            self._current_date = current_day
            self._daily_pnl = Decimal("0")

    # =========================================================================
    # Main Backtest Methods
    # =========================================================================

    def run(
        self,
        strategy: BaseStrategy,
        symbol: str,
        start: datetime,
        end: datetime,
        data_path: str = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            strategy: Strategy to test
            symbol: Trading symbol
            start: Start datetime
            end: End datetime
            data_path: Optional path to data file

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Starting backtest: {strategy.__class__.__name__} on {symbol}")
        logger.info(f"Period: {start} to {end}")

        # Reset state
        self.reset()

        # Load data
        if data_path:
            loader = HistoricalDataLoader(data_path)
        elif self.data_loader:
            loader = self.data_loader
        else:
            raise ValueError("No data loader provided")

        with loader:
            trades = loader.load_trades(symbol, start, end)

        if not trades:
            logger.warning("No trades found for backtest period")
            return self._create_result(symbol, strategy.__class__.__name__, start, end)

        logger.info(f"Loaded {len(trades)} trades")

        # Process trades
        trade_buffer = []
        last_process_time = trades[0].timestamp

        for trade in trades:
            # Check for new day
            self._check_new_day(trade.timestamp)

            # Buffer trades for aggregation
            trade_buffer.append(trade)

            # Process every 100ms or when buffer is full
            if (trade.timestamp - last_process_time).total_seconds() >= 0.1 or len(trade_buffer) >= 100:
                # Generate signal from strategy
                signal = strategy.generate_signal(
                    symbol=symbol,
                    trades=trade_buffer,
                    orderbook=None,  # Simplified for trade-based backtest
                )

                if signal:
                    current_price = trade_buffer[-1].price
                    self._process_signal(signal, current_price, trade.timestamp)

                # Update equity
                self._update_equity(trade.timestamp, trade_buffer[-1].price)

                # Reset buffer
                trade_buffer = []
                last_process_time = trade.timestamp

        # Close any open position at end
        if self._position and trades:
            self._close_position(trades[-1].price, trades[-1].timestamp, "end_of_backtest")

        return self._create_result(symbol, strategy.__class__.__name__, start, end)

    def run_on_bars(
        self,
        strategy: BaseStrategy,
        symbol: str,
        bars: List[OHLCV],
    ) -> BacktestResult:
        """
        Run backtest on OHLCV bars.

        Args:
            strategy: Strategy to test
            symbol: Trading symbol
            bars: List of OHLCV bars

        Returns:
            BacktestResult with performance metrics
        """
        if not bars:
            raise ValueError("No bars provided")

        logger.info(f"Starting bar-based backtest: {strategy.__class__.__name__}")
        logger.info(f"Bars: {len(bars)} ({bars[0].timestamp} to {bars[-1].timestamp})")

        # Reset state
        self.reset()

        # Skip warmup period
        warmup = min(self.config.warmup_bars, len(bars) - 1)

        for i, bar in enumerate(bars[warmup:], warmup):
            # Check for new day
            self._check_new_day(bar.timestamp)

            # Get historical bars for strategy
            history = bars[max(0, i - 100):i + 1]

            # Generate signal
            signal = strategy.generate_signal_from_bars(
                symbol=symbol,
                bars=history,
            )

            if signal:
                # Use close price for execution
                self._process_signal(signal, bar.close, bar.timestamp)

            # Update equity with close price
            self._update_equity(bar.timestamp, bar.close)

        # Close any open position at end
        if self._position and bars:
            self._close_position(bars[-1].close, bars[-1].timestamp, "end_of_backtest")

        return self._create_result(
            symbol,
            strategy.__class__.__name__,
            bars[0].timestamp,
            bars[-1].timestamp,
        )

    def _create_result(
        self,
        symbol: str,
        strategy_name: str,
        start: datetime,
        end: datetime,
    ) -> BacktestResult:
        """Create backtest result."""
        # Calculate metrics
        metrics = calculate_metrics(self._trades)

        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy_name,
            start_time=start,
            end_time=end,
            config=self.config,
            initial_capital=self.config.initial_capital,
            final_capital=self._capital,
            metrics=metrics,
            trades=self._trades.copy(),
            equity_curve=self._equity_curve.copy(),
            max_drawdown=self._max_drawdown,
            max_drawdown_pct=self._max_drawdown_pct,
        )


# =============================================================================
# Walk-Forward Optimization
# =============================================================================

@dataclass
class WalkForwardResult:
    """Walk-forward optimization result."""
    results: List[BacktestResult]

    @property
    def combined_trades(self) -> List[TradeRecord]:
        """All trades from all periods."""
        trades = []
        for result in self.results:
            trades.extend(result.trades)
        return trades

    @property
    def combined_metrics(self) -> PerformanceMetrics:
        """Metrics for all periods combined."""
        return calculate_metrics(self.combined_trades)

    @property
    def total_return(self) -> Decimal:
        """Total return across all periods."""
        return sum(r.total_return for r in self.results)

    @property
    def avg_return_pct(self) -> float:
        """Average return per period."""
        if not self.results:
            return 0.0
        return sum(r.total_return_pct for r in self.results) / len(self.results)


def walk_forward_test(
    engine: BacktestEngine,
    strategy: BaseStrategy,
    symbol: str,
    start: datetime,
    end: datetime,
    train_period: timedelta,
    test_period: timedelta,
    data_path: str = None,
) -> WalkForwardResult:
    """
    Perform walk-forward testing.

    Args:
        engine: Backtest engine
        strategy: Strategy to test
        symbol: Trading symbol
        start: Start datetime
        end: End datetime
        train_period: Training period duration
        test_period: Testing period duration
        data_path: Path to data file

    Returns:
        WalkForwardResult with all test periods
    """
    results = []
    current = start + train_period  # Skip first training period

    while current + test_period <= end:
        # Test on current period
        test_start = current
        test_end = current + test_period

        logger.info(f"Walk-forward test period: {test_start} to {test_end}")

        result = engine.run(
            strategy=strategy,
            symbol=symbol,
            start=test_start,
            end=test_end,
            data_path=data_path,
        )

        results.append(result)
        current += test_period

    return WalkForwardResult(results=results)


# =============================================================================
# Monte Carlo Simulation
# =============================================================================

def monte_carlo_simulation(
    trades: List[TradeRecord],
    initial_capital: Decimal,
    num_simulations: int = 1000,
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation on trade results.

    Shuffles trade order to assess strategy robustness.

    Args:
        trades: List of trade records
        initial_capital: Starting capital
        num_simulations: Number of simulations

    Returns:
        Dictionary with simulation statistics
    """
    import random

    if not trades:
        return {"error": "No trades to simulate"}

    final_capitals = []
    max_drawdowns = []

    for _ in range(num_simulations):
        # Shuffle trades
        shuffled = trades.copy()
        random.shuffle(shuffled)

        # Simulate equity curve
        capital = initial_capital
        peak = initial_capital
        max_dd = Decimal("0")

        for trade in shuffled:
            capital += trade.net_pnl
            if capital > peak:
                peak = capital
            dd = peak - capital
            if dd > max_dd:
                max_dd = dd

        final_capitals.append(float(capital))
        max_drawdowns.append(float(max_dd))

    # Calculate statistics
    final_capitals.sort()
    max_drawdowns.sort()

    return {
        "num_simulations": num_simulations,
        "final_capital": {
            "mean": sum(final_capitals) / len(final_capitals),
            "median": final_capitals[len(final_capitals) // 2],
            "std": (sum((x - sum(final_capitals) / len(final_capitals)) ** 2 for x in final_capitals) / len(final_capitals)) ** 0.5,
            "percentile_5": final_capitals[int(0.05 * len(final_capitals))],
            "percentile_95": final_capitals[int(0.95 * len(final_capitals))],
            "worst": final_capitals[0],
            "best": final_capitals[-1],
        },
        "max_drawdown": {
            "mean": sum(max_drawdowns) / len(max_drawdowns),
            "median": max_drawdowns[len(max_drawdowns) // 2],
            "percentile_95": max_drawdowns[int(0.95 * len(max_drawdowns))],
            "worst": max_drawdowns[-1],
        },
    }
