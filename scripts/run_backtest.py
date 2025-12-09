#!/usr/bin/env python3
"""
Backtest runner script.

Run backtests on historical data with different strategies.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.backtest import (
    BacktestEngine,
    BacktestConfig,
    HistoricalDataLoader,
    walk_forward_test,
    monte_carlo_simulation,
)
from src.strategy.orderbook_imbalance import OrderBookImbalanceStrategy
from src.strategy.volume_spike import VolumeSpikeStrategy
from src.strategy.base import CompositeStrategy


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level=level,
    )


def get_strategy(strategy_name: str, config: dict = None):
    """Get strategy by name."""
    config = config or {}

    strategies = {
        "imbalance": OrderBookImbalanceStrategy,
        "volume": VolumeSpikeStrategy,
    }

    if strategy_name == "combined":
        imbalance = OrderBookImbalanceStrategy(config.get("imbalance", {}))
        volume = VolumeSpikeStrategy(config.get("volume", {}))
        return CompositeStrategy(config, [imbalance, volume])

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return strategies[strategy_name](config.get(strategy_name, {}))


def run_basic_backtest(args):
    """Run a basic backtest."""
    logger.info("=" * 60)
    logger.info("RUNNING BACKTEST")
    logger.info("=" * 60)

    # Create config
    config = BacktestConfig(
        initial_capital=args.capital,
        leverage=args.leverage,
        slippage_bps=args.slippage,
        commission_rate=args.commission / 100,
        max_position_pct=args.max_position,
    )

    # Create engine
    engine = BacktestEngine(config)

    # Get strategy
    strategy = get_strategy(args.strategy)

    # Run backtest
    result = engine.run(
        strategy=strategy,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        data_path=args.data_path,
    )

    # Print results
    print(result.summary())

    # Save equity curve if requested
    if args.output:
        save_results(result, args.output)

    return result


def run_walk_forward(args):
    """Run walk-forward test."""
    logger.info("=" * 60)
    logger.info("RUNNING WALK-FORWARD TEST")
    logger.info("=" * 60)

    config = BacktestConfig(
        initial_capital=args.capital,
        leverage=args.leverage,
    )

    engine = BacktestEngine(config)
    strategy = get_strategy(args.strategy)

    result = walk_forward_test(
        engine=engine,
        strategy=strategy,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        train_period=timedelta(days=args.train_days),
        test_period=timedelta(days=args.test_days),
        data_path=args.data_path,
    )

    # Print results for each period
    for i, period_result in enumerate(result.results):
        print(f"\n--- Period {i + 1} ---")
        print(f"Return: {period_result.total_return_pct:+.2f}%")
        print(f"Trades: {period_result.metrics.total_trades}")
        print(f"Win Rate: {period_result.metrics.win_rate:.1f}%")

    # Combined metrics
    print("\n" + "=" * 40)
    print("COMBINED RESULTS")
    print("=" * 40)
    print(f"Total Periods: {len(result.results)}")
    print(f"Total Return: ${float(result.total_return):,.2f}")
    print(f"Avg Return/Period: {result.avg_return_pct:.2f}%")
    print(f"Total Trades: {result.combined_metrics.total_trades}")
    print(f"Win Rate: {result.combined_metrics.win_rate:.1f}%")

    return result


def run_monte_carlo(args):
    """Run Monte Carlo simulation on backtest results."""
    logger.info("=" * 60)
    logger.info("RUNNING MONTE CARLO SIMULATION")
    logger.info("=" * 60)

    # First run a backtest
    config = BacktestConfig(initial_capital=args.capital, leverage=args.leverage)
    engine = BacktestEngine(config)
    strategy = get_strategy(args.strategy)

    result = engine.run(
        strategy=strategy,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        data_path=args.data_path,
    )

    if not result.trades:
        logger.error("No trades to simulate")
        return None

    # Run Monte Carlo
    mc_result = monte_carlo_simulation(
        trades=result.trades,
        initial_capital=args.capital,
        num_simulations=args.simulations,
    )

    # Print results
    print("\n" + "=" * 40)
    print("MONTE CARLO SIMULATION RESULTS")
    print("=" * 40)
    print(f"Simulations: {mc_result['num_simulations']}")
    print()
    print("Final Capital:")
    print(f"  Mean:        ${mc_result['final_capital']['mean']:,.2f}")
    print(f"  Median:      ${mc_result['final_capital']['median']:,.2f}")
    print(f"  Std Dev:     ${mc_result['final_capital']['std']:,.2f}")
    print(f"  5th %:       ${mc_result['final_capital']['percentile_5']:,.2f}")
    print(f"  95th %:      ${mc_result['final_capital']['percentile_95']:,.2f}")
    print(f"  Worst Case:  ${mc_result['final_capital']['worst']:,.2f}")
    print(f"  Best Case:   ${mc_result['final_capital']['best']:,.2f}")
    print()
    print("Max Drawdown:")
    print(f"  Mean:        ${mc_result['max_drawdown']['mean']:,.2f}")
    print(f"  95th %:      ${mc_result['max_drawdown']['percentile_95']:,.2f}")
    print(f"  Worst Case:  ${mc_result['max_drawdown']['worst']:,.2f}")

    return mc_result


def save_results(result, output_path: str):
    """Save backtest results to file."""
    import json

    output = {
        "summary": {
            "symbol": result.symbol,
            "strategy": result.strategy_name,
            "start": result.start_time.isoformat(),
            "end": result.end_time.isoformat(),
            "initial_capital": float(result.initial_capital),
            "final_capital": float(result.final_capital),
            "total_return_pct": result.total_return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
        },
        "metrics": {
            "total_trades": result.metrics.total_trades,
            "winning_trades": result.metrics.winning_trades,
            "losing_trades": result.metrics.losing_trades,
            "win_rate": result.metrics.win_rate,
            "profit_factor": result.metrics.profit_factor,
            "sharpe_ratio": result.metrics.sharpe_ratio,
            "sortino_ratio": result.metrics.sortino_ratio,
        },
        "equity_curve": result.equity_curve,
        "trades": [
            {
                "symbol": t.symbol,
                "side": t.side,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "entry_price": float(t.entry_price),
                "exit_price": float(t.exit_price),
                "quantity": float(t.quantity),
                "pnl": float(t.pnl),
                "commission": float(t.commission),
            }
            for t in result.trades
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def parse_date(date_str: str) -> datetime:
    """Parse date string."""
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Cannot parse date: {date_str}")


def main():
    parser = argparse.ArgumentParser(description="Run backtests on historical data")

    # Mode
    parser.add_argument(
        "mode",
        choices=["backtest", "walk-forward", "monte-carlo"],
        help="Backtest mode",
    )

    # Required arguments
    parser.add_argument("--symbol", "-s", required=True, help="Trading symbol (e.g., BTCUSDT)")
    parser.add_argument("--start", required=True, type=parse_date, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, type=parse_date, help="End date (YYYY-MM-DD)")

    # Data
    parser.add_argument(
        "--data-path",
        default="data/raw/market_data.db",
        help="Path to data file",
    )

    # Strategy
    parser.add_argument(
        "--strategy",
        "-t",
        default="imbalance",
        choices=["imbalance", "volume", "combined"],
        help="Strategy to test",
    )

    # Capital
    parser.add_argument("--capital", "-c", type=float, default=100, help="Initial capital (USD)")
    parser.add_argument("--leverage", "-l", type=int, default=10, help="Leverage")

    # Execution simulation
    parser.add_argument("--slippage", type=float, default=1.0, help="Slippage (bps)")
    parser.add_argument("--commission", type=float, default=0.04, help="Commission (%)")
    parser.add_argument("--max-position", type=float, default=0.5, help="Max position size (fraction)")

    # Walk-forward parameters
    parser.add_argument("--train-days", type=int, default=7, help="Training period (days)")
    parser.add_argument("--test-days", type=int, default=1, help="Test period (days)")

    # Monte Carlo
    parser.add_argument("--simulations", type=int, default=1000, help="Number of Monte Carlo simulations")

    # Output
    parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Validate dates
    if args.start >= args.end:
        logger.error("Start date must be before end date")
        sys.exit(1)

    # Run appropriate mode
    if args.mode == "backtest":
        run_basic_backtest(args)
    elif args.mode == "walk-forward":
        run_walk_forward(args)
    elif args.mode == "monte-carlo":
        run_monte_carlo(args)


if __name__ == "__main__":
    main()
