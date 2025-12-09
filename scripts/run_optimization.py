#!/usr/bin/env python3
"""
Walk-Forward Optimization Script.

Run hyperparameter optimization for trading strategies using Optuna
with walk-forward validation to avoid overfitting.

Usage:
    # Simple optimization
    python scripts/run_optimization.py --strategy orderbook_imbalance --trials 50

    # Walk-forward optimization
    python scripts/run_optimization.py --strategy mean_reversion --walk-forward --trials 100

    # Multiple strategies
    python scripts/run_optimization.py --strategy all --trials 50

    # Custom date range
    python scripts/run_optimization.py --strategy grid_trading --start 2024-01-01 --end 2024-06-01
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add(
        "logs/optimization_{time}.log",
        level="DEBUG",
        rotation="100 MB",
    )


def run_simple_optimization(
    strategy_name: str,
    n_trials: int,
    symbol: str,
    data_path: str,
    objective: str,
    output_dir: str,
) -> dict:
    """Run simple Optuna optimization."""
    from src.optimization.optuna_optimizer import (
        StrategyOptimizer,
        OptimizationConfig,
        STRATEGY_PARAM_FUNCTIONS,
    )

    if strategy_name not in STRATEGY_PARAM_FUNCTIONS:
        available = list(STRATEGY_PARAM_FUNCTIONS.keys())
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")

    config = OptimizationConfig(
        n_trials=n_trials,
        symbol=symbol,
        data_path=data_path,
        objective_metric=objective,
    )

    optimizer = StrategyOptimizer(strategy_name, config)
    best_params = optimizer.optimize()

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_path / f"{strategy_name}_opt_{timestamp}.json"

    optimizer.save_results(str(result_file))

    # Get importance
    importance = optimizer.get_importance()
    if importance:
        logger.info("Parameter importance:")
        for param, score in sorted(importance.items(), key=lambda x: -x[1]):
            logger.info(f"  {param}: {score:.4f}")

    return {
        "strategy": strategy_name,
        "best_params": best_params,
        "best_value": optimizer.study.best_value if optimizer.study else None,
        "importance": importance,
        "result_file": str(result_file),
    }


def run_walk_forward_optimization(
    strategy_name: str,
    n_trials: int,
    symbol: str,
    data_path: str,
    objective: str,
    output_dir: str,
    start_date: datetime,
    end_date: datetime,
    train_days: int,
    test_days: int,
) -> dict:
    """Run walk-forward optimization."""
    from src.optimization.walk_forward import (
        WalkForwardOptimizer,
        WalkForwardConfig,
    )
    from src.optimization.optuna_optimizer import STRATEGY_PARAM_FUNCTIONS
    from src.strategy import (
        OrderBookImbalanceStrategy,
        VolumeSpikeStrategy,
        GridTradingStrategy,
        MeanReversionStrategy,
    )

    # Map strategy names to classes
    STRATEGY_CLASSES = {
        "orderbook_imbalance": OrderBookImbalanceStrategy,
        "volume_spike": VolumeSpikeStrategy,
        "grid_trading": GridTradingStrategy,
        "mean_reversion": MeanReversionStrategy,
    }

    if strategy_name not in STRATEGY_CLASSES:
        available = list(STRATEGY_CLASSES.keys())
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")

    config = WalkForwardConfig(
        train_period_days=train_days,
        test_period_days=test_days,
        step_days=test_days,
        optimization_trials=n_trials,
        optimization_metric=objective,
        symbol=symbol,
        data_path=data_path,
        output_dir=output_dir,
        save_results=True,
    )

    optimizer = WalkForwardOptimizer(
        strategy_name=strategy_name,
        strategy_class=STRATEGY_CLASSES[strategy_name],
        param_space_fn=STRATEGY_PARAM_FUNCTIONS[strategy_name],
        config=config,
    )

    result = optimizer.run(start_date, end_date)

    # Log results
    logger.info("=" * 60)
    logger.info(f"Walk-Forward Optimization Results: {strategy_name}")
    logger.info("=" * 60)
    logger.info(f"Total Return: {result.total_return:.2f}%")
    logger.info(f"Avg Sharpe: {result.avg_sharpe:.2f}")
    logger.info(f"Avg Win Rate: {result.avg_win_rate:.1f}%")
    logger.info(f"Total Trades: {result.total_trades}")
    logger.info(f"Profitable Windows: {result.windows_profitable}/{len(result.windows)}")

    # Parameter stability
    stability = optimizer.get_param_stability()
    if stability:
        logger.info("\nParameter Stability (lower CV = more stable):")
        for param, stats in sorted(stability.items(), key=lambda x: x[1]["cv"]):
            logger.info(
                f"  {param}: mean={stats['mean']:.4f}, "
                f"std={stats['std']:.4f}, CV={stats['cv']:.2f}"
            )

    # Latest params
    latest_params = optimizer.get_latest_params()
    logger.info(f"\nLatest optimized params: {latest_params}")

    return {
        "strategy": strategy_name,
        "total_return": result.total_return,
        "avg_sharpe": result.avg_sharpe,
        "windows_profitable": result.windows_profitable,
        "total_windows": len(result.windows),
        "latest_params": latest_params,
        "param_stability": stability,
    }


def run_all_strategies(args) -> list:
    """Run optimization for all available strategies."""
    from src.optimization.optuna_optimizer import STRATEGY_PARAM_FUNCTIONS

    strategies = ["orderbook_imbalance", "volume_spike", "grid_trading", "mean_reversion"]
    results = []

    for strategy in strategies:
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimizing: {strategy}")
        logger.info("=" * 60)

        try:
            if args.walk_forward:
                result = run_walk_forward_optimization(
                    strategy_name=strategy,
                    n_trials=args.trials,
                    symbol=args.symbol,
                    data_path=args.data_path,
                    objective=args.objective,
                    output_dir=args.output_dir,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    train_days=args.train_days,
                    test_days=args.test_days,
                )
            else:
                result = run_simple_optimization(
                    strategy_name=strategy,
                    n_trials=args.trials,
                    symbol=args.symbol,
                    data_path=args.data_path,
                    objective=args.objective,
                    output_dir=args.output_dir,
                )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to optimize {strategy}: {e}")
            results.append({"strategy": strategy, "error": str(e)})

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run strategy optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --strategy orderbook_imbalance --trials 50
  %(prog)s --strategy mean_reversion --walk-forward
  %(prog)s --strategy all --trials 100 --objective sharpe_ratio
  %(prog)s --strategy grid_trading --start 2024-01-01 --end 2024-06-01
        """,
    )

    # Strategy selection
    parser.add_argument(
        "--strategy", "-s",
        required=True,
        help="Strategy to optimize (orderbook_imbalance, volume_spike, grid_trading, mean_reversion, all)",
    )

    # Optimization settings
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)",
    )
    parser.add_argument(
        "--objective", "-o",
        default="sharpe_ratio",
        choices=["sharpe_ratio", "sortino_ratio", "total_return", "profit_factor", "win_rate"],
        help="Optimization objective (default: sharpe_ratio)",
    )

    # Walk-forward settings
    parser.add_argument(
        "--walk-forward", "-wf",
        action="store_true",
        help="Use walk-forward optimization (recommended)",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=30,
        help="Training window in days (default: 30)",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=7,
        help="Test window in days (default: 7)",
    )

    # Date range
    parser.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=datetime.now() - timedelta(days=90),
        help="Start date (default: 90 days ago)",
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=datetime.now(),
        help="End date (default: today)",
    )

    # Data settings
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--data-path",
        default="data/raw/market_data.db",
        help="Path to market data file",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="data/optimization",
        help="Output directory for results",
    )

    # Misc
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    args.start_date = args.start
    args.end_date = args.end

    # Setup logging
    setup_logging(args.verbose)

    logger.info("Strategy Optimization")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Trials: {args.trials}")
    logger.info(f"Objective: {args.objective}")
    logger.info(f"Walk-Forward: {args.walk_forward}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Date range: {args.start_date.date()} to {args.end_date.date()}")

    try:
        if args.strategy == "all":
            results = run_all_strategies(args)
        elif args.walk_forward:
            results = [run_walk_forward_optimization(
                strategy_name=args.strategy,
                n_trials=args.trials,
                symbol=args.symbol,
                data_path=args.data_path,
                objective=args.objective,
                output_dir=args.output_dir,
                start_date=args.start_date,
                end_date=args.end_date,
                train_days=args.train_days,
                test_days=args.test_days,
            )]
        else:
            results = [run_simple_optimization(
                strategy_name=args.strategy,
                n_trials=args.trials,
                symbol=args.symbol,
                data_path=args.data_path,
                objective=args.objective,
                output_dir=args.output_dir,
            )]

        # Save summary
        summary_path = Path(args.output_dir) / f"optimization_summary_{datetime.now():%Y%m%d_%H%M%S}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nSummary saved to: {summary_path}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("=" * 60)
        for r in results:
            if "error" in r:
                logger.error(f"{r['strategy']}: FAILED - {r['error']}")
            elif args.walk_forward:
                logger.info(
                    f"{r['strategy']}: return={r.get('total_return', 0):.2f}%, "
                    f"sharpe={r.get('avg_sharpe', 0):.2f}, "
                    f"profitable={r.get('windows_profitable', 0)}/{r.get('total_windows', 0)}"
                )
            else:
                logger.info(
                    f"{r['strategy']}: best_{args.objective}={r.get('best_value', 0):.4f}"
                )

    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
