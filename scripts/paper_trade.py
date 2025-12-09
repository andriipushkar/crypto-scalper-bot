#!/usr/bin/env python3
"""
Paper Trading Script.

Runs the trading bot on Binance Testnet for paper trading.
Uses real market data but executes on testnet.

Usage:
    python scripts/paper_trade.py

Requirements:
    - Set BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET in .env
    - Configure strategies in config/settings.yaml

Press Ctrl+C to stop.
"""

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.utils.logger import setup_logger
from src.core.engine import TradingEngine
from src.strategy import OrderBookImbalanceStrategy, VolumeSpikeStrategy
from src.risk import RiskManager
from src.execution import BinanceFuturesAPI, OrderExecutor


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        return yaml.safe_load(f)


def load_risk_config(config_path: str = "config/risk.yaml") -> dict:
    """Load risk configuration."""
    path = Path(config_path)

    if not path.exists():
        logger.warning(f"Risk config not found: {config_path}, using defaults")
        return {}

    with open(path) as f:
        return yaml.safe_load(f)


async def setup_api(config: dict) -> BinanceFuturesAPI:
    """Setup and configure Binance API."""
    exchange_config = config.get("exchange", {})
    trading_config = config.get("trading", {})

    testnet = exchange_config.get("testnet", True)

    api = BinanceFuturesAPI(testnet=testnet)

    if not api.has_credentials:
        logger.error("API credentials not set!")
        logger.error("Set BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET in .env")
        raise ValueError("Missing API credentials")

    await api.connect()

    # Configure leverage and margin for each symbol
    leverage = trading_config.get("leverage", 10)
    margin_type = trading_config.get("margin_type", "CROSSED")

    for symbol in trading_config.get("symbols", ["BTCUSDT"]):
        try:
            await api.set_leverage(symbol, leverage)
            await api.set_margin_type(symbol, margin_type)
        except Exception as e:
            logger.warning(f"Failed to configure {symbol}: {e}")

    # Log account info
    balance = await api.get_balance("USDT")
    positions = await api.get_positions()

    logger.info(f"Account balance: ${balance:.2f} USDT")
    logger.info(f"Open positions: {len(positions)}")

    return api


def create_strategies(config: dict) -> list:
    """Create strategy instances from config."""
    strategies = []
    strategies_config = config.get("strategies", {})

    # Order Book Imbalance Strategy
    imbalance_config = strategies_config.get("orderbook_imbalance", {})
    if imbalance_config.get("enabled", True):
        strategies.append(OrderBookImbalanceStrategy(imbalance_config))

    # Volume Spike Strategy
    volume_config = strategies_config.get("volume_spike", {})
    if volume_config.get("enabled", True):
        strategies.append(VolumeSpikeStrategy(volume_config))

    return strategies


async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Load configs
    config = load_config()
    risk_config = load_risk_config()

    # Setup logging
    log_config = config.get("logging", {})
    file_config = log_config.get("file", {})

    setup_logger(
        level=log_config.get("level", "INFO"),
        log_file=file_config.get("path") if file_config.get("enabled") else None,
        rotation=file_config.get("rotation", "1 day"),
        retention=file_config.get("retention", "30 days"),
    )

    logger.info("=" * 60)
    logger.info("CRYPTO SCALPER BOT - PAPER TRADING")
    logger.info("=" * 60)

    exchange_config = config.get("exchange", {})
    trading_config = config.get("trading", {})

    logger.info(f"Environment: {'TESTNET' if exchange_config.get('testnet', True) else 'MAINNET'}")
    logger.info(f"Symbols: {trading_config.get('symbols', ['BTCUSDT'])}")
    logger.info("-" * 60)

    # Initialize components
    engine = None
    api = None

    try:
        # Setup API
        api = await setup_api(config)

        # Create risk manager
        risk_manager = RiskManager(risk_config)

        # Create executor
        executor = OrderExecutor(
            api=api,
            risk_manager=risk_manager,
            use_limit_orders=trading_config.get("orders", {}).get("default_type") == "LIMIT",
        )

        # Create strategies
        strategies = create_strategies(config)

        if not strategies:
            logger.error("No strategies enabled!")
            return

        # Create and configure engine
        engine = TradingEngine(config)

        for strategy in strategies:
            engine.add_strategy(strategy)

        engine.set_risk_manager(risk_manager)
        engine.set_executor(executor)

        # Setup shutdown handler
        loop = asyncio.get_event_loop()

        def shutdown_handler():
            logger.info("Shutdown signal received")
            asyncio.create_task(engine.stop())

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, shutdown_handler)

        # Start engine
        logger.info("Starting trading engine...")
        await engine.start()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

    finally:
        # Cleanup
        if engine and engine.is_running:
            await engine.stop()

        if api:
            await api.close()

        logger.info("Paper trading stopped")


if __name__ == "__main__":
    asyncio.run(main())
