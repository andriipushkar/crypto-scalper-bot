#!/usr/bin/env python3
"""
Run Paper Trading with Live Data

Uses Binance WebSocket for real-time 1m candles and runs strategies.
"""

import asyncio
import signal
import sys
from datetime import datetime
from typing import List
from decimal import Decimal

sys.path.insert(0, '/home/sssmmmddd/Code/pro/crypto-scalper-bot')

from loguru import logger
from src.trading.live_trading import LiveTradingLoop

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)


# =============================================================================
# Strategy Implementations (simplified for live trading)
# =============================================================================

def cluster_analysis_strategy(eng, symbol: str, candle, history: List):
    """Cluster Analysis - POC and Value Area."""
    if len(history) < 20:
        return

    volume_prices = []
    for c in history[-20:]:
        typical_price = (c.high + c.low + c.close) / 3
        volume_prices.append((typical_price, c.volume))

    total_vol = sum(v for _, v in volume_prices)
    poc = sum(p * v for p, v in volume_prices) / total_vol if total_vol > 0 else candle.close

    closes = sorted([c.close for c in history[-20:]])
    va_low = closes[3]
    va_high = closes[16]

    has_position = symbol in eng.positions
    current_price = candle.close

    if not has_position:
        qty = (eng.get_available_balance() * 0.1) / current_price
        if qty > 0.001:
            if current_price < va_low and current_price < poc * 0.99:
                logger.info(f"[CLUSTER] BUY signal: {symbol} @ {current_price:.2f}")
                asyncio.create_task(eng.place_order(symbol, "BUY", qty))
            elif current_price > va_high and current_price > poc * 1.01:
                logger.info(f"[CLUSTER] SELL signal: {symbol} @ {current_price:.2f}")
                asyncio.create_task(eng.place_order(symbol, "SELL", qty))


def mean_reversion_strategy(eng, symbol: str, candle, history: List):
    """Mean Reversion - Z-score based."""
    if len(history) < 25:
        return

    import statistics
    closes = [c.close for c in history[-20:]]
    mean = sum(closes) / len(closes)
    std = statistics.stdev(closes) if len(closes) > 1 else 0.001

    z_score = (candle.close - mean) / std if std > 0 else 0

    has_position = symbol in eng.positions
    current_price = candle.close

    if not has_position:
        qty = (eng.get_available_balance() * 0.1) / current_price
        if qty > 0.001:
            if z_score < -2.0:
                logger.info(f"[MEAN_REV] BUY signal: {symbol} Z={z_score:.2f}")
                asyncio.create_task(eng.place_order(symbol, "BUY", qty))
            elif z_score > 2.0:
                logger.info(f"[MEAN_REV] SELL signal: {symbol} Z={z_score:.2f}")
                asyncio.create_task(eng.place_order(symbol, "SELL", qty))


def volume_spike_strategy(eng, symbol: str, candle, history: List):
    """Volume Spike Detection."""
    if len(history) < 20:
        return

    volumes = [c.volume for c in history[-20:]]
    avg_volume = sum(volumes) / len(volumes)
    price_change = (candle.close - candle.open) / candle.open

    has_position = symbol in eng.positions
    current_price = candle.close

    is_volume_spike = candle.volume > avg_volume * 3.0

    if not has_position and is_volume_spike:
        qty = (eng.get_available_balance() * 0.1) / current_price
        if qty > 0.001:
            if price_change > 0.001:
                logger.info(f"[VOL_SPIKE] BUY signal: {symbol} Vol={candle.volume/avg_volume:.1f}x")
                asyncio.create_task(eng.place_order(symbol, "BUY", qty))
            elif price_change < -0.001:
                logger.info(f"[VOL_SPIKE] SELL signal: {symbol} Vol={candle.volume/avg_volume:.1f}x")
                asyncio.create_task(eng.place_order(symbol, "SELL", qty))


def order_flow_strategy(eng, symbol: str, candle, history: List):
    """Order Flow Analysis."""
    if len(history) < 50:
        return

    closes = [c.close for c in history[-50:]]
    current_price = candle.close
    has_position = symbol in eng.positions

    if has_position:
        return

    sma20 = sum(closes[-20:]) / 20
    sma50 = sum(closes[-50:]) / 50

    uptrend = sma20 > sma50 and closes[-1] > sma20
    downtrend = sma20 < sma50 and closes[-1] < sma20

    if not uptrend and not downtrend:
        return

    range_size = candle.high - candle.low
    if range_size == 0:
        return

    close_position = (candle.close - candle.low) / range_size

    avg_vol = sum(c.volume for c in history[-10:]) / 10
    vol_ratio = candle.volume / avg_vol if avg_vol > 0 else 1

    prev_candle = history[-1]
    prev_range = prev_candle.high - prev_candle.low
    prev_close_pos = (prev_candle.close - prev_candle.low) / prev_range if prev_range > 0 else 0.5

    aggressive_buying = close_position > 0.65 and prev_close_pos > 0.55 and vol_ratio > 1.3 and uptrend
    aggressive_selling = close_position < 0.35 and prev_close_pos < 0.45 and vol_ratio > 1.3 and downtrend

    qty = (eng.get_available_balance() * 0.1) / current_price

    if qty > 0.001:
        if aggressive_buying:
            logger.info(f"[ORDER_FLOW] BUY signal: {symbol} ClosePos={close_position:.2f}")
            asyncio.create_task(eng.place_order(symbol, "BUY", qty))
        elif aggressive_selling:
            logger.info(f"[ORDER_FLOW] SELL signal: {symbol} ClosePos={close_position:.2f}")
            asyncio.create_task(eng.place_order(symbol, "SELL", qty))


# =============================================================================
# Main
# =============================================================================

async def main():
    print("=" * 60)
    print("   PAPER TRADING - LIVE MODE")
    print("   Press Ctrl+C to stop")
    print("=" * 60)

    # Symbols to trade
    symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
        "DOGEUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT",
    ]

    # Create trading loop
    loop = LiveTradingLoop(
        symbols=symbols,
        timeframe="1m",
        initial_balance=1000.0,
        leverage=10,
    )

    # Add strategies
    loop.add_strategy(cluster_analysis_strategy)
    loop.add_strategy(mean_reversion_strategy)
    loop.add_strategy(volume_spike_strategy)
    loop.add_strategy(order_flow_strategy)

    print(f"\nSymbols: {', '.join(symbols)}")
    print(f"Strategies: {len(loop.strategies)}")
    print(f"Initial Balance: $1,000")
    print(f"Leverage: 10x")
    print("\nWaiting for candles...\n")

    # Handle shutdown
    def shutdown(sig, frame):
        print("\n\nShutting down...")
        asyncio.create_task(loop.stop())

    signal.signal(signal.SIGINT, shutdown)

    # Status printer
    async def print_status():
        while loop.running:
            await asyncio.sleep(60)  # Print status every minute
            status = loop.get_status()
            balance = status["balance"]
            print(f"\n[STATUS] Candles: {status['candles_processed']} | "
                  f"Signals: {status['signals_generated']} | "
                  f"Positions: {status['positions']} | "
                  f"Trades: {status['trades']} | "
                  f"Balance: ${balance['total']:.2f}")

    # Start
    asyncio.create_task(print_status())
    await loop.start()


if __name__ == "__main__":
    asyncio.run(main())
