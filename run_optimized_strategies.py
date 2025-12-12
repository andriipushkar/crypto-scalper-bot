#!/usr/bin/env python3
"""
Optimized strategies backtest.
Fixes for: Grid Trading, DCA & Grid, Advanced Orderbook, DCA
"""

import asyncio
import sys
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import statistics

sys.path.insert(0, '/home/sssmmmddd/Code/pro/crypto-scalper-bot')

from src.backtesting.data import HistoricalDataLoader, OHLCV
from src.backtesting.engine import BacktestEngine, BacktestConfig, OrderSide


@dataclass
class StrategyResult:
    name: str
    total_trades: int
    win_rate: float
    total_pnl: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    final_balance: float


def close_position(eng: BacktestEngine, symbol: str):
    """Helper to close position by placing opposite order."""
    if symbol not in eng.positions:
        return
    pos = eng.positions[symbol]
    # Place opposite order to close
    if pos.side == "long":
        eng.place_order(symbol, OrderSide.SELL, pos.quantity)
    else:
        eng.place_order(symbol, OrderSide.BUY, pos.quantity)


# ============================================================================
# OPTIMIZED STRATEGIES
# ============================================================================

def create_grid_trading_optimized():
    """
    Grid Trading V2 - OPTIMIZED

    Improvements:
    1. Trend filter - only trade in trend direction
    2. Dynamic grid spacing based on volatility
    3. Maximum loss per grid cycle
    4. Position size limits
    5. Grid reset on trend change
    """
    state = {
        "base_price": None,
        "grid_orders": {},
        "candle_count": 0,
        "last_trend": None,
    }

    def strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
        if len(history) < 50:
            return

        state["candle_count"] += 1
        current_price = candle.close

        # Calculate trend using EMA
        closes = [c.close for c in history[-50:]]
        ema_20 = sum(closes[-20:]) / 20
        ema_50 = sum(closes[-50:]) / 50

        # Determine trend
        trend = "up" if ema_20 > ema_50 else "down"

        # Calculate volatility for dynamic grid
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0.01

        # Dynamic grid size (1-3% based on volatility)
        grid_size = max(0.01, min(0.03, volatility * 2))

        # Reset grid on trend change
        if state["last_trend"] and state["last_trend"] != trend:
            if symbol in eng.positions:
                close_position(eng, symbol)
            state["grid_orders"] = {}
            state["base_price"] = None

        state["last_trend"] = trend

        # Initialize base price
        if state["base_price"] is None:
            state["base_price"] = current_price
            return

        base = state["base_price"]
        has_position = symbol in eng.positions

        # Maximum 5 grid levels
        max_levels = 5

        # Only trade in trend direction
        if trend == "up":
            for i in range(1, max_levels + 1):
                buy_level = base * (1 - i * grid_size)
                sell_level = base * (1 + i * grid_size * 0.5)
                level_key = f"{symbol}_buy_{i}"

                if current_price <= buy_level and level_key not in state["grid_orders"]:
                    size_pct = 0.08 / i
                    qty = (eng.get_available_balance() * size_pct) / current_price
                    if qty > 0:
                        order = eng.place_order(symbol, OrderSide.BUY, qty)
                        if order and order.filled:
                            state["grid_orders"][level_key] = True
                            max_loss_price = base * (1 - (max_levels + 1) * grid_size)
                            eng.set_stop_loss(symbol, max_loss_price)
                            eng.set_take_profit(symbol, sell_level)

                if has_position and current_price >= sell_level:
                    close_position(eng, symbol)
                    state["grid_orders"] = {}
                    state["base_price"] = current_price
                    break

        else:  # Downtrend
            for i in range(1, max_levels + 1):
                sell_level = base * (1 + i * grid_size)
                buy_level = base * (1 - i * grid_size * 0.5)
                level_key = f"{symbol}_sell_{i}"

                if current_price >= sell_level and level_key not in state["grid_orders"]:
                    size_pct = 0.08 / i
                    qty = (eng.get_available_balance() * size_pct) / current_price
                    if qty > 0:
                        order = eng.place_order(symbol, OrderSide.SELL, qty)
                        if order and order.filled:
                            state["grid_orders"][level_key] = True
                            max_loss_price = base * (1 + (max_levels + 1) * grid_size)
                            eng.set_stop_loss(symbol, max_loss_price)
                            eng.set_take_profit(symbol, buy_level)

                if has_position and current_price <= buy_level:
                    close_position(eng, symbol)
                    state["grid_orders"] = {}
                    state["base_price"] = current_price
                    break

        # Update base price periodically
        if state["candle_count"] % 24 == 0:
            if not has_position:
                state["base_price"] = current_price
                state["grid_orders"] = {}

    return strategy


def create_dca_grid_optimized():
    """
    DCA & Grid V2 - OPTIMIZED

    Improvements:
    1. Max 3 DCA levels with decreasing size
    2. Trend confirmation before entry
    3. Time-based exit (max hold period)
    4. Tighter overall stop loss (15% max)
    """
    state = {
        "entry_price": None,
        "dca_count": 0,
        "max_dca": 3,
        "entry_candle": 0,
        "candle_count": 0,
    }

    def strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
        if len(history) < 30:
            return

        state["candle_count"] += 1
        current_price = candle.close
        has_position = symbol in eng.positions

        # Indicators
        closes = [c.close for c in history[-30:]]
        sma_10 = sum(closes[-10:]) / 10
        sma_30 = sum(closes[-30:]) / 30

        # RSI calculation
        gains, losses = [], []
        for i in range(1, 15):
            diff = closes[-i] - closes[-i-1]
            gains.append(max(0, diff))
            losses.append(max(0, -diff))
        avg_gain = sum(gains) / len(gains) if gains else 0.001
        avg_loss = sum(losses) / len(losses) if losses else 0.001
        rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50

        # Trend filter
        uptrend = sma_10 > sma_30

        if not has_position:
            entry_signal = (
                rsi < 35 and
                current_price < sma_10 * 0.99 and
                uptrend
            )

            if entry_signal:
                qty = (eng.get_available_balance() * 0.05) / current_price
                if qty > 0:
                    order = eng.place_order(symbol, OrderSide.BUY, qty)
                    if order and order.filled:
                        state["entry_price"] = current_price
                        state["dca_count"] = 0
                        state["entry_candle"] = state["candle_count"]
                        eng.set_stop_loss(symbol, current_price * 0.85)
                        eng.set_take_profit(symbol, current_price * 1.08)
        else:
            pos = eng.positions[symbol]
            avg_entry = pos.entry_price

            # DCA logic
            if state["dca_count"] < state["max_dca"] and uptrend and state["entry_price"] is not None:
                dca_drops = [0.03, 0.05, 0.08]

                if state["dca_count"] < len(dca_drops):
                    target_drop = sum(dca_drops[:state["dca_count"] + 1])
                    dca_price = state["entry_price"] * (1 - target_drop)

                    if current_price <= dca_price:
                        sizes = [0.04, 0.03, 0.02]
                        size_pct = sizes[state["dca_count"]] if state["dca_count"] < len(sizes) else 0.02

                        qty = (eng.get_available_balance() * size_pct) / current_price
                        if qty > 0:
                            order = eng.place_order(symbol, OrderSide.BUY, qty)
                            if order and order.filled:
                                state["dca_count"] += 1
                                new_avg = pos.entry_price
                                eng.set_take_profit(symbol, new_avg * 1.05)
                                eng.set_stop_loss(symbol, new_avg * 0.88)

            # Time-based exit
            hold_time = state["candle_count"] - state["entry_candle"]
            if state["entry_candle"] > 0 and hold_time > 72 and current_price > avg_entry * 0.98:
                close_position(eng, symbol)
                state["entry_price"] = None

            # Exit if trend reverses strongly
            if not uptrend and current_price < sma_30 * 0.98:
                close_position(eng, symbol)
                state["entry_price"] = None

    return strategy


def create_advanced_orderbook_optimized():
    """
    Advanced Orderbook V2 - OPTIMIZED

    Improvements:
    1. Multiple confirmation signals required
    2. Volume spike confirmation
    3. Momentum filter
    4. Tighter stops
    """
    state = {
        "last_signal": None,
        "signal_candle": 0,
        "candle_count": 0,
    }

    def strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
        if len(history) < 30:
            return

        state["candle_count"] += 1
        current_price = candle.close

        # Support/resistance
        highs = [c.high for c in history[-20:]]
        lows = [c.low for c in history[-20:]]
        closes = [c.close for c in history[-20:]]

        recent_highs = sorted(highs[-10:], reverse=True)[:3]
        recent_lows = sorted(lows[-10:])[:3]
        resistance = sum(recent_highs) / 3
        support = sum(recent_lows) / 3

        # Volume analysis
        volumes = [c.volume for c in history[-20:]]
        avg_vol = sum(volumes) / len(volumes)
        vol_spike = candle.volume > avg_vol * 1.8

        # Momentum
        roc = (current_price - closes[-5]) / closes[-5] * 100

        # Candle patterns
        is_bullish_candle = candle.close > candle.open
        is_bearish_candle = candle.close < candle.open
        candle_body = abs(candle.close - candle.open)
        candle_range = candle.high - candle.low
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        strong_candle = body_ratio > 0.6

        has_position = symbol in eng.positions

        # Signal cooldown
        if state["last_signal"] and (state["candle_count"] - state["signal_candle"]) < 3:
            return

        if not has_position:
            qty = (eng.get_available_balance() * 0.08) / current_price
            if qty <= 0:
                return

            near_support = current_price < support * 1.01
            long_signal = (
                near_support and
                vol_spike and
                is_bullish_candle and
                strong_candle and
                roc > -1
            )

            near_resistance = current_price > resistance * 0.99
            short_signal = (
                near_resistance and
                vol_spike and
                is_bearish_candle and
                strong_candle and
                roc < 1
            )

            if long_signal:
                order = eng.place_order(symbol, OrderSide.BUY, qty)
                if order and order.filled:
                    eng.set_stop_loss(symbol, support * 0.99)
                    target = current_price + (resistance - current_price) * 0.5
                    eng.set_take_profit(symbol, target)
                    state["last_signal"] = "long"
                    state["signal_candle"] = state["candle_count"]

            elif short_signal:
                order = eng.place_order(symbol, OrderSide.SELL, qty)
                if order and order.filled:
                    eng.set_stop_loss(symbol, resistance * 1.01)
                    target = current_price - (current_price - support) * 0.5
                    eng.set_take_profit(symbol, target)
                    state["last_signal"] = "short"
                    state["signal_candle"] = state["candle_count"]

    return strategy


def create_dca_optimized():
    """
    DCA V2 - OPTIMIZED

    Improvements:
    1. Trend-following DCA
    2. RSI filter for entries
    3. Maximum position size limit
    4. Profit-taking at defined levels
    """
    state = {
        "positions_count": 0,
        "max_positions": 5,
        "candle_count": 0,
        "entry_prices": [],
    }

    def strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
        if len(history) < 30:
            return

        state["candle_count"] += 1
        current_price = candle.close

        if state["candle_count"] % 4 != 0:
            return

        closes = [c.close for c in history[-30:]]
        ema_10 = sum(closes[-10:]) / 10
        ema_30 = sum(closes[-30:]) / 30
        uptrend = ema_10 > ema_30

        # RSI
        gains, losses = [], []
        for i in range(1, 15):
            diff = closes[-i] - closes[-i-1]
            gains.append(max(0, diff))
            losses.append(max(0, -diff))
        avg_gain = sum(gains) / len(gains) if gains else 0.001
        avg_loss = sum(losses) / len(losses) if losses else 0.001
        rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50

        has_position = symbol in eng.positions

        if not has_position:
            if uptrend and 35 < rsi < 50:
                qty = (eng.get_available_balance() * 0.06) / current_price
                if qty > 0:
                    order = eng.place_order(symbol, OrderSide.BUY, qty)
                    if order and order.filled:
                        state["positions_count"] = 1
                        state["entry_prices"] = [current_price]
                        eng.set_stop_loss(symbol, current_price * 0.92)
                        eng.set_take_profit(symbol, current_price * 1.06)
        else:
            pos = eng.positions[symbol]
            avg_entry = pos.entry_price

            # DCA
            if (state["positions_count"] > 0 and
                state["positions_count"] < state["max_positions"] and
                uptrend and
                state["entry_prices"] and
                len(state["entry_prices"]) > 0 and
                current_price < state["entry_prices"][-1] * 0.98 and
                rsi < 40):

                size_pct = 0.04 / state["positions_count"]
                qty = (eng.get_available_balance() * size_pct) / current_price

                if qty > 0:
                    order = eng.place_order(symbol, OrderSide.BUY, qty)
                    if order and order.filled:
                        state["positions_count"] += 1
                        state["entry_prices"].append(current_price)
                        new_avg = pos.entry_price
                        eng.set_stop_loss(symbol, new_avg * 0.90)
                        eng.set_take_profit(symbol, new_avg * 1.05)

            # Take profit
            if current_price > avg_entry * 1.04:
                close_position(eng, symbol)
                state["positions_count"] = 0
                state["entry_prices"] = []

            # Emergency exit
            if not uptrend and current_price < ema_30 * 0.98:
                close_position(eng, symbol)
                state["positions_count"] = 0
                state["entry_prices"] = []

    return strategy


# ============================================================================
# Main Runner
# ============================================================================

STRATEGIES = {
    "grid_v2": ("📐 Grid Trading V2", create_grid_trading_optimized()),
    "dca_grid_v2": ("💰 DCA & Grid V2", create_dca_grid_optimized()),
    "orderbook_v2": ("📊 Advanced Orderbook V2", create_advanced_orderbook_optimized()),
    "dca_v2": ("💰 DCA V2", create_dca_optimized()),
}


async def run_backtest(name: str, display_name: str, strategy_func, data, config) -> StrategyResult:
    print(f"\n{'='*60}")
    print(f"Running: {display_name}")
    print(f"{'='*60}")

    engine = BacktestEngine(config)
    result = engine.run(strategy=strategy_func, data=data, warmup_period=50)

    return StrategyResult(
        name=display_name,
        total_trades=result.total_trades,
        win_rate=round(result.win_rate, 2),
        total_pnl=round(result.total_return, 2),
        total_return_pct=round(result.total_return_pct, 2),
        max_drawdown_pct=round(result.max_drawdown_pct, 2),
        sharpe_ratio=round(result.sharpe_ratio, 2),
        profit_factor=round(result.profit_factor, 2) if result.profit_factor != float('inf') else 999,
        final_balance=round(result.final_balance, 2)
    )


async def main():
    print("=" * 70)
    print("   OPTIMIZED STRATEGIES BACKTEST")
    print("   Period: 2025-01-01 to 2025-12-12")
    print("=" * 70)

    symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
        "DOGEUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT",
        "DOTUSDT", "LINKUSDT", "UNIUSDT"
    ]

    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 12)

    print("\n📥 Loading cached data...")
    data_loader = HistoricalDataLoader()
    all_data = {}

    for symbol in symbols:
        try:
            candles = await data_loader.get_data(
                symbol=symbol, timeframe="1h",
                start_time=start_date, end_time=end_date,
                use_cache=True
            )
            if candles:
                all_data[symbol] = candles
                print(f"   ✓ {symbol}: {len(candles)} candles")
        except Exception as e:
            print(f"   ✗ {symbol}: {e}")

    config = BacktestConfig(
        initial_balance=10000.0,
        commission_rate=0.0004,
        slippage=0.0001,
        leverage=10,
    )

    results = []

    # Original strategies for comparison
    from run_all_strategies_backtest import (
        create_grid_trading_strategy,
        create_dca_grid_strategy,
        create_advanced_orderbook_strategy,
        create_dca_strategy,
    )

    original_strategies = {
        "grid_v1": ("📐 Grid V1 (OLD)", create_grid_trading_strategy()),
        "dca_grid_v1": ("💰 DCA&Grid V1 (OLD)", create_dca_grid_strategy()),
        "orderbook_v1": ("📊 Orderbook V1 (OLD)", create_advanced_orderbook_strategy()),
        "dca_v1": ("💰 DCA V1 (OLD)", create_dca_strategy()),
    }

    # Run original strategies
    print("\n" + "=" * 70)
    print("   ORIGINAL STRATEGIES (baseline)")
    print("=" * 70)

    for key, (display_name, strategy_func) in original_strategies.items():
        try:
            result = await run_backtest(key, display_name, strategy_func, all_data, config)
            results.append(result)
            print(f"   Trades: {result.total_trades} | Win: {result.win_rate}% | P&L: ${result.total_pnl} | MaxDD: {result.max_drawdown_pct}%")
        except Exception as e:
            print(f"   Error: {e}")

    # Run optimized strategies
    print("\n" + "=" * 70)
    print("   OPTIMIZED STRATEGIES (new)")
    print("=" * 70)

    for key, (display_name, strategy_func) in STRATEGIES.items():
        try:
            result = await run_backtest(key, display_name, strategy_func, all_data, config)
            results.append(result)
            print(f"   Trades: {result.total_trades} | Win: {result.win_rate}% | P&L: ${result.total_pnl} | MaxDD: {result.max_drawdown_pct}%")
        except Exception as e:
            print(f"   Error: {e}")

    # Summary
    print("\n")
    print("=" * 110)
    print("   COMPARISON: ORIGINAL vs OPTIMIZED")
    print("=" * 110)
    print(f"{'Strategy':<30} {'Trades':>8} {'Win%':>8} {'P&L':>12} {'Return%':>10} {'MaxDD%':>8} {'Sharpe':>8}")
    print("-" * 110)

    for r in results:
        color = "✓" if r.total_return_pct > 0 else "✗"
        print(f"{color} {r.name:<28} {r.total_trades:>8} {r.win_rate:>7.1f}% ${r.total_pnl:>10.2f} {r.total_return_pct:>9.1f}% {r.max_drawdown_pct:>7.1f}% {r.sharpe_ratio:>7.2f}")

    print("-" * 110)

    # Calculate improvements
    print("\n📊 IMPROVEMENTS SUMMARY:")

    pairs = [
        ("Grid", "Grid V1", "Grid Trading V2"),
        ("DCA&Grid", "DCA&Grid V1", "DCA & Grid V2"),
        ("Orderbook", "Orderbook V1", "Advanced Orderbook V2"),
        ("DCA", "DCA V1", "DCA V2"),
    ]

    for name, old_pattern, new_pattern in pairs:
        old = next((r for r in results if old_pattern in r.name), None)
        new = next((r for r in results if new_pattern in r.name), None)

        if old and new:
            pnl_diff = new.total_pnl - old.total_pnl
            dd_improvement = old.max_drawdown_pct - new.max_drawdown_pct
            print(f"\n   {name}:")
            print(f"      P&L: ${old.total_pnl:.0f} → ${new.total_pnl:.0f} ({'+' if pnl_diff > 0 else ''}{pnl_diff:.0f})")
            print(f"      MaxDD: {old.max_drawdown_pct:.1f}% → {new.max_drawdown_pct:.1f}% (improved by {dd_improvement:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
