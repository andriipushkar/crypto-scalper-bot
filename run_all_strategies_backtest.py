#!/usr/bin/env python3
"""
Run backtests for all trading strategies.
Downloads real historical data from Binance and tests each strategy.
"""

import asyncio
import sys
from datetime import datetime
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
import statistics

# Add src to path
sys.path.insert(0, '/home/sssmmmddd/Code/pro/crypto-scalper-bot')

from src.backtesting.data import HistoricalDataLoader, OHLCV
from src.backtesting.engine import BacktestEngine, BacktestConfig, OrderSide


@dataclass
class StrategyResult:
    """Result of a strategy backtest."""
    name: str
    total_trades: int
    win_rate: float
    total_pnl: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    final_balance: float


# ============================================================================
# Helper Functions for Strategy Improvements
# ============================================================================

INITIAL_BALANCE = 10000.0  # For drawdown calculation

def get_dynamic_position_size(eng: BacktestEngine, base_pct: float = 0.1) -> float:
    """Dynamic position sizing based on current drawdown."""
    current_balance = eng.get_available_balance()
    # Calculate drawdown from peak (approximate using initial balance)
    peak = max(INITIAL_BALANCE, current_balance)
    drawdown_pct = (peak - current_balance) / peak * 100 if peak > 0 else 0

    # Reduce position size during drawdown
    if drawdown_pct > 30:
        return base_pct * 0.3  # 3% instead of 10%
    elif drawdown_pct > 20:
        return base_pct * 0.5  # 5% instead of 10%
    elif drawdown_pct > 10:
        return base_pct * 0.7  # 7% instead of 10%
    return base_pct


def apply_progressive_trailing(eng: BacktestEngine, symbol: str, current_price: float):
    """Progressive trailing stop - trail 50% of profit."""
    if symbol not in eng.positions:
        return

    pos = eng.positions[symbol]
    entry_price = pos.entry_price

    if pos.side == OrderSide.BUY:
        profit_pct = (current_price - entry_price) / entry_price * 100
        if profit_pct > 0.5:  # Start trailing after 0.5% profit
            # Trail at 50% of current profit
            trail_pct = profit_pct * 0.5
            new_sl = entry_price * (1 + trail_pct / 100)
            if new_sl > (pos.stop_loss or 0):
                eng.set_stop_loss(symbol, new_sl)
    else:  # SHORT
        profit_pct = (entry_price - current_price) / entry_price * 100
        if profit_pct > 0.5:
            trail_pct = profit_pct * 0.5
            new_sl = entry_price * (1 - trail_pct / 100)
            if pos.stop_loss is None or new_sl < pos.stop_loss:
                eng.set_stop_loss(symbol, new_sl)


def is_low_liquidity_hour(candle: OHLCV) -> bool:
    """Check if current hour is low liquidity (00:00-04:00 UTC)."""
    hour = candle.timestamp.hour
    return 0 <= hour < 4


def calculate_atr(history: List, period: int = 14) -> float:
    """Calculate Average True Range."""
    if len(history) < period + 1:
        return 0

    tr_list = []
    for i in range(1, period + 1):
        high = history[-i].high
        low = history[-i].low
        prev_close = history[-i-1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)

    return sum(tr_list) / len(tr_list) if tr_list else 0


# ============================================================================
# Strategy Implementations
# ============================================================================

def create_volume_spike_strategy(volume_multiplier: float = 3.0):
    """Volume Spike strategy - enter on volume spikes."""
    def strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
        if len(history) < 20:
            return

        # Calculate average volume
        volumes = [c.volume for c in history[-20:]]
        avg_volume = sum(volumes) / len(volumes)

        # Price movement
        price_change = (candle.close - candle.open) / candle.open

        has_position = symbol in eng.positions
        current_price = candle.close

        # Volume spike detection
        is_volume_spike = candle.volume > avg_volume * volume_multiplier

        if not has_position and is_volume_spike:
            qty = (eng.get_available_balance() * 0.1) / current_price
            if qty > 0:
                if price_change > 0.001:  # Bullish volume spike
                    order = eng.place_order(symbol, OrderSide.BUY, qty)
                    if order and order.filled:
                        eng.set_stop_loss(symbol, current_price * 0.99)
                        eng.set_take_profit(symbol, current_price * 1.02)
                elif price_change < -0.001:  # Bearish volume spike
                    order = eng.place_order(symbol, OrderSide.SELL, qty)
                    if order and order.filled:
                        eng.set_stop_loss(symbol, current_price * 1.01)
                        eng.set_take_profit(symbol, current_price * 0.98)

    return strategy


def create_mean_reversion_strategy(lookback: int = 20, std_multiplier: float = 2.0):
    """Mean Reversion strategy - trade against extreme moves."""
    def strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
        if len(history) < lookback + 5:
            return

        closes = [c.close for c in history[-lookback:]]
        mean = sum(closes) / len(closes)
        std = statistics.stdev(closes) if len(closes) > 1 else 0.001

        z_score = (candle.close - mean) / std if std > 0 else 0

        has_position = symbol in eng.positions
        current_price = candle.close

        if not has_position:
            qty = (eng.get_available_balance() * 0.1) / current_price
            if qty > 0:
                if z_score < -std_multiplier:  # Oversold - buy
                    order = eng.place_order(symbol, OrderSide.BUY, qty)
                    if order and order.filled:
                        eng.set_stop_loss(symbol, current_price * 0.985)
                        eng.set_take_profit(symbol, mean)  # Target mean
                elif z_score > std_multiplier:  # Overbought - sell
                    order = eng.place_order(symbol, OrderSide.SELL, qty)
                    if order and order.filled:
                        eng.set_stop_loss(symbol, current_price * 1.015)
                        eng.set_take_profit(symbol, mean)

    return strategy


def create_hybrid_scalping_strategy():
    """Hybrid Scalping V3 - trend + RSI + 2-candle confirmation."""
    def strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
        if len(history) < 50:
            return

        closes = [c.close for c in history[-50:]]
        volumes = [c.volume for c in history[-50:]]
        current_price = candle.close
        has_position = symbol in eng.positions

        if has_position:
            return

        # === TREND FILTER ===
        sma_20 = sum(closes[-20:]) / 20
        sma_50 = sum(closes[-50:]) / 50
        uptrend = sma_20 > sma_50 and closes[-1] > sma_20
        downtrend = sma_20 < sma_50 and closes[-1] < sma_20

        if not uptrend and not downtrend:
            return

        # === RSI ===
        gains, losses = [], []
        for i in range(1, 15):
            diff = closes[-i] - closes[-i-1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(diff))

        avg_gain = sum(gains) / len(gains) if gains else 0.001
        avg_loss = sum(losses) / len(losses) if losses else 0.001
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))

        # === VOLUME & CANDLE ===
        avg_vol = sum(volumes[-20:]) / 20
        good_volume = candle.volume > avg_vol * 1.2
        bullish_candle = candle.close > candle.open
        bearish_candle = candle.close < candle.open
        prev = history[-1]
        prev_bullish = prev.close > prev.open
        prev_bearish = prev.close < prev.open

        qty = (eng.get_available_balance() * 0.1) / current_price

        if good_volume and qty > 0:
            if uptrend and rsi < 60 and bullish_candle and prev_bullish:
                order = eng.place_order(symbol, OrderSide.BUY, qty)
                if order and order.filled:
                    eng.set_stop_loss(symbol, current_price * 0.988)
                    eng.set_take_profit(symbol, current_price * 1.03)
            elif downtrend and rsi > 40 and bearish_candle and prev_bearish:
                order = eng.place_order(symbol, OrderSide.SELL, qty)
                if order and order.filled:
                    eng.set_stop_loss(symbol, current_price * 1.012)
                    eng.set_take_profit(symbol, current_price * 0.97)

    return strategy


def create_impulse_scalping_strategy():
    """Impulse Scalping V11 - with all improvements."""

    def strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
        if len(history) < 50:
            return

        if is_low_liquidity_hour(candle):
            return

        closes = [c.close for c in history[-50:]]
        volumes = [c.volume for c in history[-50:]]
        current_price = candle.close
        has_position = symbol in eng.positions

        if has_position:
            apply_progressive_trailing(eng, symbol, current_price)
            return

        # === TREND FILTER ===
        sma20 = sum(closes[-20:]) / 20
        sma50 = sum(closes[-50:]) / 50
        uptrend = sma20 > sma50 and current_price > sma20
        downtrend = sma20 < sma50 and current_price < sma20

        if not uptrend and not downtrend:
            return

        # === EMA for MACD ===
        def ema(data, period):
            multiplier = 2 / (period + 1)
            ema_val = data[0]
            for price in data[1:]:
                ema_val = (price - ema_val) * multiplier + ema_val
            return ema_val

        ema_12 = ema(closes[-26:], 12)
        ema_26 = ema(closes[-26:], 26)
        macd_line = ema_12 - ema_26

        prev_closes = closes[:-1]
        prev_ema_12 = ema(prev_closes[-26:], 12)
        prev_ema_26 = ema(prev_closes[-26:], 26)
        prev_macd = prev_ema_12 - prev_ema_26

        macd_bullish = macd_line > 0 and prev_macd <= 0
        macd_bearish = macd_line < 0 and prev_macd >= 0

        # === VOLUME & CANDLE ===
        avg_volume = sum(volumes[-20:]) / 20
        good_volume = candle.volume > avg_volume * 1.2
        bullish_candle = candle.close > candle.open
        bearish_candle = candle.close < candle.open

        # === DYNAMIC SIZE + ATR SL/TP ===
        pos_pct = get_dynamic_position_size(eng, 0.1)
        qty = (eng.get_available_balance() * pos_pct) / current_price
        atr = calculate_atr(history, 14)
        sl_distance = atr * 1.5
        tp_distance = atr * 3.0

        if good_volume and qty > 0:
            if macd_bullish and uptrend and bullish_candle:
                order = eng.place_order(symbol, OrderSide.BUY, qty)
                if order and order.filled:
                    eng.set_stop_loss(symbol, current_price - sl_distance)
                    eng.set_take_profit(symbol, current_price + tp_distance)
            elif macd_bearish and downtrend and bearish_candle:
                order = eng.place_order(symbol, OrderSide.SELL, qty)
                if order and order.filled:
                    eng.set_stop_loss(symbol, current_price + sl_distance)
                    eng.set_take_profit(symbol, current_price - tp_distance)

    return strategy


def create_orderbook_imbalance_strategy(imbalance_threshold: float = 2.0):
    """Orderbook Imbalance V2 - trend filter + 2-candle confirmation."""
    def strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
        if len(history) < 50:
            return

        closes = [c.close for c in history[-50:]]
        current_price = candle.close
        has_position = symbol in eng.positions

        if has_position:
            return

        # === TREND FILTER ===
        sma_20 = sum(closes[-20:]) / 20
        sma_50 = sum(closes[-50:]) / 50
        uptrend = sma_20 > sma_50 and closes[-1] > sma_20
        downtrend = sma_20 < sma_50 and closes[-1] < sma_20

        if not uptrend and not downtrend:
            return

        # === VOLUME IMBALANCE ===
        buy_vol = candle.volume if candle.close > candle.open else candle.volume * 0.3
        sell_vol = candle.volume if candle.close < candle.open else candle.volume * 0.3
        imbalance = buy_vol / sell_vol if sell_vol > 0 else 2

        # === VOLUME CONFIRMATION ===
        avg_vol = sum(c.volume for c in history[-10:]) / 10
        good_volume = candle.volume > avg_vol * 1.3

        # === CANDLE CONFIRMATION ===
        prev = history[-1]
        prev_bullish = prev.close > prev.open
        prev_bearish = prev.close < prev.open
        curr_bullish = candle.close > candle.open
        curr_bearish = candle.close < candle.open

        qty = (eng.get_available_balance() * 0.1) / current_price

        if good_volume and qty > 0:
            if uptrend and imbalance > imbalance_threshold and curr_bullish and prev_bullish:
                order = eng.place_order(symbol, OrderSide.BUY, qty)
                if order and order.filled:
                    eng.set_stop_loss(symbol, current_price * 0.988)
                    eng.set_take_profit(symbol, current_price * 1.03)
            elif downtrend and imbalance < 1 / imbalance_threshold and curr_bearish and prev_bearish:
                order = eng.place_order(symbol, OrderSide.SELL, qty)
                if order and order.filled:
                    eng.set_stop_loss(symbol, current_price * 1.012)
                    eng.set_take_profit(symbol, current_price * 0.97)

    return strategy


def create_advanced_orderbook_strategy():
    """Advanced Orderbook V8 - with all improvements + wider pullback zone."""
    def strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
        if len(history) < 50:
            return

        if is_low_liquidity_hour(candle):
            return

        closes = [c.close for c in history[-50:]]
        volumes = [c.volume for c in history[-50:]]
        current_price = candle.close
        has_position = symbol in eng.positions

        if has_position:
            apply_progressive_trailing(eng, symbol, current_price)
            return

        # === TREND FILTER ===
        sma20 = sum(closes[-20:]) / 20
        sma50 = sum(closes[-50:]) / 50
        uptrend = sma20 > sma50
        downtrend = sma20 < sma50

        if not uptrend and not downtrend:
            return

        # === PULLBACK TO SMA20 (wider zone: 1%) ===
        near_sma20 = abs(candle.close - sma20) / sma20 < 0.01

        # === CANDLE CONFIRMATION ===
        bullish_candle = candle.close > candle.open
        bearish_candle = candle.close < candle.open
        prev = history[-1]
        prev_bullish = prev.close > prev.open
        prev_bearish = prev.close < prev.open

        # === VOLUME ===
        avg_vol = sum(volumes[-20:]) / 20
        good_volume = candle.volume > avg_vol * 1.2

        # === DYNAMIC POSITION SIZE ===
        pos_pct = get_dynamic_position_size(eng, 0.1)
        qty = (eng.get_available_balance() * pos_pct) / current_price

        # === ATR-based SL/TP ===
        atr = calculate_atr(history, 14)
        sl_distance = atr * 1.5
        tp_distance = atr * 3.5  # Higher R:R for pullback

        if good_volume and qty > 0:
            if uptrend and near_sma20 and bullish_candle and prev_bullish:
                order = eng.place_order(symbol, OrderSide.BUY, qty)
                if order and order.filled:
                    eng.set_stop_loss(symbol, current_price - sl_distance)
                    eng.set_take_profit(symbol, current_price + tp_distance)
            elif downtrend and near_sma20 and bearish_candle and prev_bearish:
                order = eng.place_order(symbol, OrderSide.SELL, qty)
                if order and order.filled:
                    eng.set_stop_loss(symbol, current_price + sl_distance)
                    eng.set_take_profit(symbol, current_price - tp_distance)

    return strategy


def create_print_tape_strategy():
    """Print Tape - whale trade detection simulation."""
    def strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
        if len(history) < 15:
            return

        # Simulate CVD (Cumulative Volume Delta)
        cvd = 0
        for c in history[-15:]:
            delta = c.volume if c.close > c.open else -c.volume
            cvd += delta

        # Whale detection - unusually large candle
        avg_range = sum(abs(c.high - c.low) for c in history[-10:]) / 10
        current_range = abs(candle.high - candle.low)
        is_whale_candle = current_range > avg_range * 2

        has_position = symbol in eng.positions
        current_price = candle.close

        if not has_position and is_whale_candle:
            qty = (eng.get_available_balance() * 0.1) / current_price
            if qty > 0:
                if cvd > 0 and candle.close > candle.open:  # Positive CVD + bullish
                    order = eng.place_order(symbol, OrderSide.BUY, qty)
                    if order and order.filled:
                        eng.set_stop_loss(symbol, current_price * 0.99)
                        eng.set_take_profit(symbol, current_price * 1.018)
                elif cvd < 0 and candle.close < candle.open:  # Negative CVD + bearish
                    order = eng.place_order(symbol, OrderSide.SELL, qty)
                    if order and order.filled:
                        eng.set_stop_loss(symbol, current_price * 1.01)
                        eng.set_take_profit(symbol, current_price * 0.982)

    return strategy


def create_cluster_analysis_strategy():
    """Cluster Analysis - POC and Value Area simulation."""
    def strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
        if len(history) < 20:
            return

        # Simulate POC (Point of Control) - price with most volume
        # Use typical price weighted by volume
        volume_prices = []
        for c in history[-20:]:
            typical_price = (c.high + c.low + c.close) / 3
            volume_prices.append((typical_price, c.volume))

        total_vol = sum(v for _, v in volume_prices)
        poc = sum(p * v for p, v in volume_prices) / total_vol if total_vol > 0 else candle.close

        # Value Area (70% of volume)
        closes = sorted([c.close for c in history[-20:]])
        va_low = closes[3]  # ~15%
        va_high = closes[16]  # ~85%

        has_position = symbol in eng.positions
        current_price = candle.close

        # Trade based on POC and Value Area
        if not has_position:
            qty = (eng.get_available_balance() * 0.1) / current_price
            if qty > 0:
                if current_price < va_low and current_price < poc * 0.99:  # Below value area
                    order = eng.place_order(symbol, OrderSide.BUY, qty)
                    if order and order.filled:
                        eng.set_stop_loss(symbol, current_price * 0.985)
                        eng.set_take_profit(symbol, poc)
                elif current_price > va_high and current_price > poc * 1.01:  # Above value area
                    order = eng.place_order(symbol, OrderSide.SELL, qty)
                    if order and order.filled:
                        eng.set_stop_loss(symbol, current_price * 1.015)
                        eng.set_take_profit(symbol, poc)

    return strategy


def create_dca_strategy(interval: int = 5):
    """DCA Strategy - Dollar Cost Averaging."""
    candle_count = {"count": 0}

    def strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
        if len(history) < 20:
            return

        candle_count["count"] += 1

        # DCA every N candles
        if candle_count["count"] % interval != 0:
            return

        # Trend detection for direction
        sma = sum(c.close for c in history[-20:]) / 20
        current_price = candle.close

        has_position = symbol in eng.positions

        qty = (eng.get_available_balance() * 0.05) / current_price  # Smaller position for DCA
        if qty > 0:
            if not has_position:
                if current_price < sma:  # Below average - start accumulating
                    order = eng.place_order(symbol, OrderSide.BUY, qty)
                    if order and order.filled:
                        eng.set_stop_loss(symbol, current_price * 0.95)  # Wider SL for DCA
                        eng.set_take_profit(symbol, sma * 1.02)
            else:
                # Add to position if price dropped more
                pos = eng.positions[symbol]
                if current_price < pos.entry_price * 0.98:  # 2% below entry
                    order = eng.place_order(symbol, OrderSide.BUY, qty * 0.5)

    return strategy


def create_grid_trading_strategy(grid_size: float = 0.01, levels: int = 5):
    """Grid Trading Strategy."""
    grid_state = {"base_price": None, "orders": {}}

    def strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
        if len(history) < 10:
            return

        current_price = candle.close

        # Initialize grid base price
        if grid_state["base_price"] is None:
            grid_state["base_price"] = current_price
            return

        base = grid_state["base_price"]
        has_position = symbol in eng.positions

        # Calculate grid levels
        for i in range(-levels, levels + 1):
            if i == 0:
                continue

            grid_price = base * (1 + i * grid_size)
            level_key = f"{symbol}_{i}"

            # Buy at lower levels, sell at higher
            if i < 0 and current_price <= grid_price:
                if level_key not in grid_state["orders"]:
                    qty = (eng.get_available_balance() * 0.05) / current_price
                    if qty > 0:
                        order = eng.place_order(symbol, OrderSide.BUY, qty)
                        if order and order.filled:
                            grid_state["orders"][level_key] = True
                            eng.set_take_profit(symbol, base)  # Target base price

            elif i > 0 and current_price >= grid_price and has_position:
                # Close position at profit
                eng.close_position(symbol)
                grid_state["orders"] = {}  # Reset
                break

    return strategy


def create_dca_grid_strategy():
    """Combined DCA + Grid strategy."""
    state = {"entry_price": None, "dca_count": 0, "max_dca": 3}

    def strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
        if len(history) < 20:
            return

        current_price = candle.close
        has_position = symbol in eng.positions

        # Trend filter
        sma_20 = sum(c.close for c in history[-20:]) / 20

        if not has_position:
            # Initial entry
            if current_price < sma_20 * 0.99:  # Below SMA
                qty = (eng.get_available_balance() * 0.1) / current_price
                if qty > 0:
                    order = eng.place_order(symbol, OrderSide.BUY, qty)
                    if order and order.filled:
                        state["entry_price"] = current_price
                        state["dca_count"] = 0
                        eng.set_stop_loss(symbol, current_price * 0.92)  # Wide SL
                        eng.set_take_profit(symbol, current_price * 1.05)
        else:
            # DCA if price drops
            if state["dca_count"] < state["max_dca"]:
                drop_level = 0.02 * (state["dca_count"] + 1)  # 2%, 4%, 6%
                if current_price < state["entry_price"] * (1 - drop_level):
                    qty = (eng.get_available_balance() * 0.1) / current_price
                    if qty > 0:
                        order = eng.place_order(symbol, OrderSide.BUY, qty)
                        if order and order.filled:
                            state["dca_count"] += 1
                            # Update average entry
                            pos = eng.positions[symbol]
                            avg_entry = pos.entry_price
                            eng.set_take_profit(symbol, avg_entry * 1.03)

    return strategy


def create_order_flow_strategy():
    """Order Flow V2.1 - trend filter + order flow analysis."""
    def strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
        if len(history) < 50:
            return

        closes = [c.close for c in history[-50:]]
        current_price = candle.close
        has_position = symbol in eng.positions

        if has_position:
            return

        # === TREND FILTER (SMA20 vs SMA50) ===
        sma20 = sum(closes[-20:]) / 20
        sma50 = sum(closes[-50:]) / 50

        uptrend = sma20 > sma50 and closes[-1] > sma20
        downtrend = sma20 < sma50 and closes[-1] < sma20

        if not uptrend and not downtrend:
            return

        # === ORDER FLOW ANALYSIS ===
        range_size = candle.high - candle.low
        if range_size == 0:
            return

        close_position = (candle.close - candle.low) / range_size

        avg_vol = sum(c.volume for c in history[-10:]) / 10
        vol_ratio = candle.volume / avg_vol if avg_vol > 0 else 1

        prev_candle = history[-1]
        prev_range = prev_candle.high - prev_candle.low
        prev_close_pos = (prev_candle.close - prev_candle.low) / prev_range if prev_range > 0 else 0.5

        aggressive_buying = (
            close_position > 0.65 and prev_close_pos > 0.55 and
            vol_ratio > 1.3 and uptrend
        )
        aggressive_selling = (
            close_position < 0.35 and prev_close_pos < 0.45 and
            vol_ratio > 1.3 and downtrend
        )

        qty = (eng.get_available_balance() * 0.1) / current_price

        if qty > 0:
            if aggressive_buying:
                order = eng.place_order(symbol, OrderSide.BUY, qty)
                if order and order.filled:
                    eng.set_stop_loss(symbol, current_price * 0.988)
                    eng.set_take_profit(symbol, current_price * 1.03)
            elif aggressive_selling:
                order = eng.place_order(symbol, OrderSide.SELL, qty)
                if order and order.filled:
                    eng.set_stop_loss(symbol, current_price * 1.012)
                    eng.set_take_profit(symbol, current_price * 0.97)

    return strategy


# ============================================================================
# Main Backtest Runner
# ============================================================================

STRATEGIES = {
    "volume_spike": ("📢 Volume Spike", create_volume_spike_strategy()),
    "hybrid_scalping": ("🎯 Hybrid Scalping", create_hybrid_scalping_strategy()),
    "impulse_scalping": ("⚡ Impulse Scalping", create_impulse_scalping_strategy()),
    "order_flow": ("📊 Order Flow", create_order_flow_strategy()),
    "orderbook_imbalance": ("📈 Orderbook Imbalance", create_orderbook_imbalance_strategy()),
    "advanced_orderbook": ("📊 Advanced Orderbook", create_advanced_orderbook_strategy()),
    "print_tape": ("📜 Print Tape", create_print_tape_strategy()),
    "cluster_analysis": ("🔬 Cluster Analysis", create_cluster_analysis_strategy()),
    # REMOVED: dca_grid (-96.4%), grid_trading (-191.2%)
    "mean_reversion": ("🔄 Mean Reversion", create_mean_reversion_strategy()),
}


async def run_strategy_backtest(
    name: str,
    display_name: str,
    strategy_func: Callable,
    data: Dict[str, List[OHLCV]],
    config: BacktestConfig
) -> StrategyResult:
    """Run backtest for a single strategy."""
    print(f"\n{'='*60}")
    print(f"Running: {display_name}")
    print(f"{'='*60}")

    engine = BacktestEngine(config)

    result = engine.run(
        strategy=strategy_func,
        data=data,
        warmup_period=50
    )

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
    print("   MULTI-STRATEGY BACKTEST - TOP 50 COINS")
    print("   Period: 2025-01-01 to 2025-12-01")
    print("   Initial Balance: $10,000")
    print("=" * 70)

    # Configuration - TOP 50 Binance Futures coins by volume
    symbols = [
        # Top 10
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
        # 11-20
        "MATICUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "XLMUSDT",
        "ETCUSDT", "FILUSDT", "TRXUSDT", "NEARUSDT", "ICPUSDT",
        # 21-30
        "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "LDOUSDT",
        "AAVEUSDT", "MKRUSDT", "SNXUSDT", "CRVUSDT", "RUNEUSDT",
        # 31-40
        "GRTUSDT", "ALGOUSDT", "FTMUSDT", "SANDUSDT", "MANAUSDT",
        "AXSUSDT", "THETAUSDT", "EGLDUSDT", "EOSUSDT", "XTZUSDT",
        # 41-50
        "FLOWUSDT", "CHZUSDT", "APEUSDT", "IMXUSDT", "GMXUSDT",
        "STXUSDT", "CFXUSDT", "AGIXUSDT", "FETUSDT", "OCEANUSDT",
    ]

    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 1)

    # Download data once
    print("\n📥 Downloading historical data...")
    data_loader = HistoricalDataLoader()
    all_data: Dict[str, List[OHLCV]] = {}

    for symbol in symbols:
        print(f"   Downloading {symbol}...", end=" ", flush=True)
        try:
            candles = await data_loader.get_data(
                symbol=symbol,
                timeframe="1h",
                start_time=start_date,
                end_time=end_date,
                use_cache=True
            )
            if candles:
                all_data[symbol] = candles
                print(f"✓ {len(candles)} candles")
        except Exception as e:
            print(f"✗ Error: {e}")

    print(f"\n✓ Downloaded data for {len(all_data)} symbols")

    # Backtest config
    config = BacktestConfig(
        initial_balance=10000.0,
        commission_rate=0.0004,
        slippage=0.0001,
        leverage=10,
    )

    # Run all strategies
    results: List[StrategyResult] = []

    for key, (display_name, strategy_func) in STRATEGIES.items():
        try:
            result = await run_strategy_backtest(
                name=key,
                display_name=display_name,
                strategy_func=strategy_func,
                data=all_data,
                config=config
            )
            results.append(result)

            print(f"   Trades: {result.total_trades}")
            print(f"   Win Rate: {result.win_rate}%")
            print(f"   P&L: ${result.total_pnl} ({result.total_return_pct}%)")
            print(f"   Max DD: {result.max_drawdown_pct}%")
            print(f"   Sharpe: {result.sharpe_ratio}")

        except Exception as e:
            print(f"   ✗ Error: {e}")

    # Summary
    print("\n")
    print("=" * 100)
    print("   SUMMARY - ALL STRATEGIES")
    print("=" * 100)
    print(f"{'Strategy':<25} {'Trades':>8} {'Win%':>8} {'P&L':>12} {'Return%':>10} {'MaxDD%':>8} {'Sharpe':>8} {'PF':>6}")
    print("-" * 100)

    # Sort by return
    results.sort(key=lambda x: x.total_return_pct, reverse=True)

    for r in results:
        print(f"{r.name:<25} {r.total_trades:>8} {r.win_rate:>7.1f}% ${r.total_pnl:>10.2f} {r.total_return_pct:>9.1f}% {r.max_drawdown_pct:>7.1f}% {r.sharpe_ratio:>7.2f} {r.profit_factor:>6.2f}")

    print("-" * 100)

    # Best performers
    if results:
        best_return = max(results, key=lambda x: x.total_return_pct)
        best_sharpe = max(results, key=lambda x: x.sharpe_ratio)
        best_winrate = max(results, key=lambda x: x.win_rate)

        print(f"\n🏆 BEST PERFORMERS:")
        print(f"   Best Return: {best_return.name} ({best_return.total_return_pct}%)")
        print(f"   Best Sharpe: {best_sharpe.name} ({best_sharpe.sharpe_ratio})")
        print(f"   Best Win Rate: {best_winrate.name} ({best_winrate.win_rate}%)")


if __name__ == "__main__":
    asyncio.run(main())
