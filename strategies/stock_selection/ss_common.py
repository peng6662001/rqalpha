from __future__ import annotations

from datetime import datetime, time, timedelta

import numpy as np

from rqalpha.apis import (
    all_instruments,
    history_bars,
    is_st_stock,
    is_suspended,
    order_target_percent,
    update_universe,
)
from rqalpha.environment import Environment


POOL_SIZE = 300
BENCHMARK = "000300.XSHG"


def build_stock_pool(pool_size, min_listed_days=120, liquidity_lookback=20):
    trading_dt = Environment.get_instance().trading_dt
    cutoff_date = trading_dt.date() - timedelta(days=min_listed_days)
    universe = all_instruments("CS")
    if universe.empty:
        return []

    universe = universe[
        universe["listed_date"].map(
            lambda d: (d.date() if hasattr(d, "date") else d) <= cutoff_date
        )
    ]
    if "exchange" in universe.columns:
        universe = universe[universe["exchange"].isin(["XSHG", "XSHE"])]

    liquid_candidates = []
    for order_book_id in universe["order_book_id"].tolist():
        try:
            if is_st_stock(order_book_id) or is_suspended(order_book_id):
                continue
        except Exception:
            continue

        turnover = history_bars(order_book_id, liquidity_lookback, "1d", "total_turnover")
        if turnover is None or len(turnover) < liquidity_lookback:
            continue
        turnover = np.asarray(turnover, dtype=float)
        if np.isnan(turnover).any() or np.any(turnover <= 0):
            continue
        liquid_candidates.append((order_book_id, float(turnover.mean())))

    liquid_candidates.sort(key=lambda item: item[1], reverse=True)
    return [order_book_id for order_book_id, _ in liquid_candidates[:pool_size]]


def setup_context(
    context,
    lookback,
    top_n=5,
    cash_buffer=0.05,
    defensive_filter=False,
    rebalance_frequency="daily",
    hold_buffer_rank=None,
):
    context.stock_pool = build_stock_pool(POOL_SIZE)
    context.lookback = lookback
    context.top_n = top_n
    context.cash_buffer = cash_buffer
    context.defensive_filter = defensive_filter
    context.last_rebalance = None
    context.rebalance_frequency = rebalance_frequency
    context.hold_buffer_rank = hold_buffer_rank
    update_universe(context.stock_pool + [BENCHMARK])


def should_rebalance(context):
    frequency = getattr(context, "rebalance_frequency", "daily")
    now = context.now.date()
    if frequency == "daily":
        return True
    if frequency == "weekly":
        key = now.isocalendar()[:2]
    elif frequency == "biweekly":
        year, week, _ = now.isocalendar()
        key = (year, week // 2)
    elif frequency == "monthly":
        key = (now.year, now.month)
    else:
        raise ValueError(f"Unsupported rebalance_frequency: {frequency}")
    if context.last_rebalance == key:
        return False
    context.last_rebalance = key
    return True


def close_history(order_book_id, bars):
    data = history_bars(order_book_id, bars, "1d", "close")
    if data is None or len(data) < bars:
        return None
    data = np.asarray(data, dtype=float)
    if np.isnan(data).any() or np.any(data <= 0):
        return None
    return data


def high_low_close_history(order_book_id, bars):
    high = history_bars(order_book_id, bars, "1d", "high")
    low = history_bars(order_book_id, bars, "1d", "low")
    close = history_bars(order_book_id, bars, "1d", "close")
    if high is None or low is None or close is None:
        return None, None, None
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    if len(high) < bars or len(low) < bars or len(close) < bars:
        return None, None, None
    if np.isnan(high).any() or np.isnan(low).any() or np.isnan(close).any():
        return None, None, None
    return high, low, close


def total_return(prices):
    return float(prices[-1] / prices[0] - 1.0)


def daily_returns(prices):
    return prices[1:] / prices[:-1] - 1.0


def volatility(prices):
    returns = daily_returns(prices)
    if len(returns) == 0:
        return None
    return float(np.std(returns))


def downside_volatility(prices):
    returns = daily_returns(prices)
    negative = returns[returns < 0]
    if len(negative) == 0:
        return 0.0
    return float(np.std(negative))


def max_drawdown(prices):
    peaks = np.maximum.accumulate(prices)
    drawdowns = prices / peaks - 1.0
    return float(abs(drawdowns.min()))


def rsi(prices, period=14):
    deltas = np.diff(prices)
    if len(deltas) < period:
        return None
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def breakout_position(high, low, close):
    price_range = float(high.max() - low.min())
    if price_range <= 0:
        return None
    return float((close[-1] - low.min()) / price_range)


def moving_average_gap(prices, window):
    if len(prices) < window:
        return None
    ma = prices[-window:].mean()
    if ma <= 0:
        return None
    return float(prices[-1] / ma - 1.0)


def benchmark_is_healthy(context):
    prices = close_history(BENCHMARK, 120)
    if prices is None:
        return True
    medium_term = prices[-1] / prices[-60] - 1.0
    long_ma = prices[-120:].mean()
    return bool(prices[-1] > long_ma and medium_term > -0.03)


def rebalance(context, selected):
    target_weight = (1.0 - context.cash_buffer) / max(len(selected), 1)
    selected = set(selected)
    current_positions = set(context.portfolio.positions.keys())
    trade_universe = current_positions | selected
    for order_book_id in trade_universe:
        if order_book_id in selected:
            order_target_percent(order_book_id, target_weight)
        else:
            order_target_percent(order_book_id, 0)


def active_positions(context):
    return {
        order_book_id
        for order_book_id, position in context.portfolio.positions.items()
        if getattr(position, "quantity", 0) > 0
    }


def select_with_hold_buffer(context, ranked):
    top_n = context.top_n
    buffer_rank = getattr(context, "hold_buffer_rank", None)
    if not buffer_rank or buffer_rank <= top_n:
        return [order_book_id for order_book_id, _ in ranked[:top_n]]

    ranked_ids = [order_book_id for order_book_id, _ in ranked]
    rank_map = {order_book_id: idx + 1 for idx, order_book_id in enumerate(ranked_ids)}
    retained = [
        order_book_id
        for order_book_id in active_positions(context)
        if rank_map.get(order_book_id, float("inf")) <= buffer_rank
    ]
    retained = sorted(retained, key=lambda order_book_id: rank_map[order_book_id])[:top_n]

    selected = list(retained)
    for order_book_id in ranked_ids:
        if len(selected) >= top_n:
            break
        if order_book_id not in selected:
            selected.append(order_book_id)
    return selected


def run_monthly_selection(context, score_func):
    if not should_rebalance(context):
        return

    if context.defensive_filter and not benchmark_is_healthy(context):
        rebalance(context, [])
        return

    scores = {}
    for order_book_id in context.stock_pool:
        score = score_func(context, order_book_id)
        if score is None or not np.isfinite(score):
            continue
        scores[order_book_id] = float(score)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected = select_with_hold_buffer(context, ranked)
    rebalance(context, selected)
