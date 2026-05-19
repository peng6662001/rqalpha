import sys
from pathlib import Path

import numpy as np
from rqalpha.apis import *

sys.path.insert(0, str((Path.cwd() / "strategies" / "stock_selection").resolve()))

from ss_common import (
    BENCHMARK,
    breakout_position,
    build_stock_pool,
    close_history,
    daily_returns,
    high_low_close_history,
    max_drawdown,
    moving_average_gap,
)


__config__ = {
    "base": {
        "benchmark": BENCHMARK,
        "accounts": {
            "stock": 1000000,
        },
    }
}


def init(context):
    context.benchmark = BENCHMARK
    context.pool_size = 300
    context.stock_pool = []
    context.top_n = 5
    context.hold_buffer_rank = 8
    context.rebalance_frequency = "weekly"
    context.max_vol_20 = 0.050
    context.max_dd_60 = 0.24
    context.min_ret_20 = 0.08
    context.min_ret_60 = 0.12
    context.min_breakout_60 = 0.70
    context.min_avg_turnover_20 = 300000000.0
    context.min_price = 8.0
    context.max_price = 120.0
    context.max_limit_hits_10 = 1
    context.cash_buffer = 0.05
    context.pending_selected = []
    context.last_pool_key = None
    update_universe([context.benchmark])


def _volatility(prices):
    returns = daily_returns(prices)
    if len(returns) == 0:
        return None
    return float(np.std(returns))


def _avg_turnover(order_book_id, bars):
    turnover = history_bars(order_book_id, bars, "1d", "total_turnover")
    if turnover is None or len(turnover) < bars:
        return None
    turnover = np.asarray(turnover, dtype=float)
    if np.isnan(turnover).any() or np.any(turnover <= 0):
        return None
    return float(turnover.mean())


def _recent_limit_hits(order_book_id, bars):
    fields = ["close", "high", "low"]
    data = history_bars(order_book_id, bars + 1, "1d", fields)
    if data is None or len(data) < bars + 1:
        return None
    close = np.asarray(data["close"], dtype=float)
    high = np.asarray(data["high"], dtype=float)
    low = np.asarray(data["low"], dtype=float)
    if np.isnan(close).any() or np.isnan(high).any() or np.isnan(low).any():
        return None

    hit_count = 0
    for idx in range(1, len(close)):
        prev_close = close[idx - 1]
        if prev_close <= 0:
            return None
        up_ratio = high[idx] / prev_close - 1.0
        down_ratio = low[idx] / prev_close - 1.0
        if up_ratio >= 0.097 or down_ratio <= -0.097:
            hit_count += 1
    return hit_count


def _score_stock(order_book_id):
    prices_20 = close_history(order_book_id, 20)
    prices_60 = close_history(order_book_id, 60)
    prices_120 = close_history(order_book_id, 120)
    high, low, close = high_low_close_history(order_book_id, 60)
    if prices_20 is None or prices_60 is None or prices_120 is None:
        return None
    if high is None or low is None or close is None:
        return None

    ret_20 = float(prices_20[-1] / prices_20[0] - 1.0)
    ret_60 = float(prices_60[-1] / prices_60[0] - 1.0)
    ma_gap_20 = moving_average_gap(prices_20, 20)
    ma_gap_60 = moving_average_gap(prices_60, 60)
    breakout = breakout_position(high, low, close)
    vol_20 = _volatility(prices_20)
    dd_60 = max_drawdown(prices_60)
    dd_120 = max_drawdown(prices_120)

    if (
        ma_gap_20 is None
        or ma_gap_60 is None
        or breakout is None
        or vol_20 is None
        or dd_60 is None
        or dd_120 is None
    ):
        return None

    score = (
        0.32 * ret_20
        + 0.28 * ret_60
        + 0.18 * ma_gap_20
        + 0.10 * ma_gap_60
        + 0.12 * breakout
        - 0.20 * vol_20
        - 0.10 * dd_60
        - 0.05 * dd_120
    )
    return {
        "score": float(score),
        "last_price": float(prices_20[-1]),
        "ret_20": ret_20,
        "ret_60": ret_60,
        "breakout_60": float(breakout),
        "vol_20": float(vol_20),
        "max_dd_60": float(dd_60),
    }


def _is_tradeable_candidate(context, order_book_id, metrics):
    if is_st_stock(order_book_id) or is_suspended(order_book_id):
        return False
    avg_turnover_20 = _avg_turnover(order_book_id, 20)
    if avg_turnover_20 is None or avg_turnover_20 < context.min_avg_turnover_20:
        return False
    if metrics["last_price"] < context.min_price or metrics["last_price"] > context.max_price:
        return False
    limit_hits_10 = _recent_limit_hits(order_book_id, 10)
    if limit_hits_10 is None or limit_hits_10 > context.max_limit_hits_10:
        return False
    if metrics["vol_20"] > context.max_vol_20:
        return False
    if metrics["max_dd_60"] > context.max_dd_60:
        return False
    if metrics["ret_20"] < context.min_ret_20:
        return False
    if metrics["ret_60"] < context.min_ret_60:
        return False
    if metrics["breakout_60"] < context.min_breakout_60:
        return False
    return True


def _selected_candidates(context):
    scored = []
    for order_book_id in context.stock_pool:
        metrics = _score_stock(order_book_id)
        if metrics is None:
            continue
        if not _is_tradeable_candidate(context, order_book_id, metrics):
            continue
        scored.append((order_book_id, metrics["score"]))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [order_book_id for order_book_id, _ in scored[: context.top_n]]


def before_trading(context):
    pool_key = (context.now.year, context.now.month)
    if context.last_pool_key != pool_key:
        context.stock_pool = build_stock_pool(context.pool_size)
        context.last_pool_key = pool_key
        update_universe(context.stock_pool + [context.benchmark])

    context.pending_selected = _selected_candidates(context)


def open_auction(context, bar_dict):
    if context.rebalance_frequency == "weekly":
        week_key = context.now.isocalendar()[:2]
        if getattr(context, "last_rebalance_week", None) == week_key:
            return
        context.last_rebalance_week = week_key

    selected = list(context.pending_selected)
    current_positions = {
        position.order_book_id
        for position in get_positions()
        if getattr(position, "quantity", 0) > 0
    }
    if context.hold_buffer_rank and current_positions:
        ranked = context.pending_selected
        rank_map = {order_book_id: idx + 1 for idx, order_book_id in enumerate(ranked)}
        retained = [
            order_book_id
            for order_book_id in sorted(current_positions)
            if rank_map.get(order_book_id, float("inf")) <= context.hold_buffer_rank
        ]
        selected = list(retained[: context.top_n])
        for order_book_id in ranked:
            if len(selected) >= context.top_n:
                break
            if order_book_id not in selected:
                selected.append(order_book_id)

    selected_set = set(selected)
    target_weight = (1.0 - context.cash_buffer) / max(len(selected_set), 1)
    trade_universe = sorted(current_positions | selected_set)

    for order_book_id in trade_universe:
        if order_book_id in selected_set:
            order_target_percent(order_book_id, target_weight)
        else:
            order_target_percent(order_book_id, 0.0)
