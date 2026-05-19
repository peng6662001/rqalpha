import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str((Path.cwd() / "strategies" / "stock_selection").resolve()))

from ss_common import (
    BENCHMARK,
    POOL_SIZE,
    build_stock_pool,
    close_history,
    daily_returns,
    high_low_close_history,
    moving_average_gap,
    run_monthly_selection,
    update_universe,
)


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_int(name, default):
    return int(os.environ.get(name, default))


def init(context):
    context.stock_pool = []
    context.lookback = 260
    context.top_n = _env_int("RQALPHA_TOP_N", 5)
    context.cash_buffer = _env_float("RQALPHA_CASH_BUFFER", 0.05)
    context.defensive_filter = False
    context.last_rebalance = None
    context.rebalance_frequency = "monthly"
    context.hold_buffer_rank = _env_int("RQALPHA_HOLD_BUFFER_RANK", 8)
    context.residual_window = _env_int("RQALPHA_RESIDUAL_WINDOW", 120)
    context.skip_recent = _env_int("RQALPHA_SKIP_RECENT", 20)
    context.lookback_52w = _env_int("RQALPHA_52W_WINDOW", 252)
    context.residual_weight = _env_float("RQALPHA_RESIDUAL_WEIGHT", 1.0)
    context.high_weight = _env_float("RQALPHA_HIGH_WEIGHT", 0.8)
    context.trend_weight = _env_float("RQALPHA_TREND_WEIGHT", 0.25)
    context.vol_penalty = _env_float("RQALPHA_VOL_PENALTY", 0.20)
    context.beta_penalty = _env_float("RQALPHA_BETA_PENALTY", 0.10)


def before_trading(context):
    context.stock_pool = build_stock_pool(POOL_SIZE)
    update_universe(context.stock_pool + [BENCHMARK])


def _residual_momentum(context, order_book_id):
    bars = context.residual_window + context.skip_recent + 5
    stock = close_history(order_book_id, bars)
    bench = close_history(BENCHMARK, bars)
    if stock is None or bench is None:
        return None
    stock_mid = stock[-1 - context.skip_recent] / stock[0] - 1.0
    bench_mid = bench[-1 - context.skip_recent] / bench[0] - 1.0
    stock_ret = daily_returns(stock[-context.residual_window - 1 :])
    bench_ret = daily_returns(bench[-context.residual_window - 1 :])
    if len(stock_ret) != len(bench_ret) or len(stock_ret) == 0:
        beta = 1.0
    else:
        var = np.var(bench_ret)
        beta = 1.0 if var <= 0 else float(np.cov(stock_ret, bench_ret)[0, 1] / var)
    return stock_mid - bench_mid - context.beta_penalty * beta


def score_stock(context, order_book_id):
    high, low, close = high_low_close_history(order_book_id, context.lookback_52w)
    if close is None:
        return None
    residual = _residual_momentum(context, order_book_id)
    if residual is None:
        return None
    highest = float(high.max())
    if highest <= 0:
        return None
    proximity = float(close[-1] / highest)
    ma_gap = moving_average_gap(close, min(60, len(close)))
    if ma_gap is None:
        ma_gap = 0.0
    vol = float(np.std(daily_returns(close[-61:]))) if len(close) >= 61 else float(np.std(daily_returns(close)))
    return (
        context.residual_weight * residual
        + context.high_weight * proximity
        + context.trend_weight * ma_gap
        - context.vol_penalty * vol
    )


def handle_bar(context, bar_dict):
    run_monthly_selection(context, score_stock)
