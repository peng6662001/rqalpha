import os
import sys
from pathlib import Path

sys.path.insert(0, str((Path.cwd() / "strategies" / "stock_selection").resolve()))

from ss_common import (
    BENCHMARK,
    POOL_SIZE,
    build_stock_pool,
    breakout_position,
    close_history,
    daily_returns,
    high_low_close_history,
    moving_average_gap,
    rebalance,
    run_monthly_selection,
    total_return,
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
    context.bull_ma_window = _env_int("RQALPHA_BULL_MA_WINDOW", 120)
    context.bull_ret_window = _env_int("RQALPHA_BULL_RET_WINDOW", 20)
    context.bull_ret_threshold = _env_float("RQALPHA_BULL_RET_THRESHOLD", 0.00)
    context.bear_ma_gap = _env_float("RQALPHA_BEAR_MA_GAP", -0.03)
    context.momentum_window = _env_int("RQALPHA_MOM_WINDOW", 120)
    context.skip_recent = _env_int("RQALPHA_SKIP_RECENT", 20)
    context.reversal_window = _env_int("RQALPHA_REV_WINDOW", 5)
    context.long_confirm_window = _env_int("RQALPHA_LONG_CONFIRM_WINDOW", 60)
    context.vol_window = _env_int("RQALPHA_VOL_WINDOW", 20)
    context.momentum_weight = _env_float("RQALPHA_MOM_WEIGHT", 1.0)
    context.breakout_weight = _env_float("RQALPHA_BREAKOUT_WEIGHT", 0.35)
    context.vol_penalty = _env_float("RQALPHA_VOL_PENALTY", 0.30)
    context.reversal_penalty = _env_float("RQALPHA_REVERSAL_PENALTY", 0.25)
    context.reversal_weight = _env_float("RQALPHA_REVERSAL_WEIGHT", 1.0)
    context.long_confirm_weight = _env_float("RQALPHA_LONG_CONFIRM_WEIGHT", 0.50)
    context.oversold_bonus = _env_float("RQALPHA_OVERSOLD_BONUS", 0.20)


def before_trading(context):
    context.stock_pool = build_stock_pool(POOL_SIZE)
    update_universe(context.stock_pool + [BENCHMARK])


def _market_regime(context):
    bars = max(context.bull_ma_window, context.bull_ret_window) + 5
    prices = close_history(BENCHMARK, bars)
    if prices is None:
        return "bull"
    ma_gap = moving_average_gap(prices, context.bull_ma_window)
    bench_ret = prices[-1] / prices[-1 - context.bull_ret_window] - 1.0
    if ma_gap is None:
        return "bull"
    if ma_gap <= context.bear_ma_gap:
        return "bear"
    if prices[-1] > prices[-context.bull_ma_window :].mean() and bench_ret >= context.bull_ret_threshold:
        return "bull"
    return "neutral"


def _momentum_score(context, order_book_id):
    bars = context.momentum_window + context.skip_recent + 5
    prices = close_history(order_book_id, bars)
    high, low, close = high_low_close_history(order_book_id, bars)
    if prices is None or high is None:
        return None
    start = prices[0]
    end = prices[-1 - context.skip_recent]
    if start <= 0 or end <= 0:
        return None
    mid_term_momentum = end / start - 1.0
    recent_reversal = prices[-1] / prices[-1 - context.skip_recent] - 1.0
    vol = daily_returns(prices[-context.vol_window - 1 :]).std()
    breakout = breakout_position(high[-252:], low[-252:], close[-252:]) if len(close) >= 252 else breakout_position(high, low, close)
    if breakout is None:
        breakout = 0.0
    return (
        context.momentum_weight * mid_term_momentum
        + context.breakout_weight * breakout
        - context.vol_penalty * float(vol)
        - context.reversal_penalty * recent_reversal
    )


def _reversal_score(context, order_book_id):
    bars = max(context.long_confirm_window, context.reversal_window, context.vol_window) + 5
    prices = close_history(order_book_id, bars)
    if prices is None:
        return None
    short_ret = prices[-1] / prices[-1 - context.reversal_window] - 1.0
    long_ret = prices[-1] / prices[-1 - context.long_confirm_window] - 1.0
    ma_gap = moving_average_gap(prices, min(20, len(prices)))
    if ma_gap is None:
        ma_gap = 0.0
    vol = daily_returns(prices[-context.vol_window - 1 :]).std()
    return (
        context.reversal_weight * (-short_ret)
        + context.long_confirm_weight * long_ret
        + context.oversold_bonus * (-ma_gap)
        - context.vol_penalty * float(vol)
    )


def score_stock(context, order_book_id):
    regime = _market_regime(context)
    if regime == "bull":
        return _momentum_score(context, order_book_id)
    return _reversal_score(context, order_book_id)


def handle_bar(context, bar_dict):
    if _market_regime(context) == "bear" and os.environ.get("RQALPHA_BEAR_TO_CASH", "0") == "1":
        rebalance(context, [])
        return
    run_monthly_selection(context, score_stock)
