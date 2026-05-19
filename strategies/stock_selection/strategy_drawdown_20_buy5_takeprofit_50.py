import sys
from pathlib import Path

import numpy as np
from rqalpha.apis import *

sys.path.insert(0, str((Path.cwd() / "strategies" / "stock_selection").resolve()))

from ss_common import BENCHMARK, build_stock_pool, close_history


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
    context.pool_size = 500
    context.stock_pool = []
    context.top_n = 5
    context.lookback = 120
    context.entry_drawdown = 0.20
    context.take_profit = 0.50
    context.cash_buffer = 0.02
    context.min_price = 5.0
    context.max_price = 120.0
    context.min_avg_turnover_20 = 200000000.0
    context.last_pool_key = None
    update_universe([context.benchmark])


def _avg_turnover(order_book_id, bars):
    turnover = history_bars(order_book_id, bars, "1d", "total_turnover")
    if turnover is None or len(turnover) < bars:
        return None
    turnover = np.asarray(turnover, dtype=float)
    if np.isnan(turnover).any() or np.any(turnover <= 0):
        return None
    return float(turnover.mean())


def _candidate_score(context, order_book_id):
    prices = close_history(order_book_id, context.lookback)
    if prices is None:
        return None

    current_price = float(prices[-1])
    if current_price < context.min_price or current_price > context.max_price:
        return None

    avg_turnover_20 = _avg_turnover(order_book_id, 20)
    if avg_turnover_20 is None or avg_turnover_20 < context.min_avg_turnover_20:
        return None

    peak_price = float(np.max(prices))
    if peak_price <= 0:
        return None
    drawdown = 1.0 - current_price / peak_price
    if drawdown < context.entry_drawdown:
        return None

    ret_20 = float(prices[-1] / prices[-20] - 1.0) if len(prices) >= 20 else 0.0
    # 先按回撤深度选，再偏好近期开始企稳的票
    score = drawdown + 0.15 * ret_20
    return {
        "score": float(score),
        "drawdown": float(drawdown),
        "ret_20": float(ret_20),
        "price": current_price,
    }


def _refresh_pool(context):
    pool_key = (context.now.year, context.now.month)
    if context.last_pool_key == pool_key:
        return
    context.stock_pool = build_stock_pool(context.pool_size)
    context.last_pool_key = pool_key
    update_universe(context.stock_pool + [context.benchmark])


def _rank_candidates(context):
    ranked = []
    for order_book_id in context.stock_pool:
        try:
            if is_st_stock(order_book_id) or is_suspended(order_book_id):
                continue
        except Exception:
            continue

        metrics = _candidate_score(context, order_book_id)
        if metrics is None:
            continue
        ranked.append((order_book_id, metrics))

    ranked.sort(
        key=lambda item: (
            item[1]["score"],
            item[1]["drawdown"],
            item[1]["ret_20"],
        ),
        reverse=True,
    )
    return ranked


def _take_profit_exits(context):
    exited = set()
    for position in get_positions():
        quantity = getattr(position, "quantity", 0)
        if quantity <= 0:
            continue
        avg_price = getattr(position, "avg_price", 0)
        last_price = getattr(position, "last_price", 0)
        if avg_price <= 0 or last_price <= 0:
            continue
        profit = last_price / avg_price - 1.0
        if profit >= context.take_profit:
            logger.info(
                "止盈卖出 %s profit=%.4f avg_price=%.2f last_price=%.2f",
                position.order_book_id,
                profit,
                avg_price,
                last_price,
            )
            order_target_percent(position.order_book_id, 0.0)
            exited.add(position.order_book_id)
    return exited


def handle_bar(context, bar_dict):
    _refresh_pool(context)
    exited = _take_profit_exits(context)

    ranked = _rank_candidates(context)
    selected = []
    for order_book_id, _ in ranked:
        if order_book_id in exited:
            continue
        selected.append(order_book_id)
        if len(selected) >= context.top_n:
            break

    selected_set = set(selected)
    current_positions = {
        position.order_book_id
        for position in get_positions()
        if getattr(position, "quantity", 0) > 0
    }

    target_weight = (1.0 - context.cash_buffer) / max(len(selected_set), 1)
    trade_universe = sorted(current_positions | selected_set)
    for order_book_id in trade_universe:
        if order_book_id in exited:
            continue
        if order_book_id in selected_set:
            order_target_percent(order_book_id, target_weight)
        else:
            order_target_percent(order_book_id, 0.0)
