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
    context.market_drawdown_trigger = 0.10
    context.market_drawdown_reset = 0.06
    context.max_stock_drawdown = 0.10
    context.take_profit = 0.50
    context.cash_buffer = 0.02
    context.min_price = 5.0
    context.max_price = 120.0
    context.min_avg_turnover_20 = 200000000.0
    context.last_pool_key = None
    context.market_peak = None
    update_universe([context.benchmark])


def _refresh_pool(context):
    pool_key = (context.now.year, context.now.month)
    if context.last_pool_key == pool_key:
        return
    context.stock_pool = build_stock_pool(context.pool_size)
    context.last_pool_key = pool_key
    update_universe(context.stock_pool + [context.benchmark])


def _avg_turnover(order_book_id, bars):
    turnover = history_bars(order_book_id, bars, "1d", "total_turnover")
    if turnover is None or len(turnover) < bars:
        return None
    turnover = np.asarray(turnover, dtype=float)
    if np.isnan(turnover).any() or np.any(turnover <= 0):
        return None
    return float(turnover.mean())


def _market_drawdown(context):
    prices = close_history(context.benchmark, context.lookback)
    if prices is None:
        return 0.0
    benchmark_price = float(prices[-1])
    if context.market_peak is None:
        context.market_peak = benchmark_price
    context.market_peak = max(context.market_peak, benchmark_price)
    if context.market_peak <= 0:
        return 0.0
    return 1.0 - benchmark_price / context.market_peak


def _resilient_stock_metrics(context, order_book_id):
    prices = close_history(order_book_id, context.lookback)
    if prices is None:
        return None

    current_price = float(prices[-1])
    if current_price < context.min_price or current_price > context.max_price:
        return None

    avg_turnover = _avg_turnover(order_book_id, 20)
    if avg_turnover is None or avg_turnover < context.min_avg_turnover_20:
        return None

    peak_price = float(np.max(prices))
    if peak_price <= 0:
        return None
    stock_drawdown = 1.0 - current_price / peak_price
    if stock_drawdown > context.max_stock_drawdown:
        return None

    ret_20 = float(prices[-1] / prices[-20] - 1.0) if len(prices) >= 20 else 0.0
    ret_60 = float(prices[-1] / prices[-60] - 1.0) if len(prices) >= 60 else ret_20
    score = (1.0 - stock_drawdown) + 0.25 * ret_20 + 0.10 * ret_60
    return {
        "score": float(score),
        "stock_drawdown": float(stock_drawdown),
        "ret_20": float(ret_20),
        "ret_60": float(ret_60),
    }


def _rank_resilient_stocks(context):
    ranked = []
    for order_book_id in context.stock_pool:
        try:
            if is_st_stock(order_book_id) or is_suspended(order_book_id):
                continue
        except Exception:
            continue

        metrics = _resilient_stock_metrics(context, order_book_id)
        if metrics is None:
            continue
        ranked.append((order_book_id, metrics))

    ranked.sort(
        key=lambda item: (
            item[1]["score"],
            -item[1]["stock_drawdown"],
            item[1]["ret_20"],
            item[1]["ret_60"],
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
                f"止盈卖出 {position.order_book_id} "
                f"profit={profit:.4f} avg_price={avg_price:.2f} last_price={last_price:.2f}"
            )
            order_target_percent(position.order_book_id, 0.0)
            exited.add(position.order_book_id)
    return exited


def handle_bar(context, bar_dict):
    _refresh_pool(context)
    exited = _take_profit_exits(context)
    drawdown = _market_drawdown(context)
    plot("market_drawdown", drawdown)

    current_positions = {
        position.order_book_id
        for position in get_positions()
        if getattr(position, "quantity", 0) > 0
    }

    if drawdown < context.market_drawdown_trigger:
        if current_positions and drawdown <= context.market_drawdown_reset:
            for order_book_id in sorted(current_positions):
                logger.info(f"大盘回撤收窄，清仓 {order_book_id} drawdown={drawdown:.4f}")
                order_target_percent(order_book_id, 0.0)
        return

    ranked = _rank_resilient_stocks(context)
    selected = []
    for order_book_id, _ in ranked:
        if order_book_id in exited:
            continue
        selected.append(order_book_id)
        if len(selected) >= context.top_n:
            break

    selected_set = set(selected)
    target_weight = (1.0 - context.cash_buffer) / max(len(selected_set), 1)
    trade_universe = sorted(current_positions | selected_set)

    if selected:
        logger.info(
            f"大盘回撤超过10%，买入抗跌股 {selected[0]} 等{len(selected)}只 "
            f"drawdown={drawdown:.4f}"
        )

    for order_book_id in trade_universe:
        if order_book_id in exited:
            continue
        if order_book_id in selected_set:
            order_target_percent(order_book_id, target_weight)
        else:
            order_target_percent(order_book_id, 0.0)
