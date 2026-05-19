import sys
from pathlib import Path

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
    context.market_peak = None
    context.entry_drawdown = 0.05
    context.reset_drawdown = 0.03
    context.lookback = 20
    context.min_price = 5.0
    context.max_price = 120.0
    context.min_avg_turnover_20 = 200000000.0
    context.last_pool_key = None
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
    turnover = [float(x) for x in turnover if x and x > 0]
    if len(turnover) < bars:
        return None
    return sum(turnover) / len(turnover)


def _worst_stock_score(context, order_book_id):
    prices = close_history(order_book_id, context.lookback)
    if prices is None:
        return None

    current_price = float(prices[-1])
    if current_price < context.min_price or current_price > context.max_price:
        return None

    avg_turnover = _avg_turnover(order_book_id, 20)
    if avg_turnover is None or avg_turnover < context.min_avg_turnover_20:
        return None

    ret_20 = float(prices[-1] / prices[0] - 1.0)
    return ret_20


def _select_worst_stock(context):
    worst_order_book_id = None
    worst_score = None
    for order_book_id in context.stock_pool:
        try:
            if is_st_stock(order_book_id) or is_suspended(order_book_id):
                continue
        except Exception:
            continue

        score = _worst_stock_score(context, order_book_id)
        if score is None:
            continue
        if worst_score is None or score < worst_score:
            worst_score = score
            worst_order_book_id = order_book_id
    return worst_order_book_id, worst_score


def _market_drawdown(context):
    prices = close_history(context.benchmark, 120)
    if prices is None:
        return 0.0
    benchmark_price = float(prices[-1])
    if context.market_peak is None:
        context.market_peak = benchmark_price
    context.market_peak = max(context.market_peak, benchmark_price)
    if context.market_peak <= 0:
        return 0.0
    return 1.0 - benchmark_price / context.market_peak


def handle_bar(context, bar_dict):
    _refresh_pool(context)
    drawdown = _market_drawdown(context)
    plot("market_drawdown", drawdown)

    current_positions = [
        position for position in get_positions() if getattr(position, "quantity", 0) > 0
    ]
    current_holding = current_positions[0].order_book_id if current_positions else None

    if drawdown < context.entry_drawdown:
        if current_holding is not None and drawdown <= context.reset_drawdown:
            logger.info(f"大盘回撤收窄，清仓 {current_holding} drawdown={drawdown:.4f}")
            order_target_percent(current_holding, 0.0)
        return

    target_order_book_id, score = _select_worst_stock(context)
    if target_order_book_id is None:
        return

    if current_holding is not None and current_holding != target_order_book_id:
        order_target_percent(current_holding, 0.0)

    logger.info(
        f"大盘回撤触发，买入跌幅最深股票 {target_order_book_id} "
        f"drawdown={drawdown:.4f} ret20={score:.4f}"
    )
    order_target_percent(target_order_book_id, 1.0)
