from __future__ import annotations

import sys
from pathlib import Path

from rqalpha.apis import *

sys.path.insert(0, str((Path.cwd() / "strategies" / "stock_selection").resolve()))

from ss_common import BENCHMARK, benchmark_is_healthy
from stable_compound_common import (
    DEFAULT_CASH_BUFFER,
    DEFAULT_EVENT_PATH,
    DEFAULT_HOLD_BUFFER_RANK,
    DEFAULT_POOL_SIZE,
    DEFAULT_TOP_N,
    collect_candidate_frame,
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
    context.pool_size = DEFAULT_POOL_SIZE
    context.top_n = DEFAULT_TOP_N
    context.hold_buffer_rank = DEFAULT_HOLD_BUFFER_RANK
    context.cash_buffer = DEFAULT_CASH_BUFFER
    context.rebalance_frequency = "weekly"
    context.event_path = DEFAULT_EVENT_PATH
    context.pending_ranked = []
    context.last_selection_key = None
    context.benchmark_healthy = True
    update_universe([context.benchmark])


def before_trading(context):
    week_key = context.now.isocalendar()[:2]
    if context.last_selection_key == week_key:
        return
    frame = collect_candidate_frame(
        trading_date=context.now.date(),
        pool_size=context.pool_size,
        event_path=context.event_path,
    )
    context.benchmark_healthy = benchmark_is_healthy(context)
    frame = frame[frame["tradeable"]].copy()
    context.pending_ranked = frame.to_dict("records")
    context.last_selection_key = week_key
    update_universe([context.benchmark] + frame["order_book_id"].tolist())


def open_auction(context, bar_dict):
    week_key = context.now.isocalendar()[:2]
    if getattr(context, "last_rebalance_week", None) == week_key:
        return
    context.last_rebalance_week = week_key

    ranked_ids = [row["order_book_id"] for row in context.pending_ranked]
    current_positions = {
        position.order_book_id
        for position in get_positions()
        if getattr(position, "quantity", 0) > 0
    }

    if context.benchmark_healthy:
        selected = _select_with_hold_buffer(
            ranked_ids=ranked_ids,
            current_positions=current_positions,
            top_n=context.top_n,
            hold_buffer_rank=context.hold_buffer_rank,
        )
    else:
        selected = []
    target_weight = (1.0 - context.cash_buffer) / max(len(selected), 1)
    selected_set = set(selected)
    trade_universe = sorted(current_positions | selected_set)

    for order_book_id in trade_universe:
        if order_book_id in selected_set:
            order_target_percent(order_book_id, target_weight)
        else:
            order_target_percent(order_book_id, 0.0)


def _select_with_hold_buffer(ranked_ids, current_positions, top_n, hold_buffer_rank):
    if not hold_buffer_rank or hold_buffer_rank <= top_n:
        return ranked_ids[:top_n]

    rank_map = {order_book_id: idx + 1 for idx, order_book_id in enumerate(ranked_ids)}
    retained = [
        order_book_id
        for order_book_id in sorted(current_positions)
        if rank_map.get(order_book_id, float("inf")) <= hold_buffer_rank
    ][:top_n]

    selected = list(retained)
    for order_book_id in ranked_ids:
        if len(selected) >= top_n:
            break
        if order_book_id not in selected:
            selected.append(order_book_id)
    return selected
