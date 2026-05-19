from __future__ import annotations

import json
import sys
from pathlib import Path

from rqalpha.apis import *

sys.path.insert(0, str((Path.cwd() / "strategies" / "stock_selection").resolve()))

from ss_common import BENCHMARK, benchmark_is_healthy
from stable_compound_common import (
    DEFAULT_EVENT_PATH,
    DEFAULT_POOL_SIZE,
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
    context.event_path = DEFAULT_EVENT_PATH
    context.generated = False
    update_universe([context.benchmark])


def after_trading(context):
    if context.generated:
        return

    frame = collect_candidate_frame(
        trading_date=context.now.date(),
        pool_size=context.pool_size,
        event_path=context.event_path,
    )
    selected = frame[frame["tradeable"]].copy()
    recommended = selected.head(5)
    backup = selected.iloc[5:10].copy()

    payload = {
        "selection_date": context.now.date().isoformat(),
        "benchmark": context.benchmark,
        "benchmark_healthy": benchmark_is_healthy(context),
        "framework": "stable_compound_midterm",
        "manual_checklist": [
            "先看行业景气和资金是否还在",
            "再确认盈利改善或业绩事件支撑",
            "只在趋势未坏、缩量整理或二次启动时考虑买入",
            "遇到趋势破位或逻辑失效优先减仓",
        ],
        "positioning_defaults": {
            "holdings": "5-10",
            "single_position": "8%-20%",
            "cash_buffer": "keep some dry powder, avoid blind full allocation",
        },
        "filters": {
            "avg_turnover_20_min": 200000000.0,
            "price_range": [5.0, 150.0],
            "max_limit_hits_10": 1,
            "max_vol_20": 0.055,
            "max_drawdown_120": 0.35,
            "min_ret_60": 0.02,
            "min_breakout_60": 0.55,
            "ma_gap_120_floor": -0.02,
        },
        "recommended": _to_records(recommended),
        "backup": _to_records(backup),
        "top_candidates": _to_records(selected.head(15)),
    }

    output_path = Path.cwd() / ".temp" / f"stable_compound_selection_{context.now.date().isoformat()}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"stable compound selector wrote: {output_path}")
    for idx, row in enumerate(payload["recommended"], 1):
        logger.info(
            f"recommended rank={idx} {row['order_book_id']} {row['symbol']} "
            f"score={row['composite_score']:.4f} fundamental={row['fundamental_score']:.4f} "
            f"trend={row['trend_score']:.4f} risk={row['risk_score']:.4f}"
        )
    context.generated = True


def _to_records(frame):
    columns = [
        "order_book_id",
        "symbol",
        "industry_name",
        "last_price",
        "composite_score",
        "fundamental_score",
        "prosperity_score",
        "trend_score",
        "risk_score",
        "ret_20",
        "ret_60",
        "ma_gap_20",
        "ma_gap_60",
        "ma_gap_120",
        "near_high_252",
        "vol_20",
        "max_dd_60",
        "max_dd_120",
        "event_family",
        "forecast_type",
        "days_since_event",
    ]
    return frame[columns].to_dict("records")
