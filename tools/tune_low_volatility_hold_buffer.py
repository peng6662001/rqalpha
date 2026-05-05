from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from rqalpha import run_file


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = Path.home() / ".rqalpha" / "bundle"
OUTPUT = REPO_ROOT / ".temp" / "low_volatility_hold_buffer.json"
STRATEGY = "strategies/stock_selection/strategy_low_volatility.py"

PERIODS = [
    ("2014-01-01", "2016-12-31"),
    ("2017-01-01", "2019-12-31"),
    ("2020-01-01", "2022-12-31"),
    ("2023-01-01", "2024-12-31"),
]

PARAM_GRID = [
    {"hold_buffer_rank": None},
    {"hold_buffer_rank": 10},
    {"hold_buffer_rank": 15},
    {"hold_buffer_rank": 20},
]

BASE_ENV = {
    "RQALPHA_LOW_VOL_TOP_N": "5",
    "RQALPHA_LOW_VOL_LOOKBACK": "80",
    "RQALPHA_LOW_VOL_CASH_BUFFER": "0.02",
    "RQALPHA_LOW_VOL_VOL_WINDOW": "60",
    "RQALPHA_LOW_VOL_TREND_FILTER": "off",
    "RQALPHA_LOW_VOL_TREND_WINDOW": "60",
    "RQALPHA_LOW_VOL_MA_WINDOW": "120",
    "RQALPHA_LOW_VOL_REBALANCE_FREQUENCY": "daily",
}


@dataclass
class PeriodResult:
    period: str
    total_returns: float
    annualized_returns: float
    max_drawdown: float
    sharpe: float
    calmar: float | None


def base_config(start: str, end: str):
    return {
        "base": {
            "start_date": start,
            "end_date": end,
            "frequency": "1d",
            "matching_type": "current_bar",
            "accounts": {"stock": 1000000},
            "benchmark": "000300.XSHG",
            "data_bundle_path": str(BUNDLE_ROOT),
        },
        "extra": {"log_level": "error"},
        "mod": {
            "sys_progress": {"enabled": False},
            "sys_analyser": {"enabled": True, "plot": False},
        },
    }


def with_env(params: dict):
    keys = list(BASE_ENV) + ["RQALPHA_LOW_VOL_HOLD_BUFFER_RANK"]
    old_env = {key: os.environ.get(key) for key in keys}
    os.environ.update(BASE_ENV)
    if params["hold_buffer_rank"] is None:
        os.environ.pop("RQALPHA_LOW_VOL_HOLD_BUFFER_RANK", None)
    else:
        os.environ["RQALPHA_LOW_VOL_HOLD_BUFFER_RANK"] = str(params["hold_buffer_rank"])
    return old_env


def restore_env(old_env: dict):
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_one(params: dict, start: str, end: str) -> PeriodResult:
    old_env = with_env(params)
    try:
        result = run_file(STRATEGY, config=base_config(start, end))
    finally:
        restore_env(old_env)

    summary = result["sys_analyser"]["summary"]
    mdd = summary["max_drawdown"]
    calmar = None if not mdd else summary["annualized_returns"] / mdd
    return PeriodResult(
        period=f"{start}->{end}",
        total_returns=summary["total_returns"],
        annualized_returns=summary["annualized_returns"],
        max_drawdown=mdd,
        sharpe=summary["sharpe"],
        calmar=calmar,
    )


def aggregate(params: dict, results: list[PeriodResult]):
    calmar_values = [r.calmar for r in results if r.calmar is not None]
    positive_periods = sum(1 for r in results if r.total_returns > 0)
    return {
        "params": params,
        "period_results": [asdict(r) for r in results],
        "positive_period_ratio": positive_periods / len(results),
        "avg_calmar": sum(calmar_values) / len(calmar_values),
        "min_calmar": min(calmar_values),
        "avg_sharpe": sum(r.sharpe for r in results) / len(results),
        "worst_drawdown": max(r.max_drawdown for r in results),
        "avg_annualized_returns": sum(r.annualized_returns for r in results) / len(results),
    }


def sort_results(results: list[dict]):
    return sorted(
        results,
        key=lambda r: (
            r["positive_period_ratio"],
            r["avg_calmar"],
            r["min_calmar"],
            r["avg_sharpe"],
            -r["worst_drawdown"],
        ),
        reverse=True,
    )


def write_results(results: list[dict]):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps({"periods": PERIODS, "results": sort_results(results)}, ensure_ascii=False, indent=2)
    )


def main():
    all_results = []
    for params in PARAM_GRID:
        period_results = [run_one(params, start, end) for start, end in PERIODS]
        all_results.append(aggregate(params, period_results))
        write_results(all_results)

    best = sort_results(all_results)[0]
    print(f"wrote {OUTPUT}")
    print(json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()
