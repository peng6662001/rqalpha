from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from rqalpha import run_file


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = Path.home() / ".rqalpha" / "bundle"
OUTPUT = REPO_ROOT / ".temp" / "low_volatility_tuning.json"
STRATEGY = "strategies/stock_selection/strategy_low_volatility.py"

PERIODS = [
    ("2014-01-01", "2016-12-31"),
    ("2017-01-01", "2019-12-31"),
    ("2020-01-01", "2022-12-31"),
    ("2023-01-01", "2024-12-31"),
]

PARAM_GRID = [
    {"top_n": 5, "lookback": 80, "cash_buffer": 0.05},
    {"top_n": 10, "lookback": 80, "cash_buffer": 0.05},
    {"top_n": 15, "lookback": 80, "cash_buffer": 0.05},
]


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
        "mod": {"sys_progress": {"enabled": False}, "sys_analyser": {"enabled": True}},
    }


def run_one(params: dict, start: str, end: str) -> PeriodResult:
    old_env = {
        "RQALPHA_LOW_VOL_TOP_N": os.environ.get("RQALPHA_LOW_VOL_TOP_N"),
        "RQALPHA_LOW_VOL_LOOKBACK": os.environ.get("RQALPHA_LOW_VOL_LOOKBACK"),
        "RQALPHA_LOW_VOL_CASH_BUFFER": os.environ.get("RQALPHA_LOW_VOL_CASH_BUFFER"),
    }
    os.environ["RQALPHA_LOW_VOL_TOP_N"] = str(params["top_n"])
    os.environ["RQALPHA_LOW_VOL_LOOKBACK"] = str(params["lookback"])
    os.environ["RQALPHA_LOW_VOL_CASH_BUFFER"] = str(params["cash_buffer"])
    try:
        result = run_file(STRATEGY, config=base_config(start, end))
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
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


def main():
    all_results = []
    for params in PARAM_GRID:
        period_results = [run_one(params, start, end) for start, end in PERIODS]
        all_results.append(aggregate(params, period_results))

    all_results.sort(
        key=lambda r: (
            r["positive_period_ratio"],
            r["avg_calmar"],
            r["min_calmar"],
            r["avg_sharpe"],
            -r["worst_drawdown"],
        ),
        reverse=True,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"periods": PERIODS, "results": all_results}, ensure_ascii=False, indent=2))
    print(f"wrote {OUTPUT}")
    print(json.dumps(all_results[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
