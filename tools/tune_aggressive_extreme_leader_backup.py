from __future__ import annotations

import itertools
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from rqalpha import run_file


REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGY = REPO_ROOT / "strategies" / "stock_selection" / "strategy_market_drawdown_scale_in_aggressive_extreme_leader_backup.py"
OUTPUT = REPO_ROOT / ".temp" / "tune_aggressive_extreme_leader_backup.json"
BUNDLE_ROOT = Path.home() / ".rqalpha" / "bundle"
YEARS = range(2020, 2026)


@dataclass
class YearResult:
    year: int
    total_returns: float
    annualized_returns: float
    max_drawdown: float
    sharpe: float


def base_config(start: str, end: str) -> dict:
    return {
        "base": {
            "strategy_file": str(STRATEGY),
            "start_date": start,
            "end_date": end,
            "frequency": "1d",
            "matching_type": "current_bar",
            "benchmark": "000300.XSHG",
            "accounts": {"stock": 1000000},
            "data_bundle_path": str(BUNDLE_ROOT),
        },
        "extra": {"log_level": "error"},
        "mod": {
            "sys_progress": {"enabled": False},
            "sys_analyser": {"enabled": True},
            "sys_simulation": {
                "enabled": True,
                "matching_type": "current_bar",
                "slippage_model": "PriceRatioSlippage",
                "slippage": 0.001,
            },
            "sys_transaction_cost": {
                "enabled": True,
                "stock_commission_multiplier": 0.375,
                "tax_multiplier": 1.0,
                "cn_stock_min_commission": 5,
            },
        },
    }


def run_one_year(year: int, env_overrides: dict[str, str]) -> YearResult:
    old_values = {k: os.environ.get(k) for k in env_overrides}
    try:
        os.environ.update(env_overrides)
        result = run_file(str(STRATEGY), config=base_config(f"{year}-01-01", f"{year}-12-31"))
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    summary = result["sys_analyser"]["summary"]
    return YearResult(
        year=year,
        total_returns=summary["total_returns"],
        annualized_returns=summary["annualized_returns"],
        max_drawdown=summary["max_drawdown"],
        sharpe=summary["sharpe"],
    )


def aggregate(label: str, params: dict[str, float | str], rows: list[YearResult]) -> dict:
    return {
        "label": label,
        "params": params,
        "years": [asdict(r) for r in rows],
        "aggregate": {
            "avg_total_return": sum(r.total_returns for r in rows) / len(rows),
            "avg_annualized_returns": sum(r.annualized_returns for r in rows) / len(rows),
            "worst_year_return": min(r.total_returns for r in rows),
            "best_year_return": max(r.total_returns for r in rows),
            "worst_max_drawdown": max(r.max_drawdown for r in rows),
            "positive_years": sum(1 for r in rows if r.total_returns > 0),
            "avg_sharpe": sum(r.sharpe for r in rows) / len(rows),
        },
    }


def env_params(params: dict[str, float | str]) -> dict[str, str]:
    mapping = {}
    for key, value in params.items():
        mapping[key] = str(value)
    return mapping


def sweep() -> list[dict]:
    baseline = {
        "RQALPHA_PRIMARY": "300059.XSHE",
        "RQALPHA_BACKUP": "159949.XSHE",
        "RQALPHA_ENTRY_DRAWDOWN": 0.02,
        "RQALPHA_RESET_DRAWDOWN": 0.015,
        "RQALPHA_LOOKBACK": 30,
        "RQALPHA_BULL_GAP": 0.02,
        "RQALPHA_SUPER_BULL_GAP": 0.06,
        "RQALPHA_SWITCH_DELTA": 0.03,
        "RQALPHA_PROFIT_TRIM_1": 0.60,
        "RQALPHA_PROFIT_TRIM_2": 0.95,
        "RQALPHA_PROFIT_EXIT": 1.50,
        "RQALPHA_TRAILING_ACTIVATION": 0.75,
        "RQALPHA_TRAILING_STOP": 0.21,
        "RQALPHA_TRAILING_EXIT": 0.32,
        "RQALPHA_LOCK1_PROFIT": 0.85,
        "RQALPHA_LOCK1_CAP": 0.92,
        "RQALPHA_LOCK2_PROFIT": 1.20,
        "RQALPHA_LOCK2_CAP": 0.78,
        "RQALPHA_LOCK3_PROFIT": 1.60,
        "RQALPHA_LOCK3_CAP": 0.50,
    }

    coarse_grid = {
        "RQALPHA_ENTRY_DRAWDOWN": [0.015, 0.02, 0.03],
        "RQALPHA_LOOKBACK": [20, 30, 40],
        "RQALPHA_SWITCH_DELTA": [0.015, 0.03, 0.05],
        "RQALPHA_PROFIT_TRIM_1": [0.45, 0.60, 0.75],
        "RQALPHA_PROFIT_TRIM_2": [0.80, 0.95, 1.10],
        "RQALPHA_PROFIT_EXIT": [1.20, 1.50, 1.80],
        "RQALPHA_TRAILING_ACTIVATION": [0.60, 0.75, 0.90],
        "RQALPHA_TRAILING_STOP": [0.16, 0.21, 0.26],
        "RQALPHA_TRAILING_EXIT": [0.26, 0.32, 0.38],
        "RQALPHA_SUPER_BULL_GAP": [0.04, 0.06, 0.08],
    }

    candidate_params = []

    for key, values in coarse_grid.items():
        for value in values:
            params = dict(baseline)
            params[key] = value
            candidate_params.append((f"coarse_{key}_{value}", params))

    candidate_params.extend(
        [
            (
                "combo_fast_entry_loose_trail",
                {
                    **baseline,
                    "RQALPHA_ENTRY_DRAWDOWN": 0.015,
                    "RQALPHA_LOOKBACK": 20,
                    "RQALPHA_SWITCH_DELTA": 0.015,
                    "RQALPHA_PROFIT_TRIM_1": 0.75,
                    "RQALPHA_PROFIT_TRIM_2": 1.10,
                    "RQALPHA_PROFIT_EXIT": 1.80,
                    "RQALPHA_TRAILING_ACTIVATION": 0.90,
                    "RQALPHA_TRAILING_STOP": 0.26,
                    "RQALPHA_TRAILING_EXIT": 0.38,
                    "RQALPHA_SUPER_BULL_GAP": 0.04,
                },
            ),
            (
                "combo_fast_switch_bull",
                {
                    **baseline,
                    "RQALPHA_ENTRY_DRAWDOWN": 0.015,
                    "RQALPHA_LOOKBACK": 20,
                    "RQALPHA_SWITCH_DELTA": 0.01,
                    "RQALPHA_PROFIT_TRIM_1": 0.75,
                    "RQALPHA_PROFIT_TRIM_2": 1.10,
                    "RQALPHA_PROFIT_EXIT": 1.80,
                    "RQALPHA_TRAILING_ACTIVATION": 0.90,
                    "RQALPHA_TRAILING_STOP": 0.26,
                    "RQALPHA_TRAILING_EXIT": 0.38,
                    "RQALPHA_SUPER_BULL_GAP": 0.04,
                },
            ),
            (
                "combo_hold_winners_longer",
                {
                    **baseline,
                    "RQALPHA_PROFIT_TRIM_1": 0.85,
                    "RQALPHA_PROFIT_TRIM_2": 1.25,
                    "RQALPHA_PROFIT_EXIT": 2.00,
                    "RQALPHA_TRAILING_ACTIVATION": 1.00,
                    "RQALPHA_TRAILING_STOP": 0.28,
                    "RQALPHA_TRAILING_EXIT": 0.40,
                    "RQALPHA_LOCK1_PROFIT": 1.00,
                    "RQALPHA_LOCK2_PROFIT": 1.40,
                    "RQALPHA_LOCK3_PROFIT": 1.90,
                },
            ),
        ]
    )

    outputs = []
    for label, params in candidate_params:
        rows = [run_one_year(year, env_params(params)) for year in YEARS]
        item = aggregate(label, params, rows)
        outputs.append(item)
        print(label, json.dumps(item["aggregate"], ensure_ascii=False))

    outputs.sort(key=lambda item: item["aggregate"]["avg_total_return"], reverse=True)
    return outputs


def main():
    results = sweep()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"strategy": STRATEGY.name, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {OUTPUT}")
    print(json.dumps(results[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
