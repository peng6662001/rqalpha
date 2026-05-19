from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from rqalpha import run_file


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = Path.home() / ".rqalpha" / "bundle"
YEARS = range(2020, 2026)
OUTPUT = REPO_ROOT / ".temp" / "tune_external_research_strategies_2020_2025.json"
STRATEGIES = {
    "state_switch": REPO_ROOT / "strategies" / "stock_selection" / "strategy_state_switch_momentum_reversal.py",
    "high_residual": REPO_ROOT / "strategies" / "stock_selection" / "strategy_52week_high_residual_momentum.py",
}


@dataclass
class YearResult:
    year: int
    total_returns: float
    annualized_returns: float
    max_drawdown: float
    sharpe: float


def base_config(strategy_file: Path, start: str, end: str) -> dict:
    return {
        "base": {
            "strategy_file": str(strategy_file),
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
                "stock_min_commission": 5,
            },
        },
    }


def run_one_year(strategy_file: Path, year: int, env_overrides: dict[str, str]) -> YearResult:
    old_values = {k: os.environ.get(k) for k in env_overrides}
    try:
        os.environ.update(env_overrides)
        result = run_file(str(strategy_file), config=base_config(strategy_file, f"{year}-01-01", f"{year}-12-31"))
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


def aggregate(name: str, strategy_key: str, params: dict[str, float | str], rows: list[YearResult]) -> dict:
    return {
        "name": name,
        "strategy_key": strategy_key,
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
    return {k: str(v) for k, v in params.items()}


def strategy_variants():
    return {
        "state_switch": [
            (
                "state_switch_base",
                {
                    "RQALPHA_TOP_N": 5,
                    "RQALPHA_HOLD_BUFFER_RANK": 8,
                    "RQALPHA_BULL_RET_THRESHOLD": 0.00,
                    "RQALPHA_BEAR_MA_GAP": -0.03,
                    "RQALPHA_MOM_WEIGHT": 1.0,
                    "RQALPHA_BREAKOUT_WEIGHT": 0.35,
                    "RQALPHA_VOL_PENALTY": 0.30,
                    "RQALPHA_REVERSAL_PENALTY": 0.25,
                    "RQALPHA_LONG_CONFIRM_WEIGHT": 0.50,
                },
            ),
            (
                "state_switch_bear_cash",
                {
                    "RQALPHA_TOP_N": 5,
                    "RQALPHA_HOLD_BUFFER_RANK": 8,
                    "RQALPHA_BEAR_TO_CASH": 1,
                    "RQALPHA_BEAR_MA_GAP": -0.02,
                    "RQALPHA_BULL_RET_THRESHOLD": 0.01,
                },
            ),
            (
                "state_switch_faster_reversal",
                {
                    "RQALPHA_TOP_N": 5,
                    "RQALPHA_HOLD_BUFFER_RANK": 10,
                    "RQALPHA_REV_WINDOW": 3,
                    "RQALPHA_LONG_CONFIRM_WINDOW": 40,
                    "RQALPHA_REVERSAL_WEIGHT": 1.2,
                    "RQALPHA_LONG_CONFIRM_WEIGHT": 0.35,
                    "RQALPHA_OVERSOLD_BONUS": 0.30,
                },
            ),
            (
                "state_switch_stronger_momentum",
                {
                    "RQALPHA_TOP_N": 5,
                    "RQALPHA_HOLD_BUFFER_RANK": 8,
                    "RQALPHA_SKIP_RECENT": 10,
                    "RQALPHA_BREAKOUT_WEIGHT": 0.50,
                    "RQALPHA_REVERSAL_PENALTY": 0.10,
                    "RQALPHA_MOM_WEIGHT": 1.15,
                },
            ),
        ],
        "high_residual": [
            (
                "high_residual_base",
                {
                    "RQALPHA_TOP_N": 5,
                    "RQALPHA_HOLD_BUFFER_RANK": 8,
                    "RQALPHA_RESIDUAL_WEIGHT": 1.0,
                    "RQALPHA_HIGH_WEIGHT": 0.8,
                    "RQALPHA_TREND_WEIGHT": 0.25,
                    "RQALPHA_VOL_PENALTY": 0.20,
                },
            ),
            (
                "high_residual_high_focus",
                {
                    "RQALPHA_TOP_N": 5,
                    "RQALPHA_HOLD_BUFFER_RANK": 8,
                    "RQALPHA_RESIDUAL_WEIGHT": 0.8,
                    "RQALPHA_HIGH_WEIGHT": 1.1,
                    "RQALPHA_TREND_WEIGHT": 0.20,
                    "RQALPHA_VOL_PENALTY": 0.15,
                },
            ),
            (
                "high_residual_resid_focus",
                {
                    "RQALPHA_TOP_N": 5,
                    "RQALPHA_HOLD_BUFFER_RANK": 10,
                    "RQALPHA_RESIDUAL_WEIGHT": 1.3,
                    "RQALPHA_HIGH_WEIGHT": 0.6,
                    "RQALPHA_TREND_WEIGHT": 0.30,
                    "RQALPHA_BETA_PENALTY": 0.15,
                },
            ),
            (
                "high_residual_tighter_top3",
                {
                    "RQALPHA_TOP_N": 3,
                    "RQALPHA_HOLD_BUFFER_RANK": 5,
                    "RQALPHA_RESIDUAL_WEIGHT": 1.0,
                    "RQALPHA_HIGH_WEIGHT": 0.9,
                    "RQALPHA_TREND_WEIGHT": 0.35,
                    "RQALPHA_VOL_PENALTY": 0.10,
                },
            ),
        ],
    }


def main():
    only_strategy = os.environ.get("RQALPHA_ONLY_STRATEGY")
    only_variant = os.environ.get("RQALPHA_ONLY_VARIANT")
    years = YEARS
    if os.environ.get("RQALPHA_ONLY_YEAR"):
        y = int(os.environ["RQALPHA_ONLY_YEAR"])
        years = [y]
    all_results = []
    for strategy_key, variants in strategy_variants().items():
        if only_strategy and strategy_key != only_strategy:
            continue
        strategy_file = STRATEGIES[strategy_key]
        for name, params in variants:
            if only_variant and name != only_variant:
                continue
            rows = [run_one_year(strategy_file, year, env_params(params)) for year in years]
            item = aggregate(name, strategy_key, params, rows)
            all_results.append(item)
            print(name, json.dumps(item["aggregate"], ensure_ascii=False))
    all_results.sort(key=lambda item: item["aggregate"]["avg_total_return"], reverse=True)
    OUTPUT.write_text(json.dumps({"results": all_results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {OUTPUT}")
    if all_results:
        print("best", json.dumps(all_results[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
