from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rqalpha import run_file


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = Path.home() / ".rqalpha" / "bundle"
OUTPUT = REPO_ROOT / ".temp" / "strategy_period_stability.json"

PERIODS = [
    ("2014-01-01", "2016-12-31"),
    ("2017-01-01", "2019-12-31"),
    ("2020-01-01", "2022-12-31"),
    ("2023-01-01", "2024-12-31"),
]


@dataclass
class PeriodResult:
    period: str
    total_returns: float
    annualized_returns: float
    max_drawdown: float
    sharpe: float
    calmar: float | None


def strategy_specs():
    stock_config = {
        "base": {
            "frequency": "1d",
            "matching_type": "current_bar",
            "accounts": {"stock": 1000000},
        },
        "extra": {"log_level": "error"},
    }
    return [
        ("strategy_dual_thrust_existing", "strategies/stock_selection/strategy_dual_thrust_existing.py", "000300.XSHG"),
        ("strategy_turtle_existing", "strategies/stock_selection/strategy_turtle_existing.py", "000300.XSHG"),
        ("example_buy_and_hold", "rqalpha/examples/buy_and_hold.py", "000300.XSHG"),
        ("example_golden_cross", "rqalpha/examples/golden_cross.py", "000300.XSHG"),
        ("example_macd", "rqalpha/examples/macd.py", "000300.XSHG"),
        ("example_rsi", "rqalpha/examples/rsi.py", "000300.XSHG"),
        ("strategy_momentum_20", "strategies/stock_selection/strategy_momentum_20.py", "000300.XSHG"),
        ("strategy_momentum_60", "strategies/stock_selection/strategy_momentum_60.py", "000300.XSHG"),
        ("strategy_dual_momentum", "strategies/stock_selection/strategy_dual_momentum.py", "000300.XSHG"),
        ("strategy_low_volatility", "strategies/stock_selection/strategy_low_volatility.py", "000300.XSHG"),
        ("strategy_low_vol_momentum", "strategies/stock_selection/strategy_low_vol_momentum.py", "000300.XSHG"),
        ("strategy_mean_reversion_5", "strategies/stock_selection/strategy_mean_reversion_5.py", "000300.XSHG"),
        ("strategy_rsi_reversion", "strategies/stock_selection/strategy_rsi_reversion.py", "000300.XSHG"),
        ("strategy_breakout_55", "strategies/stock_selection/strategy_breakout_55.py", "000300.XSHG"),
        ("strategy_ma_strength", "strategies/stock_selection/strategy_ma_strength.py", "000300.XSHG"),
        ("strategy_quality_trend", "strategies/stock_selection/strategy_quality_trend.py", "000300.XSHG"),
    ], stock_config


def run_one(path: str, benchmark: str, start: str, end: str, base_config: dict):
    config = json.loads(json.dumps(base_config))
    config["base"]["start_date"] = start
    config["base"]["end_date"] = end
    config["base"]["benchmark"] = benchmark
    config["base"]["data_bundle_path"] = str(BUNDLE_ROOT)
    config["mod"] = {"sys_progress": {"enabled": False}, "sys_analyser": {"enabled": True}}
    result = run_file(path, config=config)
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


def aggregate(name: str, results: list[PeriodResult]):
    calmar_values = [r.calmar for r in results if r.calmar is not None]
    positive_periods = sum(1 for r in results if r.total_returns > 0)
    return {
        "name": name,
        "period_results": [asdict(r) for r in results],
        "positive_period_ratio": positive_periods / len(results),
        "avg_calmar": sum(calmar_values) / len(calmar_values),
        "min_calmar": min(calmar_values),
        "avg_sharpe": sum(r.sharpe for r in results) / len(results),
        "worst_drawdown": max(r.max_drawdown for r in results),
        "avg_annualized_returns": sum(r.annualized_returns for r in results) / len(results),
    }


def main():
    specs, base_config = strategy_specs()
    all_results = []
    for name, path, benchmark in specs:
        period_results = []
        for start, end in PERIODS:
            period_results.append(run_one(path, benchmark, start, end, base_config))
        all_results.append(aggregate(name, period_results))

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
    print(f"strategies={len(all_results)} periods={len(PERIODS)}")


if __name__ == "__main__":
    main()
