from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "strategies" / "stock_selection").resolve()))
sys.path.insert(0, str((ROOT / "tools").resolve()))

from ml_next_day_lgbm_baseline import DEFAULT_OUTPUT_DIR, build_dataset
from run_low_turnover_lgbm_ranker_report import (
    BEST_PARAMS,
    BEST_SPEC,
    PROFILE_EXPOSURE,
    PROFILE_FILTERS,
    aggregate,
    backtest_with_trace,
    cost_summary,
    load_or_train_scores,
)
from search_low_turnover_lgbm_targets import add_open_forward_labels, train_predict_year


def main() -> int:
    parser = argparse.ArgumentParser(description="对比低换手 LightGBM Ranker 的 risk 与 steady 两个 profile")
    parser.add_argument("--years", default="2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025")
    parser.add_argument("--profiles", default="risk,steady", help="逗号分隔 profile，默认 risk,steady")
    parser.add_argument("--score-cache", default="", help="可选 pickle 路径；存在则复用年度打分，不存在则训练后写入")
    parser.add_argument("--output", default="", help="JSON 输出路径，默认写入 .temp")
    args = parser.parse_args()

    years = [int(item.strip()) for item in args.years.split(",") if item.strip()]
    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_dataset_cache.pkl"

    dataset, feature_cols = build_dataset(limit_stocks=None, cache_path=cache_path)
    dataset = add_open_forward_labels(dataset)

    score_cache_path = Path(args.score_cache) if args.score_cache else None
    scored_by_year = load_or_train_scores(dataset, feature_cols, BEST_SPEC, years, score_cache_path)
    profile_payloads = {}
    for profile in profiles:
        rows = []
        for year in years:
            metrics, trace = backtest_with_trace(
                scored_by_year[year],
                risk_filter=PROFILE_FILTERS[profile],
                exposure_rule=PROFILE_EXPOSURE[profile],
                **BEST_PARAMS,
            )
            metrics["year"] = year
            metrics["drawdown"] = trace["drawdown"]
            rows.append(metrics)
        profile_payloads[profile] = {
            "risk_filter": PROFILE_FILTERS[profile],
            "exposure_rule": PROFILE_EXPOSURE[profile],
            "years": rows,
            "aggregate": aggregate(rows),
        }

    year_table = []
    for year in years:
        row = {"year": year}
        base_return = None
        for profile in profiles:
            metrics = next(item for item in profile_payloads[profile]["years"] if item["year"] == year)
            row[f"{profile}_return"] = metrics["total_returns"]
            row[f"{profile}_max_drawdown"] = metrics["max_drawdown"]
            row[f"{profile}_avg_exposure"] = metrics.get("avg_exposure", 1.0)
            if base_return is None:
                base_return = metrics["total_returns"]
            else:
                row[f"{profile}_vs_{profiles[0]}_return_delta"] = metrics["total_returns"] - base_return
        year_table.append(row)

    payload = {
        "strategy": "low_turnover_lgbm_ranker_h10_step5_top5_buffer20",
        "spec": BEST_SPEC,
        "params": BEST_PARAMS,
        "profiles": profile_payloads,
        "year_table": year_table,
        "cost_assumptions": cost_summary(),
    }
    output_path = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / f"low_turnover_lgbm_profile_compare_{'_'.join(profiles)}_{'_'.join(map(str, years))}.json"
    )
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(payload, profiles, output_path)
    return 0


def print_report(payload: dict, profiles: list[str], output_path: Path) -> None:
    print()
    print("低换手 LightGBM Ranker Profile 对照")
    print("参数: 10日收益排序 / 每5个交易日调仓 / Top5 等权 / Top20 缓冲 / 无杠杆")
    for profile in profiles:
        agg = payload["profiles"][profile]["aggregate"]
        print(
            f"{profile}: 最弱年份 {_pct(agg['min_total_return'])}, "
            f"平均收益 {_pct(agg['avg_total_return'])}, "
            f"最差回撤 {_pct(agg['worst_max_drawdown'])}, "
            f"平均仓位 {_pct(agg['avg_exposure'])}"
        )
    print()
    print("逐年对照:")
    for row in payload["year_table"]:
        line = [str(row["year"])]
        for profile in profiles:
            line.append(
                f"{profile} 收益 {_pct(row[f'{profile}_return'])} / "
                f"回撤 {_pct(row[f'{profile}_max_drawdown'])} / "
                f"仓位 {_pct(row[f'{profile}_avg_exposure'])}"
            )
        if len(profiles) >= 2:
            last_profile = profiles[-1]
            line.append(f"{last_profile}-{profiles[0]} 收益差 {_pct(row[f'{last_profile}_vs_{profiles[0]}_return_delta'])}")
        print(" | ".join(line))
    print()
    print(f"完整 JSON: {output_path}")


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
