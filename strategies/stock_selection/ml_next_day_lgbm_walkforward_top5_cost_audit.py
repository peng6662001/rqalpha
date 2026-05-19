import json

import pandas as pd

from ml_next_day_lgbm_baseline import DEFAULT_OUTPUT_DIR, build_dataset, compute_metrics, lgb


TOP_N = 5
YEARS = range(2010, 2021)
SCENARIOS = {
    "no_cost": {
        "buy_commission": 0.0,
        "sell_commission": 0.0,
        "sell_tax": 0.0,
        "buy_slippage": 0.0,
        "sell_slippage": 0.0,
    },
    "mild_cost": {
        "buy_commission": 0.0003,
        "sell_commission": 0.0003,
        "sell_tax": 0.0010,
        "buy_slippage": 0.0010,
        "sell_slippage": 0.0010,
    },
    "stress_cost": {
        "buy_commission": 0.0003,
        "sell_commission": 0.0003,
        "sell_tax": 0.0010,
        "buy_slippage": 0.0020,
        "sell_slippage": 0.0020,
    },
}


def train_predict_year(dataset, feature_cols, year):
    train_end = pd.Timestamp(f"{year - 1}-12-31")
    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year}-12-31")

    labeled = dataset.dropna(subset=["next_ret_1", "benchmark_next_ret_1"]).copy()
    train_df = labeled[labeled["date"] <= train_end].copy()
    test_df = labeled[(labeled["date"] >= test_start) & (labeled["date"] <= test_end)].copy()
    if train_df.empty or test_df.empty:
        return None

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_df[feature_cols], train_df["next_ret_1"])
    test_df["pred"] = model.predict(test_df[feature_cols])
    return test_df


def backtest_topn_with_costs(scored, scenario, top_n=TOP_N):
    daily = []
    benchmark = []
    scored = scored.sort_values(["date", "pred"], ascending=[True, False])
    roundtrip_cost = (
        scenario["buy_commission"]
        + scenario["sell_commission"]
        + scenario["sell_tax"]
        + scenario["buy_slippage"]
        + scenario["sell_slippage"]
    )
    diagnostics = {
        "avg_invested_ratio": 0.0,
        "days": 0,
        "avg_names_bought": 0.0,
    }

    invested_ratios = []
    names_bought = []
    for _, group in scored.groupby("date"):
        picks = group.head(top_n).copy()
        if picks.empty:
            continue
        executable = picks[picks["next_open_buyable"]].copy()
        invested_ratio = len(executable) / float(top_n)
        invested_ratios.append(invested_ratio)
        names_bought.append(len(executable))

        if executable.empty:
            daily.append(0.0)
            benchmark.append(float(group["benchmark_next_ret_1"].iloc[0]))
            continue

        gross = executable["next_open_to_close_ret"] - roundtrip_cost
        realized = float(gross.mean() * invested_ratio)
        daily.append(realized)
        benchmark.append(float(group["benchmark_next_ret_1"].iloc[0]))

    diagnostics["days"] = len(invested_ratios)
    diagnostics["avg_invested_ratio"] = sum(invested_ratios) / len(invested_ratios) if invested_ratios else 0.0
    diagnostics["avg_names_bought"] = sum(names_bought) / len(names_bought) if names_bought else 0.0
    metrics = compute_metrics(daily, benchmark)
    metrics.update(diagnostics)
    return metrics


def aggregate_metrics(rows):
    return {
        "avg_total_return": sum(r["total_returns"] for r in rows) / len(rows),
        "avg_annualized_returns": sum(r["annualized_returns"] for r in rows) / len(rows),
        "worst_year_return": min(r["total_returns"] for r in rows),
        "best_year_return": max(r["total_returns"] for r in rows),
        "worst_max_drawdown": max(r["max_drawdown"] for r in rows),
        "positive_years": sum(1 for r in rows if r["total_returns"] > 0),
        "avg_invested_ratio": sum(r["avg_invested_ratio"] for r in rows) / len(rows),
        "avg_names_bought": sum(r["avg_names_bought"] for r in rows) / len(rows),
    }


def main():
    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_dataset_cache.pkl"
    dataset, feature_cols = build_dataset(limit_stocks=None, cache_path=cache_path)

    scored_by_year = {}
    for year in YEARS:
        scored = train_predict_year(dataset, feature_cols, year)
        if scored is not None:
            scored_by_year[year] = scored

    summary = {"strategy": "ml_next_day_lgbm_walkforward_top5_cost_audit", "period": "2010-2020", "scenarios": {}}
    for name, scenario in SCENARIOS.items():
        rows = []
        for year, scored in scored_by_year.items():
            metrics = backtest_topn_with_costs(scored.copy(), scenario, top_n=TOP_N)
            metrics["year"] = year
            rows.append(metrics)
            print(name, year, json.dumps(metrics, ensure_ascii=False))
        summary["scenarios"][name] = {
            "assumptions": scenario,
            "years": rows,
            "aggregate": aggregate_metrics(rows),
        }
        print(name, json.dumps(summary["scenarios"][name]["aggregate"], ensure_ascii=False))

    out = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_walkforward_top5_cost_audit_2010_2020.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
