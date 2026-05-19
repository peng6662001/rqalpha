from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from lightgbm.basic import LightGBMError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "strategies" / "stock_selection").resolve()))

from ml_next_day_lgbm_baseline import BUNDLE_DIR, DEFAULT_OUTPUT_DIR, build_dataset, compute_metrics, lgb


YEARS = (2021, 2023, 2025)
LABEL_HORIZONS = (3, 5, 10, 20)
LABEL_MODES = ("ret", "excess", "cls")
MODEL_KINDS = ("reg", "rank", "cls")
REBALANCE_STEPS = (5, 10, 20)
TOP_NS = (1, 3, 5, 10)
HOLD_BUFFERS = (0, 10, 20)
SCENARIO = {
    "buy_commission": 0.0003,
    "sell_commission": 0.0003,
    "sell_tax": 0.0010,
    "buy_slippage": 0.0010,
    "sell_slippage": 0.0010,
}
BUY_COST = SCENARIO["buy_commission"] + SCENARIO["buy_slippage"]
SELL_COST = SCENARIO["sell_commission"] + SCENARIO["sell_tax"] + SCENARIO["sell_slippage"]
LGBM_BACKEND: str | None = None
LGBM_GPU_PARAMS: dict[str, int] = {}


def main() -> None:
    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = DEFAULT_OUTPUT_DIR / "ml_next_day_lgbm_dataset_cache.pkl"
    dataset, feature_cols = build_dataset(limit_stocks=None, cache_path=cache_path)
    dataset = add_open_forward_labels(dataset)

    scored_cache = {}
    train_specs = build_train_specs()
    for spec in train_specs:
        key = spec_key(spec)
        scored_cache[key] = {}
        for year in YEARS:
            print(f"train {key} year={year}", flush=True)
            scored_cache[key][year] = train_predict_year(dataset, feature_cols, spec, year)

    results = []
    out = DEFAULT_OUTPUT_DIR / "low_turnover_lgbm_search_2021_2023_2025.json"
    for spec in train_specs:
        key = spec_key(spec)
        for step in REBALANCE_STEPS:
            for top_n in TOP_NS:
                for hold_buffer in HOLD_BUFFERS:
                    if hold_buffer and hold_buffer <= top_n:
                        continue
                    rows = []
                    for year in YEARS:
                        metrics = backtest_low_turnover(
                            scored_cache[key][year],
                            top_n=top_n,
                            rebalance_step=step,
                            hold_buffer=hold_buffer,
                        )
                        metrics["year"] = year
                        rows.append(metrics)
                    record = {
                        **spec,
                        "rebalance_step": step,
                        "top_n": top_n,
                        "hold_buffer": hold_buffer,
                        "years": rows,
                        "aggregate": aggregate(rows),
                    }
                    results.append(record)
                    out.write_text(
                        json.dumps(
                            {
                                "strategy_family": "low_turnover_lgbm_open_to_open",
                                "years": list(YEARS),
                                "cost_assumptions": SCENARIO,
                                "results": results,
                                "best_result": best_record(results),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    agg = record["aggregate"]
                    print(
                        f"{key} step={step} top={top_n} buffer={hold_buffer} "
                        f"min={agg['min_total_return']:.2%} avg={agg['avg_total_return']:.2%} "
                        f"mdd={agg['worst_max_drawdown']:.2%} gt50={agg['all_years_gt_50']}",
                        flush=True,
                    )

    results = sorted(
        results,
        key=lambda r: (
            r["aggregate"]["all_years_gt_50"],
            r["aggregate"]["min_total_return"],
            r["aggregate"]["avg_total_return"],
            -r["aggregate"]["worst_max_drawdown"],
        ),
        reverse=True,
    )
    payload = {
        "strategy_family": "low_turnover_lgbm_open_to_open",
        "years": list(YEARS),
        "cost_assumptions": SCENARIO,
        "results": results,
        "best_result": results[0] if results else None,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")
    print(json.dumps(payload["best_result"], ensure_ascii=False, indent=2))


def add_open_forward_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    needed = ["order_book_id", "date", "open", "next_open", "next_open_buyable"]
    for horizon in LABEL_HORIZONS:
        needed.append(f"forward_open_ret_{horizon}")

    frames = []
    ids = dataset["order_book_id"].drop_duplicates().tolist()
    with h5py.File(BUNDLE_DIR / "stocks.h5", "r") as store:
        for order_book_id in ids:
            if order_book_id not in store:
                continue
            raw = pd.DataFrame.from_records(store[order_book_id][:])
            if raw.empty:
                continue
            raw["date"] = pd.to_datetime(raw["datetime"].astype(str).str[:8])
            raw = raw.sort_values("date").reset_index(drop=True)
            frame = pd.DataFrame(
                {
                    "order_book_id": order_book_id,
                    "date": raw["date"],
                    "open": raw["open"].astype(float),
                    "next_open": raw["open"].astype(float).shift(-1),
                    "next_limit_up": raw["limit_up"].astype(float).shift(-1),
                    "next_limit_down": raw["limit_down"].astype(float).shift(-1),
                }
            )
            frame["next_open_buyable"] = (
                (frame["next_open"] < frame["next_limit_up"] * 0.999)
                & (frame["next_open"] > frame["next_limit_down"] * 1.001)
                & frame["next_open"].notna()
            )
            for horizon in LABEL_HORIZONS:
                exit_open = raw["open"].astype(float).shift(-(horizon + 1))
                frame[f"forward_open_ret_{horizon}"] = exit_open / frame["next_open"] - 1.0
            frames.append(frame[needed])

    forward = pd.concat(frames, ignore_index=True)
    drop_cols = [col for col in needed if col in dataset.columns and col not in ("order_book_id", "date")]
    dataset = dataset.drop(columns=drop_cols).merge(forward, on=["order_book_id", "date"], how="left")
    benchmark_forward = build_benchmark_forward(dataset)
    for horizon in LABEL_HORIZONS:
        dataset[f"target_ret_{horizon}"] = dataset[f"forward_open_ret_{horizon}"]
        dataset[f"target_excess_{horizon}"] = dataset[f"forward_open_ret_{horizon}"] - dataset["date"].map(
            benchmark_forward[horizon]
        )
        dataset[f"target_cls_{horizon}"] = (dataset[f"forward_open_ret_{horizon}"] > 0.02).astype(int)
    return dataset


def build_benchmark_forward(dataset: pd.DataFrame) -> dict[int, dict[pd.Timestamp, float]]:
    daily = dataset.groupby("date")["benchmark_next_ret_1"].first().sort_index()
    out = {}
    for horizon in LABEL_HORIZONS:
        forward = pd.Series(index=daily.index, dtype=float)
        values = daily.to_numpy(dtype=float)
        for idx, date in enumerate(daily.index):
            window = values[idx : idx + horizon]
            if len(window) < horizon or np.isnan(window).any():
                continue
            forward.loc[date] = float(np.prod(1.0 + window) - 1.0)
        out[horizon] = dict(forward.dropna())
    return out


def build_train_specs() -> list[dict]:
    return [
        {"label_horizon": 3, "label_mode": "ret", "model_kind": "rank"},
        {"label_horizon": 5, "label_mode": "ret", "model_kind": "rank"},
        {"label_horizon": 10, "label_mode": "ret", "model_kind": "rank"},
        {"label_horizon": 20, "label_mode": "ret", "model_kind": "rank"},
        {"label_horizon": 5, "label_mode": "excess", "model_kind": "reg"},
        {"label_horizon": 10, "label_mode": "excess", "model_kind": "reg"},
        {"label_horizon": 20, "label_mode": "excess", "model_kind": "reg"},
        {"label_horizon": 5, "label_mode": "excess", "model_kind": "rank"},
        {"label_horizon": 10, "label_mode": "excess", "model_kind": "rank"},
        {"label_horizon": 20, "label_mode": "excess", "model_kind": "rank"},
        {"label_horizon": 5, "label_mode": "cls", "model_kind": "cls"},
        {"label_horizon": 10, "label_mode": "cls", "model_kind": "cls"},
    ]


def spec_key(spec: dict) -> str:
    return f"h{spec['label_horizon']}_{spec['label_mode']}_{spec['model_kind']}"


def train_predict_year(dataset: pd.DataFrame, feature_cols: list[str], spec: dict, year: int) -> pd.DataFrame:
    horizon = spec["label_horizon"]
    label_col = f"target_{spec['label_mode']}_{horizon}"
    labeled = dataset.dropna(subset=[label_col, "benchmark_next_ret_1"]).copy()
    train_df = labeled[labeled["date"] <= pd.Timestamp(f"{year - 1}-12-31")].copy()
    test_df = labeled[(labeled["date"] >= pd.Timestamp(f"{year}-01-01")) & (labeled["date"] <= pd.Timestamp(f"{year}-12-31"))].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError(f"empty train/test for {label_col} {year}")

    model_kind = spec["model_kind"]
    if model_kind == "reg":
        model = build_lgbm_estimator(
            lgb.LGBMRegressor,
            objective="regression",
            n_estimators=220,
            learning_rate=0.04,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=4,
            verbosity=-1,
        )
        model.fit(train_df[feature_cols], train_df[label_col])
        test_df["pred"] = model.predict(test_df[feature_cols])
    elif model_kind == "rank":
        grouped = train_df.sort_values(["date", "order_book_id"]).copy()
        grouped["_rank_label"] = (
            grouped.groupby("date")[label_col]
            .rank(method="first", pct=True)
            .mul(9)
            .round()
            .astype(int)
        )
        group_sizes = grouped.groupby("date").size().tolist()
        model = build_lgbm_estimator(
            lgb.LGBMRanker,
            objective="lambdarank",
            metric="ndcg",
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=4,
            verbosity=-1,
        )
        model.fit(grouped[feature_cols], grouped["_rank_label"], group=group_sizes)
        test_df["pred"] = model.predict(test_df[feature_cols])
    elif model_kind == "cls":
        model = build_lgbm_estimator(
            lgb.LGBMClassifier,
            objective="binary",
            n_estimators=220,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=4,
            verbosity=-1,
        )
        model.fit(train_df[feature_cols], train_df[label_col])
        test_df["pred"] = model.predict_proba(test_df[feature_cols])[:, 1]
    else:
        raise ValueError(f"unsupported model kind: {model_kind}")
    return test_df


def build_lgbm_estimator(cls, **kwargs):
    global LGBM_BACKEND
    backend = LGBM_BACKEND or detect_lgbm_backend()
    params = dict(kwargs)
    if backend == "gpu":
        params["device_type"] = "gpu"
        params.update(LGBM_GPU_PARAMS)
    return cls(**params)


def detect_lgbm_backend() -> str:
    global LGBM_BACKEND, LGBM_GPU_PARAMS
    if LGBM_BACKEND is not None:
        return LGBM_BACKEND

    sample_x = np.random.RandomState(0).randn(256, 6)
    sample_y = np.random.RandomState(1).randn(256)
    gpu_candidates = [
        {"gpu_platform_id": 1, "gpu_device_id": 0},
        {"gpu_platform_id": 0, "gpu_device_id": 0},
        {"gpu_platform_id": 0, "gpu_device_id": 1},
        {},
    ]
    last_exc = None
    for gpu_params in gpu_candidates:
        try:
            probe = lgb.LGBMRegressor(
                n_estimators=4,
                num_leaves=15,
                learning_rate=0.1,
                random_state=42,
                n_jobs=1,
                verbosity=-1,
                device_type="gpu",
                **gpu_params,
            )
            probe.fit(sample_x, sample_y)
            LGBM_BACKEND = "gpu"
            LGBM_GPU_PARAMS = gpu_params
            if gpu_params:
                print(f"LightGBM backend: GPU (OpenCL) {gpu_params}", flush=True)
            else:
                print("LightGBM backend: GPU (OpenCL) auto", flush=True)
            break
        except (LightGBMError, RuntimeError, ValueError) as exc:
            last_exc = exc
    if LGBM_BACKEND is None:
        LGBM_BACKEND = "cpu"
        LGBM_GPU_PARAMS = {}
        print(f"LightGBM backend: CPU fallback ({last_exc!r})", flush=True)
    return LGBM_BACKEND


def backtest_low_turnover(scored: pd.DataFrame, top_n: int, rebalance_step: int, hold_buffer: int) -> dict:
    scored = scored.sort_values(["date", "pred"], ascending=[True, False]).copy()
    all_dates = sorted(pd.to_datetime(scored["date"].drop_duplicates()))
    rebalance_dates = all_dates[::rebalance_step]
    by_date = {date: group.copy() for date, group in scored.groupby("date")}

    holdings: dict[str, float] = {}
    cash = 1.0
    equity_curve = []
    benchmark_returns = []
    turnover_values = []

    for idx, current_date in enumerate(all_dates):
        group = by_date[current_date]
        if current_date in rebalance_dates:
            ranked = group.sort_values("pred", ascending=False)
            ranked_ids = ranked["order_book_id"].tolist()
            rank_map = {order_book_id: rank + 1 for rank, order_book_id in enumerate(ranked_ids)}
            target_ids = select_with_buffer(ranked_ids, holdings.keys(), top_n, hold_buffer, rank_map)
            buyable = set(ranked.loc[ranked["next_open_buyable"], "order_book_id"])
            target_ids = [order_book_id for order_book_id in target_ids if order_book_id in buyable]
            target_weight = 1.0 / len(target_ids) if target_ids else 0.0
            targets = {order_book_id: target_weight for order_book_id in target_ids}

            sell_turnover = sum(max(holdings.get(order_book_id, 0.0) - targets.get(order_book_id, 0.0), 0.0) for order_book_id in set(holdings) | set(targets))
            buy_turnover = sum(max(targets.get(order_book_id, 0.0) - holdings.get(order_book_id, 0.0), 0.0) for order_book_id in set(holdings) | set(targets))
            cost = min(sell_turnover * SELL_COST + buy_turnover * BUY_COST, 0.99)
            if targets:
                holdings = {order_book_id: weight * (1.0 - cost) for order_book_id, weight in targets.items()}
                cash = 0.0
            else:
                holdings = {}
                cash = max(1.0 - cost, 0.0)
            turnover_values.append(sell_turnover + buy_turnover)

        next_returns = group.set_index("order_book_id")["next_ret_1"].to_dict()
        portfolio_ret = 0.0
        next_holdings = {}
        for order_book_id, weight in holdings.items():
            ret = next_returns.get(order_book_id)
            if ret is None or pd.isna(ret):
                next_holdings[order_book_id] = weight
                continue
            new_weight = weight * (1.0 + float(ret))
            portfolio_ret += new_weight - weight
            next_holdings[order_book_id] = new_weight
        gross_equity = cash + sum(next_holdings.values())
        if gross_equity <= 0:
            return failed_metrics()
        holdings = {order_book_id: weight / gross_equity for order_book_id, weight in next_holdings.items()}
        cash = cash / gross_equity
        equity_curve.append(gross_equity - 1.0)
        benchmark_returns.append(float(group["benchmark_next_ret_1"].iloc[0]))

    portfolio_returns = np.asarray(equity_curve, dtype=float)
    metrics = compute_metrics(portfolio_returns, benchmark_returns)
    metrics["avg_turnover"] = float(np.mean(turnover_values)) if turnover_values else 0.0
    metrics["rebalance_count"] = len(turnover_values)
    return metrics


def select_with_buffer(ranked_ids, current_ids, top_n: int, hold_buffer: int, rank_map: dict[str, int]) -> list[str]:
    if not hold_buffer:
        return ranked_ids[:top_n]
    selected = [
        order_book_id
        for order_book_id in current_ids
        if rank_map.get(order_book_id, 10**9) <= hold_buffer
    ][:top_n]
    for order_book_id in ranked_ids:
        if len(selected) >= top_n:
            break
        if order_book_id not in selected:
            selected.append(order_book_id)
    return selected


def failed_metrics() -> dict:
    return {
        "total_returns": -1.0,
        "annualized_returns": -1.0,
        "excess_annual_returns": -1.0,
        "max_drawdown": 1.0,
        "sharpe": 0.0,
        "benchmark_total_returns": 0.0,
        "days": 0,
        "avg_turnover": 0.0,
        "rebalance_count": 0,
    }


def best_record(results: list[dict]) -> dict | None:
    if not results:
        return None
    return sorted(
        results,
        key=lambda r: (
            r["aggregate"]["all_years_gt_50"],
            r["aggregate"]["min_total_return"],
            r["aggregate"]["avg_total_return"],
            -r["aggregate"]["worst_max_drawdown"],
        ),
        reverse=True,
    )[0]


def aggregate(rows: list[dict]) -> dict:
    return {
        "min_total_return": min(row["total_returns"] for row in rows),
        "avg_total_return": sum(row["total_returns"] for row in rows) / len(rows),
        "worst_max_drawdown": max(row["max_drawdown"] for row in rows),
        "positive_years": sum(1 for row in rows if row["total_returns"] > 0),
        "all_years_gt_50": all(row["total_returns"] > 0.5 for row in rows),
        "avg_turnover": sum(row.get("avg_turnover", 0.0) for row in rows) / len(rows),
    }


if __name__ == "__main__":
    main()
