from __future__ import annotations

import json
from pathlib import Path

from tune_aggressive_extreme_leader_backup import aggregate, env_params, run_one_year


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / ".temp" / "tune_aggressive_extreme_leader_backup_ladder.json"
YEARS = range(2020, 2026)


def main():
    base = {
        "RQALPHA_PRIMARY": "300059.XSHE",
        "RQALPHA_BACKUP": "159949.XSHE",
        "RQALPHA_ENTRY_DRAWDOWN": 0.03,
        "RQALPHA_RESET_DRAWDOWN": 0.015,
        "RQALPHA_LOOKBACK": 20,
        "RQALPHA_BULL_GAP": 0.02,
        "RQALPHA_SUPER_BULL_GAP": 0.04,
        "RQALPHA_SWITCH_DELTA": 0.015,
        "RQALPHA_LEVEL1_DRAWDOWN": 0.02,
        "RQALPHA_LEVEL2_DRAWDOWN": 0.05,
        "RQALPHA_LEVEL3_DRAWDOWN": 0.08,
        "RQALPHA_LEVEL1_TARGET": 0.50,
        "RQALPHA_LEVEL2_TARGET": 0.80,
        "RQALPHA_LEVEL3_TARGET": 1.00,
        "RQALPHA_TREND_DOWN_CAP": 0.35,
        "RQALPHA_PROFIT_TRIM_1": 0.75,
        "RQALPHA_PROFIT_TRIM_2": 1.10,
        "RQALPHA_PROFIT_EXIT": 1.80,
        "RQALPHA_TRAILING_ACTIVATION": 0.90,
        "RQALPHA_TRAILING_STOP": 0.26,
        "RQALPHA_TRAILING_EXIT": 0.38,
        "RQALPHA_LOCK1_PROFIT": 0.85,
        "RQALPHA_LOCK1_CAP": 0.92,
        "RQALPHA_LOCK2_PROFIT": 1.20,
        "RQALPHA_LOCK2_CAP": 0.78,
        "RQALPHA_LOCK3_PROFIT": 1.60,
        "RQALPHA_LOCK3_CAP": 0.50,
    }

    variants = [
        ("ladder_base", {}),
        ("ladder_early_full", {"RQALPHA_LEVEL2_DRAWDOWN": 0.04, "RQALPHA_LEVEL3_DRAWDOWN": 0.06, "RQALPHA_LEVEL1_TARGET": 0.60, "RQALPHA_LEVEL2_TARGET": 1.00}),
        ("ladder_more_gradual", {"RQALPHA_LEVEL1_TARGET": 0.35, "RQALPHA_LEVEL2_TARGET": 0.70}),
        ("ladder_ultra_aggressive", {"RQALPHA_LEVEL1_DRAWDOWN": 0.015, "RQALPHA_LEVEL2_DRAWDOWN": 0.03, "RQALPHA_LEVEL3_DRAWDOWN": 0.05, "RQALPHA_LEVEL1_TARGET": 0.70, "RQALPHA_LEVEL2_TARGET": 1.00}),
        ("ladder_high_downcap", {"RQALPHA_TREND_DOWN_CAP": 0.50}),
        ("ladder_full_downcap", {"RQALPHA_TREND_DOWN_CAP": 0.70}),
        ("ladder_aggr_plus_downcap", {"RQALPHA_LEVEL1_DRAWDOWN": 0.015, "RQALPHA_LEVEL2_DRAWDOWN": 0.03, "RQALPHA_LEVEL3_DRAWDOWN": 0.05, "RQALPHA_LEVEL1_TARGET": 0.70, "RQALPHA_LEVEL2_TARGET": 1.00, "RQALPHA_TREND_DOWN_CAP": 0.50}),
        ("ladder_entry03_super08_aggr", {"RQALPHA_SUPER_BULL_GAP": 0.08, "RQALPHA_LEVEL1_DRAWDOWN": 0.015, "RQALPHA_LEVEL2_DRAWDOWN": 0.03, "RQALPHA_LEVEL3_DRAWDOWN": 0.05, "RQALPHA_LEVEL1_TARGET": 0.70, "RQALPHA_LEVEL2_TARGET": 1.00, "RQALPHA_TREND_DOWN_CAP": 0.50}),
    ]

    results = []
    for label, delta in variants:
        params = dict(base)
        params.update(delta)
        rows = [run_one_year(year, env_params(params)) for year in YEARS]
        item = aggregate(label, params, rows)
        results.append(item)
        print(label, json.dumps(item["aggregate"], ensure_ascii=False))

    results.sort(key=lambda item: item["aggregate"]["avg_total_return"], reverse=True)
    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {OUTPUT}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
