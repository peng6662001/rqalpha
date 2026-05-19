from __future__ import annotations

import json
from pathlib import Path

from tune_aggressive_extreme_leader_backup import aggregate, env_params, run_one_year


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / ".temp" / "tune_aggressive_extreme_leader_backup_refined.json"
YEARS = range(2020, 2026)


def run_params(label: str, params: dict[str, float | str]) -> dict:
    rows = [run_one_year(year, env_params(params)) for year in YEARS]
    item = aggregate(label, params, rows)
    print(label, json.dumps(item["aggregate"], ensure_ascii=False))
    return item


def main():
    base = {
        "RQALPHA_PRIMARY": "300059.XSHE",
        "RQALPHA_BACKUP": "159949.XSHE",
        "RQALPHA_ENTRY_DRAWDOWN": 0.015,
        "RQALPHA_RESET_DRAWDOWN": 0.015,
        "RQALPHA_LOOKBACK": 20,
        "RQALPHA_BULL_GAP": 0.02,
        "RQALPHA_SUPER_BULL_GAP": 0.04,
        "RQALPHA_SWITCH_DELTA": 0.015,
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
        ("ref_super_008", {"RQALPHA_SUPER_BULL_GAP": 0.08}),
        ("ref_entry_003", {"RQALPHA_ENTRY_DRAWDOWN": 0.03}),
        ("ref_trim_later", {"RQALPHA_PROFIT_TRIM_1": 0.85, "RQALPHA_PROFIT_TRIM_2": 1.25, "RQALPHA_PROFIT_EXIT": 2.00}),
        ("ref_looser_trail", {"RQALPHA_TRAILING_ACTIVATION": 1.00, "RQALPHA_TRAILING_STOP": 0.28, "RQALPHA_TRAILING_EXIT": 0.40}),
        ("ref_tighter_switch", {"RQALPHA_SWITCH_DELTA": 0.01}),
        ("ref_lock_later", {"RQALPHA_LOCK1_PROFIT": 1.00, "RQALPHA_LOCK2_PROFIT": 1.40, "RQALPHA_LOCK3_PROFIT": 1.90}),
        ("ref_entry003_super008", {"RQALPHA_ENTRY_DRAWDOWN": 0.03, "RQALPHA_SUPER_BULL_GAP": 0.08}),
        ("ref_entry003_super008_loose", {"RQALPHA_ENTRY_DRAWDOWN": 0.03, "RQALPHA_SUPER_BULL_GAP": 0.08, "RQALPHA_TRAILING_ACTIVATION": 1.00, "RQALPHA_TRAILING_STOP": 0.28, "RQALPHA_TRAILING_EXIT": 0.40}),
        (
            "ref_all_hold",
            {
                "RQALPHA_ENTRY_DRAWDOWN": 0.03,
                "RQALPHA_SUPER_BULL_GAP": 0.08,
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
        ("ref_look15_super004", {"RQALPHA_LOOKBACK": 15, "RQALPHA_SUPER_BULL_GAP": 0.04, "RQALPHA_SWITCH_DELTA": 0.01}),
        ("ref_look15_super008", {"RQALPHA_LOOKBACK": 15, "RQALPHA_SUPER_BULL_GAP": 0.08}),
        ("ref_look10_super008", {"RQALPHA_LOOKBACK": 10, "RQALPHA_SUPER_BULL_GAP": 0.08}),
    ]

    results = []
    for label, delta in variants:
        params = dict(base)
        params.update(delta)
        results.append(run_params(label, params))

    results.sort(key=lambda item: item["aggregate"]["avg_total_return"], reverse=True)
    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {OUTPUT}")
    print(json.dumps(results[:8], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
