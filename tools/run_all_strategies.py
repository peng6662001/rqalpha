from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from rqalpha import run_file, run_func


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = Path.home() / ".rqalpha" / "bundle"
TEST_ROOT = REPO_ROOT / "tests" / "integration_tests" / "test_backtest_results"
RESULT_PATH = REPO_ROOT / ".temp" / "strategy_backtest_results.json"
FUTURE_INFO_PATH = TEST_ROOT / "resources" / "future_info.json"
CUSTOM_STRATEGY_ROOT = REPO_ROOT / "strategies" / "stock_selection"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_module(path: Path):
    module_name = "codex_" + "_".join(path.with_suffix("").parts[-4:])
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_config(config: dict[str, Any], reload_futures_info: bool = False) -> dict[str, Any]:
    config = copy.deepcopy(config)
    base = config.setdefault("base", {})
    base["data_bundle_path"] = str(BUNDLE_ROOT)

    if reload_futures_info:
        # Match the test fixture behavior for futures-related checks.
        with FUTURE_INFO_PATH.open("r", encoding="utf-8") as f:
            future_info = json.load(f)
        # Some bundled test resources are list-shaped and are not accepted by
        # base.future_info in normal backtest config; in that case rely on the
        # bundle default future_info.json instead of injecting invalid config.
        if isinstance(future_info, dict):
            base["future_info"] = future_info

    mod = config.setdefault("mod", {})
    mod.setdefault("sys_progress", {})
    mod["sys_progress"]["enabled"] = False
    analyser = mod.setdefault("sys_analyser", {})
    analyser["enabled"] = True
    return config


def _extract_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = copy.deepcopy(result.get("sys_analyser", {}).get("summary", {}))
    if "STOCK" in summary and "stock" not in summary:
        summary["stock"] = summary["STOCK"]
    if "FUTURE" in summary and "future" not in summary:
        summary["future"] = summary["FUTURE"]
    summary["strategy_name"] = summary.get("strategy_name") or summary.get("strategy_file")
    return summary


def _serialize(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return str(obj)


def _run_test_module(path: Path) -> list[dict[str, Any]]:
    module = _load_module(path)
    outputs: list[dict[str, Any]] = []

    for func_name, func in inspect.getmembers(module, inspect.isfunction):
        if not func_name.startswith("test_"):
            continue

        testcase_name = f"{path.stem}#{func_name}"
        resource_candidates = [
            TEST_ROOT / "resources" / testcase_name,
            TEST_ROOT / "resources" / path.stem,
        ]
        resources_path = next((p for p in resource_candidates if p.exists()), None)

        last_summary: dict[str, Any] | None = None

        def run_and_capture(**kwargs):
            nonlocal last_summary
            config = _prepare_config(kwargs.pop("config"), kwargs.pop("reload_futures_info", False))
            result = run_func(config=config, **kwargs)
            last_summary = _extract_summary(result)
            return last_summary

        try:
            params = []
            for p in inspect.signature(func).parameters:
                if p == "run_and_assert_result":
                    params.append(run_and_capture)
                elif p == "resources_path":
                    params.append(str(resources_path) if resources_path else "")
                else:
                    raise RuntimeError(f"unsupported test fixture parameter: {p}")
            summary = func(*params)
            if summary is None:
                summary = last_summary or {"note": "completed_without_return"}
            outputs.append(
                {
                    "name": testcase_name,
                    "kind": "integration_test_strategy",
                    "status": "ok",
                    "summary": _serialize(summary),
                }
            )
        except Exception as exc:
            outputs.append(
                {
                    "name": testcase_name,
                    "kind": "integration_test_strategy",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
    return outputs


def _example_specs() -> list[dict[str, Any]]:
    stock_config = {
        "base": {
            "start_date": "2016-06-01",
            "end_date": "2016-12-01",
            "frequency": "1d",
            "matching_type": "current_bar",
            "benchmark": "000300.XSHG",
            "accounts": {"stock": 100000},
        },
        "extra": {"log_level": "error"},
    }
    turtle_config = {
        "base": {
            "start_date": "2008-07-01",
            "end_date": "2014-09-01",
            "frequency": "1d",
            "matching_type": "current_bar",
            "benchmark": "000300.XSHG",
            "accounts": {"stock": 1000000},
        },
        "extra": {"log_level": "error"},
    }
    future_config = {
        "base": {
            "start_date": "2016-01-01",
            "end_date": "2016-06-30",
            "frequency": "1d",
            "matching_type": "current_bar",
            "benchmark": None,
            "accounts": {"future": 1000000},
        },
        "extra": {"log_level": "error"},
    }
    return [
        {"name": "example_buy_and_hold", "path": REPO_ROOT / "rqalpha" / "examples" / "buy_and_hold.py", "config": stock_config},
        {"name": "example_golden_cross", "path": REPO_ROOT / "rqalpha" / "examples" / "golden_cross.py", "config": stock_config},
        {"name": "example_macd", "path": REPO_ROOT / "rqalpha" / "examples" / "macd.py", "config": stock_config},
        {"name": "example_rsi", "path": REPO_ROOT / "rqalpha" / "examples" / "rsi.py", "config": stock_config},
        {"name": "example_subscribe_event", "path": REPO_ROOT / "rqalpha" / "examples" / "subscribe_event.py", "config": stock_config},
        {"name": "example_turtle", "path": REPO_ROOT / "rqalpha" / "examples" / "turtle.py", "config": turtle_config},
        {"name": "example_IF_macd", "path": REPO_ROOT / "rqalpha" / "examples" / "IF_macd.py", "config": future_config},
    ]


def _custom_stock_selection_specs() -> list[dict[str, Any]]:
    stock_selection_config = {
        "base": {
            "start_date": "2014-01-01",
            "end_date": "2024-12-31",
            "frequency": "1d",
            "matching_type": "current_bar",
            "benchmark": "000300.XSHG",
            "accounts": {"stock": 1000000},
        },
        "extra": {"log_level": "error"},
    }
    names = [
        "strategy_momentum_20.py",
        "strategy_momentum_60.py",
        "strategy_dual_momentum.py",
        "strategy_low_volatility.py",
        "strategy_low_vol_momentum.py",
        "strategy_mean_reversion_5.py",
        "strategy_rsi_reversion.py",
        "strategy_breakout_55.py",
        "strategy_ma_strength.py",
        "strategy_quality_trend.py",
    ]
    specs = []
    for filename in names:
        specs.append(
            {
                "name": filename.replace(".py", ""),
                "path": CUSTOM_STRATEGY_ROOT / filename,
                "config": stock_selection_config,
            }
        )
    return specs


def _run_example(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        result = run_file(str(spec["path"]), config=_prepare_config(spec["config"]))
        return {
            "name": spec["name"],
            "kind": "example_strategy",
            "status": "ok",
            "summary": _serialize(_extract_summary(result)),
        }
    except Exception as exc:
        return {
            "name": spec["name"],
            "kind": "example_strategy",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def main() -> int:
    if not BUNDLE_ROOT.exists():
        raise RuntimeError(f"bundle missing: {BUNDLE_ROOT}")

    results: list[dict[str, Any]] = []

    for path in sorted(TEST_ROOT.glob("test_*.py")):
        results.extend(_run_test_module(path))

    for spec in _example_specs():
        results.append(_run_example(spec))

    for spec in _custom_stock_selection_specs():
        results.append(_run_example(spec))

    _ensure_parent(RESULT_PATH)
    RESULT_PATH.write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ok = sum(1 for r in results if r["status"] == "ok")
    err = len(results) - ok
    print(f"wrote {RESULT_PATH}")
    print(f"success={ok} error={err} total={len(results)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
