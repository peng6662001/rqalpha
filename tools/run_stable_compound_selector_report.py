from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STRATEGY = ROOT / "strategies" / "stock_selection" / "strategy_next_day_stock_selector_stable_compound.py"
TEMP_DIR = ROOT / ".temp"
LOG_PATH = TEMP_DIR / "stable_compound_selector_report.log"


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 A 股中线稳健复利候选股报告")
    parser.add_argument("--date", required=True, help="筛选日期，格式 YYYY-MM-DD，需为本地 bundle 覆盖的交易日")
    parser.add_argument("--top", type=int, default=5, help="展示前 N 只推荐股，默认 5")
    parser.add_argument("--backup", type=int, default=5, help="展示备选股数量，默认 5")
    parser.add_argument("--capital", type=int, default=1000000, help="回测账户资金，仅用于启动 RQAlpha")
    parser.add_argument("--timeout", type=int, default=180, help="RQAlpha 最长运行秒数，默认 180")
    parser.add_argument("--verbose", action="store_true", help="直接显示 RQAlpha 原始日志，便于排查卡住原因")
    parser.add_argument("--use-cache", action="store_true", help="如果 JSON 已存在，直接读取缓存，不重新运行 RQAlpha")
    parser.add_argument("--log-file", default=str(LOG_PATH), help="运行日志文件路径")
    args = parser.parse_args()

    output_path = TEMP_DIR / f"stable_compound_selection_{args.date}.json"
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "rqalpha",
        "run",
        "-f",
        str(STRATEGY),
        "-s",
        args.date,
        "-e",
        args.date,
        "--account",
        "stock",
        str(args.capital),
        "--benchmark",
        "000300.XSHG",
    ]

    if args.use_cache and output_path.exists():
        _append_log(log_path, f"use-cache date={args.date} output={output_path}")
    else:
        started = time.monotonic()
        _append_log(log_path, f"start date={args.date} timeout={args.timeout} command={_quote_command(command)}")
        try:
            if args.verbose:
                result = subprocess.run(command, cwd=ROOT, timeout=args.timeout)
            else:
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            _append_log(log_path, f"timeout date={args.date} elapsed={elapsed:.2f}s")
            raise RuntimeError(f"RQAlpha 运行超过 {args.timeout} 秒，已停止。可加 --verbose 查看卡住前日志。") from exc

        elapsed = time.monotonic() - started
        if result.returncode != 0:
            _append_log(log_path, f"failed date={args.date} elapsed={elapsed:.2f}s returncode={result.returncode}")
            if not args.verbose:
                if result.stdout:
                    print(result.stdout, file=sys.stderr)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
            raise subprocess.CalledProcessError(result.returncode, command)
        _append_log(log_path, f"success date={args.date} elapsed={elapsed:.2f}s output={output_path}")
    if not output_path.exists():
        raise FileNotFoundError(f"没有找到输出文件: {output_path}")

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    print_report(payload, top_n=args.top, backup_n=args.backup, output_path=output_path)
    return 0


def print_report(payload: dict, top_n: int, backup_n: int, output_path: Path) -> None:
    print()
    print("A股中线稳健复利候选股报告")
    print(f"筛选日期: {payload.get('selection_date')}")
    print(f"市场基准: {payload.get('benchmark')}")
    print(f"大盘状态: {'健康，可选股' if payload.get('benchmark_healthy') else '偏弱，默认收缩仓位'}")
    print()

    print(f"前 {top_n} 只推荐:")
    for idx, row in enumerate(payload.get("recommended", [])[:top_n], 1):
        print(_format_row(idx, row))

    backups = payload.get("backup", [])[:backup_n]
    if backups:
        print()
        print(f"备选 {len(backups)} 只:")
        for idx, row in enumerate(backups, 1):
            print(_format_row(idx, row))

    print()
    print("默认执行纪律: 5 只等权为主，保留少量现金；趋势破位、逻辑失效或大盘转弱时减仓。")
    print(f"完整 JSON: {output_path}")


def _format_row(idx: int, row: dict) -> str:
    ret_20 = _pct(row.get("ret_20"))
    ret_60 = _pct(row.get("ret_60"))
    vol_20 = _pct(row.get("vol_20"))
    max_dd_120 = _pct(row.get("max_dd_120"))
    score = row.get("composite_score", 0.0)
    return (
        f"{idx}. {row.get('symbol')} {row.get('order_book_id')} "
        f"综合分 {score:.4f} | 20日 {ret_20} | 60日 {ret_60} | "
        f"20日波动 {vol_20} | 120日回撤 {max_dd_120} | "
        f"{row.get('event_family', '')}/{row.get('forecast_type', '')}"
    )


def _pct(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:.2f}%"


def _append_log(path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.write_text("", encoding="utf-8") if not path.exists() else None
    with path.open("a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] {message}\n")


def _quote_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"RQAlpha 运行失败: {exc}", file=sys.stderr)
        raise SystemExit(exc.returncode)
