from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / ".temp" / "public_pead"
OUTPUT_CSV = DATA_DIR / "public_pead_events_2020_2025.csv"
OUTPUT_SUMMARY = DATA_DIR / "public_pead_events_2020_2025_summary.json"


FORECAST_METRIC_PRIORITY = {
    "4": 1,  # 归母净利润
    "2": 2,  # 净利润
    "5": 3,  # 扣非净利润
    "6": 4,  # 营业收入
    "1": 5,  # 主营业务收入
    "3": 6,  # EPS
    "8": 7,
    "9": 8,
    "7": 9,
}

FORECAST_DIRECTION_MAP = {
    "预增": 1.0,
    "略增": 0.5,
    "续盈": 0.4,
    "扭亏": 1.0,
    "减亏": 0.6,
    "预减": -0.6,
    "略减": -0.4,
    "首亏": -1.0,
    "续亏": -0.7,
    "增亏": -1.0,
    "不确定": 0.0,
}


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def normalize_order_book_id_from_cninfo(sec_code, page_column):
    code = str(sec_code).split(".")[0]
    if code.endswith(".0"):
        code = code[:-2]
    code = code.zfill(6)
    page_column = str(page_column)
    suffix = ".SH" if page_column.startswith("SH") else ".SZ" if page_column.startswith("SZ") else ""
    return code + suffix


def prepare_cninfo_notices() -> pd.DataFrame:
    frames = []
    for family, filename in [
        ("performance_express", "cninfo_performance_express_2020_2025.csv"),
        ("earnings_forecast", "cninfo_earnings_forecast_2020_2025.csv"),
    ]:
        df = load_csv(filename).copy()
        df["family"] = family
        df["order_book_id"] = [
            normalize_order_book_id_from_cninfo(sec_code, page_column)
            for sec_code, page_column in zip(df["secCode"], df["pageColumn"])
        ]
        df["announcement_date"] = pd.to_datetime(df["announcement_date"], errors="coerce").dt.normalize()
        df["title_text"] = df["title_text"].fillna("").astype(str)
        df["pdf_url"] = df["pdf_url"].fillna("").astype(str)
        if "announcementId" in df.columns:
            df["announcementId"] = df["announcementId"].astype(str)
        if family == "performance_express":
            df = df[df["title_text"].str.contains("业绩快报", na=False)].copy()
        if family == "earnings_forecast":
            include_pattern = r"(?:业绩预告|预增|预减|预亏|扭亏|预盈|首亏|略增|略减)"
            must_have_pattern = r"(?:业绩|利润|亏损|扭亏|预盈|预亏)"
            df = df[df["title_text"].str.contains(include_pattern, na=False, regex=True)].copy()
            df = df[df["title_text"].str.contains(must_have_pattern, na=False, regex=True)].copy()
            df = df[~df["title_text"].str.contains("说明会", na=False)].copy()
        df["is_revision_notice"] = df["title_text"].str.contains("更正|修正|补充", na=False)
        df = df.sort_values(["order_book_id", "announcement_date", "is_revision_notice", "announcementId"], ascending=[True, True, False, False])
        df = df.drop_duplicates(subset=["announcementId"], keep="first")
        grouped = (
            df.groupby(["order_book_id", "announcement_date", "family"], as_index=False)
            .agg(
                cninfo_title=("title_text", "first"),
                cninfo_pdf_url=("pdf_url", "first"),
                cninfo_announcement_id=("announcementId", "first"),
                cninfo_revision_notice=("is_revision_notice", "max"),
                cninfo_notice_count=("announcementId", "count"),
            )
        )
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def prepare_appointment_table() -> pd.DataFrame:
    df = load_csv("eastmoney_appointment_dates_2020_2025.csv").copy()
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce").dt.normalize()
    for col in ["FIRST_APPOINT_DATE", "FIRST_CHANGE_DATE", "SECOND_CHANGE_DATE", "THIRD_CHANGE_DATE", "ACTUAL_PUBLISH_DATE", "APPOINT_PUBLISH_DATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.sort_values(["order_book_id", "REPORT_DATE", "ACTUAL_PUBLISH_DATE", "APPOINT_PUBLISH_DATE"])
    grouped = (
        df.groupby(["order_book_id", "REPORT_DATE"], as_index=False)
        .agg(
            first_appoint_date=("FIRST_APPOINT_DATE", "first"),
            first_change_date=("FIRST_CHANGE_DATE", "first"),
            second_change_date=("SECOND_CHANGE_DATE", "first"),
            third_change_date=("THIRD_CHANGE_DATE", "first"),
            actual_publish_date=("ACTUAL_PUBLISH_DATE", "max"),
            appoint_publish_date=("APPOINT_PUBLISH_DATE", "max"),
            appoint_notice_count=("REPORT_DATE", "size"),
        )
    )
    return grouped


def aggregate_forecasts() -> pd.DataFrame:
    df = load_csv("eastmoney_earnings_forecast_2020_2025.csv").copy()
    df["NOTICE_DATE"] = pd.to_datetime(df["NOTICE_DATE"], errors="coerce").dt.normalize()
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce").dt.normalize()
    df["metric_priority"] = df["PREDICT_FINANCE_CODE"].astype(str).map(FORECAST_METRIC_PRIORITY).fillna(99)
    df["forecast_ratio_mean"] = pd.to_numeric(df.get("PREDICT_HBMEAN"), errors="coerce")
    if "PREDICT_RATIO_LOWER" in df.columns and "PREDICT_RATIO_UPPER" in df.columns:
        ratio_mid = (pd.to_numeric(df["PREDICT_RATIO_LOWER"], errors="coerce") + pd.to_numeric(df["PREDICT_RATIO_UPPER"], errors="coerce")) / 2.0
        df["forecast_ratio_mean"] = df["forecast_ratio_mean"].fillna(ratio_mid)
    amt_mid = (pd.to_numeric(df.get("PREDICT_AMT_LOWER"), errors="coerce") + pd.to_numeric(df.get("PREDICT_AMT_UPPER"), errors="coerce")) / 2.0
    df["forecast_amount_mid"] = amt_mid
    df["direction_score"] = df["PREDICT_TYPE"].map(FORECAST_DIRECTION_MAP).fillna(0.0)
    df = df.sort_values(["order_book_id", "REPORT_DATE", "NOTICE_DATE", "metric_priority", "forecast_ratio_mean"], ascending=[True, True, True, True, False])

    rows = []
    for (order_book_id, report_date, notice_date), group in df.groupby(["order_book_id", "REPORT_DATE", "NOTICE_DATE"], sort=False):
        primary = group.iloc[0]
        ratio_mean = primary["forecast_ratio_mean"]
        if pd.isna(ratio_mean):
            ratio_mean = group["forecast_ratio_mean"].dropna().mean()
        surprise_score = np.nan
        if pd.notna(ratio_mean):
            surprise_score = float(np.clip(ratio_mean / 100.0, -5.0, 5.0))
        elif pd.notna(primary["direction_score"]):
            surprise_score = float(primary["direction_score"])
        rows.append(
            {
                "order_book_id": order_book_id,
                "event_family": "earnings_forecast",
                "event_date": notice_date,
                "report_date": report_date,
                "report_period": primary.get("QDATE") or primary.get("DATATYPE"),
                "security_name": primary.get("SECURITY_NAME_ABBR"),
                "primary_metric_code": str(primary.get("PREDICT_FINANCE_CODE")),
                "primary_metric_name": primary.get("PREDICT_FINANCE"),
                "forecast_type": primary.get("PREDICT_TYPE"),
                "forecast_ratio_mean": ratio_mean,
                "forecast_amount_mid": primary.get("forecast_amount_mid"),
                "forecast_content": primary.get("PREDICT_CONTENT"),
                "change_reason_explain": primary.get("CHANGE_REASON_EXPLAIN"),
                "surprise_score": surprise_score,
                "surprise_direction": float(np.sign(surprise_score)) if pd.notna(surprise_score) and surprise_score != 0 else 0.0,
                "eastmoney_record_count": int(len(group)),
                "metric_list": "|".join(sorted(group["PREDICT_FINANCE"].dropna().astype(str).unique())),
                "is_revision_event": bool(group["PREDICT_TYPE"].astype(str).str.contains("修正|更正", na=False).any()),
                "eastmoney_notice_date": notice_date,
            }
        )
    return pd.DataFrame(rows)


def aggregate_performance_express() -> pd.DataFrame:
    df = load_csv("eastmoney_performance_express_2020_2025.csv").copy()
    df["NOTICE_DATE"] = pd.to_datetime(df["NOTICE_DATE"], errors="coerce").dt.normalize()
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce").dt.normalize()
    for col in ["YSTZ", "JLRTBZCL", "DJDYSHZ", "DJDJLHZ", "BASIC_EPS", "PARENT_NETPROFIT", "TOTAL_OPERATE_INCOME"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rows = []
    for (order_book_id, report_date, notice_date), group in df.groupby(["order_book_id", "REPORT_DATE", "NOTICE_DATE"], sort=False):
        row = group.iloc[0]
        profit_yoy = row.get("JLRTBZCL")
        revenue_yoy = row.get("YSTZ")
        if pd.notna(profit_yoy):
            surprise_score = float(np.clip(profit_yoy / 100.0, -5.0, 5.0))
        elif pd.notna(revenue_yoy):
            surprise_score = float(np.clip(revenue_yoy / 100.0, -5.0, 5.0))
        else:
            surprise_score = np.nan
        rows.append(
            {
                "order_book_id": order_book_id,
                "event_family": "performance_express",
                "event_date": notice_date,
                "report_date": report_date,
                "report_period": row.get("QDATE") or row.get("DATATYPE"),
                "security_name": row.get("SECURITY_NAME_ABBR"),
                "primary_metric_code": "express",
                "primary_metric_name": "业绩快报",
                "forecast_type": None,
                "forecast_ratio_mean": None,
                "forecast_amount_mid": None,
                "forecast_content": None,
                "change_reason_explain": None,
                "surprise_score": surprise_score,
                "surprise_direction": float(np.sign(surprise_score)) if pd.notna(surprise_score) and surprise_score != 0 else 0.0,
                "eastmoney_record_count": int(len(group)),
                "metric_list": "YSTZ|JLRTBZCL",
                "is_revision_event": False,
                "eastmoney_notice_date": notice_date,
                "express_revenue_yoy": revenue_yoy,
                "express_profit_yoy": profit_yoy,
                "express_eps": row.get("BASIC_EPS"),
                "express_parent_netprofit": row.get("PARENT_NETPROFIT"),
                "express_operate_income": row.get("TOTAL_OPERATE_INCOME"),
            }
        )
    return pd.DataFrame(rows)


def enrich_with_cninfo(events: pd.DataFrame, cninfo: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce").dt.normalize()
    merged = out.merge(
        cninfo,
        left_on=["order_book_id", "event_date", "event_family"],
        right_on=["order_book_id", "announcement_date", "family"],
        how="left",
    )
    merged["official_title"] = merged["cninfo_title"].fillna("")
    merged["official_pdf_url"] = merged["cninfo_pdf_url"].fillna("")
    merged["official_announcement_id"] = merged["cninfo_announcement_id"].fillna("")
    merged["has_official_notice_match"] = merged["official_announcement_id"] != ""
    merged["is_revision_event"] = merged["is_revision_event"].fillna(False) | merged["cninfo_revision_notice"].fillna(False)
    merged = merged.drop(columns=["announcement_date", "family", "cninfo_title", "cninfo_pdf_url", "cninfo_announcement_id", "cninfo_revision_notice", "cninfo_notice_count"], errors="ignore")
    return merged


def enrich_with_appointments(events: pd.DataFrame, appointments: pd.DataFrame) -> pd.DataFrame:
    out = events.merge(
        appointments,
        left_on=["order_book_id", "report_date"],
        right_on=["order_book_id", "REPORT_DATE"],
        how="left",
    )
    out = out.drop(columns=["REPORT_DATE"], errors="ignore")
    out["days_from_first_appoint_to_event"] = (
        pd.to_datetime(out["event_date"], errors="coerce") - pd.to_datetime(out["first_appoint_date"], errors="coerce").dt.normalize()
    ).dt.days
    return out


def finalize_events(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["event_id"] = (
        out["order_book_id"].astype(str)
        + "|"
        + out["event_family"].astype(str)
        + "|"
        + out["report_date"].astype(str)
        + "|"
        + out["event_date"].astype(str)
    )
    out["report_year"] = pd.to_datetime(out["report_date"], errors="coerce").dt.year
    out["event_year"] = pd.to_datetime(out["event_date"], errors="coerce").dt.year
    out["event_quarter"] = pd.to_datetime(out["report_date"], errors="coerce").dt.quarter
    out["surprise_bucket"] = pd.cut(
        out["surprise_score"],
        bins=[-np.inf, -0.3, -0.1, 0.1, 0.3, np.inf],
        labels=["very_negative", "negative", "neutral", "positive", "very_positive"],
    ).astype(str)
    out = out.sort_values(["event_date", "order_book_id", "event_family", "report_date"]).reset_index(drop=True)
    return out


def build_event_table() -> tuple[pd.DataFrame, dict]:
    cninfo = prepare_cninfo_notices()
    appointments = prepare_appointment_table()
    forecast_events = aggregate_forecasts()
    express_events = aggregate_performance_express()
    events = pd.concat([forecast_events, express_events], ignore_index=True, sort=False)
    events = enrich_with_cninfo(events, cninfo)
    events = enrich_with_appointments(events, appointments)
    events = finalize_events(events)

    summary = {
        "rows": int(len(events)),
        "families": events["event_family"].value_counts(dropna=False).to_dict(),
        "official_notice_match_ratio": float(events["has_official_notice_match"].mean()),
        "revision_event_ratio": float(events["is_revision_event"].fillna(False).mean()),
        "surprise_bucket_counts": events["surprise_bucket"].value_counts(dropna=False).to_dict(),
        "date_range": {
            "min_event_date": str(events["event_date"].min()),
            "max_event_date": str(events["event_date"].max()),
        },
    }
    return events, summary


def main() -> None:
    events, summary = build_event_table()
    events.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    OUTPUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {OUTPUT_CSV}")
    print(f"saved {OUTPUT_SUMMARY}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
