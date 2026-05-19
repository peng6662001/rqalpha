from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from requests import RequestException


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / ".temp" / "public_pead"
EASTMONEY_API = "https://datacenter-web.eastmoney.com/api/data/v1/get"
CNINFO_NOTICE_API = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_PDF_PREFIX = "https://static.cninfo.com.cn/"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36"

EASTMONEY_REPORTS = {
    "performance_express": {
        "report_name": "RPT_FCI_PERFORMANCEE",
        "referer": "https://data.eastmoney.com/bbsj/yjkb.html",
        "sort_columns": "UPDATE_DATE,SECURITY_CODE",
        "sort_types": "-1,-1",
    },
    "earnings_forecast": {
        "report_name": "RPT_PUBLIC_OP_NEWPREDICT",
        "referer": "https://data.eastmoney.com/bbsj/yjyg.html",
        "sort_columns": "NOTICE_DATE,SECURITY_CODE",
        "sort_types": "-1,-1",
    },
    "appointment_dates": {
        "report_name": "RPT_PUBLIC_BS_APPOIN",
        "referer": "https://data.eastmoney.com/bbsj/yysj.html",
        "sort_columns": "FIRST_APPOINT_DATE,SECURITY_CODE",
        "sort_types": "1,1",
    },
}

CNINFO_KEYWORD_GROUPS = {
    "performance_express": {
        "slug": "performance_express",
        "keywords": ["业绩快报"],
    },
    "earnings_forecast": {
        "slug": "earnings_forecast",
        "keywords": ["业绩预告", "预增", "预减", "预亏", "扭亏", "预盈", "首亏", "略增", "略减"],
    },
}


def quarter_end_dates(start_year: int, end_year: int) -> list[str]:
    quarters = ["03-31", "06-30", "09-30", "12-31"]
    return [f"{year}-{quarter}" for year in range(start_year, end_year + 1) for quarter in quarters]


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s


def request_with_retry(method, url: str, *, retries: int = 5, backoff: float = 1.5, **kwargs):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = method(url, **kwargs)
            resp.raise_for_status()
            return resp
        except RequestException as exc:
            last_exc = exc
            if attempt == retries:
                raise
            sleep_s = backoff * attempt
            print(f"retrying {url} after error: {exc} (attempt {attempt}/{retries}, sleep={sleep_s:.1f}s)")
            time.sleep(sleep_s)
    raise last_exc


def fetch_eastmoney_table(
    sess: requests.Session,
    report_name: str,
    referer: str,
    report_date: str,
    sort_columns: str,
    sort_types: str,
    page_size: int = 500,
    pause: float = 0.12,
) -> list[dict]:
    rows: list[dict] = []
    page_number = 1
    total_pages = None
    while True:
        params = {
            "reportName": report_name,
            "columns": "ALL",
            "pageNumber": str(page_number),
            "pageSize": str(page_size),
            "sortColumns": sort_columns,
            "sortTypes": sort_types,
            "source": "WEB",
            "client": "WEB",
            "filter": f"(REPORT_DATE='{report_date}')",
        }
        r = request_with_retry(sess.get, EASTMONEY_API, params=params, headers={"Referer": referer}, timeout=30)
        payload = r.json()
        result = payload.get("result") or {}
        if total_pages is None:
            total_pages = int(result.get("pages") or 0)
        page_rows = result.get("data") or []
        rows.extend(page_rows)
        if page_number >= total_pages or not page_rows:
            break
        page_number += 1
        time.sleep(pause)
    return rows


def clean_eastmoney_df(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["source"] = "eastmoney"
    df["dataset"] = dataset_name
    if "SECUCODE" in df.columns:
        df["order_book_id"] = df["SECUCODE"]
    elif "SECURITY_CODE" in df.columns:
        df["order_book_id"] = df["SECURITY_CODE"]
    for col in ["NOTICE_DATE", "REPORT_DATE", "UPDATE_DATE", "FIRST_APPOINT_DATE", "ACTUAL_PUBLISH_DATE", "APPOINT_PUBLISH_DATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "SECURITY_TYPE" in df.columns:
        df = df[df["SECURITY_TYPE"] == "A股"].copy()
    return df


def fetch_cninfo_keyword(
    sess: requests.Session,
    keyword: str,
    start_date: str,
    end_date: str,
    page_size: int = 50,
    pause: float = 0.15,
) -> list[dict]:
    rows: list[dict] = []
    page_num = 1
    total_pages = None
    headers = {
        "Referer": "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search&checkedCategory=category_gqbd_szsh",
        "X-Requested-With": "XMLHttpRequest",
    }
    while True:
        data = {
            "pageNum": str(page_num),
            "pageSize": str(page_size),
            "column": "szse",
            "tabName": "fulltext",
            "plate": "szsh",
            "stock": "",
            "searchkey": keyword,
            "secid": "",
            "category": "",
            "trade": "",
            "seDate": f"{start_date}~{end_date}",
            "sortName": "nothing",
            "sortType": "desc",
            "isHLtitle": "true",
        }
        r = request_with_retry(sess.post, CNINFO_NOTICE_API, headers=headers, data=data, timeout=30)
        payload = r.json()
        if total_pages is None:
            total_announcement = int(payload.get("totalAnnouncement") or 0)
            total_pages = int(math.ceil(total_announcement / page_size)) if total_announcement else 0
        page_rows = payload.get("announcements") or []
        rows.extend(page_rows)
        if page_num >= total_pages or not page_rows:
            break
        page_num += 1
        time.sleep(pause)
    return rows


def clean_cninfo_df(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["source"] = "cninfo"
    df["dataset"] = dataset_name
    df["order_book_id"] = df["secCode"].astype(str) + df["pageColumn"].map(
        lambda v: ".SH" if str(v).startswith("SH") else ".SZ" if str(v).startswith("SZ") else ""
    )
    df["announcement_date"] = pd.to_datetime(df["announcementTime"], unit="ms", errors="coerce")
    df["title_text"] = df["announcementTitle"].fillna("").str.replace(r"<[^>]+>", "", regex=True)
    df["pdf_url"] = df["adjunctUrl"].fillna("").map(lambda v: CNINFO_PDF_PREFIX + v if v else "")
    if dataset_name == "performance_express":
        df = df[df["title_text"].str.contains("业绩快报", na=False)].copy()
    elif dataset_name == "earnings_forecast":
        include_pattern = r"(?:业绩预告|预增|预减|预亏|扭亏|预盈|首亏|略增|略减)"
        must_have_pattern = r"(?:业绩|利润|亏损|扭亏|预盈|预亏)"
        df = df[df["title_text"].str.contains(include_pattern, na=False)].copy()
        df = df[df["title_text"].str.contains(must_have_pattern, na=False)].copy()
        df = df[~df["title_text"].str.contains("说明会", na=False)].copy()
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_public_pead_dataset(start_year: int, end_year: int, skip_eastmoney: bool = False, skip_cninfo: bool = False) -> dict:
    sess = session()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_dates = quarter_end_dates(start_year, end_year)

    eastmoney_outputs: dict[str, pd.DataFrame] = {}
    for dataset_name, spec in EASTMONEY_REPORTS.items():
        path = OUTPUT_DIR / f"eastmoney_{dataset_name}_{start_year}_{end_year}.csv"
        if skip_eastmoney and path.exists():
            df = pd.read_csv(path)
            eastmoney_outputs[dataset_name] = df
            continue
        rows: list[dict] = []
        for report_date in report_dates:
            page_rows = fetch_eastmoney_table(
                sess,
                report_name=spec["report_name"],
                referer=spec["referer"],
                report_date=report_date,
                sort_columns=spec["sort_columns"],
                sort_types=spec["sort_types"],
            )
            rows.extend(page_rows)
            print(f"eastmoney {dataset_name} {report_date}: rows={len(page_rows)}")
        df = clean_eastmoney_df(pd.DataFrame(rows), dataset_name)
        eastmoney_outputs[dataset_name] = df
        write_csv(df, path)

    cninfo_outputs: dict[str, pd.DataFrame] = {}
    for dataset_name, spec in CNINFO_KEYWORD_GROUPS.items():
        slug = spec["slug"]
        final_path = OUTPUT_DIR / f"cninfo_{slug}_{start_year}_{end_year}.csv"
        if skip_cninfo and final_path.exists():
            df = pd.read_csv(final_path)
            cninfo_outputs[dataset_name] = df
            continue
        year_frames: list[pd.DataFrame] = []
        for year in range(start_year, end_year + 1):
            year_path = OUTPUT_DIR / f"cninfo_{slug}_{year}_{year}.csv"
            if year_path.exists():
                year_df = pd.read_csv(year_path)
                year_frames.append(year_df)
                print(f"cninfo {dataset_name} {year}: reused {len(year_df)} rows")
                continue
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            rows: list[dict] = []
            for keyword in spec["keywords"]:
                rows.extend(fetch_cninfo_keyword(sess, keyword=keyword, start_date=start_date, end_date=end_date))
            year_df = clean_cninfo_df(pd.DataFrame(rows), dataset_name)
            if "announcementId" in year_df.columns:
                year_df = year_df.sort_values(["announcementTime", "announcementId"]).drop_duplicates(subset=["announcementId"], keep="last")
            write_csv(year_df, year_path)
            year_frames.append(year_df)
            print(f"cninfo {dataset_name} {year}: rows={len(year_df)}")
        df = pd.concat(year_frames, ignore_index=True) if year_frames else pd.DataFrame()
        cninfo_outputs[dataset_name] = df
        write_csv(df, final_path)

    summary = {
        "built_at": pd.Timestamp.utcnow().isoformat(),
        "start_year": start_year,
        "end_year": end_year,
        "files": {
            "eastmoney_performance_express": str(OUTPUT_DIR / f"eastmoney_performance_express_{start_year}_{end_year}.csv"),
            "eastmoney_earnings_forecast": str(OUTPUT_DIR / f"eastmoney_earnings_forecast_{start_year}_{end_year}.csv"),
            "eastmoney_appointment_dates": str(OUTPUT_DIR / f"eastmoney_appointment_dates_{start_year}_{end_year}.csv"),
            "cninfo_performance_express": str(OUTPUT_DIR / f"cninfo_performance_express_{start_year}_{end_year}.csv"),
            "cninfo_earnings_forecast": str(OUTPUT_DIR / f"cninfo_earnings_forecast_{start_year}_{end_year}.csv"),
        },
        "counts": {
            "eastmoney_performance_express": int(len(eastmoney_outputs["performance_express"])),
            "eastmoney_earnings_forecast": int(len(eastmoney_outputs["earnings_forecast"])),
            "eastmoney_appointment_dates": int(len(eastmoney_outputs["appointment_dates"])),
            "cninfo_performance_express": int(len(cninfo_outputs["performance_express"])),
            "cninfo_earnings_forecast": int(len(cninfo_outputs["earnings_forecast"])),
        },
    }
    summary_path = OUTPUT_DIR / f"summary_{start_year}_{end_year}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a public-web PEAD prototype dataset from Eastmoney and CNINFO.")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=date.today().year - 1)
    parser.add_argument("--skip-eastmoney", action="store_true")
    parser.add_argument("--skip-cninfo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end_year < args.start_year:
        raise SystemExit("end-year must be >= start-year")
    summary = build_public_pead_dataset(
        args.start_year,
        args.end_year,
        skip_eastmoney=args.skip_eastmoney,
        skip_cninfo=args.skip_cninfo,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
