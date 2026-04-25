import argparse
import calendar
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import sqlalchemy as sa


# Leave this empty for now; you can fill company IDs later.
TARGET_COMPANY_IDS: List[int] = []


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "keydevid",
            "transcriptid",
            "headline",
            "mostimportantdateutc",
            "companyid",
            "componenttext",
            "block_num",
            "section",
            "pretty_text",
            "Id",
        ]
    )


class WRDSClient:
    def __init__(self, username: str, password: str):
        user_q = quote_plus(username)
        pass_q = quote_plus(password)
        uri = (
            f"postgresql+psycopg2://{user_q}:{pass_q}"
            "@wrds-pgdata.wharton.upenn.edu:9737/wrds?sslmode=require"
        )
        self.engine = sa.create_engine(uri)

    def raw_sql(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(sa.text(query), self.engine)

    def close(self) -> None:
        self.engine.dispose()


def load_all_earnings_calls(
    db: WRDSClient, year: int, month: int, target_company_ids: List[int] | None = None
) -> pd.DataFrame:
    """Fetch and structure all earnings call transcripts for one month."""
    month_start = f"{year}-{month:02d}-01"
    month_end = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"
    relevant_columns = (
        "companyid, keydevid, transcriptid, headline, keydeveventtypename, "
        "transcriptcollectiontypeid, transcriptcreationdate_utc"
    )
    company_filter = ""
    if target_company_ids:
        ids = ",".join(str(int(x)) for x in target_company_ids)
        company_filter = f" and companyid in ({ids})"

    sql_query = f"""
    select {relevant_columns}
    from ciq.wrds_transcript_detail
    where mostimportantdateutc between '{month_start}' and '{month_end}'
    {company_filter}
    """
    calls = db.raw_sql(sql_query)
    if calls.empty:
        return _empty_result()

    calls = calls[calls["keydeveventtypename"] == "Earnings Calls"].copy()
    if calls.empty:
        return _empty_result()

    calls["companyid"] = calls["companyid"].astype(int)
    company_ids = calls.groupby("keydevid")["companyid"].unique().reset_index()
    calls = calls.drop(columns=["companyid"])

    # Keep the latest transcript version with preferred collection type.
    calls = calls[calls["transcriptcollectiontypeid"].isin([1, 2, 7, 8])].copy()
    if calls.empty:
        return _empty_result()
    calls["transcriptcollectiontypeid"] = calls["transcriptcollectiontypeid"].replace(
        {8: 1, 1: 2, 2: 3, 7: 4}
    )

    def select_type(obj: pd.DataFrame) -> pd.Series:
        last_date = obj["transcriptcreationdate_utc"].max()
        return (
            obj[obj["transcriptcreationdate_utc"] == last_date]
            .sort_values("transcriptcollectiontypeid")
            .iloc[0]
        )

    calls = calls.groupby("keydevid").apply(select_type).reset_index(drop=True)
    if calls.empty:
        return _empty_result()

    keydevid_list = ",".join(f"{int(x)}" for x in calls["keydevid"].unique())
    sql_query2 = f"""
    select keydevid, mostimportantdateutc
    from ciq.ciqkeydev
    where keydevid in ({keydevid_list})
    """
    call_detail_time = db.raw_sql(sql_query2)
    calls = calls.merge(call_detail_time, on="keydevid", how="left")

    transcript_id_list = ",".join(f"{int(x)}" for x in calls["transcriptid"].unique())
    sql_query3 = f"""
    select transcriptid, componenttext, componentorder, transcriptcomponentid,
           transcriptcomponenttypeid, transcriptpersonid
    from ciq_transcripts.ciqtranscriptcomponent
    where transcriptid in ({transcript_id_list})
    order by transcriptid, componentorder
    """
    comps = db.raw_sql(sql_query3)
    if comps.empty:
        return _empty_result()

    sql_query4 = """
    select transcriptpersonid, transcriptpersonname
    from ciq_transcripts.ciqtranscriptperson
    """
    speaker_info = db.raw_sql(sql_query4)
    speaker_info["transcriptpersonname_id"] = (
        speaker_info["transcriptpersonname"]
        + speaker_info["transcriptpersonid"].apply(lambda x: f"_<id: {int(x)}>")
    )
    comps = comps.merge(speaker_info, on="transcriptpersonid", how="left")
    comps["transcriptpersonname_id"] = comps["transcriptpersonname_id"].fillna(
        "Unknown Person"
    )

    def break_into_blocks(obj: pd.DataFrame) -> pd.DataFrame:
        obj = obj.sort_values(["transcriptid", "componentorder"]).copy()

        pre = obj[obj["transcriptcomponenttypeid"] == 2].copy()
        pre["componenttext"] = pre["componenttext"].fillna("").apply(
            lambda x: str(x).split("\r\n")
        )
        pre = pre.explode(column="componenttext")
        pre["block_num"] = pre.groupby("transcriptid").cumcount()
        pre["section"] = "Pre"
        pre["pretty_text"] = (
            "[Presentation] "
            + pre["transcriptpersonname_id"].astype(str)
            + "\n"
            + pre["componenttext"].astype(str)
        )

        qa = obj[obj["transcriptcomponenttypeid"].isin([3, 4])].copy()
        qa["Ques"] = qa["transcriptcomponenttypeid"] == 3
        qa["transcriptpersonname_id"] = np.where(
            qa["Ques"],
            "Q--" + qa["transcriptpersonname_id"].astype(str),
            "A--" + qa["transcriptpersonname_id"].astype(str),
        )
        qa["block_num"] = qa["Ques"].cumsum()
        qa["pretty_text"] = (
            qa["transcriptpersonname_id"].astype(str)
            + "\n"
            + qa["componenttext"].fillna("").astype(str)
        )
        qablocks = (
            qa.groupby(["block_num", "transcriptid"])["componenttext"]
            .apply(lambda x: "\n".join(x.fillna("").astype(str)))
            .reset_index(name="QA")
        )
        qablocks_pretty = (
            qa.groupby(["block_num", "transcriptid"])["pretty_text"]
            .apply(lambda x: "\n".join(x.astype(str)))
            .reset_index(name="QA_pretty")
        )
        qa = qa.merge(qablocks, on=["block_num", "transcriptid"], how="left")
        qa = qa.merge(qablocks_pretty, on=["block_num", "transcriptid"], how="left")
        qa["componenttext"] = qa["QA"]
        qa["pretty_text"] = "[Q&A] " + qa["QA_pretty"]
        qa = qa.sort_values(["transcriptid", "componentorder"])
        qa = qa.drop_duplicates(["block_num", "transcriptid"], keep="first")
        qa["block_num"] = qa.groupby("transcriptid").cumcount()
        qa["section"] = "QA"

        relev_cols = ["transcriptid", "componenttext", "block_num", "section", "pretty_text"]
        return pd.concat([pre[relev_cols], qa[relev_cols]], ignore_index=True)

    comps = break_into_blocks(comps)
    if comps.empty:
        return _empty_result()

    calls = calls.drop(
        columns=["transcriptcreationdate_utc", "transcriptcollectiontypeid", "keydeveventtypename"]
    )
    calls = calls.merge(company_ids, on="keydevid", how="left")
    calls[["keydevid", "transcriptid"]] = calls[["keydevid", "transcriptid"]].astype(int)
    calls = calls.merge(comps, on="transcriptid", how="left")

    # Expand multi-company transcripts to one row per company ID.
    calls = calls.explode("companyid").reset_index(drop=True)
    calls["companyid"] = pd.to_numeric(calls["companyid"], errors="coerce")
    calls = calls.dropna(subset=["block_num", "section"]).copy()
    calls["block_num"] = pd.to_numeric(calls["block_num"], errors="coerce").astype("Int64")
    calls = calls[calls["block_num"].notna()].copy()
    calls["block_num"] = calls["block_num"].astype(int)
    calls["Id"] = calls.apply(
        lambda x: f"{int(x['keydevid'])}_{int(x['transcriptid'])}_{x['section']}_{int(x['block_num'])}",
        axis=1,
    )
    return calls


def parse_company_ids(company_ids_arg: str | None) -> List[int]:
    if company_ids_arg is None or company_ids_arg.strip() == "":
        return TARGET_COMPANY_IDS
    return [int(x.strip()) for x in company_ids_arg.split(",") if x.strip()]


def resolve_company_ids_by_names(db: WRDSClient, company_names: List[str]) -> List[int]:
    resolved_ids: List[int] = []
    for raw_name in company_names:
        name = raw_name.strip()
        if not name:
            continue
        escaped = name.replace("'", "''")
        sql = f"""
        select distinct companyid, companyname
        from ciq.wrds_transcript_detail
        where keydeveventtypename = 'Earnings Calls'
          and companyname is not null
          and lower(companyname) like lower('%{escaped}%')
        order by companyname
        limit 10
        """
        matches = db.raw_sql(sql)
        if matches.empty:
            print(f"No company match found for: {name}")
            continue
        selected = matches.iloc[0]
        resolved_ids.append(int(selected["companyid"]))
        print(
            f"Resolved '{name}' -> {selected['companyname']} (companyid={int(selected['companyid'])})"
        )
    return sorted(set(resolved_ids))


def resolve_company_ids_by_tickers_best_calls(
    db: WRDSClient,
    tickers: List[str],
    start_year: int,
    end_year: int,
) -> List[int]:
    cleaned = sorted(set([str(x).strip().upper() for x in tickers if str(x).strip()]))
    if not cleaned:
        return []

    ticker_batches: List[List[str]] = []
    batch_size = 250
    for i in range(0, len(cleaned), batch_size):
        ticker_batches.append(cleaned[i : i + batch_size])

    mapping_rows: List[pd.DataFrame] = []
    for batch in ticker_batches:
        in_list = ",".join("'" + t.replace("'", "''") + "'" for t in batch)
        sql = f"""
        select ticker, companyid, ticker_startdate, ticker_enddate
        from ciq.wrds_ciqsymbol_primary
        where ticker in ({in_list}) and ticker is not null
        """
        part = db.raw_sql(sql)
        if not part.empty:
            mapping_rows.append(part)

    if not mapping_rows:
        return []

    mapping_df = pd.concat(mapping_rows, ignore_index=True)
    mapping_df["companyid"] = pd.to_numeric(mapping_df["companyid"], errors="coerce")
    mapping_df = mapping_df.dropna(subset=["companyid"]).copy()
    if mapping_df.empty:
        return []
    mapping_df["companyid"] = mapping_df["companyid"].astype(int)

    candidate_ids = sorted(set(mapping_df["companyid"].tolist()))
    ids_sql = ",".join(str(int(x)) for x in candidate_ids)
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    calls_sql = f"""
    select companyid, count(distinct keydevid) as n_calls
    from ciq.wrds_transcript_detail
    where keydeveventtypename = 'Earnings Calls'
      and mostimportantdateutc between '{start_date}' and '{end_date}'
      and companyid in ({ids_sql})
    group by companyid
    """
    calls_df = db.raw_sql(calls_sql)
    call_map = {int(r.companyid): int(r.n_calls) for r in calls_df.itertuples(index=False)}

    mapping_df["ticker_startdate"] = pd.to_datetime(
        mapping_df["ticker_startdate"], errors="coerce"
    )
    mapping_df["ticker_enddate"] = pd.to_datetime(
        mapping_df["ticker_enddate"], errors="coerce"
    )
    mapping_df["n_calls"] = mapping_df["companyid"].map(lambda x: call_map.get(int(x), 0))

    # Prefer the company ID with most observed earnings calls in the study window.
    mapping_df = mapping_df.sort_values(
        ["ticker", "n_calls", "ticker_enddate", "ticker_startdate"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    best = mapping_df.drop_duplicates(subset=["ticker"], keep="first").copy()
    hit_count = int((best["n_calls"] > 0).sum())
    print(
        f"Resolved {len(best)} tickers to company IDs; "
        f"{hit_count} have at least one earnings call in {start_year}-{end_year}."
    )
    return sorted(set(best["companyid"].astype(int).tolist()))


def month_range(start_year: int, end_year: int) -> Iterable[tuple[int, int]]:
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield year, month


def _dedup_transcripts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "Id" in df.columns:
        return df.drop_duplicates(subset=["Id"], keep="first").reset_index(drop=True)
    subset = [
        c
        for c in ["keydevid", "transcriptid", "companyid", "section", "block_num", "componenttext"]
        if c in df.columns
    ]
    if subset:
        return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def _safe_int_list(series: pd.Series) -> List[int]:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(int).unique().tolist()
    return sorted(set(vals))


def fetch_month_with_split_retry(
    db: WRDSClient, year: int, month: int, company_ids: List[int]
) -> Tuple[pd.DataFrame, List[int]]:
    failed_ids: List[int] = []
    outputs: List[pd.DataFrame] = []
    chunks: List[List[int]] = [company_ids]
    while chunks:
        batch = chunks.pop()
        if not batch:
            continue
        try:
            data = load_all_earnings_calls(db, year, month, batch)
            if not data.empty:
                outputs.append(data)
        except Exception:
            if len(batch) <= 1:
                failed_ids.extend(batch)
            else:
                mid = len(batch) // 2
                chunks.append(batch[:mid])
                chunks.append(batch[mid:])
    out = pd.concat(outputs, ignore_index=True) if outputs else _empty_result()
    return out, sorted(set(failed_ids))


def backfill_q1_transcripts(
    db: WRDSClient, company_ids: List[int], year: int = 2018
) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
    outputs: List[pd.DataFrame] = []
    failed_by_month: Dict[int, List[int]] = {}
    for month in [1, 2, 3]:
        print(f"Backfilling month: {year}-{month:02d}")
        month_df, failed_ids = fetch_month_with_split_retry(
            db=db, year=year, month=month, company_ids=company_ids
        )
        month_df = _dedup_transcripts(month_df)
        failed_by_month[month] = failed_ids
        print(
            f"Backfill {year}-{month:02d} rows={len(month_df)}, failed_company_ids={len(failed_ids)}"
        )
        if not month_df.empty:
            outputs.append(month_df)
    out = pd.concat(outputs, ignore_index=True) if outputs else _empty_result()
    out = _dedup_transcripts(out)
    return out, failed_by_month


def run_scoring_script(
    scoring_script: Path,
    input_csv: Path,
    output_dir: Path,
    cache_path: Path,
    openrouter_api_key: str,
    openrouter_model: str,
    max_workers: int,
    request_timeout: int,
) -> None:
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = openrouter_api_key
    env["OPENROUTER_MODEL"] = openrouter_model
    cmd = [
        "python3",
        str(scoring_script),
        "--input",
        str(input_csv),
        "--output-dir",
        str(output_dir),
        "--cache-path",
        str(cache_path),
        "--max-workers",
        str(max_workers),
        "--request-timeout",
        str(request_timeout),
    ]
    subprocess.run(cmd, check=True, env=env)


def build_companyid_ticker_map(
    db: WRDSClient, company_ids: List[int], target_year: int = 2018
) -> pd.DataFrame:
    if not company_ids:
        return pd.DataFrame(columns=["companyid", "ticker"])
    ids_sql = ",".join(str(int(x)) for x in sorted(set(company_ids)))
    sql = f"""
    select companyid, ticker, ticker_startdate, ticker_enddate
    from ciq.wrds_ciqsymbol_primary
    where companyid in ({ids_sql}) and ticker is not null
    """
    raw = db.raw_sql(sql)
    if raw.empty:
        return pd.DataFrame(columns=["companyid", "ticker"])
    raw["ticker_startdate"] = pd.to_datetime(raw["ticker_startdate"], errors="coerce")
    raw["ticker_enddate"] = pd.to_datetime(raw["ticker_enddate"], errors="coerce")
    year_start = pd.Timestamp(f"{target_year}-01-01")
    year_end = pd.Timestamp(f"{target_year}-12-31")
    raw = raw[
        (raw["ticker_startdate"].isna() | (raw["ticker_startdate"] <= year_end))
        & (raw["ticker_enddate"].isna() | (raw["ticker_enddate"] >= year_start))
    ].copy()
    raw = raw.sort_values(
        by=["companyid", "ticker_enddate", "ticker_startdate"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    out = raw.drop_duplicates(subset=["companyid"], keep="first")[["companyid", "ticker"]].copy()
    out["ticker"] = out["ticker"].astype(str).str.strip()
    out = out[out["ticker"] != ""]
    return out


def merge_q1_scores_into_main(
    main_csv: Path,
    q1_companyid_csv: Path,
    ticker_map_df: pd.DataFrame,
    backup_csv: Path,
    output_csv: Path,
) -> Tuple[int, int, int]:
    main_df = pd.read_csv(main_csv, low_memory=False)
    q1_df = pd.read_csv(q1_companyid_csv, low_memory=False)
    q1_df = q1_df[(q1_df["year"] == 2018) & (q1_df["quarter"] == 1)].copy()
    if q1_df.empty:
        main_df.to_csv(output_csv, index=False)
        return len(main_df), 0, len(main_df)

    has_q1_companyid = "companyid" in q1_df.columns
    has_q1_ticker = "ticker" in q1_df.columns
    if (not has_q1_companyid) and (not has_q1_ticker):
        raise ValueError(
            "Q1 scoring CSV must contain either 'ticker' or 'companyid' column."
        )

    if has_q1_companyid:
        q1_df["companyid"] = pd.to_numeric(q1_df["companyid"], errors="coerce")
        q1_df = q1_df[q1_df["companyid"].notna()].copy()
        q1_df["companyid"] = q1_df["companyid"].astype(int)
        if not ticker_map_df.empty:
            q1_df = q1_df.merge(
                ticker_map_df.rename(columns={"ticker": "_ticker_from_companyid"}),
                on="companyid",
                how="left",
            )
            if has_q1_ticker:
                q1_df["ticker"] = q1_df["ticker"].fillna("").astype(str).str.strip()
                q1_df["ticker"] = q1_df["ticker"].replace({"nan": "", "NaN": ""})
                q1_df["ticker"] = np.where(
                    q1_df["ticker"] == "",
                    q1_df["_ticker_from_companyid"].fillna("").astype(str).str.strip(),
                    q1_df["ticker"],
                )
            else:
                q1_df["ticker"] = (
                    q1_df["_ticker_from_companyid"].fillna("").astype(str).str.strip()
                )
            q1_df = q1_df.drop(columns=["_ticker_from_companyid"], errors="ignore")
        elif not has_q1_ticker:
            raise ValueError(
                "Q1 scoring CSV has companyid but no ticker, and ticker map is empty."
            )

    q1_df["ticker"] = q1_df["ticker"].fillna("").astype(str).str.strip()
    q1_df["ticker"] = q1_df["ticker"].replace({"nan": "", "NaN": ""})
    q1_df = q1_df[q1_df["ticker"] != ""].copy()

    main_cols = main_df.columns.tolist()
    if "companyid" in q1_df.columns:
        q1_df = q1_df.drop(columns=["companyid"])
    for c in main_cols:
        if c not in q1_df.columns:
            q1_df[c] = 0.0 if pd.api.types.is_numeric_dtype(main_df[c]) else ""
    q1_df = q1_df[main_cols].copy()

    backup_csv.parent.mkdir(parents=True, exist_ok=True)
    main_df.to_csv(backup_csv, index=False)

    key_cols = [c for c in ["ticker", "year", "quarter", "fiscal_period"] if c in main_df.columns]
    merged = pd.concat([main_df, q1_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    merged = merged.sort_values(
        by=[c for c in ["ticker", "year", "quarter"] if c in merged.columns],
        kind="mergesort",
    ).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return len(main_df), len(q1_df), len(merged)


def run_fetch_only(args: argparse.Namespace, wrds_password: str) -> None:
    target_company_ids = parse_company_ids(args.company_ids)
    all_transcripts = []

    db = WRDSClient(username=args.wrds_username, password=wrds_password)
    try:
        if (not target_company_ids) and args.company_tickers_csv.strip():
            ticker_csv_path = Path(args.company_tickers_csv)
            if not ticker_csv_path.exists():
                raise FileNotFoundError(f"Ticker CSV not found: {ticker_csv_path}")
            ticker_df = pd.read_csv(ticker_csv_path)
            ticker_col = args.company_ticker_column
            if ticker_col not in ticker_df.columns:
                raise ValueError(
                    f"Ticker column '{ticker_col}' not found in {ticker_csv_path}."
                )
            tickers = [
                str(x).strip()
                for x in ticker_df[ticker_col].dropna().tolist()
                if str(x).strip()
            ]
            target_company_ids = resolve_company_ids_by_tickers_best_calls(
                db=db,
                tickers=tickers,
                start_year=args.start_year,
                end_year=args.end_year,
            )
            if target_company_ids:
                print(f"Using ticker-resolved company IDs: {len(target_company_ids)}")
            else:
                print("No company IDs resolved from ticker CSV; no filter will be applied.")
        elif (not target_company_ids) and args.company_names.strip():
            target_company_ids = resolve_company_ids_by_names(
                db, [x for x in args.company_names.split(",") if x.strip()]
            )
            if target_company_ids:
                print(f"Using resolved company IDs: {target_company_ids}")
            else:
                print("No company IDs resolved from company names; no filter will be applied.")

        for year, month in month_range(args.start_year, args.end_year):
            print(f"Fetching earnings calls: {year}-{month:02d}")
            try:
                data = load_all_earnings_calls(db, year, month, target_company_ids)
                if not data.empty:
                    all_transcripts.append(data)
            except Exception as exc:
                print(f"Failed on {year}-{month:02d}: {exc}")
    finally:
        db.close()

    if not all_transcripts:
        print("No transcripts returned for the selected range.")
        _empty_result().to_csv(args.output, index=False)
        print(f"Saved empty CSV to {args.output}")
        return

    all_transcripts_df = pd.concat(all_transcripts, ignore_index=True)

    if target_company_ids:
        all_transcripts_df = all_transcripts_df[
            all_transcripts_df["companyid"].isin(target_company_ids)
        ].copy()
        print(f"Applied company filter: {len(target_company_ids)} company IDs")
    else:
        print("No company filter applied yet (TARGET_COMPANY_IDS is empty).")

    all_transcripts_df = _dedup_transcripts(all_transcripts_df)
    all_transcripts_df.to_csv(args.output, index=False)
    print(f"Saved CSV to {args.output}")


def run_backfill_q1_and_score(args: argparse.Namespace, wrds_password: str) -> None:
    base_transcripts_csv = Path(args.base_transcripts_csv)
    scoring_script = Path(args.scoring_script)
    backfill_transcripts_csv = Path(args.backfill_transcripts_csv)
    scoring_output_dir = Path(args.scoring_output_dir)
    scoring_cache_path = Path(args.scoring_cache_path)
    main_company_quarter_csv = Path(args.main_company_quarter_csv)
    merged_company_quarter_csv = Path(args.merged_company_quarter_csv)
    merged_backup_csv = Path(args.merged_backup_csv)

    if not base_transcripts_csv.exists():
        raise FileNotFoundError(f"Base transcripts CSV not found: {base_transcripts_csv}")
    if not scoring_script.exists():
        raise FileNotFoundError(f"Scoring script not found: {scoring_script}")
    if not main_company_quarter_csv.exists():
        raise FileNotFoundError(f"Main company-quarter CSV not found: {main_company_quarter_csv}")

    openrouter_api_key = args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is required for backfill_q1 workflow.")
    openrouter_model = args.openrouter_model or os.getenv("OPENROUTER_MODEL", "")
    if not openrouter_model:
        raise ValueError("OPENROUTER_MODEL is required for backfill_q1 workflow.")

    base_df = pd.read_csv(base_transcripts_csv, usecols=["companyid"])
    company_ids = _safe_int_list(base_df["companyid"])
    if not company_ids:
        raise ValueError("No company IDs found in base transcripts CSV.")

    db = WRDSClient(username=args.wrds_username, password=wrds_password)
    try:
        q1_transcripts_df, failed_by_month = backfill_q1_transcripts(
            db=db, company_ids=company_ids, year=args.backfill_year
        )
        print(f"Failed company IDs by month: {failed_by_month}")
        q1_transcripts_df.to_csv(backfill_transcripts_csv, index=False)
        print(
            f"Saved Q1 backfill transcripts to {backfill_transcripts_csv} "
            f"(rows={len(q1_transcripts_df)})"
        )
        if q1_transcripts_df.empty:
            print("Backfill returned empty transcripts; skip scoring and merge.")
            return

        run_scoring_script(
            scoring_script=scoring_script,
            input_csv=backfill_transcripts_csv,
            output_dir=scoring_output_dir,
            cache_path=scoring_cache_path,
            openrouter_api_key=openrouter_api_key,
            openrouter_model=openrouter_model,
            max_workers=args.max_workers,
            request_timeout=args.request_timeout,
        )

        q1_companyid_csv = scoring_output_dir / "company_quarter_ai_scores.csv"
        if not q1_companyid_csv.exists():
            raise FileNotFoundError(
                f"Scoring output company_quarter_ai_scores.csv not found: {q1_companyid_csv}"
            )

        q1_company_df = pd.read_csv(q1_companyid_csv, low_memory=False)
        if "companyid" in q1_company_df.columns:
            mapped_company_ids = _safe_int_list(q1_company_df["companyid"])
            if mapped_company_ids:
                ticker_map_df = build_companyid_ticker_map(
                    db=db, company_ids=mapped_company_ids, target_year=args.backfill_year
                )
                print(
                    f"Scoring output includes companyid; built ticker map for {len(mapped_company_ids)} IDs."
                )
            else:
                ticker_map_df = pd.DataFrame(columns=["companyid", "ticker"])
                print("Scoring output has companyid but no valid IDs; merge will rely on ticker column.")
        else:
            if "ticker" not in q1_company_df.columns:
                raise ValueError(
                    "Scoring output company_quarter_ai_scores.csv must contain either companyid or ticker."
                )
            ticker_map_df = pd.DataFrame(columns=["companyid", "ticker"])
            print("Scoring output has no companyid; merge will use ticker directly.")
    finally:
        db.close()

    old_n, add_n, new_n = merge_q1_scores_into_main(
        main_csv=main_company_quarter_csv,
        q1_companyid_csv=q1_companyid_csv,
        ticker_map_df=ticker_map_df,
        backup_csv=merged_backup_csv,
        output_csv=merged_company_quarter_csv,
    )
    print(
        f"Merged company-quarter rows old/add_or_replace/new: {old_n}/{add_n}/{new_n}"
    )
    print(f"Backup saved: {merged_backup_csv}")
    print(f"Merged output saved: {merged_company_quarter_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch WRDS earnings call transcripts and export CSV."
    )
    parser.add_argument("--wrds-username", required=True, help="Your WRDS username.")
    parser.add_argument(
        "--wrds-password",
        default="",
        help="Your WRDS password. If omitted, uses WRDS_PASSWORD env var.",
    )
    parser.add_argument("--start-year", type=int, default=2024, help="Start year.")
    parser.add_argument("--end-year", type=int, default=2025, help="End year.")
    parser.add_argument(
        "--company-ids",
        default="",
        help="Comma-separated company IDs. Leave empty to use TARGET_COMPANY_IDS in code.",
    )
    parser.add_argument(
        "--company-names",
        default="",
        help="Comma-separated company names. Use this if you do not have company IDs yet.",
    )
    parser.add_argument(
        "--company-tickers-csv",
        default="",
        help="CSV file path containing a ticker column to resolve company IDs.",
    )
    parser.add_argument(
        "--company-ticker-column",
        default="Ticker",
        help="Ticker column name used with --company-tickers-csv.",
    )
    parser.add_argument(
        "--output",
        default="earnings_transcripts_2024_2025.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--workflow",
        choices=["fetch", "backfill_q1"],
        default="fetch",
        help="Workflow mode: fetch transcripts only, or backfill 2018Q1 + score + merge.",
    )
    parser.add_argument(
        "--base-transcripts-csv",
        default="transcripts_2016_2023_merged500_wrds_v2.csv",
        help="Base transcripts CSV used to derive company universe for backfill_q1.",
    )
    parser.add_argument(
        "--scoring-script",
        default="ai_innovation_and_adoption_scoring.py",
        help="Scoring script path for backfill_q1.",
    )
    parser.add_argument(
        "--backfill-year",
        type=int,
        default=2018,
        help="Year for Q1 backfill workflow.",
    )
    parser.add_argument(
        "--backfill-transcripts-csv",
        default="transcripts_2018_q1_backfill_safe.csv",
        help="Temporary transcript CSV output for backfill_q1.",
    )
    parser.add_argument(
        "--scoring-output-dir",
        default="outputs_2018q1_backfill_safe",
        help="Scoring output directory for backfill_q1.",
    )
    parser.add_argument(
        "--scoring-cache-path",
        default="outputs_2018q1_backfill_safe/llm_cache.jsonl",
        help="Scoring cache path for backfill_q1.",
    )
    parser.add_argument(
        "--main-company-quarter-csv",
        default="outputs_merged500_wrds_v2/company_quarter_ai_scores.csv",
        help="Existing company-quarter CSV to merge into for backfill_q1.",
    )
    parser.add_argument(
        "--merged-company-quarter-csv",
        default="outputs_merged500_wrds_v2/company_quarter_ai_scores.csv",
        help="Final merged company-quarter CSV output for backfill_q1.",
    )
    parser.add_argument(
        "--merged-backup-csv",
        default="outputs_merged500_wrds_v2/company_quarter_ai_scores.before_q1_refetch_backup.csv",
        help="Backup path before merging backfill_q1 results.",
    )
    parser.add_argument(
        "--openrouter-api-key",
        default="",
        help="OpenRouter API key for backfill_q1 scoring. Falls back to OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--openrouter-model",
        default="",
        help="OpenRouter model for backfill_q1 scoring. Falls back to OPENROUTER_MODEL.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Scoring workers for backfill_q1.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=45,
        help="Scoring request timeout (seconds) for backfill_q1.",
    )
    args = parser.parse_args()

    if args.start_year > args.end_year:
        raise ValueError("start-year must be <= end-year")

    wrds_password = args.wrds_password or os.getenv("WRDS_PASSWORD", "")
    if not wrds_password:
        raise ValueError(
            "WRDS password is required. Use --wrds-password or set WRDS_PASSWORD env var."
        )

    if args.workflow == "fetch":
        run_fetch_only(args=args, wrds_password=wrds_password)
        return
    if args.workflow == "backfill_q1":
        run_backfill_q1_and_score(args=args, wrds_password=wrds_password)
        return
    raise ValueError(f"Unsupported workflow: {args.workflow}")


if __name__ == "__main__":
    main()
