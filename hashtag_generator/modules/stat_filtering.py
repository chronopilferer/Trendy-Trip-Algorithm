import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from utils.file_io import save_result, copy_image
from utils.io import load_json_records, load_record

logger = logging.getLogger(__name__)

def select_filtering_strategy(
    stats_df: pd.DataFrame,
    df_full: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """
    stats_df: 'field' 컬럼 포함, 통계량 데이터프레임
    df_full: index='file_name', 원본 레코드 데이터프레임
    """
    strat: Dict[str, Dict[str, Any]] = {}
    for idx, row in stats_df.iterrows():
        field = row['field']
        series = pd.to_numeric(df_full[field], errors="coerce").dropna()
        skew = float(series.skew()) if not series.empty else 0.0
        zero_ratio = float((series == 0).mean()) if not series.empty else 0.0
        
        # dynamic rules
        if zero_ratio > 0.8:
            strat[field] = {
                "method": "binary_presence",
                "threshold": row.get("percentile_95", 0)
            }
        elif abs(skew) < 0.5:
            strat[field] = {"method": "zscore", "direction": "both"}
        elif skew > 1.0:
            strat[field] = {"method": "percentile", "direction": "high"}
        else:
            strat[field] = {"method": "iqr", "direction": "both"}
    return strat

def make_threshold_rules(
    stats_df: pd.DataFrame,
    strategy: Dict[str, Dict[str, Any]],
    lower_pct: float = 0.05,
    upper_pct: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    rules: Dict[str, Dict[str, float]] = {}
    # stats_df는 'field' 컬럼을 가짐
    for _, row in stats_df.iterrows():
        field = row['field']
        strat = strategy.get(field, {})
        method = strat.get("method")
        if method == "binary_presence":
            continue
        
        if method == "iqr":
            low, high = row["IQR_low"], row["IQR_high"]
        elif method == "percentile":
            low  = row.get(f"percentile_{int(lower_pct*100)}", row["min"])
            high = row.get(f"percentile_{int(upper_pct*100)}", row["max"])
        elif method == "zscore":
            low, high = row["mean"] - 3*row["std"], row["mean"] + 3*row["std"]
        else:
            continue

        rules[field] = {"low": float(low), "high": float(high)}
    return rules

def process_stat_filtering(
    json_dir: str,
    stats_csv: str,
    output_dir: str
) -> None:
    json_dir = Path(json_dir)
    stats_df  = pd.read_csv(stats_csv)
    records   = load_json_records(json_dir)
    df_full   = pd.DataFrame(records).set_index("file_name")

    # 1) 동적 전략 선택
    strategy = select_filtering_strategy(stats_df, df_full)
    # 2) 임계값 룰 생성
    rules = make_threshold_rules(stats_df, strategy)

    # 3) 필터링 수행
    for field, strat in strategy.items():
        flag_col = f"{field}_flag"
        df_full[flag_col] = True
        series = pd.to_numeric(df_full[field], errors="coerce")

        if strat["method"] == "binary_presence":
            thresh = strat["threshold"]
            df_full.loc[series > thresh, flag_col] = False
        else:
            low, high = rules[field]["low"], rules[field]["high"]
            dirc = strat.get("direction", "both")
            if dirc in ("both", "low"):
                df_full.loc[series < low, flag_col] = False
            if dirc in ("both", "high"):
                df_full.loc[series > high, flag_col] = False

    df_full["final_decision"] = df_full.filter(like="_flag").all(axis=1)

    # 4) 결과 저장 & 이미지 복사
    pass_dir = Path(output_dir) / "pass"
    non_dir  = Path(output_dir) / "non-pass"
    for d in (pass_dir, non_dir):
        (d / "json").mkdir(parents=True, exist_ok=True)

    for jp in json_dir.glob("*.json"):
        rec = load_record(jp, defaults={})
        fid = rec.get("file_name")
        if not fid or fid not in df_full.index:
            continue
        row = df_full.loc[fid]
        rec["flags"] = {f: bool(row[f + "_flag"]) for f in strategy}
        rec["pass"]  = bool(row["final_decision"])
        tgt = pass_dir if rec["pass"] else non_dir
        save_result(rec, tgt / "json" / f"{fid}.json")
        img = rec.get("file_path")
        if img and Path(img).exists():
            copy_image(img, ".", tgt)

    logger.info("필터링 완료 ▶ %s", output_dir)
