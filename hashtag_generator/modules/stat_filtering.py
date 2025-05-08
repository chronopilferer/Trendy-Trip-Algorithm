import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
from utils.file_io import save_result, copy_image
from utils.io import load_json_records, load_record

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 1. 장소-감성 전용 정책 (하드 코딩된 임계값)
# ------------------------------------------------------------------
BASE_POLICY: Dict[str, Dict[str, Any]] = {
    "brightness_score": {"method":"percentile","direction":"both","lower_pct":0.10,"upper_pct":0.90},
    "entropy_score":    {"method":"percentile","direction":"low","lower_pct":0.20},
    "person_area_ratio":{"method":"percentile","direction":"high","upper_pct":0.90,"absolute_max":0.10},
    "food_area_ratio":  {"method":"percentile","direction":"high","upper_pct":0.90,"absolute_max":0.30},
    "text_area_ratio":  {"method":"percentile","direction":"high","upper_pct":0.90,"absolute_max":0.20},
    "scene_max":        {"method":"percentile","direction":"low","lower_pct":0.30,"absolute_min":0.20},
    "object_max":       {"method":"percentile","direction":"low","lower_pct":0.30,"absolute_min":0.18},
    "gap_avg":          {"method":"zscore","direction":"both","abs_threshold":0.07},
}

def select_filtering_strategy(
    stats_df: pd.DataFrame,
    df_full: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """
    BASE_POLICY에 정의된 필드는 그대로 사용.
    나머지 필드는 데이터 분포(skew, zero_ratio)에 따라 동적 전략 생성.
    """
    strat: Dict[str, Dict[str, Any]] = {}
    for _, row in stats_df.iterrows():
        field = row["field"]
        if field in BASE_POLICY:
            strat[field] = BASE_POLICY[field]
            continue

        series = pd.to_numeric(df_full[field], errors="coerce").dropna()
        skew = float(series.skew()) if len(series)>0 else 0.0
        zero_ratio = float((series == 0).mean()) if len(series)>0 else 0.0

        if zero_ratio > 0.8:
            strat[field] = {"method":"binary_presence","threshold":row.get("percentile_95", 0)}
        elif abs(skew) < 0.5:
            strat[field] = {"method":"zscore","direction":"both"}
        elif skew > 1.0:
            strat[field] = {"method":"percentile","direction":"high","upper_pct":0.95}
        else:
            strat[field] = {"method":"iqr","direction":"both"}
    return strat

def make_threshold_rules(
    stats_df: pd.DataFrame,
    strategy: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    각 필드 전략에 따라 low/high 한 쌍의 임계값 계산.
    percentile → stats_df의 percentile 컬럼 직접 참조
    zscore     → abs_threshold or mean±3*std
    iqr        → IQR_low/IQR_high
    binary     → threshold 값만 사용
    """
    rules: Dict[str, Dict[str, float]] = {}
    # stats_df를 field->row 매핑용으로 인덱스 설정
    stats_map = stats_df.set_index("field")

    for field, strat in strategy.items():
        m = strat["method"]

        if m == "binary_presence":
            rules[field] = {"low": None, "high": float(strat["threshold"])}
            continue

        row = stats_map.loc[field]
        # 기본 절대값
        low  = strat.get("absolute_min", -float("inf"))
        high = strat.get("absolute_max", float("inf"))

        if m == "percentile":
            lp = int(strat.get("lower_pct", 0) * 100)
            up = int(strat.get("upper_pct", 1) * 100)
            low  = max(low, row.get(f"percentile_{lp}", row["min"]))
            high = min(high, row.get(f"percentile_{up}", row["max"]))

        elif m == "iqr":
            low  = max(low, row["IQR_low"])
            high = min(high, row["IQR_high"])

        elif m == "zscore":
            if "abs_threshold" in strat:
                low, high = -strat["abs_threshold"], strat["abs_threshold"]
            else:
                low  = max(low, row["mean"] - 3*row["std"])
                high = min(high, row["mean"] + 3*row["std"])

        rules[field] = {"low": float(low), "high": float(high)}

    return rules

def process_stat_filtering(
    json_dir: str,
    stats_csv: str,
    output_dir: str
) -> None:
    """
    1) stat CSV와 JSON 레코드 로드
    2) 전략 생성 → 임계값(rule) 생성
    3) 필터링 실행 → 결과 JSON/이미지 분류 저장
    """
    json_dir = Path(json_dir)
    stats_df = pd.read_csv(stats_csv)
    records  = load_json_records(json_dir)
    df_base  = pd.DataFrame(records).set_index("file_name")

    # 1) 전략 및 판정 기준 계산
    strategy = select_filtering_strategy(stats_df, df_base)
    rules    = make_threshold_rules(stats_df, strategy)

    # 2) 필터링 수행
    df_full = df_base.copy()
    for field, strat in strategy.items():
        flag_col = f"{field}_flag"
        df_full[flag_col] = True
        series = pd.to_numeric(df_full[field], errors="coerce")

        # 바이너리 존재 여부
        if strat["method"] == "binary_presence":
            thresh = rules[field]["high"]
            df_full.loc[series > thresh, flag_col] = False
            continue

        low, high = rules[field]["low"], rules[field]["high"]
        direction = strat.get("direction", "both")

        if direction in ("both", "low"):
            df_full.loc[series < low, flag_col] = False
        if direction in ("both", "high"):
            df_full.loc[series > high, flag_col] = False

    df_full["final_decision"] = df_full.filter(like="_flag").all(axis=1)

    # 3) 결과 디렉토리에 JSON·이미지 저장
    subdir   = Path(output_dir) / "filtered_results"
    pass_dir = subdir / "pass"
    non_dir  = subdir / "non-pass"
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

    logger.info("필터링 완료 ▶ %s", subdir)
