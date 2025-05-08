import logging
import copy
from pathlib import Path
from typing import Dict, Any, Tuple, List

import pandas as pd
from utils.file_io import save_result, copy_image
from utils.io import load_json_records, load_record

logger = logging.getLogger(__name__)

def auto_percentile(
    series: pd.Series,
    target_reject: float,
    tail: str = "both"
) -> Tuple[float, float]:
    clean = series.dropna()
    if tail == "low":
        low = clean.quantile(target_reject)
        high = clean.max()
    elif tail == "high":
        low = clean.min()
        high = clean.quantile(1 - target_reject)
    else:  # both
        half = target_reject / 2
        low  = clean.quantile(half)
        high = clean.quantile(1 - half)
    return float(low), float(high)

def tune_policy_percentiles(
    df_full: pd.DataFrame,
    policy: Dict[str, Dict[str, Any]],
    target_reject: float = 0.30
) -> None:
    for field, strat in policy.items():
        if strat.get("method") != "percentile":
            continue
        series = pd.to_numeric(df_full[field], errors="coerce")
        lo_val, hi_val = auto_percentile(series, target_reject, strat["direction"])
        clean = series.dropna()
        strat["lower_pct"] = float((clean < lo_val).mean())
        strat["upper_pct"] = float((clean > hi_val).mean())

# ------------------------------------------------------------------
# 1. 장소-감성 전용 정책 (초기값)
# ------------------------------------------------------------------
BASE_POLICY: Dict[str, Dict[str, Any]] = {
    "brightness_score": {"method":"percentile","direction":"both","lower_pct":0.02,"upper_pct":0.98},
    "entropy_score":    {"method":"percentile","direction":"low","lower_pct":0.15},
    "person_area_ratio":{"method":"percentile","direction":"high","upper_pct":0.95,"absolute_max":0.10},
    "food_area_ratio":  {"method":"percentile","direction":"high","upper_pct":0.95,"absolute_max":0.30},
    "text_area_ratio":  {"method":"percentile","direction":"high","upper_pct":0.90,"absolute_max":0.20},
    "scene_max":        {"method":"percentile","direction":"low","lower_pct":0.25,"absolute_min":0.20},
    "object_max":       {"method":"percentile","direction":"low","lower_pct":0.25,"absolute_min":0.18},
    "gap_avg":          {"method":"zscore","direction":"both","abs_threshold":0.07},
}

def select_filtering_strategy(
    stats_df: pd.DataFrame,
    df_full: pd.DataFrame,
    policy: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    strat: Dict[str, Dict[str, Any]] = {}
    for _, row in stats_df.iterrows():
        f = row["field"]
        if f in policy:
            strat[f] = policy[f]
            continue
        series = pd.to_numeric(df_full[f], errors="coerce").dropna()
        skew = float(series.skew()) if not series.empty else 0.0
        zero_ratio = float((series == 0).mean()) if not series.empty else 0.0
        if zero_ratio > 0.8:
            strat[f] = {"method":"binary_presence","threshold":row.get("percentile_95",0)}
        elif abs(skew) < 0.5:
            strat[f] = {"method":"zscore","direction":"both"}
        elif skew > 1.0:
            strat[f] = {"method":"percentile","direction":"high"}
        else:
            strat[f] = {"method":"iqr","direction":"both"}
    return strat

def make_threshold_rules(
    stats_df: pd.DataFrame,
    strategy: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    rules: Dict[str, Dict[str, float]] = {}
    for _, row in stats_df.iterrows():
        f = row["field"]
        s = strategy.get(f, {})
        m = s.get("method")
        if m == "binary_presence":
            continue

        # 절대 기준
        if "abs_threshold" in s:
            low, high = -s["abs_threshold"], s["abs_threshold"]
        else:
            low = s.get("absolute_min", -float("inf"))
            high = s.get("absolute_max", float("inf"))

        # 통계 기준과 병합
        if m == "iqr":
            low = max(low, row["IQR_low"])
            high = min(high, row["IQR_high"])
        elif m == "percentile":
            lp = int(s.get("lower_pct",0)*100)
            up = int(s.get("upper_pct",1)*100)
            low  = max(low, row.get(f"percentile_{lp}", row["min"]))
            high = min(high, row.get(f"percentile_{up}", row["max"]))
        elif m == "zscore" and "abs_threshold" not in s:
            low = max(low, row["mean"] - 3*row["std"])
            high = min(high, row["mean"] + 3*row["std"])

        rules[f] = {"low": float(low), "high": float(high)}
    return rules

def process_stat_filtering(
    json_dir: str,
    stats_csv: str,
    output_dir: str,
    optimize_percents: List[float] = None,
    default_reject: float = 0.30
) -> None:
    json_dir = Path(json_dir)
    stats_df = pd.read_csv(stats_csv)
    records  = load_json_records(json_dir)
    df_base  = pd.DataFrame(records).set_index("file_name")

    # 기본값 세팅
    if optimize_percents is None:
        optimize_percents = [default_reject]

    for pct in optimize_percents:
        # 1) POLICY 복제 및 최적화
        policy = copy.deepcopy(BASE_POLICY)
        tune_policy_percentiles(df_base, policy, pct)

        # 2) 전략 및 룰 계산
        strategy = select_filtering_strategy(stats_df, df_base, policy)
        rules    = make_threshold_rules(stats_df, strategy)

        # 3) 필터링 수행
        df_full = df_base.copy()
        for field, strat in strategy.items():
            flag_col = f"{field}_flag"
            df_full[flag_col] = True
            series = pd.to_numeric(df_full[field], errors="coerce")

            if strat["method"] == "binary_presence":
                thresh = strat["threshold"]
                df_full.loc[series > thresh, flag_col] = False
                continue

            low  = rules[field]["low"]
            high = rules[field]["high"]
            dirc = strat.get("direction", "both")
            if dirc in ("both", "low"):
                df_full.loc[series < low, flag_col] = False
            if dirc in ("both", "high"):
                df_full.loc[series > high, flag_col] = False

        df_full["final_decision"] = df_full.filter(like="_flag").all(axis=1)

        # 4) 결과 저장
        subdir = Path(output_dir) / f"reject_{int(pct*100)}"
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

        logger.info("필터링 완료 ▶ %s (reject=%s)", subdir, pct)
