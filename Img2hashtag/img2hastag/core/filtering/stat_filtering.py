import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from img2hastag.utils.io import load_json_records, load_record, save_result, copy_image
from img2hastag.utils.constants import POLICY

logger = logging.getLogger(__name__)

def select_filtering_strategy(
    stats_df: pd.DataFrame,
    df_full: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    strat: Dict[str, Dict[str, Any]] = {}
    for _, row in stats_df.iterrows():
        field = row["field"]
        if field in POLICY:
            strat[field] = POLICY[field]
            continue

        series = pd.to_numeric(df_full[field], errors="coerce").dropna()
        skew = float(series.skew()) if len(series) > 0 else 0.0
        zero_ratio = float((series == 0).mean()) if len(series) > 0 else 0.0

        if zero_ratio > 0.8:
            strat[field] = {
                "method": "binary_presence",
                "threshold": row.get("percentile_95", 0)
            }
        elif abs(skew) < 0.5:
            strat[field] = {"method": "zscore", "direction": "both"}
        elif skew > 1.0:
            strat[field] = {"method": "percentile", "direction": "high", "upper_pct": 0.95}
        else:
            strat[field] = {"method": "iqr", "direction": "both"}
    return strat

def make_threshold_rules(
    stats_df: pd.DataFrame,
    strategy: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    rules: Dict[str, Dict[str, float]] = {}
    stats_map = stats_df.set_index("field")

    for field, strat in strategy.items():
        m = strat["method"]

        if m == "binary_presence":
            rules[field] = {"low": None, "high": float(strat["threshold"])}
            continue

        if m == "absolute":
            low  = strat.get("absolute_min", -float("inf"))
            high = strat.get("absolute_max", float("inf"))
            rules[field] = {"low": float(low), "high": float(high)}
            continue

        # 그 외 percentile, iqr, zscore
        row = stats_map.loc[field]
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
                low  = max(low, row["mean"] - 3 * row["std"])
                high = min(high, row["mean"] + 3 * row["std"])

        rules[field] = {"low": float(low), "high": float(high)}

    return rules

def process_stat_filtering(
    json_dir: str,
    stats_csv: str,
    output_dir: str
) -> None:
    json_dir = Path(json_dir)
    stats_df = pd.read_csv(stats_csv)
    records  = load_json_records(json_dir)
    df_base  = pd.DataFrame(records).set_index("file_name")

    strategy = select_filtering_strategy(stats_df, df_base)
    rules    = make_threshold_rules(stats_df, strategy)

    df_full = df_base.copy()
    for field, strat in strategy.items():
        flag_col = f"{field}_flag"
        df_full[flag_col] = True
        series = pd.to_numeric(df_full[field], errors="coerce")

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

    pass_dir = Path(output_dir) / "pass"
    non_dir  = Path(output_dir) / "non-pass"
    for d in (pass_dir, non_dir):
        d.mkdir(parents=True, exist_ok=True)

    for jp in json_dir.glob("*.json"):
        rec = load_record(jp, defaults={})
        
        if "flags" in rec and "pass" in rec:
            logger.info(f"[스킵] 필터링 결과 존재함: {jp.name}")
            continue

        fid = rec.get("file_name")
        if not fid or fid not in df_full.index:
            continue
        row = df_full.loc[fid]
        rec["flags"] = {f: bool(row[f + "_flag"]) for f in strategy}
        rec["pass"]  = bool(row["final_decision"])

        save_result(rec, jp)

        img = rec.get("file_path")
        tgt = pass_dir if rec["pass"] else non_dir
        if img and Path(img).exists():
            copy_image(img, '.', str(tgt))

    logger.info("필터링 완료 ▶ ")
