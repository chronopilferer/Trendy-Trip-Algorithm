import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from utils.file_io import save_result, copy_image
from utils.io import load_json_records, load_record

logger = logging.getLogger(__name__)

# 각 feature별 필터링 전략 정의 (감성 이미지 유지 기준)
# method: 필터링 방식(iqr, percentile, zscore, binary_presence)
# direction: "low"/"high"/"both" (iqr, percentile, zscore에서만 사용)
# threshold: binary_presence에서 사용 (상한값)
FIELD_FILTERING_STRATEGY: Dict[str, Dict[str, Any]] = {
    "brightness_score":   {"method": "iqr",               "direction": "both"},
    "resolution_ratio":   {"method": "iqr",               "direction": "low"},
    "entropy_score":      {"method": "iqr",               "direction": "low"},
    # 감성 이미지는 사람/음식/텍스트가 적은 것이 특징
    "person_area_ratio":  {"method": "binary_presence",    "threshold": 0},
    "food_area_ratio":    {"method": "binary_presence",    "threshold": 0},
    "text_area_ratio":    {"method": "binary_presence",    "threshold": 0.05},  
    "num_text_boxes":     {"method": "binary_presence",    "threshold": 3},     
    # 기타 scene/object score는 iqr 활용
    "scene_max":          {"method": "iqr",               "direction": "low"},
    "scene_topk_avg":     {"method": "iqr",               "direction": "low"},
    "object_max":         {"method": "iqr",               "direction": "high"},
    "object_topk_avg":    {"method": "iqr",               "direction": "high"},
    "gap_max":            {"method": "iqr",               "direction": "both"},
    "gap_avg":            {"method": "iqr",               "direction": "both"},
}

def make_threshold_rules(
    stats_df: pd.DataFrame,
    lower_pct: float = 0.05,
    upper_pct: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """
    통계량 DataFrame으로부터 {field: {low, high}} 사전을 생성
    iqr, percentile, zscore 방식 지원
    binary_presence는 threshold로 처리됨
    """
    rules: Dict[str, Dict[str, float]] = {}

    for field, row in stats_df.iterrows():
        strat = FIELD_FILTERING_STRATEGY.get(field, {})
        method = strat.get("method", "iqr")
        if method == "binary_presence":
            continue

        if method == "iqr":
            low, high = row["IQR_low"], row["IQR_high"]
        elif method == "percentile":
            low = row.get(f"percentile_{int(lower_pct*100)}", row.get("min"))
            high = row.get(f"percentile_{int(upper_pct*100)}", row.get("max"))
        elif method == "zscore":
            mean, std = row["mean"], row["std"]
            low, high = mean - 3 * std, mean + 3 * std
        else:
            continue

        rules[field] = {"low": float(low), "high": float(high)}

    return rules

def process_stat_filtering(
    json_dir: str,
    stats_csv: str,
    output_dir: str,
    lower_pct: float = 0.05,
    upper_pct: float = 0.95,
) -> None:
    """
    감성 이미지 유지 기준으로 특성별 맞춤 필터링 수행
    """
    json_dir = Path(json_dir)
    stats_csv = Path(stats_csv)
    out_base = Path(output_dir)
    pass_dir = out_base / "pass"
    nonpass_dir = out_base / "non-pass"
    for d in (out_base, pass_dir, nonpass_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) JSON 레코드 로드
    try:
        records = load_json_records(json_dir)
    except Exception as e:
        logger.error(f"JSON 로딩 실패: {e}")
        return
    if not records:
        logger.warning("필터링 대상 JSON 레코드가 없습니다.")
        return

    # 2) 통계량 로드 및 임계값 생성
    try:
        stats_df = pd.read_csv(stats_csv).set_index("field")
    except Exception as e:
        logger.error(f"통계량 CSV 로드 실패: {e}")
        return
    rules = make_threshold_rules(stats_df, lower_pct, upper_pct)

    # 3) DataFrame 변환 및 flag 계산
    df = pd.DataFrame(records)
    if "file_name" not in df.columns:
        logger.error("file_name 필드가 존재하지 않습니다.")
        return
    df.set_index("file_name", inplace=True)

    for feature, strat in FIELD_FILTERING_STRATEGY.items():
        method = strat["method"]
        flag_col = f"{feature}_flag"
        df[flag_col] = True
        series = pd.to_numeric(df.get(feature, []), errors="coerce")

        if method == "binary_presence":
            # threshold 초과 시 부적합
            threshold = strat.get("threshold", 0)
            df.loc[series > threshold, flag_col] = False
        else:
            # iqr, percentile, zscore 방식
            if feature not in rules:
                continue
            low, high = rules[feature]["low"], rules[feature]["high"]
            direction = strat.get("direction", "both")
            if direction in ("both", "low"):
                df.loc[series < low, flag_col] = False
            if direction in ("both", "high"):
                df.loc[series > high, flag_col] = False

    # 모든 flag가 True인 경우에만 최종 통과
    df["final_decision"] = df[[c for c in df.columns if c.endswith("_flag")]].all(axis=1)

    # 4) 개별 JSON 업데이트 및 파일 분류
    for json_path in json_dir.glob("*.json"):
        try:
            rec: Dict[str, Any] = load_record(json_path, defaults={})
            rec_id = rec.get("file_name")
            if not rec_id:
                logger.warning(f"file_name 누락: {json_path.name}")
                continue
            if rec_id not in df.index:
                logger.warning(f"통계 기록 없음: {rec_id}")
                continue

            row = df.loc[rec_id]
            rec["flags"] = {f: bool(row[f"{f}_flag"]) for f in FIELD_FILTERING_STRATEGY.keys()}
            rec["pass"] = bool(row["final_decision"])

            target = pass_dir if rec["pass"] else nonpass_dir
            out_json_dir = target / 'json'
            out_json_dir.mkdir(parents=True, exist_ok=True)
            save_result(rec, str(out_json_dir / f"{rec_id}.json"))
            img_path = rec.get("file_path")
            if img_path and Path(img_path).exists():
                copy_image(img_path, '.', target)
            else:
                logger.warning(f"이미지 경로 유효하지 않음: {rec_id}")

        except Exception as e:
            logger.error(f"개별 JSON 처리 실패: {json_path.name} - {e}", exc_info=True)

    logger.info(f"필터링 완료 ▶ {out_base}  (pass={pass_dir}, non-pass={nonpass_dir})")