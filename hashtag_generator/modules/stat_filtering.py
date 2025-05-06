import logging
from pathlib import Path
from typing import Dict, Any, Literal

import pandas as pd

from utils.file_io import save_result, copy_image
from utils.io import load_json_records, load_record

logger = logging.getLogger(__name__)

FIELD_DIRECTIONS: Dict[str, Literal["both", "low", "high"]] = {
    "brightness_score":   "both",
    "resolution_ratio":   "low",
    "entropy_score":      "low",
    "person_area_ratio":  "high",
    "food_area_ratio":    "high",
    "text_area_ratio":    "high",
    "num_text_boxes":     "high",
    "scene_max":          "low",
    "scene_topk_avg":     "low",
    "object_max":         "high",
    "object_topk_avg":    "high",
    "gap_max":            "both",
    "gap_avg":            "both",
}

def make_threshold_rules(
    stats_df: pd.DataFrame,
    method: str = "iqr",
    lower_pct: float = 0.05,
    upper_pct: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """
    통계량 DataFrame으로부터 {field: {low, high}} 사전을 생성
    """
    rules: Dict[str, Dict[str, float]] = {}
    for field, row in stats_df.iterrows():
        if method == "iqr":
            low, high = row["IQR_low"], row["IQR_high"]
        elif method == "percentile":
            low = row.get(f"percentile_{int(lower_pct*100)}")
            high = row.get(f"percentile_{int(upper_pct*100)}")
        elif method == "zscore":
            mean, std = row["mean"], row["std"]
            low, high = mean - 3 * std, mean + 3 * std
        else:
            raise ValueError(f"알 수 없는 method: {method}")
        rules[field] = {"low": float(low), "high": float(high)}
    return rules

def process_stat_filtering(
    json_dir: str,
    stats_csv: str,
    output_dir: str,
    method: str = "iqr",
    lower_pct: float = 0.05,
    upper_pct: float = 0.95,
) -> None:
    """
    특성‑맞춤 임계값 필터링
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
    rules = make_threshold_rules(stats_df, method, lower_pct, upper_pct)

    # 3) DataFrame 변환 및 flag 계산
    df = pd.DataFrame(records)

    for feature, bounds in rules.items():
        direction = FIELD_DIRECTIONS.get(feature, "both")
        low, high = bounds["low"], bounds["high"]
        series = pd.to_numeric(df.get(feature, []), errors="coerce")

        flag_col = f"{feature}_flag"
        df[flag_col] = True  # 기본 통과

        if direction in ("both", "low"):
            df.loc[series < low, flag_col] = False
        if direction in ("both", "high"):
            df.loc[series > high, flag_col] = False

    # 모든 flag AND 연산
    df["final_decision"] = df[[c for c in df.columns if c.endswith("_flag")]].all(axis=1)

    # 4) 개별 JSON 업데이트 및 파일 정리
    for json_path in json_dir.glob("*.json"):
        try:
            rec: Dict[str, Any] = load_record(json_path, defaults={})
            rec_id = Path(rec.get("file_path", "")).stem  # 파일명 기준 매칭
            row = df.loc[df.index == rec_id]
            if row.empty:
                logger.warning(f"통계 기록 없음: {rec_id}")
                continue
            row = row.iloc[0]

            # flags 저장
            rec["flags"] = {
                f: bool(row[f"{f}_flag"]) for f in rules.keys()
            }
            rec["pass"] = bool(row["final_decision"])

            # JSON 저장
            target_dir = pass_dir if rec["pass"] else nonpass_dir
            out_json = target_dir / f"{rec_id}.json"
            save_result(rec, str(out_json))

            # 이미지 복사
            img_path = rec.get("file_path")
            if img_path and Path(img_path).exists():
                copy_image(img_path, str(target_dir), rec_id)
            else:
                logger.warning(f"이미지 경로 유효하지 않음: {rec_id}")

        except Exception as e:
            logger.error(f"개별 JSON 처리 실패: {json_path.name} - {e}", exc_info=True)

    logger.info(f"필터링 완료 ▶ {out_base}  (pass={pass_dir}, non-pass={nonpass_dir})")
