import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from utils.file_io import save_result, copy_image
from utils.io import load_json_records, load_record

logger = logging.getLogger(__name__)

def make_threshold_rules(
    stats_df: pd.DataFrame,
    method: str = "iqr",
    lower_pct: float = 0.05,
    upper_pct: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """
    통계량 DataFrame으로부터 threshold_rules 생성.
    method: 'iqr', 'percentile', 'zscore' 지원
    """
    rules: Dict[str, Dict[str, float]] = {}
    for field, row in stats_df.iterrows():
        if method == "iqr":
            low, high = row["IQR_low"], row["IQR_high"]
        elif method == "percentile":
            low = row.get(f"percentile_{int(lower_pct * 100)}")
            high = row.get(f"percentile_{int(upper_pct * 100)}")
        elif method == "zscore":
            mean, std = row["mean"], row["std"]
            low, high = mean - 3*std, mean + 3*std
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
    upper_pct: float = 0.95
) -> None:
    """
    통계 기반 필터링:
    1) json_dir 내 JSON 로드 → records 리스트
    2) 통계량 CSV 로드 → threshold_rules 생성
    3) records → DataFrame → 플래그 계산
    4) 개별 JSON 업데이트 후 pass/non-pass 폴더로 저장 및 이미지 복사
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

    # 2) 통계량 불러오기 및 rules 생성
    try:
        stats_df = pd.read_csv(stats_csv).set_index("field")
    except Exception as e:
        logger.error(f"통계량 CSV 로드 실패: {e}")
        return
    rules = make_threshold_rules(stats_df, method, lower_pct, upper_pct)

    # 3) DataFrame 변환 및 flag 계산
    df = pd.DataFrame(records)
    for feature in rules:
        df[f"{feature}_flag"] = True
        low, high = rules[feature]["low"], rules[feature]["high"]
        series = pd.to_numeric(df.get(feature, []), errors="coerce")
        df.loc[series < low, f"{feature}_flag"] = False
        df.loc[series > high, f"{feature}_flag"] = False
    df["final_decision"] = df[[f"{f}_flag" for f in rules]].all(axis=1)

    # 4) 개별 JSON 업데이트 및 저장
    for json_path in json_dir.glob("*.json"):
        try:
            rec: Dict[str, Any] = load_record(json_path, defaults={})
            rec_id = rec.get("image_id") or rec.get("filename")
            row = df[df.get("image_id", df.index) == rec_id]
            if row.empty:
                logger.warning(f"통계 기록 없음: {rec_id}")
                continue
            row = row.iloc[0]

            rec["flags"] = {f: bool(row[f"{f}_flag"]) for f in rules}
            rec["pass"] = bool(row["final_decision"] )

            # JSON 저장
            target_dir = pass_dir if rec["pass"] else nonpass_dir
            out_json = target_dir / f"{rec_id}.json"
            save_result(rec, str(out_json))

            # 이미지 복사
            img_path = rec.get("filepath") or rec.get("file_path")
            if img_path and Path(img_path).exists():
                copy_image(img_path, str(target_dir), rec_id)
            else:
                logger.warning(f"이미지 경로 유효하지 않음: {rec_id}")

        except Exception as e:
            logger.error(f"개별 JSON 처리 실패: {json_path.name} - {e}", exc_info=True)

    logger.info(f"필터링 완료: {out_base} (pass={pass_dir}, non-pass={nonpass_dir})")