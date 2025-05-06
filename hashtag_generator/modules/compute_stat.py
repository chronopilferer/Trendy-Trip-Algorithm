import logging
from pathlib import Path
import json
from typing import List, Optional

import pandas as pd

from utils.io import load_json_records  

logger = logging.getLogger(__name__)
DEFAULT_PERCENTILES: List[float] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

def compute_field_statistics(
    records: List[dict],
    fields: Optional[List[str]] = None,
    percentiles: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    JSON 레코드 리스트에서 지정된 fields의 통계량을 계산하여 DataFrame으로 반환합니다.

    Args:
        records: JSON에서 로드된 레코드 목록
        fields: 분석할 필드 리스트, None이면 모든 숫자형 컬럼
        percentiles: 계산할 백분위 리스트, None이면 DEFAULT_PERCENTILES 사용
    """
    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("통계 계산 대상 데이터가 없습니다.")
        return pd.DataFrame()

    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES
    if fields is None:
        fields = df.select_dtypes(include="number").columns.tolist()

    stats = []
    for f in fields:
        if f not in df.columns:
            logger.warning(f"컬럼 누락: {f}")
            continue

        series = pd.to_numeric(df[f], errors="coerce").dropna()
        if series.empty:
            logger.warning(f"값 없음: {f}")
            continue

        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        row = {
            "field": f,
            "min": series.min(),
            "max": series.max(),
            "mean": series.mean(),
            "std": series.std(),
            "IQR_low": Q1 - 1.5 * IQR,
            "IQR_high": Q3 + 1.5 * IQR,
        }
        for p in percentiles:
            row[f"percentile_{int(p*100)}"] = series.quantile(p)
        stats.append(row)

    stat_df = pd.DataFrame(stats)
    logger.info("통계량 계산 완료: %d개 필드", len(stat_df))
    return stat_df

def process_stat_compute(
    json_dir: Path,
    output_dir: Path,
    fields: Optional[List[str]] = None,
    percentiles: Optional[List[float]] = None
) -> None:
    """
    1) json_dir 내 모든 JSON 파일 로드
    2) 필드별 통계량 계산
    3) 결과를 output_dir에 CSV로 저장
    """
    try:
        records = load_json_records(json_dir)
    except Exception as e:
        logger.error(f"JSON 로드 실패: {e}")
        return

    stat_df = compute_field_statistics(records, fields, percentiles)
    if stat_df.empty:
        logger.error("통계량 결과가 없어 저장을 생략합니다.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    stat_csv = output_dir / "field_statistics.csv"
    data_csv = output_dir / "all_data.csv"

    try:
        stat_df.to_csv(stat_csv, index=False)
        pd.DataFrame(records).to_csv(data_csv, index=False)
        logger.info(f"CSV 저장 완료: {stat_csv}, {data_csv}")
    except Exception as e:
        logger.error(f"CSV 저장 실패: {e}")