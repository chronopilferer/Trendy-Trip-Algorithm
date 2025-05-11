import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

from Img2hashtag.img2hastag.utils.io import load_json_records
from Img2hashtag.img2hastag.utils.constants import DEFAULT_PERCENTILES

logger = logging.getLogger(__name__)

def compute_field_statistics(
    records: List[dict],
    fields: Optional[List[str]] = None,
    percentiles: Optional[List[float]] = None,
    index_field: str = "file_name"
) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("통계 계산 대상 데이터가 없습니다.")
        return pd.DataFrame()

    if index_field not in df.columns:
        logger.error(f"'{index_field}' 필드가 records에 존재하지 않습니다.")
        return pd.DataFrame()

    df.set_index(index_field, inplace=True)

    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES
    if fields is None:
        fields = df.select_dtypes(include="number").columns.tolist()

    stats = []
    for f in fields:
        if f not in df.columns:
            logger.warning(f"컬럼 누락: {f}")
            continue
        if not pd.api.types.is_numeric_dtype(df[f]):
            logger.warning(f"'{f}' 필드는 숫자형이 아니므로 건너뜁니다.")
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
            row[f"p{int(p*1000):03d}"] = series.quantile(p)
        stats.append(row)

    stat_df = pd.DataFrame(stats)
    logger.info("통계량 계산 완료: %d개 필드", len(stat_df))
    return stat_df


def process_stat_compute(
    json_dir: Path,
    output_dir: Path,
    fields: Optional[List[str]] = None,
    percentiles: Optional[List[float]] = None,
    index_field: str = "file_name"
) -> None:
    try:
        records = load_json_records(json_dir)
    except Exception as e:
        logger.error(f"JSON 로드 실패: {e}")
        return

    stat_df = compute_field_statistics(records, fields, percentiles, index_field)
    if stat_df.empty:
        logger.warning("통계량 결과가 없어 저장을 생략합니다.")
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