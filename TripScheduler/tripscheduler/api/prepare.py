from typing import Optional
import logging
import json

from tripscheduler.api.snap import snap_to_road
from tripscheduler.api.directions import create_matrices
from tripscheduler.api.mock import create_distance_matrix

logger = logging.getLogger(__name__)

def prepare_matrices(
    places: list[dict],
    api_key_id: str,
    api_key: str,
    use_mock: bool = False,
    mock_matrix_path: Optional[str] = None,
    mock_raw_path: Optional[str] = None,
    time_matrix: Optional[list[list[int]]] = None,
    raw: Optional[list[list[dict]]] = None
) -> tuple[list[list[int]], list[list[dict]]]:
    """
    거리 행렬 준비 함수

    우선순위:
    1. time_matrix, raw가 직접 주어졌다면 그대로 사용
    2. use_mock=True인 경우 mock 데이터 생성
    3. 아니면 실제 API로부터 생성
    """

    if time_matrix is not None and raw is not None:
        logger.info("매트릭스 직접 주입됨 - 그대로 반환")
        return time_matrix, raw

    if use_mock:
        if mock_matrix_path and mock_raw_path:
            logger.info("파일에서 mock 매트릭스 로드 중")
            with open(mock_matrix_path, 'r', encoding='utf-8') as f:
                time_matrix = json.load(f)
            with open(mock_raw_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            return time_matrix, raw

        logger.info("mock 모드 - 하버사인 기반 거리 매트릭스 생성")
        time_matrix = create_distance_matrix(places)
        raw = [[None] * len(places) for _ in range(len(places))]
        return time_matrix, raw

    logger.info("실시간 거리 매트릭스 생성 시작")
    for p in places:
        lat, lon = p['y_cord'], p['x_cord']
        s_lat, s_lon = snap_to_road(lat, lon, api_key_id, api_key)
        p['y_cord'], p['x_cord'] = s_lat, s_lon
        logger.debug("스냅 완료: %s → (%f, %f)", p.get('name', 'Unnamed'), s_lat, s_lon)

    time_matrix, raw = create_matrices(places, api_key_id, api_key)
    return time_matrix, raw
