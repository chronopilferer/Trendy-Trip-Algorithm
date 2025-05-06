import logging
import json
from typing import Optional, Tuple, List, Dict

from tripscheduler.api.snap       import snap_to_road
from tripscheduler.api.directions import create_matrices
from tripscheduler.api.mock       import create_distance_matrix

logger = logging.getLogger(__name__)

def prepare_matrices(
    places: List[Dict],
    api_key_id: str,
    api_key: str,
    use_mock: bool = False,
    mock_raw_path: Optional[str] = None,
) -> Tuple[List[List[int]], List[List[Dict]], List[List[Optional[List[float]]]]]:
    """
    장소 리스트를 받아:
      - time_matrix, raw, path_matrix를 반환.
      - use_mock+mock_raw_path 있으면 mock으로,
      - use_mock만 있으면 하버사인 mock,
      - 아니면 실제 API (스냅→directions)
    """
    # 1) mock + raw 데이터
    if use_mock and mock_raw_path:
        logger.info("Mock(raw) 모드: %s 로드", mock_raw_path)
        with open(mock_raw_path, 'r', encoding='utf-8') as f:
            mock_response_matrix = json.load(f)
        return create_matrices(
            places, api_key_id, api_key,
            is_mock_enabled=True, mock_api_response=mock_response_matrix
        )

    # 2) mock only
    if use_mock:
        logger.info("Mock 모드: 하버사인 기반 거리만 생성")
        duration_matrix = create_distance_matrix(places)
        n = len(places)
        raw_api_response = [[None] * n for _ in range(n)]
        path_matrix = [[None] * n for _ in range(n)]
        return duration_matrix, raw_api_response, path_matrix

    # 3) 실제 API: 좌표 스냅 후 matrix 생성 snap 이 필요한가? 나중에 고민
    # for p in places:
    #     p['y_cord'], p['x_cord'] = snap_to_road(
    #         p['y_cord'], p['x_cord'],
    #         api_key_id, api_key
    #     )

    return create_matrices(
        places, api_key_id, api_key,
        is_mock_enabled=False, mock_api_response=None
    )