import logging
import json
from typing import Optional, Tuple, List, Dict

from create_matrices import create_matrices
from tripscheduler.api.mock import create_distance_matrix

logger = logging.getLogger(__name__)

def prepare_matrices(
    places: List[Dict],
    api_key_id: str,
    api_key: str,
    use_mock: bool = False,
    mock_raw_path: Optional[str] = None,
) -> Tuple[List[List[int]], List[List[Dict]], List[List[Optional[List[float]]]]]:
    """
    거리 행렬 준비 함수 (mapping_info 없이 단순화)

    Returns:
      - time_matrix: 분 단위 소요 시간
      - raw: API 응답 전체
      - path_matrix: 각 경로의 path 정보
    """
    if use_mock and mock_raw_path:
        logger.info("Mock 모드: raw 데이터 로드 후 거리·경로 매트릭스 생성")
        with open(mock_raw_path, 'r', encoding='utf-8') as f:
            raw_matrix = json.load(f)
        return create_matrices(
            places, api_key_id, api_key,
            mock=True, raw_matrix=raw_matrix
        )

    if use_mock:
        logger.info("Mock 모드: 하버사인 거리 매트릭스 사용 (path 정보 없음)")
        time_matrix = create_distance_matrix(places)
        raw = [[None] * len(places) for _ in places]
        path_matrix = [[None] * len(places) for _ in places]
        return time_matrix, raw, path_matrix

    # 실제 API 호출
    return create_matrices(
        places, api_key_id, api_key,
        mock=False, raw_matrix=None
    )

if __name__ == '__main__':

    # Mock 데이터를 로드
    with open('./directions_raw_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 실제 places 데이터를 로드
    with open('./tc.json', 'r', encoding='utf-8') as f:
        schedule = json.load(f)

    places = schedule['places']

    # mock 모드로 거리 행렬 생성
    time_matrix, raw, path_matrix = prepare_matrices(
        places=places, 
        api_key_id=None, 
        api_key=None, 
        use_mock=True, 
        mock_raw_path='./directions_raw_data.json'
    )

    # 결과를 JSON 파일로 저장
    json.dump(time_matrix, open('./time_matrix.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    json.dump(raw, open('./directions_raw_data.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    json.dump(path_matrix, open('./path_matrix.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
