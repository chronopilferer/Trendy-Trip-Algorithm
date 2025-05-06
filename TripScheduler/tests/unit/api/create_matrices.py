import requests
import time
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

def create_matrices(
    places: List[Dict],
    api_key_id: str,
    api_key: str,
    mock: bool = False,
    raw_matrix: Optional[List[List[Dict]]] = None
) -> Tuple[List[List[int]], List[List[Dict]], List[List[Optional[List[float]]]]]:
    """
    거리 및 경로 행렬 생성

    Returns:
      - time_matrix: 분 단위 소요 시간
      - raw: 원시 API 응답
      - path_matrix: 경로 path 정보
    """
    n = len(places)
    time_matrix = [[0] * n for _ in range(n)]
    raw = [[None] * n for _ in range(n)]
    path_matrix = [[None] * n for _ in range(n)]

    if mock and raw_matrix is not None:
        logger.info("Mock 모드 (raw_matrix 기반)")
        raw = raw_matrix
        for i in range(n):
            for j in range(i + 1, n):
                optimal = raw[i][j].get("route", {}).get("traoptimal", [])
                if optimal:
                    duration_ms = optimal[0].get("summary", {}).get("duration", 0)
                    t_min = int(round(duration_ms / 60000))
                    path = optimal[0].get("path")
                else:
                    logger.warning("유효하지 않은 경로: %s → %s", places[i].get('name'), places[j].get('name'))
                    t_min, path = -1, None
                time_matrix[i][j] = time_matrix[j][i] = t_min
                path_matrix[i][j] = path_matrix[j][i] = path
        return time_matrix, raw, path_matrix

    if mock:
        raise ValueError("mock=True일 경우 raw_matrix가 필요합니다.")

    headers = {
        'X-NCP-APIGW-API-KEY-ID': api_key_id,
        'X-NCP-APIGW-API-KEY': api_key
    }
    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"

    for i in range(n):
        for j in range(i + 1, n):
            start = f"{places[i]['x_cord']},{places[i]['y_cord']}"
            goal = f"{places[j]['x_cord']},{places[j]['y_cord']}"
            params = {"start": start, "goal": goal, "lang": "ko"}

            try:
                resp = requests.get(url, headers=headers, params=params, timeout=5)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                logger.error("API 실패: %s → %s: %s", start, goal, e)
                data = {}

            raw[i][j] = raw[j][i] = data
            optimal = data.get("route", {}).get("traoptimal", [])
            if optimal:
                duration_ms = optimal[0].get("summary", {}).get("duration", 0)
                t_min = int(round(duration_ms / 60000))
                path = optimal[0].get("path")
            else:
                logger.warning("유효하지 않은 경로: %s → %s", start, goal)
                t_min, path = -1, None

            time_matrix[i][j] = time_matrix[j][i] = t_min
            path_matrix[i][j] = path_matrix[j][i] = path
            time.sleep(0.1)

    logger.info("거리 매트릭스 계산 완료")
    return time_matrix, raw, path_matrix

if __name__ == '__main__':

    import json

    with open('./directions_raw_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open('./tc.json', 'r', encoding='utf-8') as f:
        schedule = json.load(f)

    places = schedule['places']

    time_matrix, raw, path_matrix = create_matrices(places=places, api_key=None, api_key_id=None, mock=True, raw_matrix=data)

    json.dump(time_matrix, open('./time_matrix.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    json.dump(raw, open('./directions_raw_data.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    json.dump(path_matrix, open('./path_matrix.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)