import requests
import time
import logging

logger = logging.getLogger(__name__)

def create_matrices(places: list[dict], api_key_id: str, api_key: str):
    """
    places 리스트에 대해:
      - time_matrix: 분(min) 단위 소요 시간
      - raw: API 응답 전체
    반환
    """
    n = len(places)
    time_matrix = [[0]*n for _ in range(n)]
    raw         = [[None]*n for _ in range(n)]

    headers = {
        'X-NCP-APIGW-API-KEY-ID': api_key_id,
        'X-NCP-APIGW-API-KEY': api_key
    }
    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"

    logger.info("총 %d개 장소에 대해 거리 행렬 계산 시작", n)

    for i, pi in enumerate(places):
        for j in range(i + 1, n):
            start = f"{pi['x_cord']},{pi['y_cord']}"
            goal  = f"{places[j]['x_cord']},{places[j]['y_cord']}"
            params = {"start": start, "goal": goal, "lang": "ko"}

            try:
                logger.debug("요청: %s → %s", start, goal)
                resp = requests.get(url, headers=headers, params=params, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                logger.debug("응답 수신 완료 (%s → %s)", start, goal)
            except requests.RequestException as e:
                logger.error("Directions API 실패: %s → %s: %s", start, goal, e)
                data = {}

            raw[i][j] = raw[j][i] = data

            optimal = data.get("route", {}).get("traoptimal", [])
            if optimal:
                duration_ms = optimal[0].get("summary", {}).get("duration", 0)
                t_min = int(round(duration_ms / 60000))
            else:
                logger.warning("유효하지 않은 경로: %s → %s", start, goal)
                t_min = -1

            time_matrix[i][j] = time_matrix[j][i] = t_min
            time.sleep(0.1)  # API 요청 간 딜레이

    logger.info("거리 행렬 계산 완료")
    return time_matrix, raw
