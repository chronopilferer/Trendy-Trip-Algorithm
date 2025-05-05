import requests
import logging

logger = logging.getLogger(__name__)

def snap_to_road(lat: float, lon: float, api_key_id: str, api_key: str) -> tuple[float, float]:
    """
    네이버 Directions API 경로 탐색을 역으로 이용해
    입력 좌표를 가장 가까운 도로 위 좌표로 스냅
    """
    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    headers = {
        'X-NCP-APIGW-API-KEY-ID': api_key_id,
        'X-NCP-APIGW-API-KEY': api_key
    }
    params = {
        "start": f"{lon},{lat}",  # lon,lat
        "goal":  f"{lon},{lat}",  # 동일 지점 → 도로까지 스냅
        "lang":  "ko"
    }

    try:
        logger.debug("도로 스냅 요청: (%f, %f)", lat, lon)
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        logger.debug("도로 스냅 응답 수신 완료")
    except requests.RequestException as e:
        logger.error("도로 스냅 실패: (%f, %f) → %s", lat, lon, e)
        return lat, lon

    route = data.get("route", {}).get("trafast", [])
    if route and route[0].get("path"):
        snapped_lon, snapped_lat = route[0]["path"][0]
        logger.info("도로 스냅 성공: (%.6f, %.6f) → (%.6f, %.6f)", lat, lon, snapped_lat, snapped_lon)
        return float(snapped_lat), float(snapped_lon)

    logger.warning("도로 스냅 실패 (경로 없음): (%.6f, %.6f)", lat, lon)
    return lat, lon
