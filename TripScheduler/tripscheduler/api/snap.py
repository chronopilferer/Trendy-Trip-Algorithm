import requests
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def snap_to_road(
    lat: float,
    lon: float,
    api_key_id: str,
    api_key: str
) -> Tuple[float, float]:
    """
    Naver Directions API를 이용해 한 점을 도로에 스냅.
    - start==goal 파라미터로 호출하면, 반환된 경로의 첫 좌표를 스냅 좌표로 사용.
    - 실패 시 원본 좌표 반환.
    """
    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": api_key_id,
        "X-NCP-APIGW-API-KEY": api_key
    }
    params = {
        "start": f"{lon},{lat}",
        "goal":  f"{lon},{lat}",
        "lang":  "ko"
    }

    try:
        api_response = requests.get(url, headers=headers, params=params, timeout=5)
        api_response.raise_for_status()
        api_response_data = api_response.json()
    except requests.RequestException as e:
        logger.error("도로 스냅 API 호출 실패 (%f,%f): %s", lat, lon, e)
        return lat, lon

    optimal = api_response_data.get("route", {}).get("traoptimal") \
           or api_response_data.get("route", {}).get("trafast") \
           or []
    if optimal and isinstance(optimal[0].get("path"), list):
        snapped_lon, snapped_lat = optimal[0]["path"][0]
        return round(snapped_lat, 7), round(snapped_lon, 7)

    logger.warning("도로 스냅 응답에 경로 없음 (%f,%f)", lat, lon)
    return lat, lon
