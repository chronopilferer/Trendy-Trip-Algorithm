import pytest
from utils import time_to_minutes, minutes_to_time_str, adjust_for_midnight, haversine_distance, is_accommodation, is_transport, determine_start_end_indices

def test_time_to_minutes_valid():
    assert time_to_minutes("08:30") == 510
    assert time_to_minutes("00:00") == 0
    assert time_to_minutes("23:59") == 1439

def test_time_to_minutes_invalid():
    with pytest.raises(ValueError):
        time_to_minutes(1230)  # 숫자 타입이면 에러 발생
    with pytest.raises(ValueError):
        time_to_minutes("abc")  # 올바르지 않은 문자열

def test_minutes_to_time_str():
    assert minutes_to_time_str(510) == "08:30"
    assert minutes_to_time_str(0) == "00:00"

def test_adjust_for_midnight():
    # 자정 넘김이 없는 경우
    start, end = adjust_for_midnight(600, 900)
    assert start == 600 and end == 900
    # 자정 넘김 처리: 종료 시간이 시작보다 작으면 1440분을 더함
    start, end = adjust_for_midnight(1380, 60)
    assert start == 1380 and end == 1500  # 60 + 1440

def test_haversine_distance():
    # 제주공항과 한라산 국립공원 사이의 거리를 계산해봄(정수값)
    d = haversine_distance(33.505413, 126.492153, 33.361000, 126.533000)
    assert isinstance(d, int)
    assert d > 0

def test_is_accommodation_and_transport():
    place1 = {"category": "accommodation"}
    place2 = {"category": "transport"}
    place3 = {"category": "restaurant"}
    assert is_accommodation(place1) is True
    assert is_transport(place2) is True
    assert not is_accommodation(place3)
    assert not is_transport(place3)

def test_determine_start_end_indices():
    # 첫날 (당일치기가 아닌 경우): 첫 노드가 transport여야 하고 accommodation은 단 하나
    places = [
        {"category": "transport"},
        {"category": "landmark"}
    ]
    day_info = {"is_first_day": True, "is_last_day": False}
    start, end = determine_start_end_indices(places, day_info)
    assert start == 0 and end is None

    # 당일치기: 첫 노드 transport, 나머지 중 transport가 딱 1개
    places = [
        {"category": "transport"},
        {"category": "restaurant"},
        {"category": "transport"}
    ]
    day_info = {"is_first_day": True, "is_last_day": True}
    start, end = determine_start_end_indices(places, day_info)
    assert start == 0 and end == 2

    # 마지막 날: 시작은 accommodation, 마지막 노드는 transport
    places = [
        {"category": "accommodation"},
        {"category": "landmark"},
        {"category": "transport"}
    ]
    day_info = {"is_first_day": False, "is_last_day": True}
    start, end = determine_start_end_indices(places, day_info)
    assert start == 0 and end == 2
