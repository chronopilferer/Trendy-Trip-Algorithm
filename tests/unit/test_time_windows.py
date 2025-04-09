import pytest
from time_windows import (
    intersect_interval, merge_intervals, subtract_intervals,
    compute_meal_intervals, compute_effective_window, compute_operational_windows,
    compute_restaurant_windows, calculate_effective_time_windows
)
from utils import time_to_minutes

def test_intersect_interval():
    # 교집합: 두 인터벌의 중간 부분이 존재
    result = intersect_interval(500, 800, 600, 900)
    assert result == (600, 800)
    # 교집합이 없는 경우
    result = intersect_interval(500, 600, 610, 700)
    assert result is None

def test_merge_intervals():
    intervals = [(100, 200), (150, 250), (300, 350)]
    merged = merge_intervals(intervals)
    assert merged == [(100, 250), (300, 350)]

def test_subtract_intervals():
    main = (100, 500)
    subs = [(150, 200), (250, 300)]
    available = subtract_intervals(main, subs)
    assert available == [(100, 150), (200, 250), (300, 500)]

def test_compute_meal_intervals():
    meal_prefs = {"lunch": ["12:00", "13:00"], "dinner": ["18:00", "19:00"]}
    global_start = time_to_minutes("08:00")
    global_end = time_to_minutes("22:00")
    intervals = compute_meal_intervals(meal_prefs, global_start, global_end)
    # 간단하게 식사 시작/종료 시간 기준으로 보정 되었는지 확인
    assert intervals["lunch"][0] <= time_to_minutes("12:00")
    assert intervals["lunch"][1] >= time_to_minutes("13:00")
    assert intervals["dinner"][0] <= time_to_minutes("18:00")
    assert intervals["dinner"][1] >= time_to_minutes("19:00")

def test_compute_effective_window():
    place = {"open_time": "09:00", "close_time": "17:00"}
    global_start = time_to_minutes("08:00")
    global_end = time_to_minutes("18:00")
    effective_open, effective_close = compute_effective_window(place, global_start, global_end)
    assert effective_open == time_to_minutes("09:00")
    assert effective_close == time_to_minutes("17:00")

def test_compute_operational_windows_without_break():
    place = {"open_time": "09:00", "close_time": "17:00", "break_time": []}
    global_start = time_to_minutes("08:00")
    global_end = time_to_minutes("18:00")
    windows = compute_operational_windows(place, global_start, global_end)
    assert windows == [(time_to_minutes("09:00"), time_to_minutes("17:00"))]

def test_compute_operational_windows_with_break():
    place = {"open_time": "09:00", "close_time": "17:00", "break_time": ["12:00", "13:00"]}
    global_start = time_to_minutes("08:00")
    global_end = time_to_minutes("18:00")
    windows = compute_operational_windows(place, global_start, global_end)
    # 두 개의 구간이 생성됨: 09:00~12:00, 13:00~17:00
    assert len(windows) == 2
    assert windows[0] == (time_to_minutes("09:00"), time_to_minutes("12:00"))
    assert windows[1] == (time_to_minutes("13:00"), time_to_minutes("17:00"))

def test_compute_restaurant_windows_success():
    # effective window: 11:00 ~ 14:00
    effective_open = time_to_minutes("11:00")
    effective_close = time_to_minutes("14:00")
    meal_intervals = {"lunch": (time_to_minutes("12:00"), time_to_minutes("13:00"))}
    windows = compute_restaurant_windows(effective_open, effective_close, meal_intervals, "식당A")
    # lunch 윈도우가 포함되어 있는지 확인
    assert any(meal == "lunch" for _, _, meal in windows)

def test_calculate_effective_time_windows_non_restaurant():
    places = [
        {"id": 1, "open_time": "09:00", "close_time": "17:00", "category": "landmark", "break_time": []}
    ]
    user = {"start_time": "08:00", "end_time": "18:00", "meal_time_preferences": {}}
    windows = calculate_effective_time_windows(places, user)
    # 비식당인 경우, 세번째 요소가 None이어야 함
    assert windows[1] == [(time_to_minutes("09:00"), time_to_minutes("17:00"), None)]
