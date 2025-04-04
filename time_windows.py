from utils import time_to_minutes

def intersect_interval(start1: int, end1: int, start2: int, end2: int):
    start = max(start1, start2)
    end = min(end1, end2)
    return (start, end) if start < end else None

def subtract_intervals(main_interval: tuple, sub_intervals: list) -> list:
    available = []
    current_start, main_end = main_interval
    # sub_intervals를 시작 시간 순으로 정렬
    for sub in sorted(sub_intervals, key=lambda x: x[0]):
        sub_start, sub_end = sub
        # sub 구간이 메인 구간과 겹치지 않으면 건너뜀
        if sub_end <= current_start or sub_start >= main_end:
            continue
        # 현재 시작과 sub 구간의 시작 사이에 여유가 있다면 구간 추가
        if sub_start > current_start:
            available.append((current_start, sub_start))
        # 현재 시작 업데이트
        current_start = max(current_start, sub_end)
    # 마지막 남은 구간 추가
    if current_start < main_end:
        available.append((current_start, main_end))
    return available

def compute_meal_intervals(meal_preferences: dict, global_start: int, global_end: int) -> dict:
    intervals = {}
    for meal, times in meal_preferences.items():
        if not times:
            continue
        meal_start = time_to_minutes(times[0])
        meal_end = time_to_minutes(times[1])
        adjusted_start = max(global_start, meal_start - 30)
        adjusted_end = min(global_end, meal_end + 30)
        intervals[meal] = (adjusted_start, adjusted_end)
    return intervals

def compute_effective_window(place: dict, global_start: int, global_end: int) -> tuple:
    place_open = time_to_minutes(place["open_time"])
    place_close = time_to_minutes(place["close_time"])
    effective_open = max(place_open, global_start)
    effective_close = min(place_close, global_end)
    return effective_open, effective_close

def compute_operational_windows(place: dict, global_start: int, global_end: int) -> list:
    effective_open, effective_close = compute_effective_window(place, global_start, global_end)
    main_interval = (effective_open, effective_close)
    
    # break_time이 정의되어 있다면, 짝수 개의 시간 문자열을 쌍으로 묶어 리스트 생성
    break_intervals = []
    if "break_time" in place and place["break_time"]:
        bt = place["break_time"]
        if len(bt) % 2 == 0:
            for i in range(0, len(bt), 2):
                b_start = time_to_minutes(bt[i])
                b_end = time_to_minutes(bt[i+1])
                # 메인 구간과 겹치는 경우만 고려
                inter = intersect_interval(effective_open, effective_close, b_start, b_end)
                if inter:
                    break_intervals.append(inter)
    
    # break_intervals가 있으면, 메인 구간에서 빼고 나머지 가용 구간 계산
    if break_intervals:
        available_segments = subtract_intervals(main_interval, break_intervals)
        return available_segments
    else:
        return [main_interval]

def compute_restaurant_windows(effective_open: int, effective_close: int, meal_intervals: dict, place_name: str):
    valid_windows = []
    for meal, (meal_start, meal_end) in meal_intervals.items():
        inter = intersect_interval(effective_open, effective_close, meal_start, meal_end)
        if inter:
            valid_windows.append((inter[0], inter[1], meal))
    if not valid_windows:
        raise ValueError(f"식당 {place_name}은(는) 식사 선호 시간에 부합하는 윈도우가 없습니다.")
    return valid_windows

def calculate_effective_time_windows(places: list, user: dict) -> dict:
    effective_windows = {}
    
    # 사용자의 전체 활동 시간을 분 단위로 변환
    global_start = time_to_minutes(user["start_time"])
    global_end = time_to_minutes(user["end_time"])
    
    # 식사 선호 시간 간격 계산 (식사 시작 30분 전 ~ 종료 30분 후)
    meal_preferences = user.get("meal_time_preferences", {})
    meal_intervals = compute_meal_intervals(meal_preferences, global_start, global_end)
    
    for place in places:
        operational_segments = compute_operational_windows(place, global_start, global_end)
        print(f"[DEBUG] {place['name']} - 영업시간: {place['open_time']} ~ {place['close_time']}")
        if "break_time" in place and place["break_time"]:
            print(f"         break_time: {place['break_time']}")
        print(f"         가용 세그먼트: {operational_segments}")
        
        if place.get("category") == "restaurant":
            restaurant_windows = []
            # 각 가용 세그먼트에 대해 식사 선호 시간과의 교집합을 구함
            for segment in operational_segments:
                seg_start, seg_end = segment
                try:
                    windows = compute_restaurant_windows(seg_start, seg_end, meal_intervals, place["name"])
                    restaurant_windows.extend(windows)
                except ValueError:
                    continue
            if not restaurant_windows:
                raise ValueError(f"식당 {place['name']}은(는) 식사 선호 시간에 부합하는 윈도우가 없습니다.")
            effective_windows[place["id"]] = restaurant_windows
        else:
            effective_windows[place["id"]] = [(start, end, None) for (start, end) in operational_segments]
    
    print("[DEBUG] 최종 계산된 시간 윈도우:")
    import pprint
    pprint.pprint(effective_windows)
    
    return effective_windows