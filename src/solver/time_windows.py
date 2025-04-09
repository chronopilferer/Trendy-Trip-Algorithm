from solver.utils import time_to_minutes, adjust_for_midnight

def intersect_interval(start1: int, end1: int, start2: int, end2: int):
    """
    두 인터벌의 교집합을 계산합니다.
    내부적으로 자정 넘김 보정을 적용한 후 시작과 종료 시간을 비교합니다.
    """
    start1, end1 = adjust_for_midnight(start1, end1)
    start2, end2 = adjust_for_midnight(start2, end2)
    
    start = max(start1, start2)
    end = min(end1, end2)
    return (start, end) if start < end else None

def merge_intervals(intervals):
    """
    겹치거나 인접한 구간을 병합합니다.
    intervals: (start, end) 튜플 리스트, 각 구간은 start < end 를 만족합니다.
    """
    if not intervals:
        return []
    
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged

def subtract_intervals(main_interval: tuple, sub_intervals: list) -> list:
    """
    메인 구간(main_interval)에서 여러 서브 구간(sub_intervals)을 제거한 후 남은 가용 구간을 반환합니다.
    개선: 서브 구간들을 병합하여 겹치거나 인접한 구간을 하나로 처리합니다.
    """
    merged_sub = merge_intervals([s for s in sub_intervals if s is not None])
    available = []
    current_start, main_end = main_interval
    
    for sub in merged_sub:
        sub_start, sub_end = sub
        if sub_end <= current_start or sub_start >= main_end:
            continue
        if sub_start > current_start:
            available.append((current_start, sub_start))
        current_start = max(current_start, sub_end)
        
    if current_start < main_end:
        available.append((current_start, main_end))
    
    return available

def compute_meal_intervals(meal_preferences: dict, global_start: int, global_end: int) -> dict:
    """
    사용자의 식사 선호 시간 정보를 기반으로 각 식사의 여유 구간(식사 시작 30분 전 ~ 식사 종료 30분 후)을 계산합니다.
    입력된 시간에 대해 유효성 검증 및 자정 넘김 처리를 적용합니다.
    """
    intervals = {}
    for meal, times in meal_preferences.items():
        if not times or len(times) != 2:
            continue
        meal_start = time_to_minutes(times[0])
        meal_end = time_to_minutes(times[1])
        meal_start, meal_end = adjust_for_midnight(meal_start, meal_end)
        
        adjusted_start = max(global_start, meal_start - 30)
        adjusted_end = min(global_end, meal_end + 30)
        
        if adjusted_start < adjusted_end:
            intervals[meal] = (adjusted_start, adjusted_end)
    return intervals

def compute_effective_window(place: dict, global_start: int, global_end: int) -> tuple:
    """
    장소의 운영 시간과 전체 활동 시간(global_start, global_end)을 고려해 유효한 인터벌을 산출합니다.
    자정 넘김 보정을 적용합니다.
    """
    place_open = time_to_minutes(place["open_time"])
    place_close = time_to_minutes(place["close_time"])
    place_open, place_close = adjust_for_midnight(place_open, place_close)
    
    effective_open = max(place_open, global_start)
    effective_close = min(place_close, global_end)
    return effective_open, effective_close

def compute_operational_windows(place: dict, global_start: int, global_end: int) -> list:
    """
    장소의 영업 시간에서 break_time (휴식 시간) 구간을 제외한 가용 구간을 계산합니다.
    break_time 구간이 겹치는 경우를 대비하여 입력 검증과 보정을 진행합니다.
    """
    effective_open, effective_close = compute_effective_window(place, global_start, global_end)
    main_interval = (effective_open, effective_close)
    
    break_intervals = []
    if "break_time" in place and place["break_time"]:
        bt = place["break_time"]
        if len(bt) % 2 == 0:
            for i in range(0, len(bt), 2):
                try:
                    b_start = time_to_minutes(bt[i])
                    b_end = time_to_minutes(bt[i+1])
                except Exception as e:
                    continue  
                b_start, b_end = adjust_for_midnight(b_start, b_end)
                inter = intersect_interval(effective_open, effective_close, b_start, b_end)
                if inter:
                    break_intervals.append(inter)
                    
    if break_intervals:
        available_segments = subtract_intervals(main_interval, break_intervals)
        return available_segments
    else:
        return [main_interval]

def compute_restaurant_windows(effective_open: int, effective_close: int, meal_intervals: dict, place_name: str):
    """
    식당의 유효 운영 시간과 사용자의 식사 선호 시간 간의 교집합을 계산하여
    식사 타입별 가능한 인터벌을 반환합니다.
    """
    valid_windows = []
    for meal, (meal_start, meal_end) in meal_intervals.items():
        inter = intersect_interval(effective_open, effective_close, meal_start, meal_end)
        if inter:
            valid_windows.append((inter[0], inter[1], meal))
    if not valid_windows:
        raise ValueError(f"식당 {place_name}은(는) 식사 선호 시간에 부합하는 윈도우가 없습니다.")
    return valid_windows

def calculate_effective_time_windows(places: list, user: dict) -> dict:
    """
    사용자와 장소 정보를 기반으로 각 장소별 가용 시간 윈도우를 계산합니다.
    - 전체 활동 시간 및 식사 선호 시간에 맞추어 보정하며, 자정 넘김 처리 및 인터벌 연산 개선을 적용합니다.
    """
    effective_windows = {}
    global_start = time_to_minutes(user["start_time"])
    global_end = time_to_minutes(user["end_time"])
    global_start, global_end = adjust_for_midnight(global_start, global_end)
    
    meal_preferences = user.get("meal_time_preferences", {})
    meal_intervals = compute_meal_intervals(meal_preferences, global_start, global_end)
    
    for place in places:
        operational_segments = compute_operational_windows(place, global_start, global_end)
        
        if place.get("category") == "restaurant":
            restaurant_windows = []
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
    
    return effective_windows