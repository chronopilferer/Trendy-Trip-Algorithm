from tripscheduler.utils.time import adjust_for_midnight

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