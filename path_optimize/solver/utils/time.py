from datetime import datetime
import re

def time_to_minutes(time_str):
    """
    주어진 "HH:MM" 형식의 문자열을 분 단위 정수로 변환합니다.
    - 올바른 형식이 아니면 ValueError를 발생시킵니다.
    """
    if not isinstance(time_str, str):
        raise ValueError("시간 입력은 문자열이어야 합니다.")
        
    if not re.match(r'^\d{1,2}:\d{2}$', time_str):
        raise ValueError("시간 형식이 올바르지 않습니다. (예: 'HH:MM')")
        
    try:
        dt = datetime.strptime(time_str, "%H:%M")
    except Exception as e:
        raise ValueError(f"time_to_minutes 변환 에러: {e}")
    
    return dt.hour * 60 + dt.minute

def minutes_to_time_str(minutes):
    # 분(min)을 HH:MM 형식 문자열로 변환
    hour = minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"

def adjust_for_midnight(start, end):
    """
    인터벌이 자정을 넘어가는 경우, 종료 시간(end)이 시작 시간(start)보다 작거나 같다면
    24시간(1440분)을 추가하여 보정합니다.
    """
    if end <= start:
        end += 1440
    return start, end

