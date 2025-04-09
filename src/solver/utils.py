from datetime import datetime
import math
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

def haversine_distance(lat1, lon1, lat2, lon2):
    # 두 지점 간의 거리(km)를 하버사인 공식을 통해 계산 후 10배하고 반올림
    R = 6371  
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return int(round(R * c))

def is_accommodation(place):
    return place.get("category") == "accommodation"

def is_transport(place):
    return place.get("category") == "transport"

def validate_first_place(places, expected_category, err_msg):
    if not places:
        raise ValueError("places 리스트가 비어있습니다.")
    
    first_cat = places[0].get("category")
    if first_cat != expected_category:
        raise ValueError(err_msg)

def determine_start_end_indices(places, day_info):
    # day_info에서 첫날과 마지막 날 여부를 명시적으로 읽습니다.
    is_first_day = day_info.get("is_first_day", False)
    is_last_day = day_info.get("is_last_day", False)
    
    if not places:
        raise ValueError("places 리스트가 비어있습니다.")
        
    # 공통: 0번째 인덱스의 카테고리가 transport 또는 accommodation 이어야 함
    if places[0].get("category") not in ("transport", "accommodation"):
        raise ValueError("첫 번째 노드의 카테고리는 transport 또는 accommodation이어야 합니다.")

    # 첫날이면서 당일치기가 아닌 경우
    # - 시작(0번 인덱스)는 반드시 transport여야 합니다.
    # - 숙소(accommodation)가 2개 이상이면 오류를 발생시키고, 그렇지 않으면 end_index는 지정하지 않습니다.
    if is_first_day and not is_last_day:
        validate_first_place(places, "transport", "첫날의 첫 번째 노드는 반드시 transport이어야 합니다.")
        start_index = 0

        accommodation_indices = [idx for idx, p in enumerate(places) if is_accommodation(p)]
        if len(accommodation_indices) >= 2:
            raise ValueError(f"첫날에는 accommodation 카테고리가 두개 이상 있으면 안 됩니다. 현재 {len(accommodation_indices)}개 발견됨.")
        end_index = None

    # 당일치기 (첫날이면서 동시에 마지막 날인 경우)
    # - 시작(0번 인덱스)는 반드시 transport여야 합니다.
    # - 첫 번째 노드를 제외한 나머지 중 transport 카테고리는 정확히 1개여야 하며, 이를 end_index로 사용합니다.
    elif is_first_day and is_last_day:
        validate_first_place(places, "transport", "당일치기의 경우, 첫 번째 노드는 반드시 transport이어야 합니다.")
        start_index = 0

        transport_indices = [idx for idx, p in enumerate(places[1:], start=1) if is_transport(p)]
        if len(transport_indices) != 1:
            raise ValueError(f"당일치기의 경우, 첫 장소 제외한 transport 카테고리가 정확히 1개 있어야 합니다. 현재 {len(transport_indices)}개 발견됨.")
        end_index = transport_indices[0]

    # 마지막 날(첫날이 아닌 경우)
    # - 시작(0번 인덱스)는 반드시 accommodation이어야 합니다.
    # - 마지막 노드는 반드시 transport여야 하며,
    # - places[:-1]에서 accommodation 카테고리는 오직 0번 인덱스만 있어야 합니다.
    elif not is_first_day and is_last_day:
        validate_first_place(places, "accommodation", "마지막 날의 시작 노드(0번 인덱스)는 반드시 accommodation이어야 합니다.")
        start_index = 0

        if not is_transport(places[-1]):
            raise ValueError("마지막 날의 마지막 노드는 반드시 transport이어야 합니다.")
        end_index = len(places) - 1

        accommodation_indices = [idx for idx, p in enumerate(places[:-1]) if is_accommodation(p)]
        if accommodation_indices != [0]:
            raise ValueError("마지막 날에는 시작 노드 외에 다른 accommodation이 없어야 합니다.")

    # 중간 날(첫날도 마지막 날도 아닌 경우)
    # - 시작(0번 인덱스)는 반드시 accommodation이어야 합니다.
    # - 중간 날에는 transport 카테고리가 전혀 존재하면 안 됩니다.
    # - 첫 번째 노드를 제외한 리스트에서 accommodation 카테고리가 2개 이상이면 오류를 발생시키고, 그렇지 않으면 end_index는 지정하지 않습니다.
    else:
        validate_first_place(places, "accommodation", "중간 날의 시작 노드(0번 인덱스)는 반드시 accommodation이어야 합니다.")
        start_index = 0

        if any(is_transport(p) for p in places):
            raise ValueError("중간 날에는 transport 카테고리가 존재하면 안 됩니다.")

        accommodation_indices = [idx for idx, p in enumerate(places[1:], start=1) if is_accommodation(p)]
        if len(accommodation_indices) >= 2:
            raise ValueError(f"중간 날에는 첫 번째 노드를 제외한 accommodation 카테고리가 두개 이상 있으면 안 됩니다. 현재 {len(accommodation_indices)}개 발견됨.")
        end_index = None

    return start_index, end_index