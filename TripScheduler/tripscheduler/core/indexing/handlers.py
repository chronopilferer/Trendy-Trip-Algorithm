from typing import Optional, Tuple, Dict, List
from tripscheduler.types import PlaceList, DayInfo, Handler
from tripscheduler.core.indexing.category import get_indices_by_category, validate_place_category

def handle_first_day(
    places: PlaceList,
    acc_indices: List[int],
    transport_indices: List[int],
    start_index: int = 0,
    end_index: int = -1
) -> Tuple[int, Optional[int]]:
    if not transport_indices:
        raise ValueError("여행 첫날에 transport 장소가 없습니다.")
    start = transport_indices[start_index]
    validate_place_category(places[start], "transport", "여행 첫날 시작 장소는 transport여야 합니다.")
    
    if len(acc_indices) > 1:
        raise ValueError("여행 첫날에 accommodation이 2개 이상 있습니다.")
    end = acc_indices[start_index] if acc_indices else None
    
    return start, end

def handle_last_day(
    places: PlaceList,
    acc_indices: List[int],
    transport_indices: List[int],
    start_index: int = 0,
    end_index: int = -1
) -> Tuple[Optional[int], int]:
    if len(acc_indices) > 1:
        raise ValueError("여행 마지막날에 accommodation이 2개 이상 있습니다.")
    if len(transport_indices) != 1:
        raise ValueError("여행 마지막날에 transport 장소가 1개여야 합니다.")

    start = acc_indices[start_index] if acc_indices else None
    if start is not None:
        validate_place_category(places[start], "accommodation", "여행 마지막날 시작 장소는 accommodation여야 합니다.")

    end = transport_indices[0]
    validate_place_category(places[end], "transport", "여행 마지막날 종료 장소는 transport여야 합니다.")

    return start, end

def handle_mid_day(
    places: PlaceList,
    acc_indices: List[int],
    _transport_indices: List[int],
    start_index: int = 0,
    end_index: int = 1
) -> Tuple[Optional[int], Optional[int]]:
    count = len(acc_indices)
    if count > 2:
        raise ValueError("중간여행일에 accommodation이 3개 이상입니다.")

    if count == 2:
        start, end = acc_indices  
    elif count == 1:
        idx = acc_indices[0]
        start, end = idx, None
    else:  
        start, end = None, None

    if start is not None:
        validate_place_category(places[start], "accommodation", "중간날 시작 장소는 accommodation이어야 합니다.")
    if end is not None:
        validate_place_category(places[end],   "accommodation", "중간날 종료 장소는 accommodation이어야 합니다.")

    return start, end

def handle_one_day_trip(
    places: PlaceList,
    _acc_indices: List[int],
    transport_indices: List[int],
    start_index: int = 0,
    end_index: int = -1
) -> Tuple[int, int]:
    if len(transport_indices) != 2:
        raise ValueError("당일치기 여행에는 transport 장소가 정확히 2개 있어야 합니다.")

    start = transport_indices[start_index]
    end = transport_indices[end_index]

    validate_place_category(places[start], "transport", "당일치기 여행 시작 장소는 transport여야 합니다.")
    validate_place_category(places[end],   "transport", "당일치기 여행 종료 장소는 transport여야 합니다.")

    return start, end

handlers: Dict[Tuple[bool, bool], Handler] = {
    (True,  False): handle_first_day,
    (True,  True):  handle_one_day_trip,
    (False, False): handle_mid_day,
    (False, True):  handle_last_day,
}

def determine_start_end_indices(
    places: PlaceList,
    day_info: DayInfo,
    start_index: int = 0,
    end_index: int = -1
) -> Tuple[Optional[int], Optional[int]]:
    """
    day_info에 따라 시작/종료 노드를 결정합니다.
    - is_first_day / is_last_day 조합에 따라 핸들러 결정
    - 핸들러 내부에서 accommodation, transport 인덱스를 검사
    """
    acc_indices = get_indices_by_category(places, "accommodation")
    transport_indices = get_indices_by_category(places, "transport")

    key = (day_info.get("is_first_day", False), day_info.get("is_last_day", False))
    handler = handlers.get(key)

    if not handler:
        raise ValueError(f"유효하지 않은 day_info 조합: {key}")

    try:
        return handler(places, acc_indices, transport_indices, start_index, end_index)
    except Exception as e:
        raise ValueError(f"시작/종료 노드 결정 실패: {e}") from e
