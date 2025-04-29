from typing import List, Optional, Tuple

from utils.types import START_INDEX, END_INDEX, Handler

def split_restaurant_nodes(places, windows_map):
    new_places, new_wins = [], []
    for place in places:
        pid = place["id"]
        wins = windows_map.get(pid)
        if wins is None:
            raise ValueError(f"장소 {pid}의 유효 시간 윈도우가 없습니다.")
        if place.get("category") == "restaurant" and len(wins) > 1:
            for o, c, meal in wins:
                node = {**place}
                label = meal or "default"
                node.update({
                    "name": f"{place['name']} ({label})",
                    "id": f"{pid}_{label}",
                    "org_id": pid
                })
                new_places.append(node)
                new_wins.append((o, c, meal))
        else:
            new_places.append(place)
            new_wins.append(wins[0] if wins else (None, None, None))
    return new_places, new_wins

def get_indices_by_category(places: List[dict], category: str) -> List[int]:
    """ 카테고리에 해당하는 장소의 인덱스 반환 """
    return [i for i, p in enumerate(places) if p.get("category") == category]

def validate_place_category(place: dict, expected_category: str, err_msg: str) -> None:
    """ 카테고리 적합 검사 """
    if place.get("category") != expected_category:
        raise ValueError(err_msg)

def handle_first_day(
    places: List[dict],
    acc_indices: List[int],
    transport_indices: List[int]
) -> Tuple[int, Optional[int]]:
    """ 
    첫날의 경우 시작 노드는 반드시 transport 카테고리, 
    마지막 노드의 경우 숙소 카테고리 만약 없다면 None 반환 
    """
    if not transport_indices:
        raise ValueError("여행 첫날에 transport 장소가 없습니다.")
    start = transport_indices[START_INDEX]
    validate_place_category(places[start], "transport", "여행 첫날 시작 장소는 transport여야 합니다.")
    
    if len(acc_indices) > 1:
        raise ValueError("여행 첫날에 accommodation이 2개 이상 있습니다.")
    end = acc_indices[START_INDEX] if acc_indices else None
    
    return start, end

def handle_one_day_trip(
    places: List[dict],
    _acc_indices: List[int],
    transport_indices: List[int]
) -> Tuple[int, int]:
    """
    당일치기 여행의 경우 시작·종료 노드는 반드시 transport 카테고리여야 하며,
    transport 노드가 정확히 2개여야 한다.
    """
    if len(transport_indices) != 2:
        raise ValueError("당일치기 여행에는 transport 장소가 정확히 2개 있어야 합니다.")

    start = transport_indices[START_INDEX]
    end = transport_indices[END_INDEX]

    validate_place_category(places[start], "transport", "당일치기 여행 시작 장소는 transport여야 합니다.")
    validate_place_category(places[end],   "transport", "당일치기 여행 종료 장소는 transport여야 합니다.")

    return start, end

def handle_last_day(
    places: List[dict],
    acc_indices: List[int],
    transport_indices: List[int]
) -> Tuple[Optional[int], int]:
    """
    마지막 날일 경우 시작 노드의 경우 숙소 카테고리 만약 없다면 None 반환 
    종료 노드는 transport 카테고리가 정확히 1개여야 하며, 해당 노드를 종료 노드로 설정
    """
    if len(acc_indices) > 1:
        raise ValueError("여행 마지막날에 accommodation이 2개 이상 있습니다.")
    if len(transport_indices) != 1:
        raise ValueError("여행 마지막날에 transport 장소가 1개여야 합니다.")

    start = acc_indices[START_INDEX] if acc_indices else None
    if start is not None:
        validate_place_category(places[start], "accommodation", "여행 마지막날 시작 장소는 accommodation여야 합니다.")

    end = transport_indices[0]
    validate_place_category(places[end], "transport", "여행 마지막날 종료 장소는 transport여야 합니다.")

    return start, end

def handle_mid_day(
    places: List[dict],
    acc_indices: List[int],
    _transport_indices: List[int]
) -> Tuple[Optional[int], Optional[int]]:
    """
    중간 날일 경우 accommodation 노드 개수에 따라 분기 
    2개: 순서에 따라 지정, 1개: 노드 위치가 START_INDEX면 start, END_INDEX면 end 0개: (None, None)
    """
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

def determine_start_end_indices(
    places: List[dict], day_info: dict
) -> Tuple[Optional[int], Optional[int]]:
    
    handlers: dict[Tuple[bool, bool], Handler] = {
        (True, False): handle_first_day,
        (True, True): handle_one_day_trip,
        (False, True): handle_last_day,
        (False, False): handle_mid_day,
    }

    acc_indices = get_indices_by_category(places, "accommodation")
    transport_indices = get_indices_by_category(places, "transport")

    key = (day_info.get("is_first_day", False), day_info.get("is_last_day", False))
    handler = handlers.get(key)
    
    if not handler:
        raise ValueError("유효하지 않은 day_info 조합입니다.")
    return handler(places, acc_indices, transport_indices)