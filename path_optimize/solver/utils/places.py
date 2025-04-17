
def validate_first_place(places, expected_category, err_msg):
    if not places:
        raise ValueError("places 리스트가 비어있습니다.")
    
    first_cat = places[0].get("category")
    if first_cat != expected_category:
        raise ValueError(err_msg)

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
                    "id": f"{pid}_{label}"
                })
                new_places.append(node)
                new_wins.append((o, c, meal))
        else:
            new_places.append(place)
            new_wins.append(wins[0] if wins else (None, None, None))
    return new_places, new_wins

def is_accommodation(place):
    return place.get("category") == "accommodation"

def is_transport(place):
    return place.get("category") == "transport"

def validate_non_empty(places):
    if not places:
        raise ValueError("places 리스트가 비어있습니다.")

def get_indices_by_category(places, category_check):
    return [idx for idx, p in enumerate(places) if category_check(p)]

def handle_first_day(places, accommodation_indices):
    # 시작은 무조건 transport
    validate_first_place(places, "transport", "여행 첫날 시작 장소는 transport여야 합니다.")
    start_index = 0

    # accommodation이 하나만 있어야 종료 인덱스로
    end_index = None
    if accommodation_indices:
        if len(accommodation_indices) > 1:
            raise ValueError("여행 첫날에 accommodation이 2개 이상 있습니다.")
        end_index = accommodation_indices[0]

    return start_index, end_index

def handle_one_day_trip(places, transport_indices):
    # 당일치기: 시작은 transport, 종료도 transport(시작 제외)
    validate_first_place(places, "transport", "당일치기 여행 시작 장소는 transport여야 합니다.")
    start_index = 0

    other_transports = [i for i in transport_indices if i != 0]
    if len(other_transports) != 1:
        raise ValueError("당일치기 여행에는 시작 장소를 제외한 transport가 1개여야 합니다.")
    end_index = other_transports[0]

    return start_index, end_index

def handle_last_day(places, transport_indices):
    # 시작은 0번이 accommodation이면 그걸, 아니면 None
    start_index = 0 if is_accommodation(places[0]) else None

    # 종료는 무조건 transport가 하나
    if not transport_indices:
        raise ValueError("여행 마지막날에 transport 장소가 없습니다.")
    if len(transport_indices) > 1:
        raise ValueError("여행 마지막날에 transport 장소가 2개 이상 있습니다.")
    end_index = transport_indices[0]

    return start_index, end_index

def handle_mid_day(places, accommodation_indices):
    # 시작은 0번이 accommodation이면, 종료는 마지막이 accommodation이면
    start_index = 0 if is_accommodation(places[0]) else None
    end_index   = (len(places) - 1) if is_accommodation(places[-1]) else None

    # 장소가 하나밖에 없으면 시작만 0으로
    if len(places) == 1:
        start_index = 0

    # accommodation이 너무 많으면 에러
    if len(accommodation_indices) > 2:
        raise ValueError("중간여행일에 accommodation이 3개 이상입니다.")

    return start_index, end_index

def determine_start_end_indices(places, day_info):
    # 1. 입력 검사 및 인덱스 추출
    validate_non_empty(places)
    accommodation_indices = get_indices_by_category(places, is_accommodation)
    transport_indices     = get_indices_by_category(places, is_transport)

    is_first = day_info.get("is_first_day", False)
    is_last  = day_info.get("is_last_day", False)

    # 2. 각 케이스별 핸들러 호출
    if is_first and not is_last:
        return handle_first_day(places, accommodation_indices)
    if is_first and is_last:
        return handle_one_day_trip(places, transport_indices)
    if not is_first and is_last:
        return handle_last_day(places, transport_indices)
    # 그 외 중간 여행일
    return handle_mid_day(places, accommodation_indices)