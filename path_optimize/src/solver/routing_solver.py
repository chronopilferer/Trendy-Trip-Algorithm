import pprint
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from solver.utils import haversine_distance, time_to_minutes, minutes_to_time_str, determine_start_end_indices

import pprint

def create_distance_matrix(places):
    n = len(places)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = places[i]["x_cord"], places[i]["y_cord"]
            lat2, lon2 = places[j]["x_cord"], places[j]["y_cord"]
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            matrix[i][j] = distance
            matrix[j][i] = distance
    return matrix


def split_restaurant_nodes(places, effective_windows):
    new_places = []
    new_effective_windows = []
    
    for place in places:
        place_id = place["id"]

        if place_id not in effective_windows:
            raise ValueError(f"장소 {place_id}의 유효 시간 윈도우가 없습니다.")

        windows = effective_windows.get(place_id, [])

        print()
        pprint.pprint(f"[DEBUG] split_restaurant_nodes: 장소 {place['name']}의 유효 시간 윈도우")
        pprint.pprint(windows)
        print()
        
        # 카테고리가 'restaurant'이고, 윈도우가 여러 개인 경우 개별 노드로 분리.
        if place.get("category") == "restaurant" and len(windows) > 1:
            for win in windows:
                # 원본 장소 정보를 복사하여 새로운 노드를 생성.
                new_node = place.copy()
                # 윈도우의 meal 정보가 None이면 "default"로 처리
                meal_label = win[2] if win[2] is not None else "default"
                # 장소 이름과 id에 meal 정보를 추가해 구분
                new_node["name"] = f"{place['name']} ({meal_label})"
                new_node["id"] = f"{place_id}_{meal_label}"
                new_places.append(new_node)
                new_effective_windows.append(win)
        else:
            new_places.append(place)
            # 윈도우 리스트가 비어있지 않다면 첫 번째 윈도우 사용, 그렇지 않으면 None 처리
            new_effective_windows.append(windows[0] if windows else None)
    
    # 디버그: 분리된 유효 시간 윈도우 출력
    # print()
    # print("[DEBUG] split_restaurant_nodes: 분리된 유효 시간 윈도우")
    # pprint.pprint(new_effective_windows)
    # print()
    
    return new_places, new_effective_windows


def add_optional_disjunctions(routing, manager, selected_places, start_idx, end_idx, dummy_name="dummy", penalty=1000):
    """
    각 장소의 is_mandatory 값을 확인하여 선택적(필수 아님) 장소에 대해 disjunction 제약을 추가합니다.
    start_idx, end_idx, 그리고 더미 노드(이름이 "dummy")는 제외합니다.
    penalty 값은 해당 노드를 방문하지 않을 때 발생하는 비용입니다.
    """
    for i, place in enumerate(selected_places):
        # 이미 시작 노드, 종료 노드, 또는 더미 노드라면 스킵
        if i == start_idx or i == end_idx or place.get("name") == dummy_name:
            continue
        if not place.get("is_mandatory", True):
            idx = manager.NodeToIndex(i)
            routing.AddDisjunction([idx], penalty)


def add_dummy_end_node(selected_places, selected_eff_windows, global_start_val, global_end_val):
    """
    종료 노드가 없는 경우 더미 종료 노드를 추가
    """
    dummy_node = {
        "name": "dummy",
        "category": "dummy",
        "service_time": 0,
        "x_cord": 0.0,
        "y_cord": 0.0
    }
    selected_places.append(dummy_node)

    dummy_eff_window = (global_start_val, global_end_val, None)
    selected_eff_windows.append(dummy_eff_window)

    return len(selected_places) - 1  


def adjust_distance_matrix_for_end(distance_matrix, num_locations, end_idx):
    """
    종료 노드를 위한 거리 행렬 조정 함수.
    
    각 노드와의 거리를 0으로 설정해서 종료 노드와의 거리를 무시하도록 함.
    """
    for i in range(num_locations):
        distance_matrix[i][end_idx] = 0
        distance_matrix[end_idx][i] = 0


def create_routing(selected_places, start_idx, end_idx):
    """
    OR-Tools의 RoutingIndexManager와 RoutingModel을 생성
    """
    num_locations = len(selected_places)
    # 1명, 시작, 종료 노드 설정
    manager = pywrapcp.RoutingIndexManager(num_locations, 1, [start_idx], [end_idx])
    routing = pywrapcp.RoutingModel(manager)
    return manager, routing


def register_transit_callback(routing, manager, distance_matrix, service_times):
    """
    두 노드 사이 이동시 소요 시간(거리+서비스 시간)을 계산하는 콜백 등록
    """
    def transit_callback(i, j):
        from_node = manager.IndexToNode(i)
        to_node = manager.IndexToNode(j)
        return distance_matrix[from_node][to_node] + service_times[from_node]
    return routing.RegisterTransitCallback(transit_callback)


def add_time_dimension_constraints(routing, transit_cb_idx, global_start_val, global_end_val,
                                   selected_eff_windows, manager, start_idx, end_idx):
    """
    시간 차원 추가 후 각 노드별로 시간 제약조건 및 소프트 제약조건을 설정
    """
    time_dim_name = "Time"
    routing.AddDimension(
        transit_cb_idx,
        1000,             # 허용 오차 (슬랙)
        global_end_val,   # 최대 시간 제한
        False,            # 시작 노드에서 시간 초기화하지 않음
        time_dim_name
    )
    time_dimension = routing.GetMutableDimension(time_dim_name)
    
    # 시작 노드와 종료 노드의 시간 제약 설정

    # 시작 노드는 사용자가 설정한 시작 시간에 고정
    user_index = 0
    start_internal = routing.Start(user_index)
    time_dimension.CumulVar(start_internal).SetRange(global_start_val, global_start_val)
    
    # 종료 노드는 사용자가 설정한 종료 시간에 느슨하게 고정 (빨리 도착은 가능 하지만 느리게 도착은 제한)
    end_internal = routing.End(user_index)
    time_dimension.CumulVar(end_internal).SetRange(0, global_end_val)
    
    # 각 노드(시작, 종료 노드는 제외)에 대해 시간 창 및 소프트 상한 제약 설정
    for i, (abs_open, abs_close, _) in enumerate(selected_eff_windows):
        if i == start_idx or i == end_idx:
            continue
        idx = manager.NodeToIndex(i)
        relaxed_open = max(global_start_val, abs_open - 10)
        relaxed_close = min(global_end_val, abs_close + 10)
        time_dimension.CumulVar(idx).SetRange(relaxed_open, relaxed_close)
        time_dimension.SetCumulVarSoftUpperBound(idx, relaxed_close, 10)
    
    return time_dimension


def extract_solution(routing, manager, solution, service_times, time_dimension, selected_places):
    """
    최종 솔루션에서 경로와 각 노드의 일정 정보 추출
    """
    route = []
    idx = routing.Start(0)
    order = 1
    
    # 시작 노드부터 종료 노드까지 경로를 순회하며 정보를 추출
    while not routing.IsEnd(idx):
        node = manager.IndexToNode(idx)
        arrival = solution.Value(time_dimension.CumulVar(idx))
        service = service_times[node]
        departure = arrival + service
        route.append({
            "order": order,
            "node": node,
            "place": selected_places[node]["name"],
            "arrival": arrival,
            "arrival_str": minutes_to_time_str(arrival),
            "service_time": service,
            "departure": departure,
            "departure_str": minutes_to_time_str(departure),
            "stay_duration": service
        })
        order += 1
        idx = solution.Value(routing.NextVar(idx))
    
    # 종료 노드 정보 추가
    node = manager.IndexToNode(idx)
    arrival = solution.Value(time_dimension.CumulVar(idx))
    service = service_times[node]
    departure = arrival + service
    route.append({
        "order": order,
        "node": node,
        "place": selected_places[node]["name"],
        "arrival": arrival,
        "arrival_str": minutes_to_time_str(arrival),
        "service_time": service,
        "departure": departure,
        "departure_str": minutes_to_time_str(departure),
        "stay_duration": service
    })
    objective = solution.ObjectiveValue()
    return route, objective


def run_model(selected_places, selected_eff_windows, day_info, user):
    """
    전체 경로 최적화 모델을 수행하는 메인 함수입니다.
    
    1. 시작 및 종료 인덱스와 글로벌 시간값을 설정합니다.
    2. 종료 노드가 없으면 더미 종료 노드를 추가합니다.
    3. 거리 행렬과 서비스 시간 리스트를 생성하고, 더미 노드에 맞게 행렬을 조정합니다.
    4. 종료 노드일 경우 거리 행렬을 조정합니다.
    5. OR-Tools의 라우팅 모델을 설정하고, 전이 콜백 및 시간 차원 제약 조건을 추가합니다.
    6. 솔루션을 탐색한 후, 경로와 목적 함수 값을 반환합니다.
    """

    # 1. 시작 및 종료 인덱스와 시간 설정
    start_idx, end_idx = determine_start_end_indices(selected_places, day_info)
    global_start_val = time_to_minutes(user["start_time"])
    global_end_val = time_to_minutes(user["end_time"])
    
    # 2. 종료 인덱스가 존재하지 않으면 더미 종료 노드 추가
    if end_idx is None:
        end_idx = add_dummy_end_node(selected_places, selected_eff_windows, global_start_val, global_end_val)
    
    # 3. 거리 행렬과 서비스 시간 리스트 생성 및 조정
    local_distance_matrix = create_distance_matrix(selected_places)
    local_service_times = [p.get("service_time", 0) for p in selected_places]
    num_locations = len(selected_places)
    
    # 4 종료 노드에 맞게 거리 행렬 조정
    adjust_distance_matrix_for_end(local_distance_matrix, num_locations, end_idx)
    
    # 5. OR-Tools 라우팅 모델 설정
    manager, routing = create_routing(selected_places, start_idx, end_idx)
    transit_cb_idx = register_transit_callback(routing, manager, local_distance_matrix, local_service_times)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)
    
    add_optional_disjunctions(routing, manager, selected_places, start_idx, end_idx, penalty=1000)

    time_dimension = add_time_dimension_constraints(
        routing, transit_cb_idx, global_start_val, global_end_val,
        selected_eff_windows, manager, start_idx, end_idx
    )
    
    # 6. 검색 파라미터 설정 및 솔루션 탐색
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    search_params.time_limit.seconds = 10
    
    solution = routing.SolveWithParameters(search_params)
    if solution:
        route, objective = extract_solution(routing, manager, solution,
                                              local_service_times, time_dimension, selected_places)
        return route, objective
    else:
        pprint.pprint("[DEBUG] run_model: 솔루션을 찾지 못함")
        return None, None