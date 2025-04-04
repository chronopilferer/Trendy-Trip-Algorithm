import pprint
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from utils import haversine_distance, time_to_minutes, minutes_to_time_str, determine_start_end_indices

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

        pprint.pprint(f"[DEBUG] split_restaurant_nodes: 장소 {place['name']}의 유효 시간 윈도우")
        pprint.pprint(windows)
        
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
    print("[DEBUG] split_restaurant_nodes: 분리된 유효 시간 윈도우")
    pprint.pprint(new_effective_windows)
    
    return new_places, new_effective_windows

def run_model(selected_places, selected_eff_windows, day_info, user):
    local_distance_matrix = create_distance_matrix(selected_places)
    local_service_times = [p.get("service_time", 0) for p in selected_places]
    num_locations = len(selected_places)
    start_idx, end_idx = determine_start_end_indices(selected_places, day_info)
    
    local_manager = pywrapcp.RoutingIndexManager(num_locations, 1, [start_idx], [end_idx])
    local_routing = pywrapcp.RoutingModel(local_manager)
    
    def local_transit_callback(i, j):
        from_node = local_manager.IndexToNode(i)
        to_node = local_manager.IndexToNode(j)
        return local_distance_matrix[from_node][to_node] + local_service_times[from_node]
    
    transit_cb_idx = local_routing.RegisterTransitCallback(local_transit_callback)
    local_routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)
    
    global_start_val = time_to_minutes(user["start_time"])
    global_end_val = time_to_minutes(user["end_time"])
    time_dim_name = "Time"
    local_routing.AddDimension(
        transit_cb_idx, 1000, global_end_val, False, time_dim_name
    )
    time_dimension = local_routing.GetMutableDimension(time_dim_name)
    
    # 시작 노드 시간 고정
    start_internal = local_routing.Start(0)
    time_dimension.CumulVar(start_internal).SetRange(global_start_val, global_start_val)
    
    # 각 노드별 시간 윈도우 설정
    for i, (abs_open, abs_close, _) in enumerate(selected_eff_windows):
        if i == start_idx or i == end_idx:
            continue
        idx = local_manager.NodeToIndex(i)
        relaxed_open = max(global_start_val, abs_open - 10)
        relaxed_close = min(global_end_val, abs_close + 10)
        time_dimension.CumulVar(idx).SetRange(relaxed_open, relaxed_close)
        time_dimension.SetCumulVarSoftUpperBound(idx, relaxed_close, 10)
    
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.time_limit.seconds = 10
    
    pprint.pprint("[DEBUG] run_model: OR-Tools 솔버 실행 전 설정")
    sol = local_routing.SolveWithParameters(search_params)
    if sol:
        route = []
        idx = local_routing.Start(0)
        order = 1
        while not local_routing.IsEnd(idx):
            node = local_manager.IndexToNode(idx)
            arrival = sol.Value(time_dimension.CumulVar(idx))
            service = local_service_times[node]
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
            idx = sol.Value(local_routing.NextVar(idx))
        node = local_manager.IndexToNode(idx)
        arrival = sol.Value(time_dimension.CumulVar(idx))
        service = local_service_times[node]
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
        # pprint.pprint("[DEBUG] run_model: 솔루션 발견, 경로 및 목적 함수 값")
        # pprint.pprint(route)
        # pprint.pprint(f"[DEBUG] Objective Value: {sol.ObjectiveValue()}")
        return route, sol.ObjectiveValue()
    else:
        pprint.pprint("[DEBUG] run_model: 솔루션을 찾지 못함")
        return None, None