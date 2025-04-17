import pprint
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from solver.utils.time import time_to_minutes
from solver.utils.format import format_visit_info
from solver.utils.distance import create_distance_matrix
from solver.utils.places import determine_start_end_indices
from solver.utils.routing import is_dummy_node, add_dummy_node

def create_routing_model(n, start_idx, end_idx):
    mgr = pywrapcp.RoutingIndexManager(n, 1, [start_idx], [end_idx])
    return mgr, pywrapcp.RoutingModel(mgr)

def register_transit(routing, mgr, matrix, service_times, places):
    def cb(i, j):
        u = mgr.IndexToNode(i)
        v = mgr.IndexToNode(j)
        if is_dummy_node(places[u]['name']) or is_dummy_node(places[v]['name']):
            return 0
        return matrix[u][v] + service_times[u]
    return routing.RegisterTransitCallback(cb)

def add_time_constraints(routing, cb_idx, gs, ge, wins, mgr, start_idx, end_idx):
    routing.AddDimension(cb_idx, 1000, ge, False, "Time")
    td = routing.GetMutableDimension("Time")
    td.CumulVar(routing.Start(0)).SetRange(gs, gs)
    td.CumulVar(routing.End(0)).SetRange(0, ge)
    for i, (o, c, _) in enumerate(wins):
        if i in (start_idx, end_idx):
            continue
        idx = mgr.NodeToIndex(i)
        lo, hi = max(gs, o-10), min(ge, c+10)
        td.CumulVar(idx).SetRange(lo, hi)
        td.SetCumulVarSoftUpperBound(idx, hi, 10)
    return td

def extract_solution(routing, mgr, sol, svc, td, places, dist_mat):
    res = []
    idx = routing.Start(0)
    order = 1
    prev_node = None
    prev_departure = None

    while not routing.IsEnd(idx):
        node = mgr.IndexToNode(idx)
        name = places[node]['name']
        if not is_dummy_node(name):
            arrival = sol.Value(td.CumulVar(idx))
            stay = svc[node]
            travel_minutes = None
            wait_minutes = None
            delay_minutes = None

            if prev_node is not None:
                travel_minutes = dist_mat[prev_node][node]
                expected_arrival = prev_departure + travel_minutes
                gap = arrival - expected_arrival

                if gap >= 0:
                    wait_minutes = gap
                    delay_minutes = None
                else:
                    wait_minutes = None
                    delay_minutes = -gap

            res.append(format_visit_info(order, node, arrival, stay, places, travel_minutes, wait_minutes, delay_minutes))
            order += 1
            prev_node = node
            prev_departure = arrival + stay
        idx = sol.Value(routing.NextVar(idx))

    node = mgr.IndexToNode(idx)
    name = places[node]['name']
    if not is_dummy_node(name):
        arrival = sol.Value(td.CumulVar(idx))
        travel_minutes = dist_mat[prev_node][node] if prev_node is not None else None
        wait_minutes = max(0, arrival - (prev_departure + travel_minutes)) if travel_minutes is not None else None
        res.append(format_visit_info(order, node, arrival, 0, places, travel_minutes, wait_minutes))

    return res, sol.ObjectiveValue()

def run_model(places, eff_wins, day_info, user):

    # 1. 시작 노드 종료 노드 결정
    start_idx, end_idx = determine_start_end_indices(places, day_info)
    
    # 2. 유효 시간 윈도우 결정 gs:global start, ge:global end
    gs, ge = time_to_minutes(user['start_time']), time_to_minutes(user['end_time'])

    # 3. 더미 노드 추가 (종료, 시작 노드가 없을 시 경로계산이 되도록 설정)
    if end_idx is None: 
        print("[DEBUG] 종료 노드가 없습니다. 더미 노드를 추가합니다.")
        end_idx = add_dummy_node(places, eff_wins, 'end', gs, ge)
    if start_idx is None:
        print("[DEBUG] 시작 노드가 없습니다. 더미 노드를 추가합니다.")
        start_idx = add_dummy_node(places, eff_wins, 'start', gs, ge)

    # 4. 거리 행렬 생성 (이후 실제 이동시간으로 대체)
    dist_mat = create_distance_matrix(places)
    
    # 5. 서비스 시간 설정 
    svc_times = [p.get('service_time', 0) for p in places]
    
    # 6. 라우팅 모델 생성 mgr: manager, routing: routing model
    n = len(places)
    mgr, routing = create_routing_model(n, start_idx, end_idx)
    
    # 7. 라우팅 모델에 거리 행렬 설정
    cb_idx = register_transit(routing, mgr, dist_mat, svc_times, places)
    
    # 8. 서비스 시간 설정
    routing.SetArcCostEvaluatorOfAllVehicles(cb_idx)
    
    # 9. 필수 장소 설정 (필수 장소에 대한 패널티 설정)
    for i, p in enumerate(places):
        if i not in (start_idx, end_idx) and not p.get('is_mandatory', True):
            routing.AddDisjunction([mgr.NodeToIndex(i)], 1000)

    # 10. 시간 제약 조건 설정 (유효 시간 윈도우)
    td = add_time_constraints(routing, cb_idx, gs, ge, eff_wins, mgr, start_idx, end_idx)

    # 11. 라우팅 모델에 대한 파라미터 설정
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    params.time_limit.seconds = 10
    
    # 12. 라우팅 모델 실행 
    sol = routing.SolveWithParameters(params)

    if sol:
        return extract_solution(routing, mgr, sol, svc_times, td, places, dist_mat)
    pprint.pprint("[DEBUG] 솔루션을 찾지 못했습니다.")
    return None, None