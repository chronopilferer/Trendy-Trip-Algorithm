from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from tripscheduler.core.routing.dummy import is_dummy_node

def create_routing_model(n: int, start_idx: int, end_idx: int):
    """Index Manager와 RoutingModel 객체를 생성해 반환"""
    mgr = pywrapcp.RoutingIndexManager(n, 1, [start_idx], [end_idx])
    return mgr, pywrapcp.RoutingModel(mgr)

def register_transit(routing, mgr, matrix, service_times, places):
    """
    거리+서비스 시간을 더한 비용 함수를 callback으로 등록.
    더미 노드끼리는 비용 0으로 처리.
    """
    def transit_callback(i, j):
        u = mgr.IndexToNode(i)
        v = mgr.IndexToNode(j)
        if is_dummy_node(places[u]['name']) or is_dummy_node(places[v]['name']):
            return 0
        return matrix[u][v] + service_times[u]
    return routing.RegisterTransitCallback(transit_callback)

def add_disjunctions(routing, mgr, places, start_idx, end_idx, penalty: int = 1000):
    """
    is_mandatory=False인 노드에 대해 Disjunction(방문 안 해도 되지만 penalty 있음) 추가
    """
    for i, p in enumerate(places):
        if i not in (start_idx, end_idx) and not p.get('is_mandatory', True):
            routing.AddDisjunction([mgr.NodeToIndex(i)], penalty)

def add_time_constraints(
    routing, transit_cb_idx, global_start, global_end,
    windows, mgr, start_idx, end_idx
):
    """
    Time Dimension 추가 및 각 노드의 시간 윈도우 설정
    windows: [(open, close, tag), ...]
    """
    routing.AddDimension(transit_cb_idx, 1000, global_end, False, "Time")
    td = routing.GetMutableDimension("Time")

    # 시작/종료 시간 고정
    td.CumulVar(routing.Start(0)).SetRange(global_start, global_start)
    td.CumulVar(routing.End(0)).SetRange(0, global_end)

    for i, (o, c, _) in enumerate(windows):
        if i in (start_idx, end_idx):
            continue
        idx = mgr.NodeToIndex(i)
        lo = max(global_start, o - 10)
        hi = min(global_end, c + 10)
        td.CumulVar(idx).SetRange(lo, hi)
        td.SetCumulVarSoftUpperBound(idx, hi, 10)

    return td

def get_default_search_parameters(time_limit_sec: int = 10):
    """기본 탐색 파라미터(FirstSolution + time limit) 생성"""
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    params.time_limit.seconds = time_limit_sec
    return params
