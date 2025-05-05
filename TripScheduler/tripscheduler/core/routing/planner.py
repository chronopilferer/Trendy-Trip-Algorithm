from .context import RoutingContext
from .builder import build_model
from .solver import solve
from .parser import parse_solution

def plan_route(
    places, windows, matrix, service_times,
    start_idx, end_idx, global_start, global_end,
    time_limit_sec: int = 10
):
    """
    1) RoutingContext 생성
    2) build_model → solve → parse_solution 순으로 최적 경로를 계산
    3) (visits, objective) 튜플을 반환
    """
    ctx = RoutingContext(
        places=places,
        windows=windows,
        matrix=matrix,
        service_times=service_times,
        start_idx=start_idx,
        end_idx=end_idx,
        global_start=global_start,
        global_end=global_end
    )

    build_model(ctx)
    sol = solve(ctx, time_limit_sec)
    visits = parse_solution(ctx, sol)
    objective = sol.ObjectiveValue() if sol else None
    return visits, objective
