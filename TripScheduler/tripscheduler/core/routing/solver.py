from .components import get_default_search_parameters
from .context import RoutingContext

def solve(ctx: RoutingContext, time_limit_sec: int = 10):
    """
    구성된 RoutingContext.routing 모델을 SolveWithParameters로 풉니다.
    """
    params = get_default_search_parameters(time_limit_sec)
    solution = ctx.routing.SolveWithParameters(params)
    return solution
