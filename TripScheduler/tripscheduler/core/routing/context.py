from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
from ortools.constraint_solver import pywrapcp

@dataclass
class RoutingContext:
    places: List[dict]
    windows: List[Tuple[int, int, Optional[str]]]
    matrix: List[List[int]]
    service_times: List[int]
    start_idx: int
    end_idx: int
    global_start: int
    global_end: int
    raw: List[List[dict]]               

    mgr: pywrapcp.RoutingIndexManager   = field(init=False)
    routing: pywrapcp.RoutingModel      = field(init=False)
    callback_index: int                 = field(init=False)
    time_dimension: Any                 = field(init=False)

def build_context(
    places: List[dict],
    windows: List[Tuple[int,int,Optional[str]]],
    matrix: List[List[int]],
    service_times: List[int],
    start_idx: int,
    end_idx: int,
    global_start: int,
    global_end: int,
    routing: pywrapcp.RoutingModel,
    mgr: pywrapcp.RoutingIndexManager,
    callback_index: int,
    time_dimension: Any,
    raw: List[List[dict]],
) -> RoutingContext:
    ctx = RoutingContext(
        places=places,
        windows=windows,
        matrix=matrix,
        service_times=service_times,
        start_idx=start_idx,
        end_idx=end_idx,
        global_start=global_start,
        global_end=global_end,
        raw=raw
    )
    ctx.routing = routing
    ctx.mgr = mgr
    ctx.callback_index = callback_index
    ctx.time_dimension = time_dimension
    return ctx
