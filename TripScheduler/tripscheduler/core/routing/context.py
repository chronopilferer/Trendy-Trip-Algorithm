from dataclasses import dataclass, field
from typing import List, Tuple, Optional
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
    path_matrix: List[List[Optional[List[float]]]] = field(default_factory=list)

    routing: pywrapcp.RoutingModel = field(init=False)
    mgr: pywrapcp.RoutingIndexManager = field(init=False)
    callback_index: int = field(init=False)
    time_dimension: pywrapcp.RoutingDimension = field(init=False)

    def attach_routing_components(
        self,
        routing: pywrapcp.RoutingModel,
        mgr: pywrapcp.RoutingIndexManager,
        callback_index: int,
        time_dimension: pywrapcp.RoutingDimension,
    ):
        self.routing = routing
        self.mgr = mgr
        self.callback_index = callback_index
        self.time_dimension = time_dimension

def build_context(
    places: List[dict],
    windows: List[Tuple[int, int, Optional[str]]],
    matrix: List[List[int]],
    service_times: List[int],
    start_idx: int,
    end_idx: int,
    global_start: int,
    global_end: int,
    routing: pywrapcp.RoutingModel,
    mgr: pywrapcp.RoutingIndexManager,
    callback_index: int,
    time_dimension: pywrapcp.RoutingDimension,
    path_matrix: Optional[List[List[Optional[List[float]]]]] = None,
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
        path_matrix=path_matrix or [],
    )
    ctx.attach_routing_components(routing, mgr, callback_index, time_dimension)
    return ctx
