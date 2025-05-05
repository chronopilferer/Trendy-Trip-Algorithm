from .components import (
    create_routing_model,
    register_transit,
    add_disjunctions,
    add_time_constraints
)
from .context import RoutingContext

def build_model(ctx: RoutingContext) -> None:
    """
    RoutingContext를 받아 OR-Tools 모델을 구성합니다.
    (mgr, routing, callback_index, time_dimension 필드를 채움)
    """
    ctx.mgr, ctx.routing = create_routing_model(
        len(ctx.places), ctx.start_idx, ctx.end_idx
    )
    ctx.callback_index = register_transit(
        ctx.routing, ctx.mgr,
        ctx.matrix, ctx.service_times, ctx.places
    )
    add_disjunctions(
        ctx.routing, ctx.mgr,
        ctx.places, ctx.start_idx, ctx.end_idx
    )
    ctx.time_dimension = add_time_constraints(
        ctx.routing, ctx.callback_index,
        ctx.global_start, ctx.global_end,
        ctx.windows, ctx.mgr,
        ctx.start_idx, ctx.end_idx
    )
