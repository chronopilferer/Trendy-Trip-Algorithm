from tripscheduler.core.routing.dummy import is_dummy_node
from tripscheduler.utils.format import format_visit_info
from tripscheduler.core.routing.context import RoutingContext
import logging

logger = logging.getLogger(__name__)

def append_segment(full_path, segment):
    if not segment:
        return

    last_full = full_path[-1][-1] if full_path else None
    first_seg = segment[0]
    
    if full_path and last_full == first_seg:
        full_path.append(segment[1:])
        logger.debug("중복 첫 점 제거 후 이어붙임: 제거된 점='%s', 추가된 점 수=%d", first_seg, len(segment) - 1)
    else:
        full_path.append(segment)
        logger.debug("이어붙임: 중복 없음 또는 불일치, 추가된 점 수=%d", len(segment))

def calc_travel_info(prev_departure, arrival, travel):
    expected = prev_departure + travel
    gap = arrival - expected
    wait = gap if gap >= 0 else None
    delay = -gap if gap < 0 else None
    return wait, delay

def make_visit(order, node, arrival, stay, travel, wait, delay, place):
    return format_visit_info(
        order=order,
        node=node,
        arrival=arrival,
        stay=stay,
        places={node: place},
        travel_minutes=travel,
        wait_minutes=wait,
        delay_minutes=delay,
        x_cord=place["x_cord"],
        y_cord=place["y_cord"]
    )

def parse_solution(ctx: RoutingContext, solution):
    visits = []
    full_path = []

    idx = ctx.routing.Start(0)
    order = 1
    prev_node = None
    prev_departure = None

    logger.info("솔루션 파싱 시작")

    while not ctx.routing.IsEnd(idx):
        node = ctx.mgr.IndexToNode(idx)
        place = ctx.places[node]
        name = place['name']

        if not is_dummy_node(name):
            arrival = solution.Value(ctx.time_dimension.CumulVar(idx))
            stay = ctx.service_times[node]
            travel = wait = delay = None

            if prev_node is not None:
                travel = ctx.matrix[prev_node][node]
                segment = ctx.path_matrix[prev_node][node] or []
                append_segment(full_path, segment)
                wait, delay = calc_travel_info(prev_departure, arrival, travel)

                logger.debug("이동: %s → %s | 도착=%d, 기대=%d, 대기=%s, 지연=%s",
                    ctx.places[prev_node]['name'], name,
                    arrival, prev_departure + travel,
                    wait if wait is not None else '-', 
                    delay if delay is not None else '-')
            else:
                logger.debug("출발: %s | 도착=%d", name, arrival)

            visits.append(make_visit(order, node, arrival, stay, travel, wait, delay, place))
            order += 1
            prev_node = node
            prev_departure = arrival + stay

        idx = solution.Value(ctx.routing.NextVar(idx))

    # End 노드 처리
    end_node = ctx.mgr.IndexToNode(idx)
    end_place = ctx.places[end_node]
    end_name = end_place['name']

    if not is_dummy_node(end_name):
        arrival = solution.Value(ctx.time_dimension.CumulVar(idx))
        travel = wait = None

        if prev_node is not None:
            travel = ctx.matrix[prev_node][end_node]
            segment = ctx.path_matrix[prev_node][end_node] or []
            append_segment(full_path, segment)
            wait = max(0, arrival - (prev_departure + travel))

            logger.info("종료 이동: %s → %s | 도착=%d, 대기=%d",
                        ctx.places[prev_node]["name"], end_name, arrival, wait)

        visits.append(make_visit(order, end_node, arrival, 0, travel, wait, None, end_place))

    logger.info("솔루션 파싱 완료. 총 %d개 장소", len(visits))
    return visits, full_path
