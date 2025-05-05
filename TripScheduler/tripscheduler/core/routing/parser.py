from tripscheduler.core.routing.dummy import is_dummy_node
from tripscheduler.utils.format import format_visit_info
from tripscheduler.core.routing.context import RoutingContext

import logging
logger = logging.getLogger(__name__)

def parse_solution(ctx: RoutingContext, solution):
    """
    OR-Tools 솔루션 객체를 읽어들여 방문 순서 리스트와 전체 궤적(path)을 반환합니다.
    Returns:
      - visits: format_visit_info로 만들어진 리스트
      - full_path: [[lng, lat], …] 형태의 전체 경로 좌표 리스트
    """
    visits = []
    full_path = []
    idx = ctx.routing.Start(0)
    order = 1
    prev_node = None
    prev_departure = None

    logger.info("솔루션 파싱 시작")

    while not ctx.routing.IsEnd(idx):
        node = ctx.mgr.IndexToNode(idx)
        name = ctx.places[node]['name']

        if not is_dummy_node(name):
            arrival = solution.Value(ctx.time_dimension.CumulVar(idx))
            stay = ctx.service_times[node]
            travel = wait = delay = None

            if prev_node is not None:
                travel = ctx.matrix[prev_node][node]
                segment = ctx.raw[prev_node][node]
                path = segment["route"]["traoptimal"][0]["path"]
                if full_path and full_path[-1] == path[0]:
                    full_path.extend(path[1:])
                else:
                    full_path.extend(path)

                expected = prev_departure + travel
                gap = arrival - expected
                if gap >= 0:
                    wait = gap
                else:
                    delay = -gap

                logger.debug(
                    "이동: %s → %s | 도착:%d, 기대:%d, 대기:%s, 지연:%s",
                    ctx.places[prev_node]["name"], name,
                    arrival, expected,
                    wait if wait is not None else "-",
                    delay if delay is not None else "-"
                )
            else:
                logger.debug("출발: %s | 도착:%d", name, arrival)

            visits.append(format_visit_info(
                order, node, arrival, stay,
                ctx.places, travel, wait, delay
            ))

            order += 1
            prev_node = node
            prev_departure = arrival + stay

        idx = solution.Value(ctx.routing.NextVar(idx))

    node = ctx.mgr.IndexToNode(idx)
    name = ctx.places[node]['name']
    if not is_dummy_node(name):
        arrival = solution.Value(ctx.time_dimension.CumulVar(idx))
        if prev_node is not None:
            travel = ctx.matrix[prev_node][node]
            segment = ctx.raw[prev_node][node]
            path = segment["route"]["traoptimal"][0]["path"]
            if full_path and full_path[-1] == path[0]:
                full_path.extend(path[1:])
            else:
                full_path.extend(path)

            wait = max(0, arrival - (prev_departure + travel))
            logger.debug(
                "종료 이동: %s → %s | 도착:%d, 대기:%s",
                ctx.places[prev_node]["name"], name, arrival, wait
            )
        visits.append(format_visit_info(
            order, node, arrival, 0,
            ctx.places, travel, wait, None
        ))

    logger.info("솔루션 파싱 완료. 총 %d개 장소", len(visits))
    return visits, full_path
