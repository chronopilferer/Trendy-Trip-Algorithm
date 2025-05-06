import logging
from tripscheduler.cli.utils import (
    load_test_case,
    generate_valid_combinations,
    build_selection_inputs
)
from tripscheduler.core.preprocessing.timewindow import calculate_effective_time_windows
from tripscheduler.core.preprocessing.restaurant import split_restaurant_nodes
from tripscheduler.scheduler import run_scheduler

logger = logging.getLogger(__name__)

def execute_full_pipeline(
    json_path: str,
    use_mock: bool = False,
    mock_raw_path: str = None
):
    """
    1) JSON 로드
    2) 시간 윈도우 계산
    3) 식당 분할
    4) 조합 생성
    5) 조합별 run_scheduler 실행
    6) 결과(result dict)와 windows 리스트 반환
    """
    data = load_test_case(json_path)
    places, user, day_info = data["places"], data["user"], data["day_info"]

    eff_windows_map = calculate_effective_time_windows(places, user)
    new_places, new_windows = split_restaurant_nodes(places, eff_windows_map)

    valid_selections = generate_valid_combinations(new_places, new_windows)

    results = {}
    for sel in valid_selections:
        sel_places, sel_windows, labels = build_selection_inputs(new_places, new_windows, sel)

        logger.info("▶ 조합 %s 실행", labels)
        try:
            visits, cost, full_path = run_scheduler(
                sel_places,
                sel_windows,
                user,
                day_info,
                use_mock=use_mock,
                mock_raw_path=mock_raw_path
            )
            results[sel] = {
                "cost": cost,
                "visits": visits,
                "path": full_path
            }
            logger.info("✔ 조합 %s 완료 (cost=%s)", labels, cost)
        except Exception as e:
            results[sel] = None
            logger.error("✖ 조합 %s 실패: %s", labels, e)

    return results, new_windows
