import json
import logging
from typing import Dict, Any, Optional
from tripscheduler.cli.utils import (
    generate_valid_combinations,
    build_selection_inputs
)
from tripscheduler.core.preprocessing.timewindow import calculate_effective_time_windows
from tripscheduler.core.preprocessing.restaurant import split_restaurant_nodes
from tripscheduler.scheduler import run_scheduler

logger = logging.getLogger(__name__)

def schedule_trip(
    data: Dict[str, Any],
    use_mock: bool = False,
    mock_raw_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    외부에서 사용할 수 있는 파이프라인 wrapper.
    - data: tc.json 형태의 dict
    - output_path: 저장할 파일 경로 (예: "results.json")
    - return: results.json의 dict 형태
    """

    places, user, day_info = data["places"], data["user"], data["day_info"]

    # 1. 시간 윈도우 계산
    eff_windows_map = calculate_effective_time_windows(places, user)

    # 2. 식당 노드 분리
    new_places, new_windows = split_restaurant_nodes(places, eff_windows_map)

    # 3. 유효한 조합 생성
    valid_selections = generate_valid_combinations(new_places, new_windows)

    # 4. 조합별 스케줄링 실행
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

    # 5. 결과 저장 (최고 결과 or 첫 번째 결과)
    first_value = next((v for v in results.values() if v), None)
    output_data = {
        "visits": first_value.get("visits", []) if first_value else [],
        "path":   first_value.get("path", []) if first_value else []
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        logger.info(f"✔ 결과 저장 완료: {output_path}")

    return output_data
