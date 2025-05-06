import argparse
import logging
import json

from tripscheduler.cli.controller import execute_full_pipeline
from tripscheduler.cli.utils import display_results
from tests.utils.run_all_test_cases_in import run_all_test_cases_in

def main():
    parser = argparse.ArgumentParser(description="Trip Scheduler CLI")
    parser.add_argument(
        "json_path",
        nargs="?",
        default='./tests/data/tc.json',
        help="테스트 케이스 JSON 파일 경로"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="모의 데이터 사용"
    )
    parser.add_argument(
        "--mock-raw",
        default="./tests/data/directions_raw_data.json",
        help="모의 raw 응답 파일"
    )

    # 폴더 전체 실행용 옵션 추가
    parser.add_argument(
        "--all",
        action="store_true",
        help="폴더 내 모든 케이스 실행"
    )
    parser.add_argument(
        "--json-dir",
        default="./tests/data/",
        help="전체 실행할 JSON 폴더"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info("Trip Scheduler 시작")

    if args.all:
        # --all 모드: run_all만 호출
        run_all_test_cases_in(
            folder_path=args.json_dir,
            use_mock=args.mock,
            mock_raw_path=args.mock_raw
        )
    else:
        # 개별 테스트 실행
        results, windows = execute_full_pipeline(
            json_path=args.json_path,
            use_mock=args.mock,
            mock_raw_path=args.mock_raw
        )
        display_results(results, windows)

        # 첫 번째(혹은 유일한) 결과만 저장
        first_value = next(iter(results.values()))
        data_to_save = {
            'visits': first_value.get('visits', []),
            'path':   first_value.get('path', [])
        }
        with open("./tests/data/results.json", "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        logger.info("결과를 tests/data/results.json에 저장했습니다")

if __name__ == "__main__":
    main()
