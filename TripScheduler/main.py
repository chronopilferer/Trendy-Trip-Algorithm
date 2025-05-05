import argparse
import logging

from tripscheduler.cli.controller import execute_full_pipeline
from tripscheduler.cli.utils import display_results
from tests.utils.run_all_test_cases_in import run_all_test_cases_in

def main():
    parser = argparse.ArgumentParser(description="Trip Scheduler CLI")
    parser.add_argument("json_path", nargs="?", default='./tests/data/tc.json', help="테스트 케이스 JSON 파일 경로")
    parser.add_argument("--mock", action="store_true", help="모의 데이터 사용")
    parser.add_argument("--mock-matrix", default="./tests/data/time_matrix.json", help="모의 거리 행렬 파일")
    parser.add_argument("--mock-raw", default="./tests/data/directions_raw_data.json", help="모의 raw 응답 파일")

    # 폴더 전체 실행용 옵션 추가
    parser.add_argument("--all", action="store_true", help="폴더 내 모든 케이스 실행")
    parser.add_argument("--json-dir", default="./tests/data/", help="전체 실행할 JSON 폴더")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info("Trip Scheduler 시작")

    if args.all:
        run_all_test_cases_in(
            folder_path=args.json_dir,
            use_mock=args.mock,
            mock_matrix_path=args.mock_matrix,
            mock_raw_path=args.mock_raw
        )
    else:
        results, windows = execute_full_pipeline(
            json_path=args.json_path,
            use_mock=args.mock,
            mock_matrix_path=args.mock_matrix,
            mock_raw_path=args.mock_raw
        )
        display_results(results, windows)

if __name__ == "__main__":
    # python main.py --all --json-dir ./tests/scenarios/base/day-arrival --mock
    # python main.py ./tests/scenarios/base/tc5_too_many_restaurants.json --mock

    main()