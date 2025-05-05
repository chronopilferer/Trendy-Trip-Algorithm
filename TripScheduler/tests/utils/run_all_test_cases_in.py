import os
from tripscheduler.cli.controller import execute_full_pipeline
from tripscheduler.cli.utils import display_results

def run_all_test_cases_in(folder_path: str, use_mock: bool, mock_matrix_path: str, mock_raw_path: str):
    print(f"\n=== [INFO] 폴더 실행 시작: {folder_path} ===\n")

    found = False  # <-- 추가

    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            print(f"🕵️ 파일 발견: {file}")  # <-- 추가
            if not file.endswith(".json") or file.startswith("~") or file.startswith("."):
                print(f"⚠️ 무시된 파일: {file}")  # <-- 추가
                continue

            found = True  # <-- 파일 하나라도 처리 대상이면 True

            json_path = os.path.join(root, file)
            print(f"\n🔹 테스트 케이스: {json_path}")

            try:
                results, windows = execute_full_pipeline(
                    json_path=json_path,
                    use_mock=use_mock,
                    mock_matrix_path=mock_matrix_path,
                    mock_raw_path=mock_raw_path
                )
                display_results(results, windows)
            except Exception as e:
                print(f"  ❌ 실행 실패: {e}")

    if not found:
        print(f"\n❗ 처리 가능한 JSON 파일이 없음: {folder_path}")
