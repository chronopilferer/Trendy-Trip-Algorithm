from TripScheduler.tripscheduler.scheduler_api import schedule_trip
import json

with open("./TripScheduler/tests/data/tc.json", encoding="utf-8") as f:
    data = json.load(f)

"""
유의사항: 
만약 mock을 사용하지 않는다면, .env 파일에 NAVER_API_CLIENT_ID, NAVER_API_CLIENT_SECRET 설정 필요
mock을 사용한다면 api 의 요청에 대한 사전 응답을 json 파일로 저장해야 함
"""

result = schedule_trip(
    data,
    use_mock=True, 
    mock_raw_path="./TripScheduler/tests/data/directions_raw_data.json",
    output_path="./results.json"
    )