import json
import pprint
import itertools

from time_windows import calculate_effective_time_windows
from routing_solver import split_restaurant_nodes, run_model

if __name__ == '__main__':

    """
    테스트 케이스 
    1. tc1_base_case_first_day.json: 첫날의 기본 케이스 (완료)
    2. tc2_base_case_middle_day.json: 중간날의 기본 케이스 (완료)
    3. tc3_base_case_day_trip.json: 당일치기 기본 케이스 (완료)
    4. tc4_base_case_last_day.json: 마지막 날의 기본 케이스 (완료)
    5. tc5_too_many_restaurants.json: 식당이 너무 많은 경우 
        - 개수를 제한 (완료)
        - 개수를 제한하지 않음 (진행중)
    6. tc6_no_restaurant.json: 식당이 없는 경우 (완료)
    7. tc7_no_places.json: 장소가 없는 경우 (완료)
    8. tc8_no_accommodation.json: 숙소가 없는 경우 (완료)
    9. tc9_no_match_meal_time.json: 선호 식사 시간에 포함되지 않는 장소가 있는 경우 (완료)
        - 오류 발생시킴 (완료)
    10. tc10_too_many_places.json: 장소가 너무 많은 경우
        - 오류 발생시킴 (완료)
    """

    test_cases = [
        'tc1_base_case_first_day.json', 
        'tc2_base_case_middle_day.json', 
        'tc3_base_case_day_trip.json', 
        'tc4_base_case_last_day.json',
        'tc5_too_many_restaurants.json',
        'tc6_no_restaurant.json',
        # 'tc7_no_places.json',
        'tc8_no_accommodation.json',
        'tc9_no_match_meal_time.json',
        # 'tc10_too_many_places.json'
        ]

    for test_case in test_cases:

        print('=' * 100)
        pprint.pprint(f"[DEBUG] 메인: 테스트 케이스 {test_case} 실행")
        print('=' * 100)

        with open(test_case, encoding='utf-8') as f:
            data = json.load(f)
        
        places = data.get("places", [])
        user = data.get("user", {})
        day_info = data.get("day_info", {})

        # 시간 윈도우 계산
        eff_windows = calculate_effective_time_windows(places, user)
        
        # 식당 노드 분리
        new_places, new_eff_windows = split_restaurant_nodes(places, eff_windows)

        # 끼니별로 그룹화
        meal_groups = {}

        for i, p in enumerate(new_places):
            if p.get("category") == "restaurant":
                meal_type = new_eff_windows[i][2]  
                if meal_type is not None:
                    meal_groups.setdefault(meal_type, []).append(i)

        pprint.pprint("[DEBUG] 메인: 끼니 기준 그룹화 결과")
        pprint.pprint(meal_groups)

        # 조합 생성
        group_indices = list(meal_groups.values())
        if group_indices:
            selections = list(itertools.product(*group_indices))
        else:
            selections = [()]
        
        pprint.pprint("[DEBUG] 메인: 끼니 기준 생성된 조합 (selections)")
        pprint.pprint(selections)

        results = {}
        for sel in selections:
            selected_indices = set(sel)
            # 식당 외의 노드 추가
            for i, p in enumerate(new_places):
                if p.get("category") != "restaurant":
                    selected_indices.add(i)
            selected_indices = sorted(selected_indices)
            pprint.pprint(f"[DEBUG] 메인: 현재 선택 조합 - {selected_indices}")
            selected_places = [new_places[i] for i in selected_indices]
            selected_eff_windows = [new_eff_windows[i] for i in selected_indices]
            
            route, obj = run_model(selected_places, selected_eff_windows, day_info, user)
            if route is not None:
                results[sel] = {"objective": obj, "route": route}
            else:
                results[sel] = None
        
        # 최종 결과 출력
        pprint.pprint("[DEBUG] 메인: 최종 결과 출력")
        for sel, result in results.items():
            meal_labels = tuple(new_eff_windows[i][2] for i in sel)
            print(f"\nResults for option {meal_labels}:")
            if result:
                print("Objective value:", result["objective"])
                print("Detailed Schedule:")
                for item in result["route"]:
                    print(f"  [{item['order']}] {item['place']}")
                    print(f"     도착: {item['arrival_str']}, 출발: {item['departure_str']}, 체류: {item['stay_duration']}분")
            else:
                print("  No solution found.")