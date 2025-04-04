import json
import pprint
import itertools

from utils import determine_start_end_indices
from time_windows import calculate_effective_time_windows
from routing_solver import create_distance_matrix, split_restaurant_nodes, run_model

if __name__ == '__main__':

    test_cases = ['tc1_base_case_first_day.json', 'tc2_base_case_middle_day.json', 'tc3_base_case_day_trip.json', 'tc4_base_case_last_day.json']

    for test_case in test_cases:

        print('=' * 100)
        pprint.pprint(f"[DEBUG] 메인: 테스트 케이스 {test_case} 실행")
        print('=' * 100)

        with open(test_case, encoding='utf-8') as f:
            data = json.load(f)
        
        places = data.get("places", [])
        user = data.get("user", {})
        day_info = data.get("day_info", {})

        # 시작/종료 인덱스 결정
        start_index, end_index = determine_start_end_indices(places, day_info)
        
        # 시간 윈도우 계산
        eff_windows = calculate_effective_time_windows(places, user)
        
        # 식당 노드 분리
        new_places, new_eff_windows = split_restaurant_nodes(places, eff_windows)

        # 거리 행렬 생성 이후 api를 통한 실제 소요시간으로 대체
        distance_matrix = create_distance_matrix(new_places)
    
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

        exit()