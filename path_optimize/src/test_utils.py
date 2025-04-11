import os
import json
import pprint
import itertools

from solver.time_windows import calculate_effective_time_windows
from solver.routing_solver import split_restaurant_nodes, run_model


def run_all_test_cases_in(folder_path: str):
    print(f"\n=== [INFO] 폴더 실행 시작: {folder_path} ===\n")

    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            if not file.endswith(".json"):
                continue

            test_case_path = os.path.join(root, file)
            print('=' * 100)
            print(f"[DEBUG] 실행 중인 테스트 케이스: {test_case_path}")
            print('=' * 100)

            with open(test_case_path, encoding='utf-8') as f:
                data = json.load(f)

            places = data.get("places", [])
            user = data.get("user", {})
            day_info = data.get("day_info", {})

            eff_windows = calculate_effective_time_windows(places, user)
            new_places, new_eff_windows = split_restaurant_nodes(places, eff_windows)

            meal_groups = {}
            for i, p in enumerate(new_places):
                if p.get("category") == "restaurant":
                    meal_type = new_eff_windows[i][2]
                    if meal_type is not None:
                        meal_groups.setdefault(meal_type, []).append(i)

            group_indices = list(meal_groups.values())
            selections = list(itertools.product(*group_indices)) if group_indices else [()]

            results = {}
            for sel in selections:
                selected_indices = set(sel)
                for i, p in enumerate(new_places):
                    if p.get("category") != "restaurant":
                        selected_indices.add(i)
                selected_indices = sorted(selected_indices)

                selected_places = [new_places[i] for i in selected_indices]
                selected_eff_windows = [new_eff_windows[i] for i in selected_indices]

                route, obj = run_model(selected_places, selected_eff_windows, day_info, user)
                results[sel] = {"objective": obj, "route": route} if route else None

            pprint.pprint("[DEBUG] 최종 결과:")
            for sel, result in results.items():
                meal_labels = tuple(new_eff_windows[i][2] for i in sel)
                print(f"\nResults for option {meal_labels}:")
                if result:
                    print("Objective value:", result["objective"])
                    for item in result["route"]:
                        print(f"  [{item['order']}] {item['place']}")
                        print(f"     도착: {item['arrival_str']}, 출발: {item['departure_str']}, 체류: {item['stay_duration']}분")
                else:
                    print("  No solution found.")