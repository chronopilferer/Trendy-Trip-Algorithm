import os
import json
import pprint
import itertools
from tabulate import tabulate

from solver.utils.time_windows import calculate_effective_time_windows
from solver.routing_solver import run_model
from solver.utils.places import split_restaurant_nodes

def run_all_test_cases_in(folder_path: str):
    print(f"\n=== [INFO] 폴더 실행 시작: {folder_path} ===\n")

    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            if not file.endswith(".json") or file == 'tc9_no_match_meal_time.json':
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
                print("\n" + "="*80)
                print(f"Option {meal_labels!r}")
                print("-"*80)

                if not result:
                    print("  (해결 불가)\n")
                    continue

                print(f"  Objective value: {result['objective']}\n")

                table_data = []

                for item in result["route"]:
                    stay   = item.get('stay_duration', '-')
                    travel = item.get('travel_time', '-')
                    wait   = item.get('wait_time', '-')
                    delay  = item.get('delay_time', '-')

                    table_data.append([
                        f"[{item['order']}]",
                        item['place'],
                        item['arrival_str'],
                        item['departure_str'],
                        stay,
                        travel,
                        wait,
                        delay
                    ])

                headers = ["순서", "장소", "도착", "출발", "체류", "이동시간", "대기시간", "지연시간"]
                print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", stralign="center"))
