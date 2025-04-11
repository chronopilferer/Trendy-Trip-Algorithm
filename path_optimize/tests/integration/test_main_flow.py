import itertools
import pytest
from time_windows import calculate_effective_time_windows
from routing_solver import split_restaurant_nodes, run_model

def test_main_flow_simulation():
    data = {
        "places": [
            {
                "id": 1,
                "name": "제주공항",
                "x_cord": 126.492153,
                "y_cord": 33.505413,
                "category": "transport",
                "open_time": "08:00",
                "close_time": "08:00",
                "service_time": 0,
                "tags": ["공항", "입국"],
                "break_time": [],
                "is_mandatory": True
            },
            {
                "id": 3,
                "name": "맛집",
                "x_cord": 126.510000,
                "y_cord": 33.470000,
                "category": "restaurant",
                "open_time": "11:00",
                "close_time": "14:00",
                "service_time": 60,
                "tags": ["식당"],
                "break_time": [],
                "is_mandatory": False
            },
                        {
                "id": 1,
                "name": "제주공항",
                "x_cord": 126.492153,
                "y_cord": 33.505413,
                "category": "transport",
                "open_time": "08:00",
                "close_time": "08:00",
                "service_time": 0,
                "tags": ["공항", "입국"],
                "break_time": [],
                "is_mandatory": True
            }
        ],
        "user": {
            "start_time": "08:00",
            "end_time": "20:00",
            "meal_time_preferences": {"lunch": ["12:00", "13:00"]}
        },
        "day_info": {"is_first_day": True, "is_last_day": True, "date": "2025-04-10", "weekday": "목요일"}
    }
    
    places = data.get("places", [])
    user = data.get("user", {})
    day_info = data.get("day_info", {})
    
    eff_windows = calculate_effective_time_windows(places, user)
    new_places, new_eff_windows = split_restaurant_nodes(places, eff_windows)
    
    # 식사 노드별 그룹화
    meal_groups = {}
    for i, p in enumerate(new_places):
        if p.get("category") == "restaurant":
            meal_type = new_eff_windows[i][2]
            if meal_type is not None:
                meal_groups.setdefault(meal_type, []).append(i)
    group_indices = list(meal_groups.values())
    if group_indices:
        selections = list(itertools.product(*group_indices))
    else:
        selections = [()]
    
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
        results[sel] = {"objective": obj, "route": route} if route is not None else None
    
    valid_results = [r for r in results.values() if r is not None]
    assert len(valid_results) > 0
