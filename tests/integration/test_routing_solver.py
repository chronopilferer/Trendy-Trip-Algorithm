import pytest
from routing_solver import split_restaurant_nodes, run_model
from time_windows import calculate_effective_time_windows
from utils import time_to_minutes

def test_run_model_basic():
    places = [
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
            "id": 2,
            "name": "한라산",
            "x_cord": 126.500000,
            "y_cord": 33.400000,
            "category": "landmark",
            "open_time": "09:00",
            "close_time": "17:00",
            "service_time": 90,
            "tags": ["자연", "등산"],
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
    ]
    user = {
        "start_time": "08:00",
        "end_time": "20:00",
        "meal_time_preferences": {"lunch": ["12:00", "13:00"]}
    }
    day_info = {"is_first_day": True, "is_last_day": False}
    
    eff_windows = calculate_effective_time_windows(places, user)
    new_places, new_eff_windows = split_restaurant_nodes(places, eff_windows)
    
    route, obj = run_model(new_places, new_eff_windows, day_info, user)
    assert route is not None
    assert obj is not None
    assert isinstance(route, list)
