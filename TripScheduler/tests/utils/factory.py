from typing import Literal, Optional, Dict, Any
import random

def make_fake_place(
    id_: str,
    category: Literal["transport", "restaurant", "accommodation", "landmark"],
    name: Optional[str] = None,
    open_time="08:00",
    close_time="20:00",
    break_time=None,
    is_mandatory: bool = False
) -> Dict[str, Any]:
    return {
        "id": id_,
        "name": name or f"{category.title()} Place {id_}",
        "x_cord": round(random.uniform(126.0, 127.0), 6),
        "y_cord": round(random.uniform(33.0, 38.0), 6),
        "category": category,
        "open_time": open_time,
        "close_time": close_time,
        "service_time": 60,
        "tags": [],
        "휴무일": [],
        "break_time": break_time or [],
        "is_mandatory": is_mandatory,
    }

def make_fake_user() -> Dict[str, Any]:
    return {
        "start_time": "08:00",
        "end_time": "21:00",
        "travel_style": "relaxed",
        "meal_time_preferences": {
            "breakfast": ["08:00", "09:00"],
            "lunch": ["12:00", "13:00"],
            "dinner": ["18:00", "19:00"]
        }
    }

def make_fake_day_info(first=False, last=False) -> Dict[str, Any]:
    return {
        "is_first_day": first,
        "is_last_day": last,
        "date": "2025-04-10",
        "weekday": "목요일"
    }

def make_sample_data() -> Dict[str, Any]:
    return {
        "places": [
            make_fake_place(1, "accommodation", name="숙소1"),
            make_fake_place(2, "restaurant", name="점심식당", open_time="11:30", close_time="14:00"),
            make_fake_place(3, "landmark", name="관광지1"),
            make_fake_place(4, "restaurant", name="저녁식당", open_time="17:00", close_time="21:00"),
            make_fake_place(5, "accommodation", name="숙소2"),
        ],
        "user": make_fake_user(),
        "day_info": make_fake_day_info()
    }
