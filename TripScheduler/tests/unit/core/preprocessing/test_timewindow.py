import pytest
from tripscheduler.core.preprocessing.timewindow import calculate_effective_time_windows
from tripscheduler.core.preprocessing.restaurant import split_restaurant_nodes
from tests.utils.factory import make_fake_place

user_base = {
    "start_time": "08:00",
    "end_time": "20:00",
    "meal_time_preferences": {
        "lunch":  ("11:30", "13:00"),
        "dinner": ("17:30", "19:00")
    }
}

def test_restaurant_with_valid_lunch():
    place = make_fake_place("p1", "restaurant", name="LunchPlace", open_time="11:00", close_time="14:00")
    wins = calculate_effective_time_windows([place], user_base)
    assert "p1" in wins
    assert wins["p1"][0][2] == "lunch"

def test_restaurant_with_break_time_excluded():
    place = make_fake_place(
        "p1", "restaurant", name="LunchPlace",
        open_time="10:00", close_time="14:00",
        break_time=["12:00", "12:30"]
    )
    wins = calculate_effective_time_windows([place], user_base)
    assert "p1" in wins
    assert any(w[2] == "lunch" for w in wins["p1"])

def test_restaurant_no_overlap_raises():
    place = make_fake_place("p2", "restaurant", name="TooLate", open_time="14:00", close_time="15:00")
    with pytest.raises(ValueError):
        calculate_effective_time_windows([place], user_base)

def test_landmark_time_window_computed():
    place = make_fake_place("p3", "landmark", name="Museum", open_time="09:00", close_time="18:00")
    wins = calculate_effective_time_windows([place], user_base)
    assert wins["p3"][0][2] is None
    assert wins["p3"][0][0] >= 480   # 08:00 in minutes
    assert wins["p3"][0][1] <= 1080  # 18:00 in minutes

def test_midnight_crossing_time():
    place = make_fake_place("p4", "restaurant", name="NightCafe", open_time="22:00", close_time="02:00")
    user = {
        "start_time": "20:00",
        "end_time": "03:00",
        "meal_time_preferences": {
            "dinner": ("23:00", "01:00")
        }
    }
    wins = calculate_effective_time_windows([place], user)
    assert "p4" in wins
    assert wins["p4"][0][2] == "dinner"

def test_restaurant_splitting():
    place = make_fake_place("r10", "restaurant", name="SplitPlace", open_time="11:00", close_time="21:00")
    user = {
        "start_time": "08:00",
        "end_time": "21:00",
        "meal_time_preferences": {
            "lunch": ("11:30", "13:00"),
            "dinner": ("18:00", "19:00")
        }
    }
    wins = calculate_effective_time_windows([place], user)
    new_places, new_wins = split_restaurant_nodes([place], wins)
    assert len(new_places) == 2
    assert new_places[0]["org_id"] == "r10"
    assert new_wins[0][2] in ["lunch", "dinner"]
