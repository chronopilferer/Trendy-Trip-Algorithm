import pytest
from tripscheduler.core.indexing.handlers import determine_start_end_indices
from tests.utils.factory import make_fake_place

def make_place_list(*categories):
    return [make_fake_place(i, cat) for i, cat in enumerate(categories)]

def test_first_day_with_transport_and_optional_accommodation():
    places = make_place_list("transport", "accommodation", "restaurant")
    day_info = {"is_first_day": True, "is_last_day": False}
    start, end = determine_start_end_indices(places, day_info)
    assert start == 0
    assert end == 1

def test_first_day_with_transport_only():
    places = make_place_list("transport", "restaurant")
    day_info = {"is_first_day": True, "is_last_day": False}
    start, end = determine_start_end_indices(places, day_info)
    assert start == 0
    assert end is None

def test_last_day_with_accommodation_and_transport():
    places = make_place_list("accommodation", "transport", "restaurant")
    day_info = {"is_first_day": False, "is_last_day": True}
    start, end = determine_start_end_indices(places, day_info)
    assert start == 0
    assert end == 1

def test_last_day_with_only_transport():
    places = make_place_list("restaurant", "transport")
    day_info = {"is_first_day": False, "is_last_day": True}
    start, end = determine_start_end_indices(places, day_info)
    assert start is None
    assert end == 1

def test_mid_day_with_two_accommodations():
    places = make_place_list("accommodation", "restaurant", "accommodation")
    day_info = {"is_first_day": False, "is_last_day": False}
    start, end = determine_start_end_indices(places, day_info)
    assert start == 0
    assert end == 2

def test_mid_day_with_one_accommodation():
    places = make_place_list("restaurant", "accommodation")
    day_info = {"is_first_day": False, "is_last_day": False}
    start, end = determine_start_end_indices(places, day_info)
    assert start == 1
    assert end is None

def test_one_day_trip():
    places = make_place_list("transport", "restaurant", "transport")
    day_info = {"is_first_day": True, "is_last_day": True}
    start, end = determine_start_end_indices(places, day_info)
    assert start == 0
    assert end == 2

def test_first_day_without_transport():
    places = make_place_list("restaurant", "accommodation")
    day_info = {"is_first_day": True, "is_last_day": False}
    with pytest.raises(ValueError, match="transport.*없습니다"):
        determine_start_end_indices(places, day_info)

def test_last_day_with_multiple_accommodations():
    places = make_place_list("accommodation", "accommodation", "transport")
    day_info = {"is_first_day": False, "is_last_day": True}
    with pytest.raises(ValueError, match="accommodation.*2.*있습니다"):
        determine_start_end_indices(places, day_info)

def test_invalid_day_info_combination():
    places = make_place_list("transport")
    day_info = {"is_first_day": True, "is_last_day": None}  
    with pytest.raises(ValueError, match="유효하지 않은 day_info 조합"):
        determine_start_end_indices(places, day_info)
