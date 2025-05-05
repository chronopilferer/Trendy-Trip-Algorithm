import pytest
from tripscheduler.core.indexing.category import get_indices_by_category, validate_place_category
from tests.utils.factory import make_fake_place

def test_get_indices_by_category_found():
    places = [
        make_fake_place(1, "restaurant"),
        make_fake_place(2, "landmark"),
        make_fake_place(3, "restaurant"),
    ]
    indices = get_indices_by_category(places, "restaurant")
    assert indices == [0, 2]

def test_get_indices_by_category_not_found():
    places = [
        make_fake_place(1, "accommodation"),
        make_fake_place(2, "transport")
    ]
    indices = get_indices_by_category(places, "restaurant")
    assert indices == []

def test_validate_place_category_success():
    place = make_fake_place(1, "restaurant")
    validate_place_category(place, "restaurant", "카테고리가 일치하지 않습니다.")  

def test_validate_place_category_fail():
    place = make_fake_place(1, "landmark")
    with pytest.raises(ValueError, match="카테고리가 일치하지 않습니다."):
        validate_place_category(place, "restaurant", "카테고리가 일치하지 않습니다.")
