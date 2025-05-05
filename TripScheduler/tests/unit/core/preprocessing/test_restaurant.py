import pytest
from tripscheduler.core.preprocessing.restaurant import split_restaurant_nodes
from tests.utils.factory import make_fake_place

def test_non_restaurant_passthrough():
    place = make_fake_place("p1", "landmark", name="Museum")
    wins = {"p1": [(600, 1200, None)]}

    new_places, new_wins = split_restaurant_nodes([place], wins)

    assert len(new_places) == 1
    assert new_places[0]["id"] == "p1"
    assert new_places[0]["name"] == "Museum"
    assert new_wins[0] == (600, 1200, None)

def test_single_window_restaurant_no_split():
    place = make_fake_place("r1", "restaurant", name="SushiBar")
    wins = {"r1": [(690, 750, "lunch")]}

    new_places, new_wins = split_restaurant_nodes([place], wins)

    assert len(new_places) == 1
    assert new_places[0]["id"] == "r1"
    assert new_places[0]["name"] == "SushiBar"
    assert new_wins[0][2] == "lunch"

def test_multi_window_restaurant_split():
    place = make_fake_place("r2", "restaurant", name="KoreanBBQ")
    wins = {"r2": [(690, 750, "lunch"), (1050, 1140, "dinner")]}

    new_places, new_wins = split_restaurant_nodes([place], wins)

    assert len(new_places) == 2

    # 첫 분할
    assert new_places[0]["id"] == "r2_lunch"
    assert "lunch" in new_places[0]["name"]
    assert new_places[0]["org_id"] == "r2"
    assert new_wins[0] == (690, 750, "lunch")

    # 두 번째 분할
    assert new_places[1]["id"] == "r2_dinner"
    assert "dinner" in new_places[1]["name"]
    assert new_places[1]["org_id"] == "r2"
    assert new_wins[1] == (1050, 1140, "dinner")


def test_missing_windows_raises():
    place = make_fake_place("r3", "restaurant", name="Broken")
    wins = {}  # No entry

    with pytest.raises(ValueError, match="유효 시간 윈도우가 없습니다"):
        split_restaurant_nodes([place], wins)
