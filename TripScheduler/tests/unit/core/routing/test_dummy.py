from tripscheduler.core.routing.dummy import add_dummy_node, is_dummy_node

def test_add_dummy_node():
    places, wins = [], []
    idx = add_dummy_node(places, wins, "start", "08:00", "21:00")
    assert idx == 0
    assert places[0]["name"] == "dummy_start"
    assert wins[0][0] == "08:00"

def test_is_dummy_node():
    assert is_dummy_node("dummy_start")
    assert is_dummy_node("dummy_end")
    assert not is_dummy_node("제주공항")
