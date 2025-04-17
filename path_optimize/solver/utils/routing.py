
def add_dummy_node(places, wins, kind, gs, ge):
    name, cat = ("dummy_start", "dummy") if kind == "start" else ("dummy_end", "dummy_end")
    node = {"name": name, "category": cat, "service_time": 0, "x_cord": 0.0, "y_cord": 0.0}
    places.append(node)
    wins.append((gs, ge, None))
    return len(places) - 1

def is_dummy_node(name):
    return name in ("dummy_start", "dummy_end")