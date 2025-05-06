from tripscheduler.utils.time import minutes_to_time_str

def format_visit_info(order, node, arrival, stay, places,
                      travel_minutes=None, wait_minutes=None, delay_minutes=None,
                      x_cord=None, y_cord=None):
    record = {
        'order'         : order,
        'place'         : places[node]['name'],
        'arrival_str'   : minutes_to_time_str(arrival),
        'departure_str' : minutes_to_time_str(arrival + stay),
        'stay_duration' : minutes_to_time_str(stay),
        'x_cord'        : x_cord if x_cord is not None else places[node].get("x_cord"),
        'y_cord'        : y_cord if y_cord is not None else places[node].get("y_cord"),
    }

    if travel_minutes is not None:
        record['travel_time'] = minutes_to_time_str(travel_minutes)
    if wait_minutes is not None and wait_minutes > 0:
        record['wait_time'] = minutes_to_time_str(wait_minutes)
    if delay_minutes is not None and delay_minutes > 0:
        record['delay_time'] = minutes_to_time_str(delay_minutes)

    return record
