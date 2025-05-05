def add_dummy_node(places, wins, kind: str, gs: int, ge: int) -> int:
    """
    경로 시작 또는 종료를 위한 더미 노드를 추가합니다.
    kind: 'start' 또는 'end'
    gs, ge: global start/end 시간
    """
    name, cat = ('dummy_start', 'dummy') if kind == 'start' else ('dummy_end', 'dummy_end')
    node = {
        'name': name,
        'category': cat,
        'service_time': 0,
        'x_cord': 0.0,
        'y_cord': 0.0
    }
    places.append(node)
    wins.append((gs, ge, None))
    return len(places) - 1

def is_dummy_node(name: str) -> bool:
    """더미 노드 이름 체크"""
    return name in ('dummy_start', 'dummy_end')
