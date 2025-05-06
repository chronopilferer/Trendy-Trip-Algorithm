import json

# 마커 매핑
coord_to_marker = {}
marker_counter = [0]

def get_marker_name(coord):
    coord = tuple(round(c, 7) for c in coord)
    if coord not in coord_to_marker:
        marker = chr(ord('a') + marker_counter[0])
        coord_to_marker[coord] = marker
        marker_counter[0] += 1
    return coord_to_marker[coord]

# JSON 로드
with open('../data/directions_raw_data.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)

route_labels = []
row_index = 1  # 순번 출력용

# 모든 내부 리스트 순회
for group in raw:
    if not isinstance(group, list):
        continue

    for idx, entry in enumerate(group):
        # 첫 번째 null은 건너뛰되, null 표시 필요하면 그대로 남겨둬도 됨
        if idx == 0:
            route_labels.append("null")
            row_index += 1
            continue

        if entry in [None, [], {}] or not isinstance(entry, dict) or entry.get('code') != 0:
            route_labels.append("null")
            row_index += 1
            continue

        traoptimal = entry.get('route', {}).get('traoptimal', [])
        if not traoptimal:
            route_labels.append("null")
            row_index += 1
            continue

        # 첫 번째 경로만 분석
        route = traoptimal[0]
        summary = route.get('summary', {})
        start = tuple(summary.get('start', {}).get('location', []))
        goal = tuple(summary.get('goal', {}).get('location', []))

        if len(start) != 2 or len(goal) != 2:
            route_labels.append("null")
            row_index += 1
            continue

        start_marker = get_marker_name(start)
        goal_marker = get_marker_name(goal)

        label = f"{start_marker} → {goal_marker}"
        route_labels.append(label)
        row_index += 1

# 최종 출력
for i, label in enumerate(route_labels, 1):
    print(f"{i}. {label}")
