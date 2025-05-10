import json
import folium

# JSON 불러오기
with open('../data/results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'lightblue']

visits = data['visits']
paths = data['path']

# 시작 위치 설정 (첫 번째 방문지 기준)
start_location = [visits[0]['y_cord'], visits[0]['x_cord']]
whole_map = folium.Map(location=start_location, zoom_start=14)

# 전체 지도에 마커 추가
for visit in visits:
    lat = visit['y_cord']
    lon = visit['x_cord']
    popup_text = (
        f"<b>{visit['order']}. {visit['place']}</b><br>"
        f"도착: {visit['arrival_str']}<br>"
        f"출발: {visit['departure_str']}<br>"
        f"체류: {visit['stay_duration']}"
    )
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_text, max_width=300),
        tooltip=visit['place'],
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(whole_map)

# 각 1-2, 2-3, ... 형태의 개별 지도 생성 및 저장
for i in range(len(visits) - 1):
    from_visit = visits[i]
    to_visit = visits[i + 1]

    if i >= len(paths):
        break  # path가 부족한 경우

    segment = paths[i]
    segment_coords = [[lat, lon] for lon, lat in segment]
    color = colors[i % len(colors)]

    # 개별 지도 생성
    seg_map = folium.Map(location=segment_coords[0], zoom_start=14)

    # 경로 그리기
    folium.PolyLine(
        locations=segment_coords,
        color=color,
        weight=5,
        opacity=0.8
    ).add_to(seg_map)

    # 출발지 마커
    folium.Marker(
        location=[from_visit['y_cord'], from_visit['x_cord']],
        tooltip=f"{from_visit['order']}. {from_visit['place']}",
        icon=folium.Icon(color='green')
    ).add_to(seg_map)

    # 도착지 마커
    folium.Marker(
        location=[to_visit['y_cord'], to_visit['x_cord']],
        tooltip=f"{to_visit['order']}. {to_visit['place']}",
        icon=folium.Icon(color='red')
    ).add_to(seg_map)

    # 파일명 예: 1-2.html
    filename = f"{from_visit['order']}-{to_visit['order']}.html"
    seg_map.save(filename)

    # 전체 지도에도 경로 추가
    folium.PolyLine(
        locations=segment_coords,
        color=color,
        weight=5,
        opacity=0.8
    ).add_to(whole_map)

# 전체 지도 저장
whole_map.save("trip_map_all.html")
