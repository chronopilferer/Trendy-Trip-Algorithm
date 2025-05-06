import json
import folium

with open('../data/results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'lightblue']

start_location = [data['visits'][0]['y_cord'], data['visits'][0]['x_cord']]
m = folium.Map(location=start_location, zoom_start=14)

for visit in data['visits']:
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
    ).add_to(m)

for i, segment in enumerate(data['path']):
    segment_coords = [[lat, lon] for lon, lat in segment]
    color = colors[i % len(colors)]
    
    folium.PolyLine(
        locations=segment_coords,
        color=color,
        weight=5,
        opacity=0.8,
    ).add_to(m)

m.save('trip_map.html')
