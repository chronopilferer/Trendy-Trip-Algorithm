import math

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return int(round(R * c)) + 10  # 여유 시간 10분 추가

def create_distance_matrix(places):
    n = len(places)
    matrix = [[0] * n for _ in range(n)]
    for i, pi in enumerate(places):
        for j in range(i + 1, n):
            lat1, lon1 = pi["x_cord"], pi["y_cord"]
            lat2, lon2 = places[j]["x_cord"], places[j]["y_cord"]
            d = haversine_distance(lat1, lon1, lat2, lon2)
            matrix[i][j] = matrix[j][i] = d
    return matrix
