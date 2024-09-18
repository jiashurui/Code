import requests


def get_nearest_roads(api_key, coordinates):
    base_url = "https://roads.googleapis.com/v1/nearestRoads"
    params = {
        "points": "|".join([f"{lat},{lng}" for lat, lng in coordinates]),
        "key": api_key
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


# 示例调用
api_key = "AIzaSyAgCU29FYbjgwppooCvPCcVet1EM0n3moE"
coordinates = [(37.444945, 138.851768)]  # 替换为你的坐标列表
nearest_roads = get_nearest_roads(api_key, coordinates)
print(nearest_roads)
