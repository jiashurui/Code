import requests


def get_intersections(lat, lon, radius=20):
    # 构建Overpass API查询
    query = f"""
    [out:json];
    (
      node(around:{radius},{lat},{lon})["highway"];
      way(around:{radius},{lat},{lon})["highway"];
      node(way(around:{radius},{lat},{lon})["highway"]);
    );
    out body;
    """

    # 发送请求到Overpass API
    response = requests.get("https://overpass.osm.jp/api/interpreter", params={'data': query})
    data = response.json()

    intersections = []

    # 解析返回的结果
    for element in data['elements']:
        if element['type'] == 'node':
            intersections.append((element['lat'], element['lon']))

    return intersections


# 示例调用：查询给定经纬度50米范围内的路口
lat, lon = 37.444945, 138.851768  # 替换为你感兴趣的经纬度
intersections = get_intersections(lat, lon, radius=20)

print("Intersections found at:")
for lat, lon in intersections:
    print(f"Latitude: {lat}, Longitude: {lon}")
