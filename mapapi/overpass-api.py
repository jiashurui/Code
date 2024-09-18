import requests

url = "http://overpass-api.de/api/interpreter"
query = """
[out:json];
(
node["highway"="traffic_signals"](around:20, 37.445614, 138.849768);
node["highway"="crossing"](around:20, 37.445614, 138.849768);
);
out body;
"""

response = requests.post(url, data={'data': query})

# 检查请求是否成功
if response.status_code == 200:
    data = response.json()
    print(data['elements'])
else:
    print(f"Error: {response}")