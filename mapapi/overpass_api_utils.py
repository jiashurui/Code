import requests

def poi_trigger(lat, lng, radius):
    url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
    node["highway"="traffic_signals"](around:{radius}, {lat}, {lng});
    node["highway"="crossing"](around:{radius}, {lat}, {lng});
    );
    out body;
    """

    response = requests.post(url, data={'data': query})
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response}")

if __name__ == "__main__":
    print(poi_trigger(37.445629, 138.849725, radius=20))