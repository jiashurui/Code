import math


# haversine
# https://note.com/b_giganteus/n/n197aea3af14a
def cal_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    r = 6371

    distance = c * r
    return distance * 1000

if __name__ == "__main__":
    # 37.445487, 138.849725
    print(cal_distance(138.849725,37.445487, 138.849725, 37.4455666))