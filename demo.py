from entity.Action import UserInfo, Coordinate
from mapapi.overpass_api_utils import poi_trigger
from prototype.model import CNN
import torch

from utils.map_utils import cal_distance

# 危険なパタン定義
# 走る・JUMP・


# 0. prepare
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(in_channels=3, out_label=6).to(device)
model.load_state_dict(torch.load('../../model/1D-CNN-3CH.pth'))
model.eval()
radius = 10
start_point = UserInfo(coord=Coordinate(latitude=10, longitude=10))
end_point = UserInfo(coord=Coordinate(latitude=11, longitude=11))

input_data = []

# 1. get data
start_gps = (1, 2)
end_gps = (1, 3)

# 2. get poi TODO online API取得 --> offline 取得
poi = poi_trigger(start_point.coord.latitude, start_point.coord.longitude, radius)

# 3. trigger
# poi not none


# 現在の位置から、最寄交差点を取得

# 最初と最後のデータから、交差点に近づくかを判断

s_distance = cal_distance(start_point.coord.latitude, start_point.coord.longitude,
                          poi.latitude, poi.longitude)
e_distance = cal_distance(end_point.coord.latitude, end_point.coord.longitude,
                          poi.latitude, poi.longitude)

# 遠くなるまたトリガー範囲外
if s_distance < e_distance | e_distance >= radius:
    exit()


# 4. recognition 認識
outputs = model(input_data)
pred = outputs.argmax(dim=1, keepdim=True)  # 获取概率最大的索引


# 5. trigger end

# 6. analysis danger
# 7. alarm
