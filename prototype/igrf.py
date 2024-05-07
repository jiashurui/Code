# International Geomagnetic Reference Field
import pymap3d as pm

# 输入经度、纬度和海拔高度（单位：度、度、米）
longitude = -122.2964
latitude = 47.6102
altitude = 0

# 计算地磁偏角和倾角
magnetic_declination, magnetic_inclination = pm.aer(latitude, longitude, altitude)

print("地磁偏角（磁北偏角）：", magnetic_declination, "度")
print("地磁倾角：", magnetic_inclination, "度")
