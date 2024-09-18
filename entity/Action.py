import dataclasses


@dataclasses.dataclass
class Coordinate:
    latitude: float
    longitude: float


@dataclasses.dataclass
class UserInfo:
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    mag_x: float
    mag_y: float
    mag_z: float
    coord: Coordinate
