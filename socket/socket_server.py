import socket
import struct
import traceback
from datetime import datetime

import numpy as np
import sys

import train.train_1d_cnn
from train import train_1d_cnn, train_mh_1d_cnn, train_lstm
from utils.config_utils import get_value_from_config
from utils.show import real_time_show_phone_data
from prototype import global_tramsform, constant

# 定义服务器地址和端口
HOST = get_value_from_config('ip')  # 本地 IP 地址
PORT = 8081  # 监听的端口
sys.path.append('../prototype')  # 将 module_a 所在的文件夹添加到路径
apply_model = 'realworld'
# apply_model = 'mHealth'
model = 'lstm'

# 接收完整数据的函数
def receive_data(conn, data_size):
    data = b''
    while len(data) < data_size:
        try:
            packet = conn.recv(data_size - len(data))  # 接收剩余字节
            if not packet:
                # 客户端断开连接，返回None
                return None
            data += packet
        except ConnectionResetError:
            print("Client forcibly closed the connection.")
            return None
        except Exception as e:
            print(f"Error receiving data: {e}")
            return None
    return data


def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # 绑定地址和端口
        server_socket.bind((HOST, PORT))

        # 开始监听连接，允许的最大连接数为5
        server_socket.listen(5)
        print(f"Server started at {HOST}:{PORT}, waiting for connections...")

        # 初始化一个存储最新数据的数组 (限制为最新的 1024 行数据)
        all_data = np.zeros((128, 3), np.float32)
        all_transformed_data = np.zeros((128, 3), np.float32)

        while True:
            conn, addr = server_socket.accept()  # 等待客户端连接
            with conn:
                print(f"Connected by {addr}")
                try:
                    while True:
                        # 定义预期的字节数 (128 行, 9 列，每个 float 4 字节)
                        data_size = 9 * 128 * 4

                        # 调用接收函数
                        data = receive_data(conn, data_size)

                        if data is None:
                            print("Connection lost or no data received.")
                            break  # 结束当前客户端的处理，等待下一个客户端

                        # 检查接收的数据长度是否匹配
                        if len(data) != data_size:
                            print(f"Expected {data_size} bytes but got {len(data)} bytes")
                            continue  # 打印日志，继续等待下一轮接收
                        print(f"Received data, time: {datetime.now()}")

                        # 将字节流解析为 float[] (确保使用与客户端一致的字节序)
                        float_array = struct.unpack(f'>{9 * 128}f', data)  # Big-endian 字节序

                        # 转换为二维数组并截取前三列
                        float_matrix = np.array([list(float_array[i:i + 9]) for i in range(0, len(float_array), 9)])

                        # 拼接新数据到 `all_data`，保留最新的 1024 行
                        all_data = np.vstack([all_data, float_matrix[:, :3]])[-1024:, :]

                        # Global Transformed
                        transformed,rpy = global_tramsform.transform_sensor_data_to_np(float_matrix)

                        # 模型预测
                        if apply_model == 'realworld':
                            if model == 'cnn':
                                pred = train_1d_cnn.apply_1d_cnn(transformed)
                            elif model == 'lstm':
                                pred = train_lstm.apply_lstm(transformed)
                            pred_label = constant.Constant.RealWorld.action_map_reverse.get(pred.item())

                        elif apply_model == 'mHealth':
                            pred = train_mh_1d_cnn.apply_1d_cnn(transformed)
                            pred_label = constant.Constant.mHealth.action_map_reverse.get(pred.item())

                        # 实时展示数据（仅展示最新数据）
                        all_transformed_data = np.vstack([all_transformed_data, transformed[:, :3]])[-1024:, :]
                        real_time_show_phone_data(all_data, all_transformed_data, pred_label, rpy)

                        # use origin data to test
                        # 将预测结果发送回客户端
                        response = struct.pack('>f', float(pred))  # 将预测结果转换为字节流
                        conn.sendall(response)  # 返回结果给客户端
                        print(f"Response sent to client, time:{datetime.now()} response:{pred_label}")  # 打印日志，确认已发送

                except Exception as e:
                    print(f"Error handling connection from {addr}: {e}")
                    traceback.print_exc()
                finally:
                    print(f"Connection with {addr} closed.")


if __name__ == '__main__':
    start_server()
