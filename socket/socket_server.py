import socket
import struct
import traceback

# 定义服务器地址和端口
HOST = '192.168.11.2'  # 本地回环地址
PORT = 8081  # 监听的端口

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
            # 客户端强制关闭连接
            print("Client forcibly closed the connection.")
            return None
        except Exception as e:
            # 处理其他接收数据时的异常
            print(f"Error receiving data: {e}")
            return None
    return data

def start_server():
    # 创建TCP/IP套接字
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # 绑定地址和端口
        server_socket.bind((HOST, PORT))

        # 开始监听连接，允许的最大连接数为5
        server_socket.listen(5)
        print(f"Server started at {HOST}:{PORT}, waiting for connections...")

        while True:
            # 等待客户端连接
            conn, addr = server_socket.accept()  # 接受连接
            with conn:
                print(f"Connected by {addr}")
                try:
                    while True:
                        # 定义预期的字节数 (128 rows * 9 cols * 4 bytes per float)
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

                        # 将字节流解析为 float[]
                        float_array = struct.unpack(f'>{9 * 128}f', data)  # 使用 Big-endian
                        # 转换为二维数组
                        float_matrix = [list(float_array[i:i + 9]) for i in range(0, len(float_array), 9)]

                        return float_matrix
                        # 打印二维数组
                        # print("Received float matrix:")
                        # for row in float_matrix:
                        #     print(row)

                except Exception as e:
                    # 捕获处理客户端连接时的其他异常
                    print(f"Error handling connection from {addr}: {e}")
                    traceback.print_exc()
                finally:
                    print(f"Connection with {addr} closed.")


if __name__ == '__main__':
    start_server()