import os
import platform
import configparser


def get_value_from_config(key, default=None):
    file_path = '../config.ini'
    # 创建configparser对象
    config = configparser.ConfigParser()

    # 读取配置文件
    config.read(file_path)

    # 获取当前操作系统
    os_type = platform.system()

    # 根据操作系统选择对应的配置
    if os_type == "Darwin":
        os_name = "Mac"
    elif os_type == "Linux":
        os_name = "Linux"
    elif os_type == "Windows":
        os_name = "Windows"
    else:
        print("Unknown OS. No environment variables set.")
        return None

    # 检查配置文件中是否有该平台的配置
    if os_name in config:
        return config[os_name].get(key, default)
    else:
        print(f"No configuration found for {os_name} in the property file.")
        return default


if __name__ == "__main__":
    # 使用示例
    print(get_value_from_config('child_data_set'))
