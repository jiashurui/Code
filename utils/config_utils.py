import os
import platform
import configparser


def set_env_from_property_file(file_path):
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
        return

    # 检查配置文件中是否有该平台的配置
    if os_name in config:
        for key in config[os_name]:
            # 设置环境变量
            os.environ[key] = config[os_name][key]
            print(f"Set {key} = {config[os_name][key]} for {os_name}")
    else:
        print(f"No configuration found for {os_name} in the property file.")


# 使用示例
set_env_from_property_file('config.properties')
