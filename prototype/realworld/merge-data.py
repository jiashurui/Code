import glob
import os
import pandas as pd

# 文件所在的根目录
root_dir = '../../data/realworld/1/'

# 定义身体位置
body_parts = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']

# 定义传感器类型，按顺序排列
sensors = ['acc', 'Gyroscope', 'MagneticField']

# 遍历每个身体位置
for body_part in body_parts:
    # 存储每个传感器的文件路径
    sensor_files = {sensor: None for sensor in sensors}

    # 遍历文件夹，寻找匹配的文件
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 检查文件名是否包含当前身体位置和传感器类型
            for sensor in sensors:
                if file.startswith(sensor) and body_part in file and file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    sensor_files[sensor] = file_path

    # 读取每个传感器的数据，并按列名进行合并
    dataframes = []
    for sensor in sensors:  # 按照顺序读取加速度 -> 角速度 -> 磁力
        file_path = sensor_files.get(sensor)
        if file_path:
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)
                df = df.iloc[:, 2:]  # 舍弃前两列，假设前两列为时间和ID（如果没有列名）
                df = df.add_prefix(sensor + '_')  # 添加前缀区分不同传感器的数据
                dataframes.append(df)
            except Exception as e:
                print(f'读取文件 {file_path} 时出错: {e}')

    # 合并所有传感器的数据，按列顺序加速度 -> 角速度 -> 磁力
    if dataframes:
        merged_data = pd.concat(dataframes, axis=1)

        # 保存合并后的文件
        output_file = f'{body_part}_merged.csv'

        # 如果保存路径不存在，则创建
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        merged_data.to_csv(os.path.join(root_dir, output_file), index=False)
        print(f'已保存合并文件: {output_file}')
