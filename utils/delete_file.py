import os
import glob

# 设定文件的目录和名称模式
directory = "../data/realworld/2/"
patternGPS = ""

# 构建完整的路径模式
path_pattern_GPS = os.path.join(directory, 'GPS*.csv')
path_pattern_LIGHT = os.path.join(directory, 'Light*.csv')
path_pattern_Mic = os.path.join(directory, 'Microphone*.csv')
path_pattern_SQL = os.path.join(directory, '*.sqlite')
path_pattern_ZIP = os.path.join(directory, '*.zip')

# 使用glob找到所有匹配的文件
list_file_patterns = [glob.glob(path_pattern_GPS),
                      glob.glob(path_pattern_LIGHT),
                      glob.glob(path_pattern_Mic),
                      glob.glob(path_pattern_SQL),
                      glob.glob(path_pattern_ZIP),]

# 遍历文件列表并删除每个文件
for pattern_list in list_file_patterns:
    for file in pattern_list:
        try:
            os.remove(file)
            print(f"文件已删除：{file}")
        except OSError as e:
            print(f"删除文件时出错：{e} - {file}")
