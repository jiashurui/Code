# 用来解压realworld的数据集,记得换路径!
import os
import zipfile
import tarfile

def unzip_file(zip_path, extract_to):
    """解压ZIP文件"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def untar_file(tar_path, extract_to):
    """解压TAR文件"""
    with tarfile.open(tar_path, 'r') as tar_ref:
        tar_ref.extractall(path=extract_to)

def extract_files(folder_path,extract_to):
    """遍历文件夹并解压所有ZIP和TAR文件"""

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if zipfile.is_zipfile(file_path):
            print(f"解压ZIP文件：{filename}")
            unzip_file(file_path, extract_to)
        elif tarfile.is_tarfile(file_path):
            print(f"解压TAR文件：{filename}")
            untar_file(file_path, extract_to)

# 指定文件夹路径
frompath = '../data/realworld/2/'
topath = '../data/realworld/2/'
extract_files(frompath, topath)
