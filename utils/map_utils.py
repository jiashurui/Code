def find_key_by_value(d, target_value):
    for key, value in d.items():
        if value == target_value:
            return key
    return None  # 如果没有找到，返回 None

# 示例字典
my_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 2}

# 查找值为 2 的键
key = find_key_by_value(my_dict, 2)
print(key)  # 输出 'b'，因为它是第一个值为 2 的键


def find_keys_by_value(d, target_value):
    keys = [key for key, value in d.items() if value == target_value]
    return keys

# 示例字典
my_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 2}

# 查找所有值为 2 的键
keys = find_keys_by_value(my_dict, 2)
print(keys)  # 输出 ['b', 'd']
