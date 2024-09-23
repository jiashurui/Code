def find_key_by_value(dictionary, value):
    # 反转字典
    reversed_dict = {v: k for k, v in dictionary.items()}

    return reversed_dict.get(value)