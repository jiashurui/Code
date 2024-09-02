import datetime

def unix_time_to_datetime(unix_timestamp):
    if unix_timestamp > 1e12:
        unix_timestamp = unix_timestamp/1000
    elif unix_timestamp < 999999999:
        print("not a unix timestamp")
        return None

    datetime_obj = datetime.datetime.fromtimestamp(unix_timestamp)

    # 将日期时间对象格式化为可读字符串
    readable_time = datetime_obj.strftime('%Y-%m-%d %H:%M:%S')

    print(f"Unix timestamp: {unix_timestamp}")
    print(f"datatime: {readable_time}")

def datetime_to_unixtime(year, month, day, hour, minute, second):
    # 创建datetime对象
    dt = datetime.datetime(year, month, day, hour, minute, second)

    # 将datetime对象转换为Unix时间戳
    unix_timestamp = int(dt.timestamp()) * 1000
    print(f"datetime: {dt}")
    print(f"unixtime: {unix_timestamp}")

    return unix_timestamp

