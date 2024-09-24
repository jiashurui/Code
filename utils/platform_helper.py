import platform


def get_os_type():
    os_type = platform.system()

    if os_type == "Darwin":
        return "Mac"
    elif os_type == "Linux":
        return "Linux"
    elif os_type == "Windows":
        return "Windows"
    else:
        return "Unknown OS"


print(get_os_type())
