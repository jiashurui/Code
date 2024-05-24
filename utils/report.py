from datetime import datetime


def save_report(config):
    now = datetime.now()

    data = f"使用以下参数:"
    for key, value in config.items():
        data += f"\n{key}: {value}"
    with open(f'../report/{now}.txt', 'w') as file:
        file.write(data)
def save_plot(plt , name):
    plt.gcf().savefig(f'../{name}.png', bbox_inches='tight')

