
#
from matplotlib import pyplot as plt
def show_me_data0(np_arr):
    fig, ax = plt.subplots()
    ax.plot(np_arr)
    ax.legend()

    # plt.xlim(0, 10000)
    plt.ylim(0, 2.5)

    plt.show()

def show_me_data1(df, col_name):
    fig, ax = plt.subplots()
    for col in col_name:
        ax.plot(df.index, df[col], label=col)
    # 设置图例
    ax.legend()
    plt.show()
def show_me_data2(df_list, col_name):
    # too many plot is hard
    for df in df_list[0:5]:
        show_me_data1(df, col_name)

