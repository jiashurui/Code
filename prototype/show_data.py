import random

import numpy as np

from prototype.constant import Constant
from prototype.dataReader import get_data, get_origin_data
from matplotlib import pyplot as plt
from utils.map_utils import find_key_by_value


# 随便展示几条数据
def show_some_data():
    rows = 2
    cols = 2

    x,y,z, places, actions, who = get_origin_data(100)
    fig, axs = plt.subplots(rows, cols)
    fig.set_size_inches(10, 10)
    random_integers = [random.randint(1, len(x)) for _ in range(16)]

    index = 0
    label_map = Constant.RealWorld.label_map
    action_map = Constant.RealWorld.action_map

    for i in range(rows):
        for j in range(cols):
            indite = random_integers[index]
            axs[i, j].plot(x[indite, :], label='x')
            axs[i, j].plot(y[indite, :], label='y')
            axs[i, j].plot(z[indite, :], label='z')

            label_text = find_key_by_value(label_map, places[indite])
            action_text = find_key_by_value(action_map, actions[indite])

            axs[i, j].set_title(f'{indite}, {label_text} , {action_text}')
            axs[i, j].legend(fontsize="x-large")
            index += 1

    # tensor2np
    plt.tight_layout()
    plt.show()


def random_choose_one_data():
    train_data, train_labels, test_data, test_labels = get_data(100)
    index = random.randint(1, train_data.size(0))
    print(f'random_choose_one_data, index:  {index}')
    return train_data[index, 0, :]


def show_fft_result():
    one_simple_data = random_choose_one_data().numpy()
    single_no_dc = one_simple_data - np.mean(one_simple_data)
    fft_result = np.fft.fft(single_no_dc)
    freqs = np.fft.fftfreq(len(one_simple_data), 1 / 100)
    window = np.hanning(len(one_simple_data))

    fig, (a1, a2, a3, a4, a5, a6, a7, a8) = plt.subplots(8, 1)
    fig.set_size_inches(12, 12)

    a1.plot(one_simple_data)
    a1.set_title('origin xyz data')

    a2.plot(single_no_dc)
    a2.set_title('origin xyz data - mean')

    a3.plot(freqs, fft_result.real)
    a3.set_title('fft real part')

    a4.plot(freqs, fft_result.imag)
    a4.set_title('fft imag part')

    a5.plot(freqs, np.abs(fft_result))
    a5.set_title('fft absolute value')

    a6.plot(freqs, np.angle(fft_result))
    a6.set_title('fft angle value')

    a7.plot(np.angle(fft_result))
    a7.set_title('fft angle value(val)')

    a8.plot(one_simple_data * window)
    a8.set_title('fft origin data with windows function')

    plt.show()

# 展示随便一个数据的fft之后的结果.
show_fft_result()

# 对比下不同数据
show_some_data()