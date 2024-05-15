def slide_window(input_list, window_size, over_lap):
    buffer = []
    all_data = []
    index = 0
    while True:
        if len(buffer) < window_size:
            if index < len(input_list):
                buffer.append(input_list[index])
                index += 1
            else:
                #flush
                all_data.append(buffer)
                break
        else:
            # flush
            all_data.append(buffer)
            buffer = []
            index = index - int(window_size * over_lap)

    return all_data

def slide_window2(input_list, window_size, over_lap):
    all_data = []

    stat_point = 0
    end_point = stat_point + window_size
    stride = int(window_size * over_lap)
    while True:
        if end_point >= len(input_list) - 1:
            #flush
            # all_data.append(input_list[stat_point: len(input_list) - 1 ])
            break

        all_data.append(input_list.iloc[stat_point: end_point])
        stat_point += stride
        end_point += stride

    return all_data

def slide_window_df(data_frame, window_size, over_lap):
    buffer = []
    all_data = []
    index = 0
    while True:
        if len(buffer) < window_size:
            if index < len(data_frame):
                buffer.append(data_frame.iloc[index])
                index += 1
            else:
                #flush
                all_data.append(buffer)
                break
        else:
            # flush
            all_data.append(buffer)
            buffer = []
            index = index - int(window_size * over_lap)

    return all_data
# usage
# l = []
# for i in range(100): l.append(i)
#
# print(slide_window(l, 4, 0.5))
