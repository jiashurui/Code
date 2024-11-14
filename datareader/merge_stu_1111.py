import pandas as pd


# データ分割
def split_stu_1111(file, data_type):
    save_path = '../data/student/1111_lab/processed/'

    df = pd.read_csv(file)

    actions = ['Turn_Left_45', 'Turn_Left_90', 'Turn_Left_135',
               'Turn_Right_45', 'Turn_Right_90', 'Turn_Right_135',
               'Raise_Left_Low', 'Raise_Left_Medium', 'Raise_Left_High',
               'Raise_Right_Low', 'Raise_Right_Medium', 'Raise_Right_High',
               'Interval', 'Interval', 'Interval']  # Interval

    subject = ['sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6']

    # Tokyo time : 2024年11月11日 15:10:00
    # UTC time : 2024年11月11日 06:10:00
    start_time = 1731305400000

    # 1 min (1 sec * 60)
    time_step = 1000 * 60

    # ６人 x 15　行動
    for i in range(6 * 15):
        end_time = start_time + time_step
        df_split = df[(df['UNIX_time(milli)'] >= start_time)]
        df_split = df_split[df_split['UNIX_time(milli)'] <= end_time]

        start_time = end_time

        subject_name = subject[i // 15]
        action_name = actions[i % 15]

        # Intervalは処理しない
        if action_name == 'Interval':
            continue
        # save data
        filename = f'{subject_name}_{data_type}_{action_name}.csv'
        df_split.to_csv(save_path + filename, index=False)


if __name__ == '__main__':
    split_stu_1111('../data/student/1111_lab/accelerometers.csv', 'acc')
    split_stu_1111('../data/student/1111_lab/angular_rate.csv', 'ang')
    split_stu_1111('../data/student/1111_lab/geopoints.csv', 'geo')
    split_stu_1111('../data/student/1111_lab/magnetic_field.csv', 'mag')
