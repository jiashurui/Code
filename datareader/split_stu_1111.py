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

    df_arr = []
    # ６人 x 15　行動
    for i in range(6 * 15):
        end_time = start_time + time_step
        df_split = df[(df['UNIX_time(milli)'] >= start_time)]
        df_split = df_split[df_split['UNIX_time(milli)'] <= end_time]

        start_time = end_time

        subject_name = subject[i // 15]
        action_name = actions[i % 15]

        sub_no = i // 15
        action_no = i % 15
        # Intervalは処理しない
        if action_name == 'Interval':
            continue

        data_frame = df_split.values
        df_new = pd.DataFrame(data_frame, columns=['Unix_time',f'{data_type}x', f'{data_type}y', f'{data_type}z'])

        df_new['label'] = action_no
        df_new['subject'] = sub_no

        df_arr.append(df_new)
        # save data
        # filename = f'{subject_name}_{data_type}_{action_name}.csv'
        # df_split.to_csv(save_path + filename, index=False)

    big_df = pd.concat(df_arr, ignore_index=True)

    return big_df

if __name__ == '__main__':
    d1 = split_stu_1111('../data/student/1111_lab/accelerometers.csv', 'acc')
    d2 = split_stu_1111('../data/student/1111_lab/angular_rate.csv', 'ang')
    d3 = split_stu_1111('../data/student/1111_lab/magnetic_field.csv', 'mag')
    # d4 = split_stu_1111('../data/student/1111_lab/geopoints.csv', 'geo')

    merged_df = pd.concat([d1, d2, d3], axis=1)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated(keep='last')]
    merged_df = merged_df[['Unix_time'] + [col for col in merged_df.columns if col != 'Unix_time']]
    merged_df.to_csv('../data/student/1111_lab/merged.csv', index=False)

    # print(merged_df)
