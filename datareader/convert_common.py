# 将 dataframe 的列名改为标准化的形式
# 默认输入是9/10列的dataframe
# Param:{df: Pandas dataframe}
# Result:{更改后的dataframe}
def convert_df_columns_name(df):
    columns_with_label = ['accx','accy','accz','gyx','gyy','gyz','magx','magy','magz','label']
    columns_without_label = ['accx','accy','accz','gyx','gyy','gyz','magx','magy','magz']
    if len(df.columns) == 9:
        df.columns = columns_without_label
    elif len(df.columns) == 10:
        df.columns = columns_with_label
    else:
        raise Exception('Column names do not match')
    return df

if __name__ == '__main__':
    print(convert_df_columns_name(None))