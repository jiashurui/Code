import pandas as pd

# 假设有三个CSV文件 'file1.csv', 'file2.csv', 'file3.csv'

csv1 = '../data/student/0726_lab/accelerometers_label.csv'
csv2 = '../data/student/0726_lab/angular_rate.csv'
csv3 = '../data/student/0726_lab/magnetic_field.csv'

csv_files = [csv1, csv2, csv3]

# 读取所有CSV文件为DataFrame列表
dfs = [pd.read_csv(file) for file in csv_files]
# 读取第一个CSV文件，保留所有列
df1 = pd.read_csv(csv_files[0])

# 读取第二和第三个CSV文件，跳过第一列（可以用参数 usecols 来指定读取的列）
df2 = pd.read_csv(csv_files[1], usecols=lambda column: column != df1.columns[0])  # 不读取第1列
df3 = pd.read_csv(csv_files[2], usecols=lambda column: column != df1.columns[0])  # 不读取第1列


# 将这些DataFrame横向拼接
merged_df = pd.concat([df1, df2, df3], axis=1)

# label 位置放最后
label = merged_df.pop('label')
merged_df['label'] = label

# 将拼接后的DataFrame导出为新的CSV文件
merged_df.to_csv('../data/student/0726_lab/merge_labeled.csv', index=False)
