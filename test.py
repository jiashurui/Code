#! python
import csv

import math
import numpy as np

all_list = []
with open('/Users/jiashurui/Desktop/homework.csv') as file:
    lines = [line.rstrip() for line in file]

    for str in lines:
        line = str.split(',')
        res = [eval(i) for i in line]
        all_list.append(res)
    # TF
    tf = []
    frequency = []
    for term_list in all_list:
        # print(term_list)
        m = []
        for idoc in range(1, 51):
            count = term_list.count(idoc)
            m.append(count)
        frequency.append(m)
        max_count = max(m)
        tf_line = []
        for fre in m:
            tf_line.append(fre / max_count)
        tf.append(tf_line)
    # print(frequency)

    matrix = np.array(frequency)
    # print(tf)

    idf_list = []
    for column in matrix.T:
        # print(column)# ndarray
        c = 0
        for val in column:
            if val != 0:
                c = c + 1
        idf = math.log10(100 / c)
        idf_list.append(idf)

    print(idf_list)
    print(tf)

    result = []
    for doc in tf:
        calc_list = []
        for i in range(0, 50):
            tf_idf = round(doc[i] * idf_list[i], 4)

            str_number = "{:.4f}".format(tf_idf)

            calc_list.append(str_number)
        print(calc_list)
        result.append(calc_list)
    print(result)

    for l in result:
        for i in range(len(l)):
            if l[i] == 0.0 or l[i] == '0.0000':
                l[i] = 0

    with open('/Users/jiashurui/Desktop/result.csv', 'w') as f:
        writer = csv.writer(f)

        for line in result:
            writer.writerow(line)
