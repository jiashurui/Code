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

    # print(idf_list)
    # print(tf)

    result = []
    for doc in tf:
        calc_list = []
        for i in range(0, 50):
            tf_idf = round(doc[i] * idf_list[i], 4)

            # str_number = "{:.4f}".format(tf_idf)

            calc_list.append(tf_idf)
        # print(calc_list)
        result.append(calc_list)
    # print(result)

    # with open('/Users/jiashurui/Desktop/result.csv', 'w') as f:
    #     writer = csv.writer(f)
    #
    #     for line in result:
    #         writer.writerow(line)

    query_frequency = []
    with open('/Users/jiashurui/Desktop/qs.csv') as fff:
        ql = [line.rstrip() for line in fff]

        for s in ql:
            line = s.split(',')
            res = [eval(i) for i in line]
            zeros_list = [0] * 50

            for ele in res:
                zeros_list[ele - 1] += 1
            query_frequency.append(zeros_list)

    query_tf = []
    for q in query_frequency:
        max_count = max(q)
        tf_line = []
        for fre in q:
            tf_line.append(fre / max_count)
        query_tf.append(tf_line)

    # print(query_tf)
    query_matrix = np.array(query_frequency)
    # print(tf)

    query_idf_list = idf_list
    # for column in query_matrix.T:
    #     # print(column)# ndarray
    #     count = 0
    #     for val in column:
    #         if val != 0:
    #             count = count + 1
    #     if count != 0:
    #         idf = math.log10(100 / count)
    #     else:
    #         idf = 0
    #     query_idf_list.append(idf)

    query_tf_idf = []
    for doc in query_tf:
        cl = []
        for i in range(0, 50):
            q_tf_idf = round(doc[i] * query_idf_list[i], 4)
            cl.append(q_tf_idf)
        # print(calc_list)
        query_tf_idf.append(cl)
    # print(query_tf_idf)

    # d1 --->d100
    # q1-->q4 |q|
    # 4 x 50
    query_tf_idf = np.array(query_tf_idf)
    # print(query_tf_idf)

    # 100 x 50
    doc_tf_idf = np.array(result)
    # print(doc_tf_idf)

    # 计算点积矩阵
    dot_product_matrix = np.dot(query_tf_idf, doc_tf_idf.T)

    # 计算查询向量的范数
    query_norms = np.linalg.norm(query_tf_idf, axis=1)

    # 计算文档向量的范数
    document_norms = np.linalg.norm(doc_tf_idf, axis=1)

    # 计算范数矩阵
    norm_matrix = np.outer(query_norms, document_norms)

    # 计算余弦相似度矩阵
    cosine_similarity_matrix = dot_product_matrix / norm_matrix

    # 输出结果
    # print(cosine_similarity_matrix)

    # 获取文档编号矩阵（文档编号从1开始）
    doc_indices = np.arange(1, cosine_similarity_matrix.shape[1] + 1)

    # 按行处理相似度矩阵
    sorted_results = []
    sorted_index = []
    for query_idx, similarities in enumerate(cosine_similarity_matrix):
        # 创建一个 (similarity, doc_index) 的列表
        sim_doc_pairs = [(sim, doc_idx) for sim, doc_idx in zip(similarities, doc_indices) if sim > 0]

        # 按相似度降序排列，若相似度相同则按文档编号升序排列
        sorted_pairs = sorted(sim_doc_pairs, key=lambda x: (-x[0], x[1]))
        sorted_results.append(sorted_pairs)
        sorted_doc_indices = [doc_idx for sim, doc_idx in sorted_pairs]
        sorted_index.append(sorted_doc_indices)

    print(sorted_results)
    with open('/Users/jiashurui/Desktop/homework2.csv', 'w') as f:
        writer = csv.writer(f)

        for line in sorted_index:
            writer.writerow(line)