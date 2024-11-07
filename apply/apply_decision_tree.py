import numpy as np
from joblib import load

from datareader import datareader_stu, realworld_datareader, mh_datareader
from datareader.datareader_stu import simple_get_stu_all_features
from datareader.mh_datareader import simple_get_mh_all_features
from datareader.realworld_datareader import simple_get_realworld_all_features
from prototype import constant

# mHealth ----> stu , realworld,
loaded_clf_mh = load('../model/decision_tree_mHealth.joblib')
loaded_clf_realworld = load('../model/decision_tree_realworld.joblib')
loaded_clf_stu = load('../model/decision_tree_stu.joblib')


def apply_to_stu():
    # Param
    slice_length = 40
    filtered_label = [2, 3]
    mapping = constant.Constant.simple_action_set.mapping_stu

    # 全局变换之后的大学生数据(全局变换按照frame进行)
    origin_data = simple_get_stu_all_features(slice_length, type='df',
                                              filtered_label=filtered_label,
                                              mapping_label=mapping, with_rpy=True)
    origin_data_np = np.array(origin_data)
    # 抽取特征
    features_list = datareader_stu.get_features(origin_data)

    train_data = np.array(features_list)

    # np round 是因为,标签在转换过程中出现了浮点数,导致astype int的时候,标签错误
    label = np.round(origin_data_np[:, 0, 9]).astype(int)

    accuracy_mh = loaded_clf_mh.score(train_data, label)
    accuracy_realworld = loaded_clf_realworld.score(train_data, label)
    accuracy_stu = loaded_clf_stu.score(train_data, label)

    print(f'Student Accuracy,mh_model : {accuracy_mh}')
    print(f'Student Accuracy,realworld_model : {accuracy_realworld}')
    print(f'Student Accuracy,stu_model : {accuracy_stu}')
############################################################################################################

def apply_to_realworld():
    simpling = 50
    features_number = 9
    slice_length = 256
    filtered_label = [0, 1, 3, 5]
    mapping = constant.Constant.simple_action_set.mapping_realworld

    # 全局变换之后RealWorld数据(全局变换按照frame进行)
    origin_data = simple_get_realworld_all_features(slice_length, type='df',
                                                    filtered_label=filtered_label,
                                                    mapping_label=mapping,
                                                    with_rpy=True)
    origin_data_np = np.array(origin_data)

    features_list = realworld_datareader.get_features(origin_data)
    train_data = np.array(features_list)

    # np round 是因为,标签在转换过程中出现了浮点数,导致astype int的时候,标签错误
    label = np.round(origin_data_np[:, 0, 9]).astype(int)

    accuracy_mh = loaded_clf_mh.score(train_data, label)
    accuracy_realworld = loaded_clf_realworld.score(train_data, label)
    accuracy_stu = loaded_clf_stu.score(train_data, label)

    print(f'Realworld Accuracy,mh_model : {accuracy_mh}')
    print(f'Realworld Accuracy,realworld_model : {accuracy_realworld}')
    print(f'Realworld Accuracy,stu_model : {accuracy_stu}')

############################################################################################################
def apply_to_mHealth():
    features_number = 9
    slice_length = 256

    filtered_label = [0, 2, 3, 5, 6, 7, 8, 9, 10]
    mapping = constant.Constant.simple_action_set.mapping_mh

    # 全局变换之后的大学生数据(全局变换按照frame进行)
    origin_data = simple_get_mh_all_features(slice_length, type='df',
                                             filtered_label=filtered_label,
                                             mapping_label=mapping, with_rpy=True)
    origin_data_np = np.array(origin_data)

    features_list = mh_datareader.get_features(origin_data)
    train_data = np.array(features_list)

    # np round 是因为,标签在转换过程中出现了浮点数,导致astype int的时候,标签错误
    label = np.round(origin_data_np[:, 0, 9]).astype(int)

    accuracy_mh = loaded_clf_mh.score(train_data, label)
    accuracy_realworld = loaded_clf_realworld.score(train_data, label)
    accuracy_stu = loaded_clf_stu.score(train_data, label)

    print(f'mHealth Accuracy,mh_model : {accuracy_mh}')
    print(f'mHealth Accuracy,realworld_model : {accuracy_realworld}')
    print(f'mHealth Accuracy,stu_model : {accuracy_stu}')


if __name__ == '__main__':
    apply_to_stu()
    apply_to_realworld()
    apply_to_mHealth()