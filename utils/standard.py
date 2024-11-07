from prototype import constant

# 对标签进行映射
def stand_label(label, dataset):
    if dataset == 'mHealth':
        label_mapping = constant.Constant.simple_action_set.mapping_mh
    elif dataset == 'realworld':
        label_mapping = constant.Constant.simple_action_set.mapping_realworld
    elif dataset == 'stu':
        label_mapping = constant.Constant.simple_action_set.mapping_stu

    return [label_mapping[label] for label in label]

