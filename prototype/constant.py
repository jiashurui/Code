class Constant:
    class RealWorld:
        placement_map = {
            'waist': 0,
            'chest': 1,
            'forearm': 2,
            'head': 3,
            'shin': 4,
            'thigh': 5,
            'upperarm': 6
        }
        action_map = {
            'climbingdown': 0,
            'climbingup': 1,
            'jumping': 2,
            'lying': 3,
            'running': 4,
            'sitting': 5,
            'standing': 6,
            'walking': 7,
        }
        action_map_reverse = {
            0: 'climbingdown',
            1: 'climbingup',
            2: 'jumping',
            3: 'lying',
            4: 'running',
            5: 'sitting',
            6: 'standing',
            7: 'walking'
        }

    class UCI:
        place_map = {}  # waist
        action_map = {
            'WALKING': 1,
            'WALKING_UPSTAIRS': 2,
            'WALKING_DOWNSTAIRS': 3,
            'SITTING': 4,
            'STANDING': 5,
            'LAYING': 6
        },
        action_map_reverse = {
            1: 'WALKING',
            2: 'WALKING_UPSTAIRS',
            3: 'WALKING_DOWNSTAIRS',
            4: 'SITTING',
            5: 'STANDING',
            6: 'LAYING'
        }

    class CHA:
        place_map = {}  # pocket (thigh)
        action_map = {
            'WALKING': 1,
            'WALKING_UPSTAIRS': 2,
            'WALKING_DOWNSTAIRS': 3,
            'SITTING': 4,
            'STANDING': 5,
            'LAYING': 6
        }

    class mHealth:
        place_map = {}  # pocket (thigh)
        action_map = {
            'Null': 0,
            'STANDING': 1,
            'Sitting': 2,
            'Lying': 3,
            'Walking': 4,
            'Climbing stairs': 5,
            'Waist bends forward': 6,
            'Frontal elevation of arms': 7,
            'Knees bending': 8,
            'Cycling': 9,
            'Jogging': 10,
            'Running': 11,
            'Jump front & back': 12
        }
        action_map_reverse = {
            0: 'Null',
            1: 'STANDING',
            2: 'Sitting',
            3: 'Lying',
            4: 'Walking',
            5: 'Climbing stairs',
            6: 'Waist bends forward',
            7: 'Frontal elevation of arms',
            8: 'Knees bending',
            9: 'Cycling',
            10: 'Jogging',
            11: 'Running',
            12: 'Jump front & back'
        }
        data_columns = ['chest_x', 'chest_y', 'chest_z',
                        'electrocardiogram_1', 'electrocardiogram_2',
                        'ankle_x', 'ankle_y', 'ankle_z',
                        'gyro_x', 'gyro_y', 'gyro_z',
                        'magnetometer_x', 'magnetometer_y', 'magnetometer_z',
                        'arm_x', 'arm_y', 'arm_z',
                        'gyro_arm_x', 'gyro_arm_y', 'gyro_arm_z',
                        'magnetometer_arm_x', 'magnetometer_arm_y', 'magnetometer_arm_z',
                        'label']

    class ChildWalk:
        place_map = {}
        action_map = {
            '歩く': 1,
            '止まる': 2,
            '走る': 3,
        }
        action_map_en = {
            'なし': 'null',
            '歩く': 'walk',
            '止まる': 'stand',
            '走る': 'run',
        }
        action_map_en_reverse = {
            0: 'null',
            1: 'walk',
            2: 'stand',
            3: 'run',
        }
        orient_map = {
            '前を向く': 0,
            '下を向く': 1,
            '振り返る(左)': 2,
            '右を向く': 3
        },

    # 長岡技大学生行動
    class uStudent:
        place_map = {}
        action_map = {
            '立つ': 1,
            'フラフラ': 2,
            'しゃがむ': 3,
            '跳ぶ': 4,
            '歩く': 5,
            '走る': 6
        }
        action_map_en = {
            'stand': 1,
            'Wandering': 2,
            'squat': 3,
            'jump': 4,
            'walk': 5,
            'run': 6
        }
        action_map_en_reverse = {
            1: 'stand',
            2: 'Wandering',
            3: 'squat',
            4: 'jump',
            5: 'walk',
            6: 'run'
        }

    class uStudent_1111:
        action_map_en_reverse = {
            0: 'Turn_Left_45',
            1: 'Turn_Left_90',
            2: 'Turn_Left_135',
            3: 'Turn_Right_45',
            4: 'Turn_Right_90',
            5: 'Turn_Right_135',
            6: 'Raise_Left_Low',
            7: 'Raise_Left_Medium',
            8: 'Raise_Left_High',
            9: 'Raise_Right_Low',
            10: 'Raise_Right_Medium',
            11: 'Raise_Right_High',
    }
    # Realworld 数据集与长冈科技大学学生交叉
    class realworld_x_uStudent:
        action_map = {
            'stand':0,
            'jump':1,
            'walk':2,
            'run':3
        }
        action_map_en_reverse = {
            0: 'stand',
            1: 'jump',
            2: 'walk',
            3: 'run'
        }
        # 将realworld的标签转换
        mapping_realworld = {
            # 'climbingdown': 0,
            # 'climbingup': 1,
            2: 1,  # 'jumping': 2,
            # 'lying': 3,
            4: 3,  # 'running': 4,
            # 'sitting': 5,
            6: 0,  # 'standing': 6,
            7: 2  # 'walking': 7,
        }
        # 将大学生的标签转换
        mapping_stu = {
            1: 0,  # 1: 'stand',
            # 2: 'Wandering',
            # 3: 'squat',
            4: 1 , # 4: 'jump',
            5: 2 , # 5: 'walk',
            6: 3 # 6: 'run'
        }
    class realworld_x_mHealth:
        action_map = {
            'stand':0,
            'jump':1,
            'walk':2,
            'run':3,
            'lying':4,
            'sitting':5,
            'climbing stairs':6,
        }
        action_map_en_reverse = {
            0: 'stand',
            1: 'jump',
            2: 'walk',
            3: 'run',
            4: 'lying',
            5: 'sitting',
            6: 'climbing stairs'
        }
        # 将realworld的标签转换
        mapping_realworld = {
            0: 6, # 'climbingdown': 0,
            1: 6, # 'climbingup': 1,
            2: 1, # 'jumping': 2,
            3 :4, # 'lying': 3,
            4: 3, # 'running': 4,
            5: 5, # 'sitting': 5,
            6: 0, # 'standing': 6,
            7: 2  # 'walking': 7,
        }
        # 将mHealth的标签转换
        mapping_mh = {
            # 0: 'Null',
            1:0,# 1: 'STANDING',
            2:5,# 2: 'Sitting',
            3:4,# 3: 'Lying',
            4:2,# 4: 'Walking',
            5:6,# 5: 'Climbing stairs',
            # 6: 'Waist bends forward',
            # 7: 'Frontal elevation of arms',
            # 8: 'Knees bending',
            # 9: 'Cycling',
            # 10: 'Jogging',
            11:3# 11: 'Running',
            # 12: 'Jump front & back'
        }

    # 最简单的行动分类
    class simple_action_set:
        action_map = {
            'stand': 0,
            'walk': 1,
            'run': 2
        }
        action_map_en_reverse = {
            0: 'stand',
            1: 'walk',
            2: 'run'
        }
        # 将realworld的标签转换
        mapping_realworld = {
            # 0: 4, # 'climbingdown': 0,
            # 1: 4, # 'climbingup': 1,
            # 2: 1,  # 'jumping': 2,
            # 'lying': 3,
            4: 2,  # 'running': 4,
            # 'sitting': 5,
            6: 0,  # 'standing': 6,
            7: 1  # 'walking': 7,
        }
        # 将大学生的标签转换
        mapping_stu = {
            1: 0,  # 1: 'stand',
            # 2: 'Wandering',
            # 3: 'squat',
            # 4: 1 , # 4: 'jump',
            5: 1 , # 5: 'walk',
            6: 2 # 6: 'run'
        }
        # 将mHealth的标签转换
        mapping_mh = {
            # 0: 'Null',
            1:0,# 1: 'STANDING',
            # 2:5,# 2: 'Sitting',
            # 3:4,# 3: 'Lying',
            4:1,# 4: 'Walking',
            # 5:6,# 5: 'Climbing stairs',
            # 6: 'Waist bends forward',
            # 7: 'Frontal elevation of arms',
            # 8: 'Knees bending',
            # 9: 'Cycling',
            # 10: 'Jogging',
            11:2,# 11: 'Running',
            # 12:1# 12: 'Jump front & back'
        }
        # 将mHealth的标签转换
