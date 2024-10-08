class Constant:
    class RealWorld:
        label_map = {
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
            1: 'STANDING',
            2:'Sitting',
            3:'Lying',
            4:'Walking',
            5:'Climbing stairs',
            6:'Waist bends forward',
            7:'Frontal elevation of arms',
            8:'Knees bending',
            9:'Cycling',
            10:'Jogging',
            11:'Running',
            12:'Jump front & back'
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
        orient_map = {
            '前を向く':0,
            '下を向く':1,
            '振り返る(左)':2,
            '右を向く':3
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