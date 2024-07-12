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
    class ChildWalk:
        place_map = {}
        action_map = {
            '歩く': 1,
            '止まる': 2,
            '走る': 3,
        }