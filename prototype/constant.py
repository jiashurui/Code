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
        place_map = {}
        action_map = {
            'WALKING': 1,
            'WALKING_UPSTAIRS': 2,
            'WALKING_DOWNSTAIRS': 3,
            'SITTING': 4,
            'STANDING': 5,
            'LAYING': 6
        }