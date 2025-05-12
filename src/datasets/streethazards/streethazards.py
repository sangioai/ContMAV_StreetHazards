########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################

    

class StreetHazardsBase:
    SPLITS = ["train", "valid", "test", "val"]

    # number of classes without void/unlabeled/anomaly
    N_CLASSES = [12]

    # 1+12+1 classes (0: unlabeled, 13 = anomaly)
    CLASS_NAMES_FULL = [
        "unlabeled",   # =   1,
        "building",    # =   2,
        "fence",       # =   3, 
        "other",       # =   4,
        "pedestrian",  # =   5, 
        "pole",        # =   6,
        "road line",   # =   7, 
        "road",        # =   8,
        "sidewalk",    # =   9,
        "vegetation",  # =  10, 
        "car",         # =  11,
        "wall",        # =  12, 
        "traffic sign",# =  13,
        "anomaly",     # =  14,
    ]
       

    CLASS_COLORS_FULL = [
        [  0,   0,   0],  # unlabeled    =   1,
        [ 70,  70,  70],  # building     =   2,
        [190, 153, 153],  # fence        =   3, 
        [250, 170, 160],  # other        =   4,
        [220,  20,  60],  # pedestrian   =   5, 
        [153, 153, 153],  # pole         =   6,
        [157, 234,  50],  # road line    =   7, 
        [128,  64, 128],  # road         =   8,
        [244,  35, 232],  # sidewalk     =   9,
        [107, 142,  35],  # vegetation   =  10, 
        [  0,   0, 142],  # car          =  11,
        [102, 102, 156],  # wall         =  12, 
        [220, 220,   0],  # traffic sign =  13,
        [ 60, 250, 240],  # anomaly      =  14,
    ]

    RGB_DIR = "rgb"

    LABELS_FULL_DIR = "labels_12"
    LABELS_FULL_COLORED_DIR = "labels_12_colored"
