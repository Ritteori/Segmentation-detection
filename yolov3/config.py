ANNOTATIONS_DIR = r"J:\ML_for_porfolio\segmentation\yolov3\VOC2012\Annotations"
LABELS_DIR = r"J:\ML_for_porfolio\segmentation\yolov3\VOC2012\yolo_labels"
IMAGES_DIR = r"J:\ML_for_porfolio\segmentation\yolov3\VOC2012\JPEGImages"
BATCH_SIZE = 16
EPOCHS = 25
IMG_SHAPE = (3,416,416)

ANCHORS = [
    [(10,13), (16,30), (33,23)],   # small scale (13×13)
    [(30,61), (62,45), (59,119)],  # medium scale (26×26)
    [(116,90), (156,198), (373,326)]  # large scale (52×52)
]