import cv2
import os

FILE_DIR = os.path.dirname(__file__)


class Detector:
    with open(os.path.join(FILE_DIR, 'coco.names'), 'rt') as f:
        class_names = f.read().rstrip('\n').split('\n')

    def __init__(self, model_type='ssd'):
        if model_type == 'ssd':
            config_path = os.path.join(FILE_DIR, 'ssd_mobilenet_v3', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
            weights_path = os.path.join(FILE_DIR, 'ssd_mobilenet_v3', 'frozen_inference_graph.pb')
        else:
            raise NotImplementedError("only ssd is currently supported")

        self.net = cv2.dnn_DetectionModel(weights_path, config_path)

        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

    def detect(self, img, conf_threshold=0.5):
        class_ids, confs, bboxes = self.net.detect(img, conf_threshold)
        class_names = [self.class_names[class_id - 1] for class_id in class_ids.flatten()]

        return class_names, confs, bboxes
