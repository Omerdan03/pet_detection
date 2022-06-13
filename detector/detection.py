import cv2
import os

FILE_DIR = os.path.dirname(__file__)

class Detector():

    with open(os.path.join(FILE_DIR, 'coco.names'), 'rt') as f:
        class_names = f.read().rstrip('\n').split('\n')

    def __init__(self, model_type='ssd'):
        if model_type == 'ssd':
            config_path = os.path.join(FILE_DIR, 'ssd_mobilenet_v3', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
            weights_path = os.path.join(FILE_DIR, 'ssd_mobilenet_v3', 'frozen_inference_graph.pb')

        self.net = cv2.dnn_DetectionModel(weights_path, config_path)

        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

    def detect(self, img, conf_threshold=0.5):
        class_ids, confs, bboxes = self.net.detect(img, conf_threshold)

        return class_ids, confs, bboxes




if __name__ == '__main__':

    detector = Detector()

    camera = cv2.VideoCapture(0)

    imgs = []
    for i in range(100):

        ret, frame = camera.read()

        if not ret:
            break

        class_ids, confs, bboxes = detector.detect(frame)

        for class_id, conf, bbox in zip(class_ids.flatten(), confs.flatten(), bboxes):
            cv2.rectangle(frame, bbox, color=(255, 0, 0), thickness=3)
            cv2.putText(frame, detector.class_names[class_id-1], (bbox[0]+10, bbox[1]+30),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 0, 0))

        imgs.append(frame)
        key = cv2.waitKey(30)
        if key == 27:
            break
        cv2.imshow('frame', frame)

    camera.release()
    cv2.destroyAllWindows()





