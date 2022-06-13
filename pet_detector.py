import os
from collections import defaultdict

import imageio
from detector import Detector
import cv2
import pandas as pd
import numpy as np


def main():

    detector = Detector(model_type='ssd')

    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        class_ids, confs, bboxes = detector.detect(frame)
        class_names = [detector.class_names[class_id -1] for class_id in class_ids.flatten()]

        if 'dog' in class_names:
            pass


        cv2.imshow('camera', frame)
        key = cv2.waitKey(30)
        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
