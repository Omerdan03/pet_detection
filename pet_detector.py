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
        if len(class_ids) != 0:  # Prevents from assessing when no objected was detected
            class_names = [detector.class_names[class_id - 1] for class_id in class_ids.flatten()]

            for i in range(len(class_names)):
                x, y, w, h = bboxes[i]
                cv2.putText(frame, class_names[i], (x, y - 15),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            if 'dog' in class_names:
                print('dog')


        cv2.imshow('camera', frame)
        key = cv2.waitKey(30)
        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
