import argparse
import cv2
import numpy as np
from datetime import datetime
import os
from os import path

from detector import Detector

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
CSV_FILE_NAME = 'detection_log.csv'


def get_args():
    parser = argparse.ArgumentParser(
        prog='pet_monitor',
        formatter_class=argparse.RawTextHelpFormatter,
        description="Runs pet_monitor.")

    parser.add_argument('-pet', metavar='pet', default="dog", type=str,
                        help='type of pet to monitor for (dog / cat)')
    parser.add_argument('-outputfolder', metavar='outputfolder', default='output', type=str,
                        help='Path where output.csv and images will be saved')
    args = parser.parse_args()
    return args


def mask_based_on_contours(contours, img_shape) -> np.ndarray:
    mask = np.zeros(img_shape).astype(bool)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            mask[y: y + h, x: x + w] = True

    return mask


def time_to_string(time):
    return time.strftime("%d_%m_%Y-%H_%M_%S")


def main():
    args = get_args()
    monitored_pet = args.pet
    outputfolder = args.outputfolder

    detector = Detector(model_type='ssd')
    camera = cv2.VideoCapture(0)

    if not path.exists(outputfolder):
        os.mkdir(outputfolder)
    if not path.exists(path.join(outputfolder, CSV_FILE_NAME)):
        with open(path.join(outputfolder, CSV_FILE_NAME), "a") as file:
            file.write('time,indication\n')
    last_check_time = datetime.now()
    while True:
        current_time = datetime.now()

        if (current_time - last_check_time).seconds > 0: # Check every second

            time_string = time_to_string(current_time)

            ret, frame = camera.read()
            indication = False
            if not ret:
                break
            class_names, confs, bboxes = detector.detect(frame, conf_threshold=0.5)
            if len(class_names) != 0:  # Prevents from assessing when no objected was detected
                for i, class_name in enumerate(class_names):
                    if class_name == monitored_pet:  # Printing only the requested class
                        indication = True
                        x, y, w, h = bboxes[i]
                        cv2.putText(frame, class_name, (x, y - 15),
                                    cv2.FONT_HERSHEY_PLAIN, 2, RED, 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 3)
            with open(path.join(outputfolder, CSV_FILE_NAME), "a") as file:
                file.write(f'{time_string}, {indication}\n')

            # cv2.putText(frame, time_string, (15, 30), cv2.FONT_HERSHEY_PLAIN, 1, BLACK, 2)
            # cv2.putText(frame, f"Press ESC to close", (15, 30), cv2.FONT_HERSHEY_PLAIN, 1, BLACK, 2)
            # cv2.imwrite(path.join(outputfolder, f'{time_string}.jpg'), frame)
            # cv2.imshow('camera', frame)
            last_check_time = current_time

        # wait for exc key to close
        key = cv2.waitKey(30)
        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
