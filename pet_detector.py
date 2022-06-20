from detector import Detector
import cv2
import numpy as np
import time

def main():

    detector = Detector(model_type='ssd')

    camera = cv2.VideoCapture(0)

    while True:
        start_time = time.time()
        ret, frame = camera.read()
        if not ret:
            break

        class_ids, confs, bboxes = detector.detect(frame, conf_threshold=0.5)
        if len(class_ids) != 0:  # Prevents from assessing when no objected was detected
            class_names = [detector.class_names[class_id - 1] for class_id in class_ids.flatten()]

            for i, class_id in enumerate(class_names):\

                if class_id == 'dog':
                    x, y, w, h = bboxes[i]
                    cv2.putText(frame, class_names[i], (x, y - 15),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        end_time = time.time()
        fps = np.round(1 / (end_time - start_time), 2)
        cv2.putText(frame, f"current fps: {fps}", (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        cv2.imshow('camera', frame)
        key = cv2.waitKey(30)
        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
