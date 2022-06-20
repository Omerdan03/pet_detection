from detector import Detector
import cv2
import numpy as np
import time


def mask_based_on_contours(contours, img_shape) -> np.ndarray:
    mask = np.zeros(img_shape).astype(bool)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            mask[y: y + h, x: x + w] = True

    return mask


def main():
    detector = Detector(model_type='ssd')

    camera = cv2.VideoCapture(0)
    max_fps = camera.get(cv2.CAP_PROP_FPS)

    # This part was an attempt in increase frame rate with movement detection
    # movement_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    while True:
        start_time = time.time()
        ret, frame = camera.read()
        if not ret:
            break

        # simple_mask = movement_detection.apply(frame)
        # movement_contours, _ = cv2.findContours(simple_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # mask = mask_based_on_contours(movement_contours, frame.shape)
        # masked_frame = mask * frame

        class_ids, confs, bboxes = detector.detect(frame, conf_threshold=0.5)
        if len(class_ids) != 0:  # Prevents from assessing when no objected was detected
            class_names = [detector.class_names[class_id - 1] for class_id in class_ids.flatten()]

            for i, class_name in enumerate(class_names):
                if class_name == 'dog':  # Printing only dogs on frame, can be changed to any class name
                    x, y, w, h = bboxes[i]
                    cv2.putText(frame, class_name, (x, y - 15),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        end_time = time.time()
        time_pass = end_time - start_time
        fps = min(np.round(1 / time_pass, 2), max_fps)
        cv2.putText(frame, f"current fps: {fps}", (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        # cv2.imshow('movement_mask', masked_frame

        cv2.imshow('camera', frame)
        key = cv2.waitKey(30)
        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
