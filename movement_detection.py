import cv2


def main():

    camera = cv2.VideoCapture(0)

    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    while True:

        ret, frame = camera.read()
        if not ret:
            break

        mask = object_detector.apply(frame)

        cv2.imshow('mask', mask)
        cv2.imshow('image', frame)
        key = cv2.waitKey(30)
        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()