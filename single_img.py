import cv2

def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    name = f'frame.jpg'
    cv2.imwrite(name, frame)


if __name__ == '__main__':
    main()