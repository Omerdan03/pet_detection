import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
TO_SCALE = 600

yolo_folder = 'yolo'
# get the categories names and colors
with open(os.path.join(yolo_folder, "coco.names"), "r") as f:
    coco_classes = f.read().split("\n")[:-1]
color_arr = np.random.randint(0, 256, size=(80, 3), dtype='int')

# build the pretrained network
NET = cv2.dnn.readNetFromDarknet(os.path.join(yolo_folder, 'yolov3.cfg'),
                                 os.path.join(yolo_folder, 'yolov3.weights'))


def get_video_stream():
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Handles the mirroring of the current frame
        frame = cv2.flip(frame, 1)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yield frame


def caption(img):
    """
    This function gets an np.array of an image and returns an np.array of image with the right tags
    :param img: np.array
    :return:  np.array
    """

    def output_coordinates_to_box_coordinates(cx, cy, w, h, img_h, img_w):
        abs_x = int((cx - w / 2) * img_w)
        abs_y = int((cy - h / 2) * img_h)
        abs_w = int(w * img_w)
        abs_h = int(h * img_h)
        return abs_x, abs_y, abs_w, abs_h

    def numpy_to_list(array):
        return [int(num) for num in array]

    print('caption started')

    # Scaling the photo so the largest dimension will be 600
    scale = TO_SCALE / max(img.shape)
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    img = cv2.resize(img, dim, cv2.INTER_AREA)
    print(f'scaled to {dim}')

    print('captioning with model')
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
    NET.setInput(blob)
    NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    unconnected = NET.getUnconnectedOutLayers()
    output_names = [NET.getLayerNames()[layer_num - 1] for layer_num in unconnected]
    large, medium, small = NET.forward(output_names)
    all_outputs = np.vstack((large, medium, small))
    objs = all_outputs[all_outputs[:, 4] > 0.1]
    boxes = [output_coordinates_to_box_coordinates(*obj[:4], *img.shape[:2]) for obj in objs]
    confidences = [float(obj[4]) for obj in objs]
    class_names = [coco_classes[np.argmax(obj[5:])] for obj in objs]
    colors = [numpy_to_list(color_arr[np.argmax(obj[5:])]) for obj in objs]
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    img_yolo = img.copy()
    print('finished captioning, putting tags on the image')
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        class_name = class_names[i]
        confidence = confidences[i]
        color = colors[i]
        text = f'{class_name} {confidence:.3}'
        cv2.rectangle(img_yolo, (x, y), (x + w, y + h), color, 5)
        cv2.putText(img_yolo, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

    print('finish tagging, return image')
    return img_yolo


def main():

    video_stream = get_video_stream()
    while True:
        arr = caption(next(video_stream))
        cv2.imshow('image', arr)


if __name__ == "__main__":
    main()
