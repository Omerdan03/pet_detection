import os
from collections import defaultdict

import imageio
import cv2
import pandas as pd
import numpy as np

TARGET = 'potted plant'
LOCATION = 'dining table'
TO_SCALE = 600

yolo_folder = 'yolo'
# get the categories names and colors
with open(os.path.join(yolo_folder, "coco.names"), "r") as f:
    coco_classes = f.read().split("\n")[:-1]
color_arr = np.random.randint(0, 256, size=(80, 3), dtype='int')

# build the pretrained network
NET = cv2.dnn.readNetFromDarknet(os.path.join(yolo_folder, 'yolov3.cfg'),
                                 os.path.join(yolo_folder, 'yolov3.weights'))


def get_video_stream(camera):
    """
    This function returns a Generator that generate images from a given camera
    :return:
    """
    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        # Handles the mirroring of the current frame
        frame = cv2.flip(frame, 1)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yield frame


def output_coordinates_to_box_coordinates(cx, cy, width, height, img_h, img_w):
    abs_x = int((cx - width / 2) * img_w)
    abs_y = int((cy - height / 2) * img_h)
    abs_w = int(width * img_w)
    abs_h = int(height * img_h)
    return abs_x, abs_y, abs_w, abs_h


def coordinates_to_box_coordinates(cx, cy, width, height):
    left_side = cx - width / 2
    right_side = cx + width / 2
    bottom_side = cy - height / 2
    top_side = cy + height / 2

    return left_side, right_side, bottom_side, top_side


def detect_objects(img: np.array):
    # cv2.imshow('image', img)
    """
    This function gets an np.array of an image and returns a dict with object name and locations. np.array of image with the right tags
    :param img: np.array
    :return:  np.array
    """
    # Scaling the photo so the largest dimension will be 600
    scale = TO_SCALE / max(img.shape)
    img_width = int(img.shape[1] * scale)
    img_height = int(img.shape[0] * scale)
    dim = (img_width, img_height)
    img = cv2.resize(img, dim, cv2.INTER_AREA)

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
    NET.setInput(blob)
    NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    unconnected = NET.getUnconnectedOutLayers()
    output_names = [NET.getLayerNames()[layer_num - 1] for layer_num in unconnected]
    large, medium, small = NET.forward(output_names)
    all_outputs = np.vstack((large, medium, small))
    objs = all_outputs[all_outputs[:, 4] > 0.1]  # Getting all outputs with certainty above 0.1
    boxes = [output_coordinates_to_box_coordinates(*obj[:4], *img.shape[:2]) for obj in objs]
    confidences = [float(obj[4]) for obj in objs]
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    objects_dict = defaultdict(list)
    for i in indices:
        obj = objs[i]
        obj_class = coco_classes[np.argmax(obj[5:])]
        obj_center = obj[:2]
        obj_box = coordinates_to_box_coordinates(*obj[:4])
        objects_dict[obj_class].append({'center': obj_center,
                                        'box': obj_box,
                                       'scaled_box': boxes[i]})

    return objects_dict


def is_in_box(location, box):
    return (location[0] >= box[0]) and (location[0] <= box[1]) and (location[1] >= box[2]) and (location[1] <= box[3])


def add_caption_to_image(img, objects_dict):
    text_color = [255, 255, 255]
    frame_color = [0, 0, 0]
    # Scaling the photo so the largest dimension will be 600
    scale = TO_SCALE / max(img.shape)
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    img = cv2.resize(img, dim, cv2.INTER_AREA)

    objects = list()
    for class_name in objects_dict:
        single_class_list = [obj for obj in objects_dict[class_name]]
        [obj.update({'class_name': class_name}) for obj in single_class_list]
        objects += single_class_list

    img_yolo = img.copy()
    for obj in objects:
        x, y, w, h = obj['scaled_box']
        class_name = obj['class_name']
        text = f'{class_name}'
        cv2.rectangle(img_yolo, (x, y), (x + w, y + h), frame_color, 5)
        cv2.putText(img_yolo, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
    cv2.putText(img_yolo, pd.Timestamp.now().strftime('%D - %H:%M:%S'), (int(width*0.6), height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    return img_yolo


def caption(img):
    """
    This function gets an np.array of an image and returns an np.array of image with the right tags
    :param img: np.array
    :return:  np.array
    """
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
    objs = all_outputs[all_outputs[:, 4] > 0.1]  # Getting all outputs with certainty above 0.1
    boxes = [output_coordinates_to_box_coordinates(*obj[:4], *img.shape[:2]) for obj in objs]
    confidences = [float(obj[4]) for obj in objs]
    class_names = [coco_classes[np.argmax(obj[5:])] for obj in objs]
    colors = [(color_arr[np.argmax(obj[5:])]).tolist() for obj in objs]
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

    camera = cv2.VideoCapture(0)
    ser = pd.Series(index=pd.DatetimeIndex([]), dtype=bool)
    video_stream = get_video_stream(camera)
    start_time = pd.Timestamp.now()
    video_images = list()
    cash = list()
    while True:
        input_img = next(video_stream)
        object_dict = detect_objects(input_img)
        img_with_caption = add_caption_to_image(input_img, object_dict)
        cash.append(img_with_caption)
        while len(cash) > 100:
            del cash[0]
        location_boxes = [obj['box'] for obj in object_dict[TARGET]]
        target_loc = [obj['center'] for obj in object_dict[LOCATION]]
        detection = any([any([is_in_box(target, location_box) for location_box in location_boxes] for target in target_loc)])
        ser[pd.Timestamp.now()] = detection
        if any(ser[-1] != ser.last('5s')):
            video_images += cash[-5:]
            cash = list()

        if pd.Timestamp.now() - start_time > pd.Timedelta('1min'):
            break
        # #
        # img_with_caption2 = caption(input_img)
        cv2.imshow('image', img_with_caption)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    ser.to_csv('output.csv')
    imageio.mimsave('output.gif', video_images)


if __name__ == "__main__":
    main()
