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
    output_names = [NET.getLayerNames()[layer_num - 1] for layer_num in unconnected.reshape(-1)]
    large, medium, small = NET.forward(output_names)
    all_outputs = np.vstack((large, medium, small))
    objs = all_outputs[all_outputs[:, 4] > 0.1]  # Getting all outputs with certainty above 0.1
    boxes = [output_coordinates_to_box_coordinates(*obj[:4], *img.shape[:2]) for obj in objs]
    confidences = [float(obj[4]) for obj in objs]
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    objects_dict = defaultdict(list)
    for i in indices.reshape(-1):
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


def main():

    camera = cv2.VideoCapture(0)
    start_time = pd.Timestamp.now()
    video_images = list()
    cash = list()
    while True:

        ret, frame = camera.read()
        if not ret:
            break


        cv2.imshow('camera', frame)

        # object_dict = detect_objects(frame)
        # if 'dog' in object_dict:
        #     dog_locations = object_dict['dog']
        #     img_with_caption = add_caption_to_image(frame, object_dict)


        #cash.append(img_with_caption)
        # while len(cash) > 100:
        #     del cash[0]
        # location_boxes = [obj['box'] for obj in object_dict[TARGET]]
        # target_loc = [obj['center'] for obj in object_dict[LOCATION]]
        # detection = any([any([is_in_box(target, location_box) for location_box in location_boxes] for target in target_loc)])
        # ser[pd.Timestamp.now()] = detection
        # if any(ser[-1] != ser.last('5s')):
        #     video_images += cash[-5:]
        #     cash = list()
        #
        # if pd.Timestamp.now() - start_time > pd.Timedelta('1min'):
        #     break
        # #
        # img_with_caption2 = caption(input_img)
        # cv2.imshow('image', img_with_caption)
        key = cv2.waitKey(30)
        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
    #imageio.mimsave('output.gif', video_images)


if __name__ == "__main__":
    main()
