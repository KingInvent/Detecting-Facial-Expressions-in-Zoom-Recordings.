import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import Facial_Expression.transforms as transforms
from skimage import io
from skimage.transform import resize
from Facial_Expression.models import *
import math
from typing import List, Mapping, Optional, Tuple, Union


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# Mediapipe facial detection initialization
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

# Loading of Facial expression model
net = VGG('VGG19')
checkpoint = torch.load(os.path.join('Facial_Expression/FER2013_VGG19', 'PrivateTest_model.t7'), map_location='cpu')
net.load_state_dict(checkpoint['net'])
net.cpu()
net.eval()

# Replace path with Video Path to be processed
cap = cv2.VideoCapture("/Users/Daniela/PycharmProjects/pythonProject1/video_zoom_class/00002.mp4")
with mp_face_detection.FaceDetection(
        min_detection_confidence=0.3, model_selection=1) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_org = image.copy()

        image_rows, image_cols, _ = image.shape

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
                relative_bounding_box = detection.location_data.relative_bounding_box

                # Conversion of medipipe coordinates to normalized coordinates
                rect_start_point = _normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                    image_rows)
                rect_end_point = _normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin + relative_bounding_box.width,
                    relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
                    image_rows)

                if rect_start_point and rect_end_point:

                    # Crop image to specified area of BBox and save the image
                    crop_img = image_org[rect_start_point[1] - 10:rect_end_point[1] + 10,
                               rect_start_point[0] - 10:rect_end_point[0] + 10]
                    cv2.imwrite("temp.jpg", crop_img)

                    raw_img = io.imread('temp.jpg')
                    gray = rgb2gray(raw_img)
                    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

                    img = gray[:, :, np.newaxis]
                    img = np.concatenate((img, img, img), axis=2)
                    img = Image.fromarray(img)
                    inputs = transform_test(img)

                    ncrops, c, h, w = np.shape(inputs)

                    inputs = inputs.view(-1, c, h, w)
                    inputs = inputs.cpu()
                    inputs = Variable(inputs, volatile=True)
                    outputs = net(inputs)
                    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

                    score = F.softmax(outputs_avg)
                    _, predicted = torch.max(outputs_avg.data, 0)
                    expression = class_names[int(predicted.cpu())]

                    cv2.putText(image, expression, (rect_start_point[0], rect_start_point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow('Facial Expression Detection', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
