import os
import shutil
import PIL
import numpy as np
import cv2
import zipfile
from PIL import Image
import tensorflow as tf
import pandas as pd
import pathlib
import gradio as gr

import matplotlib.pyplot as plt
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def dice_score(y_true, y_pred, threshold=0.5):
    smooth = 1.0
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), dtype=tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred >= threshold, [-1]), dtype=tf.float32)
    intersection = y_true_f * y_pred_f
    score = (2.0 * tf.reduce_sum(intersection) + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )
    return score


def dice_loss(y_true, y_pred):
    loss = 1.0 - dice_score(y_true, y_pred)
    return tf.cast(loss, dtype=tf.float32)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return tf.cast(bce, dtype=tf.float32) + dice


model_path = r"model\final_4block_RMSprop_swish+leakyrelu_CGAN_2000_clahe_generator_2.h5"
model = tf.keras.models.load_model(
    model_path,
    custom_objects={"bce_dice_loss": bce_dice_loss, "dice_score": dice_score},
)


def readImages(data, typeData):
    images = []
    height = int(256)
    width = int(512)
    for img in data:
        img = cv2.imread(str(img), 0)
        img = cv2.resize(img, (width, height))
        if typeData == "m":
            img = np.where(img > 0, 1, 0)
        img = np.expand_dims(img, axis=-1)
        images.append(img)
    print("(INFO..) Read Image Done")
    return np.array(images)


def normalizeImages(images):
    normalized_images = []
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(3, 3))
    height = int(256)
    width = int(512)
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (width, height))
        img = np.expand_dims(img, axis=-1)
        img = clahe.apply(img)
        img = img.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        normalized_images.append(img)
    print("(INFO..) Normalization Image Done")
    return np.array(normalized_images)

def smooth_raster_lines(im, filterRadius, filterSize, sigma):
    smoothed = np.zeros_like(im)
    contours, hierarchy = cv2.findContours(im, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]
    for countur_idx, contour in enumerate(contours):
        len_ = len(contour) + 2 * filterRadius
        idx = len(contour) - filterRadius
        x = []
        y = []    
        for i in range(len_):
            x.append(contour[(idx + i) % len(contour)][0][0])
            y.append(contour[(idx + i) % len(contour)][0][1])
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        xFilt = cv2.GaussianBlur(x, (filterSize, filterSize), sigma, sigma)
        xFilt = [q[0] for q in xFilt]
        yFilt = cv2.GaussianBlur(y, (filterSize, filterSize), sigma, sigma)
        yFilt = [q[0] for q in yFilt]
        smoothContours = []
        smooth = []
        for i in range(filterRadius, len(contour) + filterRadius):
            smooth.append([xFilt[i], yFilt[i]])
        smoothContours = np.asarray([smooth], dtype=np.int32)
        color = (0,0,0) if hierarchy[countur_idx][3] > 0 else (255,255,255)
        cv2.drawContours(smoothed, smoothContours, 0, color, -1)
    return(smoothed)

def predict(input_image):
    preprocessed_image = normalizeImages([input_image])
    segmented_image = model.predict(preprocessed_image)
    segmented_image = (segmented_image >= 0.5).astype("int")
    segmented_image = np.reshape(segmented_image, (256, 512))
    segmented_image = (segmented_image * 255).astype("uint8")

    first_image = np.reshape(preprocessed_image, (256, 512))
    first_image = (first_image * 255).astype("uint8")
    # first_image = Image.fromarray(first_image)
    # first_image = first_image.resize((2000, 942))

    # smoothing
    # method 1
    # kernel = np.ones((2,2),np.uint8)
    # segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)

    # segmented_image = cv2.GaussianBlur(segmented_image, (5,5), 0)
    # segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # method 2
    _, segmented_image = cv2.threshold(segmented_image, 125,255, cv2.THRESH_BINARY)
    segmented_image = cv2.pyrUp(segmented_image)
    for _ in range(15):
        segmented_image = cv2.medianBlur(segmented_image, 1)
        # segmented_image = cv2.GaussianBlur(segmented_image, (3,3), 0)
    # segmented_image = cv2.GaussianBlur(segmented_image, (5,5), 0)
    segmented_image = cv2.pyrDown(segmented_image)
    # _, segmented_image = cv2.threshold(segmented_image, 200,255, cv2.THRESH_BINARY)

    # method 3
    # blur1 = cv2.blur(segmented_image, (9,9))
    # blur2 = cv2.GaussianBlur(segmented_image, (3,3),0)
    # segmented_image = cv2.absdiff(blur2,blur1)
    # _, segmented_image = cv2.threshold(segmented_image, 128, 255, cv2.THRESH_BINARY_INV)

    # contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # mask = np.zeros_like(segmented_image)
    # cv2.fillPoly(mask, contours, (255, 255, 255))
    # segmented_image = cv2.bitwise_and(segmented_image, segmented_image, mask=mask)

    # canny edge detection
    canny = cv2.Canny(segmented_image, 100, 200)

    # # contour approach
    # boundary_image = np.copy(segmented_image)
    # contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     perimeter = cv2.arcLength(contour, True)
    #     approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    #     cv2.polylines(boundary_image, [approx], True, (0, 0, 255), 2)


    # dilated approach
    kernel = np.ones((1, 1), np.uint8)
    dilated_image = cv2.dilate(canny, kernel, iterations=2)
    background_image = Image.fromarray(input_image)
    background_image = background_image.resize((512, 256))
    background_image = np.array(background_image)
    foreground = np.zeros_like(background_image)
    foreground[dilated_image == 255] = [255, 0, 0]
    canny_line = cv2.addWeighted(background_image, 0.7, foreground, 1, 0)
    canny_line = Image.fromarray(canny_line)
    canny_line = canny_line.resize((2000, 942))

    # blending
    blend_image = cv2.addWeighted(first_image, 0.7, segmented_image, 1, 0)
    blend_image = Image.fromarray(blend_image)
    blend_image = blend_image.resize((2000, 942))

    # segmented_image = Image.fromarray(segmented_image)
    # segmented_image = segmented_image.resize((2000, 942))
    # blend_image = Image.blend(first_image, segmented_image, 0.5)

    return [segmented_image, blend_image, canny_line]


output_images = [gr.Image(label=f"Output {i+1}") for i in range(3)]
print(len(output_images))

gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=output_images,
).launch(share=True)
