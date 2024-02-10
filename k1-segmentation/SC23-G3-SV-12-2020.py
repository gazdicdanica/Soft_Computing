#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import cv2 
import sys
import pandas as pd

CORRECT_ANSWERS = []
ABS_DIFF = []

def get_name_and_index(image_path):
    img_name= image_path.split("/")[-1]
    img_index = img_name.split("_")[1]
    img_index = img_index[1:]
    return img_name, img_index


def process(image_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[55:995, 300:900]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)
    img_bin = cv2.bitwise_not(img_bin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    closing = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel, iterations = 4) 
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations = 7)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    closing2 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations = 3) 

    sure_bg = cv2.dilate(closing2, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)

    ret, sure_fg = cv2.threshold(dist_transform, 0.33 * dist_transform.max(), 255, 0) 
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    unique_colours = {x for l in markers for x in l}
    number_of_dittos = len(unique_colours) - 2
    
    img_name, img_index = get_name_and_index(image_path)
    img_index = int(img_index)
    img_index  -= 1

    print(img_name + "-" + str(CORRECT_ANSWERS[img_index]) + "-" + str(number_of_dittos))
    ABS_DIFF.append(abs(CORRECT_ANSWERS[img_index] - number_of_dittos))


if __name__ == "__main__":

    data = pd.read_csv('ditto_count.csv')

    for index, row in data.iterrows():
        CORRECT_ANSWERS.append(row['Broj ditto-a'])

    data_path = sys.argv[1] 
    for filename in os.listdir(data_path):
        image_path = os.path.join(data_path, filename)
        process(image_path)

    print("MAE:",str(np.mean(ABS_DIFF)))