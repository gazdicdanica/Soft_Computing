import sys
import numpy as np
import cv2 # OpenCV
from scipy import ndimage
import os
from PIL import ImageFont, ImageDraw, Image
import pandas as pd
from scipy.spatial import distance

# keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

from tensorflow.keras.optimizers import SGD
from sklearn.cluster import KMeans


alphabet = "a á b c d e é f g h i í j k l m n o ó ö ő p q r s t u ú ü ű v w x y z"


def generate_dataset(data_path):
    image = np.zeros((150,3100,3),np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    font_path = "ariblk.ttf"  
    font_size = 90
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(pil_image)
    
    draw.text((10,10), alphabet, font=font)

    
    img = np.asarray(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(data_path + 'dataset.png', img)



if __name__ == "__main__":
    generate_dataset("data/")
