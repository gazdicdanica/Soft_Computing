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

# alphabet = "a á b c d e é f g h i í j k l m n o ó ö ő p q r s t u ú ü ű v w x y z"


alphabet = ['v', 'i', 's', 'z', 'o', 'n', 't', 'l', 'á', 'r', 'a', 'e', 'g', 'é', 'u', 'b', 'd', 'y', 'k', 'm', 'c', 'f', 'n', 'j', 'ó', 'ú']
indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 25, 28, 38, 44, 46, 60, 62, 76, 78, 88, 89, 92]


expected = {}


k_means = KMeans(n_clusters=2)


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def invert(image):
    return 255-image


def dilate(image, iterations = 1):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=iterations)


def erode(image, iterations = 1):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=iterations)


def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def find_parent_letter(regions_array, contour_tuple):
    for i in range(len(regions_array)):
        region = regions_array[i]
        # x > x_c || (x + w) < (x_c + w_c)
        if region[1][0] > contour_tuple[0] or (region[1][0] + region[1][2]) < (contour_tuple[0] + contour_tuple[2] -5):
            continue
        # y < y_c
        if region[1][1] < contour_tuple[1]:
            continue
        
        return region, i
   

def select_roi(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po X osi
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour) # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if area > 100 and h < 100 and h > 30 and w > 10:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # oznaciti region pravougaonikom na originalnoj slici sa rectangle funkcijom
            region = image_bin[y:y+h+1, x:x+w+1]
            regions_array.append([resize_region(region), (x, y, w, h)])
        elif h < 30:
            ret = find_parent_letter(regions_array, (x, y, w, h))
            if ret != None:
                parent, index = ret
                region = image_bin[parent[1][1] - h : parent[1][1]+parent[1][3]+1, parent[1][0] : parent[1][0]+parent[1][2]+1]
                regions_array[index] = [resize_region(region), (parent[1][0], y, parent[1][2], parent[1][3]+h)]
    
    for region in regions_array:
        cv2.rectangle(image_orig, (region[1][0], region[1][1]), (region[1][0] + region[1][2], region[1][1] + region[1][3]), (0, 255, 0), 2)
   
    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []

    # izdvojiti sortirane parametre opisujucih pravougaonika
    # izracunati rastojanja izmedju svih susednih regiona po X osi i dodati ih u niz rastojanja
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2]) # x_next - (x_current + w_current)
        region_distances.append(distance)
    
    return image_orig, sorted_regions, region_distances


def scale_to_range(image):
    return image/255


def matrix_to_vector(image):
    return image.flatten()



def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann


def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32) # dati ulaz
    y_train = np.array(y_train, np.float32) # zeljeni izlazi na date ulaze
    
   
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    
    return ann


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result_with_spaces(outputs, alphabet, k_means):
    # odredjivanje indeksa grupe koja odgovara rastojanju izmedju reci
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    # iterativno dodavanje prepoznatih elemenata
    # dodavanje space karaktera ako je rastojanje izmedju dva slova odgovara rastojanju izmedju reci
    
    for idx, output in enumerate(outputs[1:, :]):
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[winner(output)]
   
    return result

def display_result(outputs, alphabet):
    result = alphabet[winner(outputs[0])]
    
    for idx, output in enumerate(outputs[1:, :]):
        result += alphabet[winner(output)]
   
    return result


def process(image):
    img = image_bin(image_gray(image))
    img = erode(dilate(img))
    return img


# def generate_dataset(data_path):
#     image = np.zeros((150,3100,3),np.uint8)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     pil_image = Image.fromarray(image)

#     font_path = "ariblk.ttf"  
#     font_size = 90
#     font = ImageFont.truetype(font_path, font_size)
#     draw = ImageDraw.Draw(pil_image)
    
#     draw.text((10,10), alphabet, font=font)

    
#     img = np.asarray(pil_image)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#     cv2.imwrite(data_path + 'dataset.png', img)


def train(data):
    global alphabet

    # data = load_image(data_path + "dataset.png")
    # data = invert(data)
    # processed = process(data)

    # selected_regions, letters, region_distances = select_roi(data.copy(), processed)

    # alphabet = alphabet.split(" ")
   
    inputs = prepare_for_ann(data)
    outputs = convert_output(alphabet)
    ann = create_ann(output_size=len(data))
    ann = train_ann(ann, inputs, outputs, epochs=1000)
    return ann


def read_csv(path):
    global expected
    df = pd.read_csv(path)
    for i, row in df.iterrows():
        expected[row["file"]] = row["text"]


def process_image(img_path, ann):
    global k_means
    img = load_image(img_path)[150:260, 250:850]
    processed = process(img)

    selected_regions, letters, region_distances = select_roi(img.copy(), processed)
    distances = np.array(region_distances).reshape(len(region_distances), 1)
    d = [x for x in distances if x[0] > 20]
    if(len(d) != 0):
        k_means.fit(distances)
        
    inputs = prepare_for_ann(letters)
    results = ann.predict(np.array(inputs, np.float32))
    return display_result_with_spaces(results, alphabet, k_means) if len(d) != 0 else display_result(results, alphabet)
    

def calculate_hammings(expected, actual):
    if len(expected) == len(actual):
        return distance.hamming(list(expected), list(actual)) * len(actual)
    sub_len = len(expected) if len(expected) < len(actual) else len(actual)

    dis = 0
    for i in range(sub_len):
        if expected[i] != actual[i]:
            dis += 1

    dis += abs(len(expected) - len(actual))

    return dis
    


def generate_data(data_path):
    global indexes
    img_list = []
    for img_name in os.listdir(data_path + "pictures"):
        img_path = os.path.join(data_path + "pictures", img_name)
        img = load_image(img_path)[150:260, 250:850]
        img_list.append(process(img))
    concat = cv2.hconcat(img_list)

    selected_regions, letters, region_distances = select_roi(img.copy(), concat)

    return [letters[i] for i in indexes]

    


if __name__ == "__main__":
    data_path = sys.argv[1]

    read_csv(data_path + "res.csv")

    # generate_dataset(data_path)
    data = generate_data(data_path)

    ann = train(data)

    hammings = 0

    for img_name in os.listdir(data_path + "pictures"):    
        img_path = os.path.join(data_path + "pictures", img_name)
        result = process_image(img_path, ann)
        print(img_name + "-" + expected[img_name] + "-" + result)
        
        hammings += calculate_hammings(expected[img_name], result)

    print(hammings)

        