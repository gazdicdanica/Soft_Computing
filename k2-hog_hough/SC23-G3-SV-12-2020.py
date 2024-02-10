
import os
import numpy as np
import pandas as pd
import sys
import cv2 # OpenCV
from sklearn.svm import SVC # SVM klasifikator
from sklearn.model_selection import train_test_split

hog = None
clf_svm = None

actual = {}

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def load_hog():
    global hog
    nbins = 9 # broj binova
    img = (60,120)
    cell_size = (8, 8) # broj piksela po celiji
    block_size = (3, 3) # broj celija po bloku

    hog = cv2.HOGDescriptor(_winSize=(img[1] // cell_size[1] * cell_size[1], 
                                    img[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

def train(data_path):
    global hog
    global clf_svm

    train_dir = data_path + 'pictures/'

    pos_imgs = []
    neg_imgs = []

    for img_name in os.listdir(train_dir):
        img_path = os.path.join(train_dir, img_name)
        img = load_image(img_path)
        if 'p_' in img_name:
            pos_imgs.append(img)
        elif 'n_' in img_name:
            neg_imgs.append(img)

    pos_features = []
    neg_features = []
    labels = []

    for img in pos_imgs:
        pos_features.append(hog.compute(img))
        labels.append(1)

    for img in neg_imgs:
        neg_features.append(hog.compute(img))
        labels.append(0)

    pos_features = np.array(pos_features)
    neg_features = np.array(neg_features)
    x = np.vstack((pos_features, neg_features))
    y = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    clf_svm = SVC(kernel='linear', probability=True) 
    clf_svm.fit(x_train, y_train)
    y_train_pred = clf_svm.predict(x_train)
    y_test_pred = clf_svm.predict(x_test)


def detect_line(img):
    # detekcija koordinata linije koristeci Hough transformaciju
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (5,5), 0)
    edges_img = cv2.Canny(blurred, 20, 150, apertureSize=3)
    
    # minimalna duzina linije
    min_line_length = 200
    
    # Hough transformacija
    lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=np.pi/180, threshold=10, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=20)
    
    ret = lines[0]
    for line in lines:
        if line[0][0] == line[0][2]:
            ret = line
            break
    
    x1 = ret[0][0]
    y1 = 100
    x2 = ret[0][2]
    y2 = 450
    
    return (x1, y1, x2, y2)


def classify_window(window):
    global hog 
    global clf_svm
    features = hog.compute(window).reshape(1, -1)
    return clf_svm.predict_proba(features)[0][1]

def process_image(image, step_size, window_size=(60, 120)):
    best_score = 0
    best_window = None
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            this_window = (y, x) # zbog formata rezultata
            window = image[y:y+window_size[0], x:x+window_size[1]]
            if window.shape == (window_size[0], window_size[1]):
                score = classify_window(window)
                if score > best_score:
                    best_score = score
                    best_window = this_window
    return best_score, best_window


def process_video(video_path):
    # procesiranje jednog videa
    cars_detected = 0
    line_x = 0
    
    # ucitavanje videa
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num) # indeksiranje frejmova

    last_frame = 1
    
    # analiza videa frejm po frejm
    while True:
        frame_num += 1
        
        grabbed, frame = cap.read()

        # ako frejm nije zahvacen
        if not grabbed:
            break
        
        frame = frame[100:450, 650:1300]
        
        if frame_num == 1: # ako je prvi frejm, detektuj liniju

            line_coords = detect_line(frame)
            
            line_x = line_coords[0]

        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score, window = process_image(frame_gray, step_size=20)

        if(window != (0,0) and score >= 0.99):
            start_x = window[1]
            start_y = window[0]

            if(start_x >= line_x - 10 and 220 > start_y > 100 and frame_num - last_frame > 15):
                last_frame = frame_num
                cars_detected += 1

                    

    cap.release()
    return cars_detected


def read_csv(path):
    global actual
    df = pd.read_csv(path)
    for i, row in df.iterrows():
        actual[row["Naziv_videa"]] = row["Broj_prelaza"]



if __name__ == "__main__":

    data_path = sys.argv[1]

    load_hog()
    train(data_path)

    read_csv(data_path + "counts.csv")

    abs_diff = []

    for video_name in os.listdir(data_path + "videos"):
        video_path = os.path.join(data_path + "videos", video_name)
        cars_detected = process_video(video_path)
        abs_diff.append(abs(cars_detected - actual[video_name]))
        print(video_name + "-" + str(actual[video_name]) + "-" + str(cars_detected)	)

    print("MAE: ", np.mean(abs_diff))
