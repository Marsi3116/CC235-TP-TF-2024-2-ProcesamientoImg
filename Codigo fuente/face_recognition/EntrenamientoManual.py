import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
import os
import pickle

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    return edges

def segment_skin(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin = cv2.bitwise_and(image, image, mask=skin_mask)
    return skin

dataPath = 'C:/Users/Marsi/Desktop/CURSOS/face_recognition/Data'
peopleList = os.listdir(dataPath)

facesData = []
labels = []

label = 0
for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)

    for fileName in os.listdir(personPath):
        image = cv2.imread(os.path.join(personPath, fileName))

        print(f"Procesando imagen {fileName}...")

        edges = detect_edges(image)

        skin_image = segment_skin(image)

        skin_image_resized = cv2.resize(skin_image, (64, 64))

        features = extract_features(skin_image_resized)

        facesData.append(features)
        labels.append(label)

    label += 1

facesData = np.array(facesData)
labels = np.array(labels)

clf = SVC(kernel='linear', probability=True)
clf.fit(facesData, labels)

with open('face_recognizer_svm_MANUAL.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Modelo SVM Guardado")
