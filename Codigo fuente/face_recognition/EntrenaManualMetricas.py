import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

#Función para extraer características HOG
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

#Función para detectar bordes utilizando Canny
def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    return edges

#Función para segmentar la piel usando el espacio HSV
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

X_train, X_test, y_train, y_test = train_test_split(facesData, labels, test_size=0.3, random_state=42)

clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

with open('face_recognizer_svm_MANUALMETRICAS.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Modelo SVM Guardado")

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=peopleList, yticklabels=peopleList)
plt.title("Matriz de Confusión")
plt.ylabel("Clase Real")
plt.xlabel("Clase Predicha")
plt.show()

print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=peopleList))
