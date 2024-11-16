import cv2
import os
import numpy as np
from keras_facenet import FaceNet

dataPath = 'C:/Users/Marsi/Desktop/CURSOS/face_recognition/Data'
peopleList = os.listdir(dataPath)
print('Lista de Personas: ', peopleList)

model = FaceNet()

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo imagenes de', nameDir)

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)

        # Leer la imagen y extraer el rostro
        image = cv2.imread(personPath + '/' + fileName)
        face = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extraer el embedding facial con FaceNet
        embedding = model.embeddings([face])[0]  # La salida es un vector de 128 dimensiones

        # Guardar el embedding y la etiqueta
        facesData.append(embedding)
        labels.append(label)

    label = label + 1

facesData = np.array(facesData)
labels = np.array(labels)

from sklearn.svm import SVC

print('Entrenando el clasificador...')
clf = SVC(kernel='linear', probability=True)
clf.fit(facesData, labels)

import pickle
with open('face_recognizer_svm.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Modelo SVM Guardado")
