import cv2
import os
import numpy as np
from keras_facenet import FaceNet
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

dataPath = 'C:/Users/Marsi/Desktop/CURSOS/face_recognition/Data'
peopleList = os.listdir(dataPath)
print('Lista de Personas: ', peopleList)

model = FaceNet()

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)
    print('Leyendo imagenes de', nameDir)

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)

        image = cv2.imread(os.path.join(personPath, fileName))
        face = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        embedding = model.embeddings([face])[0]  # La salida es un vector de 128 dimensiones

        facesData.append(embedding)
        labels.append(label)

    label += 1

facesData = np.array(facesData)
labels = np.array(labels)

facesData_train, facesData_test, labels_train, labels_test = train_test_split(facesData, labels, test_size=0.2, random_state=42)

clf = SVC(kernel='linear', probability=True)
clf.fit(facesData_train, labels_train)

predictions = clf.predict(facesData_test)

accuracy = accuracy_score(labels_test, predictions)
print(f"Precisión: {accuracy * 100:.2f}%")

print("\nReporte de clasificación:\n", classification_report(labels_test, predictions))

conf_matrix = confusion_matrix(labels_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.show()

with open('face_recognizer_svm_METRICAS.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Modelo SVM Guardado")
