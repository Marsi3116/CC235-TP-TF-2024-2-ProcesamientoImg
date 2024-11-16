import cv2
import os
import pickle
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

with open('face_recognizer_svm_METRICAS.pkl', 'rb') as f:
    clf = pickle.load(f)

model = FaceNet()

dataPath = 'C:/Users/Marsi/Desktop/CURSOS/face_recognition/Data'
peopleList = os.listdir(dataPath)
peopleNames = {i: name for i, name in enumerate(peopleList)}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

detector = MTCNN()

CONFIDENCE_THRESHOLD = 9.30

all_labels = []
all_predictions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, w, h = face['box']
        face_resized = rgb_frame[y:y + h, x:x + w]

        face_resized = cv2.resize(face_resized, (160, 160))

        face_embedding = model.embeddings([face_resized])[0]

        prediction = clf.predict([face_embedding])
        confidence = clf.decision_function([face_embedding])

        if np.max(confidence) < CONFIDENCE_THRESHOLD:
            label = "Desconocido"
            color = (0, 0, 255)
        else:
            label = peopleNames.get(prediction[0], "Desconocido")
            color = (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        all_labels.append(label)
        all_predictions.append(prediction[0])

    cv2.imshow('Reconocimiento Facial', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

print("\nMétricas de Reconocimiento Facial:")
print(confusion_matrix(all_labels, all_predictions))

cap.release()
cv2.destroyAllWindows()
