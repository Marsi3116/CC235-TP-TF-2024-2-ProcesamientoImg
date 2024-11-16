import cv2
import os
import pickle
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN  # Usar MTCNN para la detección

with open('face_recognizer_svm.pkl', 'rb') as f:
    clf = pickle.load(f)

model = FaceNet()

dataPath = 'C:/Users/Marsi/Desktop/CURSOS/face_recognition/Data'
peopleList = os.listdir(dataPath)
peopleNames = {i: name for i, name in enumerate(peopleList)}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#detector MTCNN
detector = MTCNN()

CONFIDENCE_THRESHOLD = 9.30

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
            confidence_text = f"Confianza: {np.max(confidence):.2f}"
        else:
            label = peopleNames.get(prediction[0], "Desconocido")
            color = (0, 255, 0)
            confidence_text = f"Confianza: {np.max(confidence):.2f}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f'{label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, confidence_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
