import cv2
import numpy as np
from skimage.feature import hog
import pickle
import os

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

with open('face_recognizer_svm_MANUAL.pkl', 'rb') as f:
    clf = pickle.load(f)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

dataPath = 'C:/Users/Marsi/Desktop/CURSOS/face_recognition/Data'
peopleList = os.listdir(dataPath)
peopleNames = {i: name for i, name in enumerate(peopleList)}

CONFIDENCE_THRESHOLD = 14.30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]

        face_resized = cv2.resize(face, (64, 64))

        features = extract_features(face_resized)

        prediction = clf.predict([features])
        confidence = clf.decision_function([features])

        if np.max(confidence) < CONFIDENCE_THRESHOLD:
            label = "Desconocido"
            confidence_text = f"Confianza: {np.max(confidence):.2f}"
        else:
            label = peopleNames.get(prediction[0], "Desconocido")
            confidence_text = f"Confianza: {np.max(confidence):.2f}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, confidence_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow('Reconocimiento Facial en Vivo', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
