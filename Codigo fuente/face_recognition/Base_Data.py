import cv2
import os
import imutils
import numpy as np
from mtcnn import MTCNN

# Nombre de la persona y rutas de datos
personName = 'Hansell Figueroa'
dataPath = 'C:/Users/Marsi/Desktop/CURSOS/face_recognition/Data'
personPath = dataPath + '/' + personName

if not os.path.exists(personPath):
    print('Carpeta Creada: ', personPath)
    os.makedirs(personPath)

detector = MTCNN()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0

if not cap.isOpened():
    print("Error: No se puede acceder a la cámara.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se recibió ningún frame de la cámara.")
            break

        frame_resized = imutils.resize(frame, width=320)

        faces = detector.detect_faces(frame_resized)

        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

            rostro = frame_resized[y:y+h, x:x+w]

            if rostro.size != 0:
                rostro = cv2.resize(rostro, (160, 160), interpolation=cv2.INTER_CUBIC)

                cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
                count += 1

        cv2.imshow('frame', frame_resized)

        k = cv2.waitKey(1)
        if k == 27 or count >= 300:
            break

cap.release()
cv2.destroyAllWindows()
