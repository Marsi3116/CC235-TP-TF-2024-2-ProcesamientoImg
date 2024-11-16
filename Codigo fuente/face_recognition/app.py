from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import imutils
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle

app = Flask(__name__)

#Configuración de la ruta para los datos
dataPath = 'C:/Users/Marsi/Desktop/CURSOS/face_recognition/Data'

detector = MTCNN()

with open('face_recognizer_svm.pkl', 'rb') as f:
    clf = pickle.load(f)

peopleList = os.listdir(dataPath)
peopleNames = {i: name for i, name in enumerate(peopleList)}  # Diccionario con las etiquetas como claves y nombres como valores

model = FaceNet()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        personName = request.form['name']
        personPath = os.path.join(dataPath, personName)

        if not os.path.exists(personPath):
            os.makedirs(personPath)

        #captura de la cámara
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        count = 0

        while True:
            ret, frame = cap.read()  # Leer un frame de la cámara
            if not ret:
                break

            #Redimensionar la imagen para mejorar el rendimiento
            frame_resized = imutils.resize(frame, width=320)

            #Detectar rostros usando MTCNN
            faces = detector.detect_faces(frame_resized)

            #Recorrer todas las detecciones y dibujar los bounding boxes
            for face in faces:
                x, y, w, h = face['box']
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

                rostro = frame_resized[y:y+h, x:x+w]

                if rostro.size != 0:
                    rostro = cv2.resize(rostro, (160, 160), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(os.path.join(personPath, f'rostro_{count}.jpg'), rostro)
                    count += 1

            cv2.imshow('Captura de Rostro', frame_resized)

            if cv2.waitKey(1) == 27 or count >= 300:
                break

        cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for('index'))

    return render_template('capture.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        peopleList = os.listdir(dataPath)
        labels = []
        facesData = []
        label = 0

        for nameDir in peopleList:
            personPath = os.path.join(dataPath, nameDir)

            for fileName in os.listdir(personPath):
                image = cv2.imread(os.path.join(personPath, fileName))
                face = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Extraer el embedding facial con FaceNet
                embedding = model.embeddings([face])[0]
                facesData.append(embedding)
                labels.append(label)

            label += 1

        facesData = np.array(facesData)
        labels = np.array(labels)

        from sklearn.svm import SVC
        clf = SVC(kernel='linear', probability=True)
        clf.fit(facesData, labels)

        #Guardar el clasificador
        with open('face_recognizer_svm.pkl', 'wb') as f:
            pickle.dump(clf, f)

        return redirect(url_for('index'))

    return render_template('train.html')

@app.route('/recognize', methods=['GET'])
def recognize():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    CONFIDENCE_THRESHOLD = len(peopleList) - 0.7

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)

        label = "Desconocido"
        confidence_text = "Esperando rostro..."
        color = (0, 0, 255)  # Rojo por defecto

        for face in faces:
            x, y, w, h = face['box']
            face_resized = rgb_frame[y:y + h, x:x + w]

            #Redimensionar el rostro para FaceNet
            face_resized = cv2.resize(face_resized, (160, 160))

            #Obtener el embedding facial del rostro
            face_embedding = model.embeddings([face_resized])[0]

            #Predecir la etiqueta del rostro usando el clasificador SVM
            prediction = clf.predict([face_embedding])
            confidence = clf.decision_function([face_embedding])


            #Si la confianza es baja, se clasifica como "Desconocido"
            if np.max(confidence) < CONFIDENCE_THRESHOLD:
                label = "Desconocido"
                color = (0, 0, 255)  #Rojo para desconocido
                confidence_text = f"Confianza: {np.max(confidence):.2f}"
            else:
                label = peopleNames.get(prediction[0], "Desconocido")  # Usar el nombre de la persona
                color = (0, 255, 0)  # Verde para reconocido
                confidence_text = f"Confianza: {np.max(confidence):.2f}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, confidence_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Reconocimiento Facial en Vivo', frame)

        # Salir si se presiona ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
