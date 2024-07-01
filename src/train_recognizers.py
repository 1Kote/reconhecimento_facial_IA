import cv2
import numpy as np
import os
from PIL import Image
import pickle

# Caminho do conjunto de dados de treinamento
training_path = 'dataset/'

# Função para obter dados das imagens de treinamento
def get_image_data(path_train):
    subdirs = [os.path.join(path_train, f) for f in os.listdir(path_train) if os.path.isdir(os.path.join(path_train, f))]
    faces = []
    ids = []
    face_names = {}
    id = 1  # ID inicial para as faces

    print("Loading faces from training set...")
    for subdir in subdirs:
        name = os.path.split(subdir)[1]  # Nome da pessoa (diretório)
        images_list = [os.path.join(subdir, f) for f in os.listdir(subdir) if not f.startswith(".")]

        for path in images_list:
            image = Image.open(path).convert('L')  # Abre a imagem e converte para escala de cinza
            face = np.array(image, 'uint8')  # Converte a imagem para um array numpy
            face = cv2.resize(face, (90, 120))  # Redimensiona a imagem do rosto
            print(str(id) + " <-- " + path)
            ids.append(id)  # Adiciona o ID à lista de IDs
            faces.append(face)  # Adiciona a face à lista de faces
            cv2.imshow("Training faces...", face)
            cv2.waitKey(50)

        if name not in face_names:
            face_names[name] = id  # Associa o nome ao ID
            id += 1

    return np.array(ids), faces, face_names
