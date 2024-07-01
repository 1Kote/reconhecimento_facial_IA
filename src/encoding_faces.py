import cv2
import numpy as np
import os
from PIL import Image
import pickle
import dlib
import sys
import imutils
import face_recognition

# Caminho onde as imagens de rosto estão localizadas
training_path = 'dataset/' 

# Nome do arquivo pickle onde serão salvas as codificações
pickle_filename = "encodings/face_encodings_custom.pickle"  

def load_encodings(path_dataset):
  list_encodings = []
  list_names = []

  # Obtém a lista de subdiretórios (um para cada pessoa)
  subdirs = [os.path.join(path_dataset, f) for f in os.listdir(path_dataset)]

  for subdir in subdirs:
    # Pega o nome do subdiretório (que é nomeado com o nome da pessoa)
    name = subdir.split(os.path.sep)[-1]  
    # Lista de caminhos para as imagens no subdiretório, ignorando arquivos ocultos
    images_list = [os.path.join(subdir, f) for f in os.listdir(subdir) if not os.path.basename(f).startswith(".")]
    
    for image_path in images_list:
      # Carrega a imagem
      img = cv2.imread(image_path)
      # Converte a imagem de BGR para RGB
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      print(name + " <-- " + image_path)

      # Obter a localização do rosto
      face_roi = face_recognition.face_locations(img, model="cnn")

      # Exibir saída do console
      if len(face_roi) > 0:
          (start_y, end_x, end_y, start_x) = face_roi[0]
          roi = img[start_y:end_y, start_x:end_x]
          roi = imutils.resize(roi, width=100)
          cv2.imshow('face', cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
          cv2.waitKey(1)  # Atualizar janela de visualização.

      # Obter a codificação do rosto
      img_encoding = face_recognition.face_encodings(img, face_roi)
      if (len(img_encoding) > 0):
        # Armazenar o nome do arquivo e a codificação do arquivo
        img_encoding = img_encoding[0]
        list_encodings.append(img_encoding)
        list_names.append(name)
      else:
        print("Não foi possível codificar o rosto da imagem => {}".format(image_path)) # Provavelmente porque não encontrou nenhum rosto na imagem

  return list_encodings, list_names

# Carregar as codificações e nomes
list_encodings, list_names = load_encodings(training_path)

print(len(list_encodings))
print(list_names)

# Armazenar as codificações e nomes em um arquivo pickle
encodings_data = {"encodings": list_encodings, "names": list_names}
with open(pickle_filename, "wb") as f:
  pickle.dump(encodings_data, f)

print('\n')
print('Rostos codificados com sucesso!')
