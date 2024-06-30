import cv2
import numpy as np
import os
import re

from helper_functions import resize_video

# Definir o detector de faces
detector = "ssd"
max_width = 800

max_samples = 20    # Para controlar quantas fotos serão tiradas.
starting_sample_number = 0 

# Função para analisar o nome da pessoa, que será o nome do subdiretório
# (pois é recomendado que um diretório não tenha espaços ou outros caracteres especiais)
def parse_name(name):
    name = re.sub(r"[^\w\s]", '', name) # Remove todos os caracteres não alfanuméricos (exceto números e letras)
    name = re.sub(r"\s+", '_', name)    # Substitui todos os espaços por um único sublinhado
    return name

# Cria a pasta final onde as fotos serão salvas (se o caminho ainda não existir)
def create_folders(final_path, final_path_full):
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    if not os.path.exists(final_path_full):
        os.makedirs(final_path_full)

# Retorna o rosto detectado usando SSD
def detect_face_ssd(network, orig_frame, show_conf=True, conf_min=0.7):
    frame = orig_frame.copy()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    network.setInput(blob)
    detections = network.forward()

    face_roi = None
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_min:
            bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = bbox.astype("int")

            if (start_x < 0 or start_y < 0 or end_x > w or end_y > h):
                continue

            face_roi = orig_frame[start_y:end_y, start_x:end_x]
            face_roi = cv2.resize(face_roi, (90, 120))
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)  # Desenha a caixa delimitadora
            if show_conf:
                text_conf = "{:.2f}%".format(confidence * 100)
                cv2.putText(frame, text_conf, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return face_roi, frame

network = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Objeto de captura de vídeo
cam = cv2.VideoCapture(0)

folder_faces = "dataset/"      # Onde os rostos recortados serão armazenados
folder_full = "dataset_full/"  # Onde serão armazenadas as fotos completas

# O usuário precisa digitar seu nome, para que os rostos sejam salvos na subpasta adequada
# Poderíamos pedir para digitar um número de ID também (que o rosto será associado com)
person_name = input('Digite seu nome: ')
person_name = parse_name(person_name)

# Junta o caminho (diretório do dataset + subpasta)
final_path = os.path.sep.join([folder_faces, person_name])
final_path_full = os.path.sep.join([folder_full, person_name])
print("Todas as fotos serão salvas em {}".format(final_path))

# Você poderia criar manualmente as pastas ou executar a função/código abaixo (ele verificará se existe. Se não, então criará a pasta)
create_folders(final_path, final_path_full)