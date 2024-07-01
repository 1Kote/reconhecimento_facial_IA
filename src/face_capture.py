import cv2
import numpy as np
import os
import re

from helper_functions import resize_video

max_width = 800  # Largura máxima do vídeo processado

max_samples = 20  # Número máximo de fotos a serem tiradas
starting_sample_number = 0  # Número inicial de amostras

# Função para padronizar o nome da pessoa, removendo caracteres especiais e substituindo espaços por sublinhados
def parse_name(name):
    name = re.sub(r"[^\w\s]", '', name)  # Remove caracteres não alfanuméricos
    name = re.sub(r"\s+", '_', name)  # Substitui espaços por sublinhados
    return name

# Cria os diretórios onde as fotos serão salvas, se ainda não existirem
def create_folders(final_path, final_path_full):
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    if not os.path.exists(final_path_full):
        os.makedirs(final_path_full)

# Detecta rosto usando SSD e retorna a região do rosto e o quadro com a detecção desenhada
def detect_face_ssd(network, orig_frame, show_conf=True, conf_min=0.7):
    frame = orig_frame.copy()  # Faz uma cópia do quadro original
    (h, w) = frame.shape[:2]  # Obtém a altura e largura do quadro
    # Cria um blob a partir da imagem redimensionada para 300x300 pixels
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    network.setInput(blob)  # Define o blob como entrada para a rede neural
    detections = network.forward()  # Executa a detecção

    face_roi = None  # Inicializa a região de interesse do rosto
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Obtém a confiança da detecção
        if confidence > conf_min:  # Verifica se a confiança é maior que o mínimo
            # Calcula as coordenadas da caixa delimitadora
            bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = bbox.astype("int")

            # Verifica se as coordenadas estão dentro dos limites do quadro
            if (start_x < 0 or start_y < 0 or end_x > w or end_y > h):
                continue

            # Extrai a região do rosto do quadro original
            face_roi = orig_frame[start_y:end_y, start_x:end_x]
            face_roi = cv2.resize(face_roi, (90, 120))  # Redimensiona a região do rosto
            # Desenha a caixa delimitadora no quadro
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            if show_conf:  # Exibe a confiança da detecção
                text_conf = "{:.2f}%".format(confidence * 100)
                cv2.putText(frame, text_conf, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return face_roi, frame  # Retorna a região do rosto e o quadro com a detecção

# Carrega a rede neural pré-treinada SSD a partir dos arquivos de configuração e pesos
network = cv2.dnn.readNetFromCaffe("ia_models/deploy.prototxt.txt", "ia_models/res10_300x300_ssd_iter_140000.caffemodel")

# Inicializa a captura de vídeo a partir da webcam (índice 0)
cam = cv2.VideoCapture(0)

# Diretórios onde os rostos recortados e as fotos completas serão armazenados
folder_faces = "dataset/"
folder_full = "dataset_full/"

# Solicita o nome da pessoa ao usuário, para que os rostos sejam salvos na subpasta correspondente
person_name = input('Digite seu nome: ')
person_name = parse_name(person_name)  # Padroniza o nome da pessoa

# Define os caminhos finais onde as fotos serão salvas
final_path = os.path.sep.join([folder_faces, person_name])
final_path_full = os.path.sep.join([folder_full, person_name])
print("Todas as fotos serão salvas em {}".format(final_path))

# Cria os diretórios se ainda não existirem
create_folders(final_path, final_path_full)

sample = 0  # Inicializa o contador de amostras
# Loop para processar cada quadro do vídeo
while(True):
    ret, frame = cam.read()  # Lê um quadro da captura de vídeo

    # Redimensiona o vídeo se uma largura máxima for especificada
    if max_width is not None:
        video_width, video_height = resize_video(frame.shape[1], frame.shape[0], max_width)
        frame = cv2.resize(frame, (video_width, video_height))

    # Detecta rostos usando SSD:
    face_roi, processed_frame = detect_face_ssd(network, frame)

    # Se um rosto for detectado, permite capturar a imagem ao pressionar 'q'
    if face_roi is not None:
        # Aguardar a tecla "q" ser pressionada para capturar a imagem
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sample += 1  # Incrementa o contador de amostras
            # Define o número da amostra considerando o número inicial
            photo_sample = sample + starting_sample_number - 1 if starting_sample_number > 0 else sample
            image_name = person_name + "." + str(photo_sample) + ".jpg"
            # Salva o rosto recortado (ROI)
            cv2.imwrite(final_path + "/" + image_name, face_roi)
            # Salva a imagem completa
            cv2.imwrite(final_path_full + "/" + image_name, frame)
            print("=> foto " + str(sample))  # Exibe a contagem de fotos

            cv2.imshow("face", face_roi)  # Mostra a imagem do rosto capturado

            # cv2.waitKey(500)  # (Comentado: pode ser usado para pausar entre capturas)

    cv2.imshow("Capturando rosto", processed_frame)  # Mostra o quadro processado
    cv2.waitKey(1)  # Aguarda por 1 milissegundo antes de continuar

    if sample >= max_samples:  # Verifica se o número máximo de amostras foi atingido
        break

print("Concluído!")
cam.release()  # Libera a captura de vídeo
cv2.destroyAllWindows()  # Fecha todas as janelas do OpenCV
