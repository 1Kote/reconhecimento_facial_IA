import cv2
import numpy as np
import pickle
import face_recognition
import csv
from datetime import datetime, timedelta
from helper_functions import resize_video
import os

# Nome do arquivo pickle onde as codificações faciais estão armazenadas
pickle_name = "encodings/face_encodings_custom.pickle"
# Nome do arquivo CSV para registro de presença
attendance_file = "records/attendance.csv"

max_width = 800
class_duration_minutes = 50  # Duração total da aula em minutos
min_presence_percentage = 0.8  # Porcentagem mínima de presença (80%)

# Carrega as codificações faciais do arquivo pickle
data_encoding = pickle.loads(open(pickle_name, "rb").read())
list_encodings = data_encoding["encodings"]
list_names = data_encoding["names"]

def recognize_faces(image, list_encodings, list_names, resizing=0.25, tolerance=0.6):
    image = cv2.resize(image, (0, 0), fx=resizing, fy=resizing)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(img_rgb)
    face_encodings = face_recognition.face_encodings(img_rgb, face_locations)
    
    if len(face_encodings) == 0:
        return [], [], []

    # Encontrar a maior face (a mais próxima)
    largest_face_index = 0
    largest_face_area = 0
    for i, (top, right, bottom, left) in enumerate(face_locations):
        area = (right - left) * (bottom - top)
        if area > largest_face_area:
            largest_face_area = area
            largest_face_index = i

    face_names = []
    conf_values = []

    encoding = face_encodings[largest_face_index]
    matches = face_recognition.compare_faces(list_encodings, encoding, tolerance=tolerance)
    name = "Not identified"
    
    face_distances = face_recognition.face_distance(list_encodings, encoding)
    best_match_index = np.argmin(face_distances)
    
    if matches[best_match_index]:
        name = list_names[best_match_index]
    
    face_names.append(name)
    conf_values.append(face_distances[best_match_index])

    # Ajusta as coordenadas dos rostos de acordo com o redimensionamento da imagem
    face_locations = np.array(face_locations)
    face_locations = face_locations / resizing

    return [face_locations[largest_face_index].astype(int)], face_names, conf_values

# Função para exibir os rostos reconhecidos na imagem
def show_recognition(frame, face_locations, face_names, conf_values):
    for face_loc, name, conf in zip(face_locations, face_names, conf_values):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        conf = "{:.8f}".format(conf)
        # Desenha o nome e a caixa delimitadora ao redor do rosto
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 255, 0), 4)
        if name != "Not identified":
            cv2.putText(frame, conf, (x1, y2 + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (20, 255, 0), 1, lineType=cv2.LINE_AA)
    return frame

# Função para registrar a presença em um arquivo CSV
def register_attendance(name, entry_time, exit_time):
    total_time = exit_time - entry_time
    total_minutes = total_time.total_seconds() / 60
    presence_percentage = (total_minutes / class_duration_minutes) * 100

    if presence_percentage >= min_presence_percentage * 100:
        status = "Presente"
    else:
        status = "Ausente"

    file_exists = os.path.isfile(attendance_file)

    with open(attendance_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Nome", "Horário de Entrada", "Horário de Saída", "Tempo Total (minutos)", "Porcentagem de Presença", "Status"])
        entry_timestamp = entry_time.strftime('%Y-%m-%d %H:%M:%S')
        exit_timestamp = exit_time.strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([name, entry_timestamp, exit_timestamp, total_minutes, presence_percentage, status])
        print(f"{name} registrado com status {status} às {entry_timestamp}")

# Inicializa a captura de vídeo
cam = cv2.VideoCapture(0)

# Dicionário para armazenar os horários de entrada e saída
attendance_records = {}

while(True):
    ret, frame = cam.read()
    
    # Redimensiona o quadro se uma largura máxima for especificada
    if max_width is not None:
        video_width, video_height = resize_video(frame.shape[1], frame.shape[0], max_width)
        frame = cv2.resize(frame, (video_width, video_height))
    
    # Reconhece rostos no quadro atual
    face_locations, face_names, conf_values = recognize_faces(frame, list_encodings, list_names, 0.25)
    processed_frame = show_recognition(frame, face_locations, face_names, conf_values)
    
    # Verifica se a tecla "E" ou "S" foi pressionada para registrar entrada ou saída
    key = cv2.waitKey(1) & 0xFF
    if key == ord('e'):
        for name in face_names:
            if name != "Not identified":
                entry_time = datetime.now()
                attendance_records[name] = {"entry_time": entry_time, "exit_time": None}
                print(f"{name} registrado com horário de entrada às {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
    elif key == ord('s'):
        for name in face_names:
            if name != "Not identified" and name in attendance_records and attendance_records[name]["entry_time"] is not None:
                exit_time = datetime.now()
                attendance_records[name]["exit_time"] = exit_time
                register_attendance(name, attendance_records[name]["entry_time"], exit_time)
                print(f"{name} registrado com horário de saída às {exit_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exibe o quadro processado
    cv2.imshow("Recognizing faces", frame)
    if key == ord('q'):
        break

print("Encerrado.")
cam.release()
cv2.destroyAllWindows()
