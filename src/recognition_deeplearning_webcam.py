import cv2
import numpy as np
import pickle
import face_recognition

# Nome do arquivo pickle onde as codificações faciais estão armazenadas
pickle_name = "face_encodings_custom.pickle"

max_width = 800

# Carrega as codificações faciais do arquivo pickle
data_encoding = pickle.loads(open(pickle_name, "rb").read())
list_encodings = data_encoding["encodings"]
list_names = data_encoding["names"]

def recognize_faces(image, list_encodings, list_names, resizing=0.25, tolerance=0.6):
    image = cv2.resize(image, (0, 0), fx=resizing, fy=resizing)
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(img_rgb)
    
    face_encodings = face_recognition.face_encodings(img_rgb, face_locations)
    
    face_names = []  # Lista para armazenar os nomes dos rostos reconhecidos
    conf_values = []  # Lista para armazenar os valores de confiança dos reconhecimentos
    
    for encoding in face_encodings:
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
    
    return face_locations.astype(int), face_names, conf_values
