import cv2
import numpy as np

def resize_video(width, height, max_width=600):
    # Define a largura máxima do vídeo processado. Se a largura original for maior, redimensiona proporcionalmente.
    
    if (width > max_width):
        # Calcula a proporção original do vídeo:
        proporcao = width / height
        # Define a nova largura do vídeo como a largura máxima:
        video_width = max_width
        # Calcula a nova altura do vídeo mantendo a proporção original:
        video_height = int(video_width / proporcao)
    else:
        # Mantém a largura e altura originais.
        video_width = width
        video_height = height

    return video_width, video_height
