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
