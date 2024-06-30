import cv2
import numpy as np

def resize_video(width, height, max_width=600):
  # max_width = em pixels. define a largura máxima do vídeo processado.
  # a altura será proporcional (definida nos cálculos abaixo)

  # se resize=True o vídeo salvo terá seu tamanho reduzido SOMENTE SE sua largura for maior que max_width
  if (width > max_width):
    # precisamos fazer a largura e a altura proporcionais (para manter a proporção do vídeo original) para que a imagem não pareça esticada
    proporcao = width / height
    # para isso, precisamos calcular a proporção (largura/altura) e usaremos esse valor para calcular a nova altura
    video_width = max_width
    video_height = int(video_width / proporcao)
  else:
    video_width = width
    video_height = height

  return video_width, video_height
