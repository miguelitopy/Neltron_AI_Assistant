# modules/vision_module.py
import cv2
import os
from datetime import datetime

CAMERA_INDEX = 0  # Altere se usar múltiplas câmeras

def capturar_imagem(caminho_imagem=None):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise Exception("Não foi possível acessar a câmera.")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("Erro ao capturar imagem da câmera.")

    if caminho_imagem is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho_imagem = f"imagem_{timestamp}.jpg"

    cv2.imwrite(caminho_imagem, frame)
    print(f"[VISÃO] Imagem salva em: {caminho_imagem}")
    return caminho_imagem
