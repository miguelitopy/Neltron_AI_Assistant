# import os
# import cv2
# import face_recognition
# import numpy as np

# USERS_DIR = "users"

# def verificar_rostos_salvos_ou_cadastrar(nome):
#     if not os.path.exists(USERS_DIR):
#         os.makedirs(USERS_DIR)

#     caminho_imagem = os.path.join(USERS_DIR, f"{nome}.jpg")

#     if not os.path.exists(caminho_imagem):
#         print(f"[INFO] Nenhum rosto encontrado para '{nome}'. Iniciando cadastro...")
#         cadastrar_rosto(nome, caminho_imagem)
#     else:
#         print(f"[INFO] Rosto de '{nome}' encontrado. Verificando se é você...")
#         if verificar_rosto(caminho_imagem):
#             print("[✅] Reconhecimento facial bem-sucedido.")
#         else:
#             print("[ERRO] Rosto não corresponde ao cadastro. Acesso negado.")
#             exit()

# def cadastrar_rosto(nome, caminho_salvar):
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("[ERRO] Não foi possível acessar a câmera.")
#         return

#     print("[INFO] Posicione seu rosto de frente para a câmera. A foto será tirada em 3 segundos...")
#     for i in range(3, 0, -1):
#         print(f"... {i}")
#         cv2.waitKey(1000)

#     ret, frame = cap.read()
#     cap.release()

#     if ret:
#         cv2.imwrite(caminho_salvar, frame)
#         print(f"[SUCESSO] Rosto salvo como '{caminho_salvar}'.")
#     else:
#         print("[ERRO] Falha ao capturar imagem.")

# def verificar_rosto(caminho_cadastrado):
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("[ERRO] Não foi possível acessar a câmera.")
#         return False

#     print("[INFO] Posicione seu rosto de frente para a câmera para verificação...")
#     cv2.waitKey(3000)
#     ret, frame = cap.read()
#     cap.release()

#     if not ret:
#         print("[ERRO] Falha ao capturar imagem.")
#         return False

#     Codifica o rosto cadastrado
#     imagem_cadastrada = face_recognition.load_image_file(caminho_cadastrado)
#     codigos_cadastrados = face_recognition.face_encodings(imagem_cadastrada)
#     if not codigos_cadastrados:
#         print("[ERRO] Não foi possível codificar o rosto salvo.")
#         return False

#     Codifica o rosto capturado ao vivo
#     codigos_atual = face_recognition.face_encodings(frame)
#     if not codigos_atual:
#         print("[ERRO] Nenhum rosto detectado na imagem atual.")
#         return False

#     Compara os dois
#     resultado = face_recognition.compare_faces([codigos_cadastrados[0]], codigos_atual[0])
#     return resultado[0]
