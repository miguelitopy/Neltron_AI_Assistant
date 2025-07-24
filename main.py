import cv2
import time
from modules.security_module import load_model, predict_helmet

def monitorar_capacete():
    interpreter = load_model()
    cap = cv2.VideoCapture(0)

    # Resolução leve para webcam (para não travar)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERRO] Não foi possível acessar a câmera.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("[INFO] Monitoramento contínuo iniciado.")

    ultima_predicao = ""
    ultima_confianca = 0.0
    tempo_ultima_predicao = 0
    intervalo_predicao = 1.0  # segundos

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Reduz para processamento (não para exibição)
        frame_proc = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            x, y, w, h = faces[0]  # Apenas primeiro rosto
            pad_x = int(w * 0.3)
            pad_y = int(h * 0.6)

            x1 = max(x - pad_x, 0)
            y1 = max(y - pad_y, 0)
            x2 = min(x + w + pad_x, frame_proc.shape[1])
            y2 = min(y + h, frame_proc.shape[0])

            helmet_region = frame_proc[y1:y2, x1:x2]

            agora = time.time()
            if agora - tempo_ultima_predicao > intervalo_predicao:
                ultima_predicao, ultima_confianca = predict_helmet(helmet_region, interpreter)
                tempo_ultima_predicao = agora

            # Mapeia coordenadas da imagem menor para a maior
            escala_x = frame.shape[1] / 320
            escala_y = frame.shape[0] / 240

            X1 = int(x1 * escala_x)
            Y1 = int(y1 * escala_y)
            X2 = int(x2 * escala_x)
            Y2 = int(y2 * escala_y)

            cor = (0, 255, 0) if ultima_predicao == "com capacete" else (0, 0, 255)
            texto = f"{ultima_predicao} ({ultima_confianca * 100:.1f}%)"

            cv2.rectangle(frame, (X1, Y1), (X2, Y2), cor, 2)
            cv2.putText(frame, texto, (X1, Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

        # Cabeçalho
        header_height = 60
        cv2.rectangle(frame, (0, 0), (frame.shape[1], header_height), (0, 0, 0), -1)
        cv2.putText(frame, "                 DX EVA - Smart Factory Safety", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        cv2.imshow("Monitoramento de Capacete", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Monitoramento encerrado.")

if __name__ == "__main__":
    monitorar_capacete()
