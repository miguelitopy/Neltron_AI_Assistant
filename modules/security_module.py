import cv2
import numpy as np
import tensorflow as tf

LABELS = ["sem capacete", "com capacete"]

def load_model(model_path='capacete_model.tflite'):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_helmet(face_img, interpreter, threshold=0.9):
    """
    Recebe uma imagem do rosto (recortado), redimensiona, normaliza e classifica com o modelo TFLite.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Redimensiona conforme o modelo (ajuste se seu modelo for 224x224, 180x180 etc.)
    img = cv2.resize(face_img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.repeat(img, 16, axis=0).astype(np.float32)  # modelo espera batch de 16

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Suporte para saída com 1 ou 2 classes
    if len(output_data) == 1:
        confidence = output_data[0]
    else:
        confidence = output_data[1]

    predicted_label = "com capacete" if confidence >= threshold else "sem capacete"
    return predicted_label, float(confidence)
