import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt

# Caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '..', 'dataset', 'archive')
IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, 'annotations')

IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 20
FINE_TUNE_EPOCHS = 10
FINE_TUNE_AT = 100  # camada a partir da qual será feito fine-tuning

def load_dataset():
    print("📦 Carregando dataset...")
    images = []
    labels = []

    for xml_file in os.listdir(ANNOTATIONS_PATH):
        if not xml_file.endswith('.xml'):
            continue

        tree = ET.parse(os.path.join(ANNOTATIONS_PATH, xml_file))
        root = tree.getroot()
        filename = root.find('filename').text
        image_path = os.path.join(IMAGES_PATH, filename)

        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0

        label = 0  # Sem capacete por padrão
        for obj in root.findall('object'):
            name = obj.find('name').text.lower()
            if name == 'helmet':
                label = 1
                break

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)

def prepare_data():
    X, y = load_dataset()
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights_dict = dict(enumerate(class_weights))
    return train_test_split(X, y, test_size=0.2, random_state=42), class_weights_dict

def build_model(fine_tune=False):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    if not fine_tune:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:FINE_TUNE_AT]:
            layer.trainable = False

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1)
    ])

    model = tf.keras.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1./255),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005 if not fine_tune else 0.0001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

def plot_metrics(history, title='Treinamento'):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title(f'Acurácia - {title}')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['precision'], label='Treino')
    plt.plot(history.history['val_precision'], label='Validação')
    plt.title(f'Precisão - {title}')
    plt.xlabel('Épocas')
    plt.ylabel('Precisão')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['recall'], label='Treino')
    plt.plot(history.history['val_recall'], label='Validação')
    plt.title(f'Recall - {title}')
    plt.xlabel('Épocas')
    plt.ylabel('Recall')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_and_convert():
    (X_train, X_test, y_train, y_test), class_weights = prepare_data()

    print("\n🧠 Etapa 1: Treinamento com base congelada...\n")
    model = build_model(fine_tune=False)
    history1 = model.fit(X_train, y_train,
                         epochs=EPOCHS,
                         batch_size=BATCH_SIZE,
                         validation_data=(X_test, y_test),
                         class_weight=class_weights)

    print("\n🔧 Etapa 2: Fine-tuning nas últimas camadas da base...\n")
    model = build_model(fine_tune=True)
    history2 = model.fit(X_train, y_train,
                         epochs=FINE_TUNE_EPOCHS,
                         batch_size=BATCH_SIZE,
                         validation_data=(X_test, y_test),
                         class_weight=class_weights)

    _, acc, precision, recall = model.evaluate(X_test, y_test)
    print(f"\n✅ Resultado Final:")
    print(f"   Acurácia : {acc * 100:.2f}%")
    print(f"   Precisão : {precision * 100:.2f}%")
    print(f"   Recall   : {recall * 100:.2f}%")

    plot_metrics(history1, title='Pré-Treinamento')
    plot_metrics(history2, title='Fine-Tuning')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    output_path = os.path.join(BASE_DIR, '..', 'capacete_model.tflite')
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"\n📁 Modelo .tflite salvo em: {output_path}")

if __name__ == '__main__':
    train_and_convert()
