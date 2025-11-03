"""
Programa de Reconocimiento de Animales usando CNN
Dataset: Animals-10 (10 categorías)
Autor: Victor
Fecha: Noviembre 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==================== CONFIGURACIÓN ====================
print("=" * 60)
print("SISTEMA DE RECONOCIMIENTO DE ANIMALES")
print("=" * 60)

# Rutas
DATASET_PATH = 'datasets/raw-img'
MODEL_PATH = 'models'
os.makedirs(MODEL_PATH, exist_ok=True)

# Hiperparámetros
IMG_SIZE = (128, 128)  # Tamaño de imagen
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2

# Categorías de animales
CATEGORIES = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina',
              'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

# Nombres en español para visualización
CATEGORIES_ES = ['Perro', 'Caballo', 'Elefante', 'Mariposa', 'Gallina',
                 'Gato', 'Vaca', 'Oveja', 'Araña', 'Ardilla']

print(f"\nCategorías: {len(CATEGORIES)}")
print(f"Tamaño de imagen: {IMG_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")

# ==================== CARGA Y PREPROCESAMIENTO ====================
print("\n" + "=" * 60)
print("CARGANDO DATASET...")
print("=" * 60)

# Data Augmentation para mejorar el modelo
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Generador de datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Generador de datos de validación
validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\nImágenes de entrenamiento: {train_generator.samples}")
print(f"Imágenes de validación: {validation_generator.samples}")
print(f"Clases detectadas: {train_generator.class_indices}")

# ==================== CONSTRUCCIÓN DEL MODELO CNN ====================
print("\n" + "=" * 60)
print("CONSTRUYENDO MODELO CNN...")
print("=" * 60)

model = keras.Sequential([
    # Bloque 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Bloque 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Bloque 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Bloque 4
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Capas densas
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(CATEGORIES), activation='softmax')
])

# Compilar modelo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nResumen del modelo:")
model.summary()

# ==================== ENTRENAMIENTO ====================
print("\n" + "=" * 60)
print("INICIANDO ENTRENAMIENTO...")
print("=" * 60)

# Callbacks para mejorar el entrenamiento
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    ),
    keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_PATH, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True
    )
]

# Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# ==================== EVALUACIÓN ====================
print("\n" + "=" * 60)
print("EVALUANDO MODELO...")
print("=" * 60)

# Evaluar en conjunto de validación
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"\nPrecisión en validación: {val_accuracy*100:.2f}%")
print(f"Pérdida en validación: {val_loss:.4f}")

# Predicciones para matriz de confusión
validation_generator.reset()
predictions = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes

# Reporte de clasificación
print("\n" + "=" * 60)
print("REPORTE DE CLASIFICACIÓN")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=CATEGORIES_ES))

# ==================== VISUALIZACIÓN ====================
print("\n" + "=" * 60)
print("GENERANDO GRÁFICAS...")
print("=" * 60)

# Gráfica 1: Accuracy
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validación', linewidth=2)
plt.title('Precisión del Modelo', fontsize=14, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfica 2: Loss
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Entrenamiento', linewidth=2)
plt.plot(history.history['val_loss'], label='Validación', linewidth=2)
plt.title('Pérdida del Modelo', fontsize=14, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfica 3: Matriz de confusión
plt.subplot(1, 3, 3)
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CATEGORIES_ES, yticklabels=CATEGORIES_ES)
plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_PATH, 'training_results.png'), dpi=300, bbox_inches='tight')
print(f"\nGráficas guardadas en: {MODEL_PATH}/training_results.png")
plt.show()

# ==================== GUARDAR MODELO ====================
print("\n" + "=" * 60)
print("GUARDANDO MODELO...")
print("=" * 60)

model.save(os.path.join(MODEL_PATH, 'animal_recognition_model.h5'))
print(f"Modelo guardado en: {MODEL_PATH}/animal_recognition_model.h5")

# ==================== FUNCIÓN DE PREDICCIÓN ====================
def predict_animal(image_path, model, show_image=True):
    """
    Predice el animal en una imagen
    
    Args:
        image_path: Ruta de la imagen
        model: Modelo entrenado
        show_image: Si mostrar la imagen con la predicción
    
    Returns:
        Predicción y probabilidad
    """
    # Cargar y preprocesar imagen
    img = keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Predicción
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    animal = CATEGORIES_ES[predicted_class]
    
    if show_image:
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Predicción: {animal}\nConfianza: {confidence*100:.2f}%',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    return animal, confidence

# ==================== EJEMPLO DE USO ====================
print("\n" + "=" * 60)
print("EJEMPLO DE PREDICCIÓN")
print("=" * 60)
print("\nPara predecir una nueva imagen, usa:")
print("predict_animal('ruta/a/tu/imagen.jpg', model)")
print("\nEjemplo:")
print("predict_animal('datasets/raw-img/gato/imagen.jpg', model)")

print("\n" + "=" * 60)
print("ENTRENAMIENTO COMPLETADO")
print("=" * 60)
print(f"\n✓ Modelo entrenado con {train_generator.samples} imágenes")
print(f"✓ Precisión alcanzada: {val_accuracy*100:.2f}%")
print(f"✓ Modelo guardado en: {MODEL_PATH}/animal_recognition_model.h5")
print(f"✓ Gráficas guardadas en: {MODEL_PATH}/training_results.png")
print("\n¡Listo para reconocer animales!")