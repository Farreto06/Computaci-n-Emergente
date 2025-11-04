import os
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Ruta descargada del dataset (modifica esta ruta si cambia)
dataset_path = r"C:\Users\Angelo\.cache\kagglehub\datasets\paultimothymooney\chest-xray-pneumonia\versions\2"
print(f"Dataset ubicado en: {dataset_path}")

# Rutas dentro del dataset para carpetas train y test
train_dir = os.path.join(dataset_path, "chest_xray", "train")
test_dir = os.path.join(dataset_path, "chest_xray", "test")

# Parámetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10

# Generadores de datos con augmentación mínima
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Modelo VGG16 base congelado
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE,3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# Evaluar en test
loss, accuracy = model.evaluate(test_generator)
print(f"Precisión en test: {accuracy*100:.2f}%")

# Guardar modelo entrenado
model.save("pneumonia_vgg16.h5")
print("Modelo guardado como pneumonia_vgg16.h5")
