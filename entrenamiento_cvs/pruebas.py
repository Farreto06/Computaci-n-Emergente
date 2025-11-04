from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
modelo = load_model("pneumonia_vgg16.h5")

# Ruta de la imagen que quieres probar (cambia el nombre si es distinto)
ruta_imagen = "torax.jpeg"

# Cargar y redimensionar la imagen
imagen = load_img(ruta_imagen, target_size=(224, 224))

# Convertir a array numpy
imagen_array = img_to_array(imagen)

# Añadir dimensión para batch
imagen_array = np.expand_dims(imagen_array, axis=0)

# Normalizar (escalado 0-1)
imagen_array = imagen_array / 255.0

# Hacer predicción
prediccion = modelo.predict(imagen_array)

# Mostrar resultado
probabilidad_neumonia = prediccion[0][0]
print(f"Probabilidad de neumonía: {probabilidad_neumonia:.4f}")

if probabilidad_neumonia > 0.8:
    print("Alto riesgo de neumonía. Se recomienda consulta médica inmediata.")
elif probabilidad_neumonia > 0.5:
    print("Posible neumonía. Se sugiere seguimiento y evaluación clínica.")
else:
    print("No se detectó evidencia clara de neumonía. Mantener controles regulares.")
