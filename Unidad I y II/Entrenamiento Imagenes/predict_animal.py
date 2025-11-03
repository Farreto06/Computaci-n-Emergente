"""
Script para Predecir Animales desde Imágenes
Uso: python predict_animal.py ruta/a/imagen.jpg
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Configuración
MODEL_PATH = 'models/best_model.h5'
IMG_SIZE = (128, 128)

# Categorías (en español)
CATEGORIES = ['Perro', 'Caballo', 'Elefante', 'Mariposa', 'Gallina',
              'Gato', 'Vaca', 'Oveja', 'Araña', 'Ardilla']

# Emojis para cada animal
EMOJIS = ['🐕', '🐴', '🐘', '🦋', '🐔', '🐱', '🐄', '🐑', '🕷️', '🐿️']

def cargar_modelo():
    """Carga el modelo entrenado"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: No se encontró el modelo en {MODEL_PATH}")
        print("   Asegúrate de haber entrenado el modelo primero.")
        sys.exit(1)
    
    print("Cargando modelo...")
    model = keras.models.load_model(MODEL_PATH)
    print("✓ Modelo cargado correctamente\n")
    return model

def predecir_animal(ruta_imagen, model, mostrar_imagen=True):
    """
    Predice qué animal hay en la imagen
    
    Args:
        ruta_imagen: Ruta de la imagen a predecir
        model: Modelo cargado
        mostrar_imagen: Si mostrar la imagen con el resultado
    
    Returns:
        animal, confianza
    """
    # Verificar que existe la imagen
    if not os.path.exists(ruta_imagen):
        print(f"❌ Error: No se encontró la imagen en {ruta_imagen}")
        return None, None
    
    # Cargar y preprocesar la imagen
    img = image.load_img(ruta_imagen, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Hacer predicción
    print("Analizando imagen...")
    predicciones = model.predict(img_array, verbose=0)
    
    # Obtener la clase con mayor probabilidad
    clase_predicha = np.argmax(predicciones[0])
    confianza = predicciones[0][clase_predicha] * 100
    
    animal = CATEGORIES[clase_predicha]
    emoji = EMOJIS[clase_predicha]
    
    # Mostrar resultado
    print("\n" + "=" * 50)
    print(f"   {emoji}  PREDICCIÓN: {animal.upper()}  {emoji}")
    print("=" * 50)
    print(f"   Confianza: {confianza:.2f}%")
    print("=" * 50)
    
    # Mostrar top 3 predicciones
    print("\nTop 3 predicciones:")
    top3_indices = np.argsort(predicciones[0])[-3:][::-1]
    for i, idx in enumerate(top3_indices, 1):
        prob = predicciones[0][idx] * 100
        print(f"   {i}. {EMOJIS[idx]} {CATEGORIES[idx]}: {prob:.2f}%")
    
    # Mostrar imagen con predicción
    if mostrar_imagen:
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        
        # Título con resultado
        color = 'green' if confianza > 70 else 'orange' if confianza > 50 else 'red'
        plt.title(f'{emoji} Predicción: {animal}\nConfianza: {confianza:.2f}%',
                 fontsize=18, fontweight='bold', color=color, pad=20)
        
        plt.tight_layout()
        plt.show()
    
    return animal, confianza

def modo_interactivo(model):
    """Modo interactivo para predecir múltiples imágenes"""
    print("\n" + "=" * 50)
    print("MODO INTERACTIVO")
    print("=" * 50)
    print("Ingresa la ruta de una imagen para predecir")
    print("Escribe 'salir' para terminar\n")
    
    while True:
        ruta = input("Ruta de la imagen: ").strip()
        
        if ruta.lower() in ['salir', 'exit', 'quit', 'q']:
            print("\n¡Hasta luego! 👋")
            break
        
        if ruta:
            predecir_animal(ruta, model)
            print("\n" + "-" * 50 + "\n")

def main():
    """Función principal"""
    print("=" * 50)
    print("🐾 RECONOCEDOR DE ANIMALES 🐾")
    print("=" * 50)
    print()
    
    # Cargar modelo
    model = cargar_modelo()
    
    # Si se pasó una imagen como argumento
    if len(sys.argv) > 1:
        ruta_imagen = sys.argv[1]
        predecir_animal(ruta_imagen, model)
    else:
        # Modo interactivo
        modo_interactivo(model)

if __name__ == "__main__":
    main()