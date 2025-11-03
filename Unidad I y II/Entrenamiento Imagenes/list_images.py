"""
Script para listar todas las imágenes del dataset
"""

import os

# Ruta del dataset
DATASET_PATH = 'datasets/raw-img'

# Categorías
CATEGORIAS = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina',
              'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

NOMBRES_ES = ['Perro', 'Caballo', 'Elefante', 'Mariposa', 'Gallina',
              'Gato', 'Vaca', 'Oveja', 'Araña', 'Ardilla']

print("=" * 70)
print("IMÁGENES DISPONIBLES EN EL DATASET")
print("=" * 70)

total_imagenes = 0

for categoria, nombre_es in zip(CATEGORIAS, NOMBRES_ES):
    ruta_categoria = os.path.join(DATASET_PATH, categoria)
    
    if os.path.exists(ruta_categoria):
        # Obtener todas las imágenes
        imagenes = [f for f in os.listdir(ruta_categoria) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        num_imagenes = len(imagenes)
        total_imagenes += num_imagenes
        
        print(f"\n📁 {nombre_es} ({categoria})")
        print(f"   Total: {num_imagenes} imágenes")
        print(f"   Ruta: {ruta_categoria}")
        
        # Mostrar las primeras 5 como ejemplo
        print(f"   Ejemplos de rutas:")
        for i, img in enumerate(imagenes[:5], 1):
            ruta_completa = os.path.join(ruta_categoria, img)
            # Convertir a formato de Windows con barras invertidas
            ruta_completa = ruta_completa.replace('/', '\\')
            print(f"      {i}. {ruta_completa}")
        
        if num_imagenes > 5:
            print(f"      ... y {num_imagenes - 5} más")

print("\n" + "=" * 70)
print(f"TOTAL: {total_imagenes} imágenes en {len(CATEGORIAS)} categorías")
print("=" * 70)

# Guardar en un archivo de texto
print("\n¿Quieres guardar todas las rutas en un archivo? (s/n): ", end='')
respuesta = input().strip().lower()

if respuesta in ['s', 'si', 'yes', 'y']:
    archivo_salida = 'rutas_imagenes.txt'
    
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        f.write("LISTA COMPLETA DE RUTAS DE IMÁGENES\n")
        f.write("=" * 70 + "\n\n")
        
        for categoria, nombre_es in zip(CATEGORIAS, NOMBRES_ES):
            ruta_categoria = os.path.join(DATASET_PATH, categoria)
            
            if os.path.exists(ruta_categoria):
                imagenes = [f for f in os.listdir(ruta_categoria) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                f.write(f"\n{nombre_es} ({categoria}) - {len(imagenes)} imágenes\n")
                f.write("-" * 70 + "\n")
                
                for img in imagenes:
                    ruta_completa = os.path.join(ruta_categoria, img)
                    ruta_completa = ruta_completa.replace('/', '\\')
                    f.write(f"{ruta_completa}\n")
    
    print(f"✓ Archivo guardado: {archivo_salida}")
    print(f"  Puedes abrirlo con: notepad {archivo_salida}")