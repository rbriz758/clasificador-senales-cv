import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split

# Importacion de modulos locales
import preprocesamiento
import segmentacion
import descriptores
import clasificador
import evaluacion

# --- Configuracion de Rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset")

# --- Configuracion de Clases (Dataset GTSRB) ---
CLASES = [13, 14, 17, 33, 38]
NOMBRES_CLASES = {
    13: "Ceda el Paso",
    14: "Stop",
    17: "Prohibido",
    33: "Giro Derecha",
    38: "Mantener Derecha"
}

def main():
    print("=== Sistema de Clasificacion de Señales de Trafico (Vision Clasica) ===")
    
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR FATAL: No se encuentra el dataset en: {DATASET_PATH}")
        return

    data = []
    labels = []
    imagenes_originales = [] # Guardamos referencia para visualizar despues
    
    print("\n--- Fase 1: Procesamiento y Extraccion de Caracteristicas ---")
    
    for clase_id in CLASES:
        path_clase = os.path.join(DATASET_PATH, str(clase_id))
        
        if not os.path.exists(path_clase):
            print(f"[AVISO] Carpeta para clase {clase_id} no encontrada. Saltando...")
            continue
            
        print(f"-> Procesando Clase {clase_id}: {NOMBRES_CLASES[clase_id]}...")
        
        color_segmentacion = 'rojo' if clase_id in [13, 14, 17] else 'azul'
        
        for img_name in os.listdir(path_clase):
            if not img_name.lower().endswith(('.png', '.jpg', '.ppm')):
                continue
                
            img_path = os.path.join(path_clase, img_name)
            
            # 1. PREPROCESAMIENTO
            img = preprocesamiento.cargar_imagen(img_path)
            if img is None: continue
            
            img_resized = preprocesamiento.resize_imagen(img, size=(64, 64))
            img_smooth = preprocesamiento.suavizado_gaussiano(img_resized)
            img_hsv = preprocesamiento.convertir_hsv(img_smooth)
            
            # 2. SEGMENTACION
            mask = segmentacion.segmentar_por_color(img_hsv, color=color_segmentacion)
            
            if cv2.countNonZero(mask) == 0:
                continue 
            
            # 3. EXTRACCION DE CARACTERISTICAS
            img_gray = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2GRAY)
            img_masked = cv2.bitwise_and(img_gray, img_gray, mask=mask)
            
            descriptor = descriptores.extraer_hog(img_masked)
            
            if descriptor is not None:
                data.append(descriptor)
                labels.append(clase_id)
                # Guardamos la imagen procesada (resized) para visualizar luego
                imagenes_originales.append(img_resized)

    # --- Verificacion de datos extraidos ---
    X = np.array(data, dtype=np.float32)
    y = np.array(labels)
    imgs = np.array(imagenes_originales)
    
    if len(X) == 0:
        print("\n[ERROR] No se extrajeron datos.")
        return

    print(f"\nTotal muestras: {len(X)}")

    # --- Fase 2: Division de Datos (80/20) ---
    print("\n--- Fase 2: Division de Datos (80/20) ---")
    # Ahora tambien dividimos el array de imagenes originales para tener correspondencia
    X_train, X_test, y_train, y_test, imgs_train, imgs_test = train_test_split(
        X, y, imgs, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Test Set: {len(X_test)} muestras")

    # --- Fase 3: Entrenamiento (SVM) ---
    print("\n--- Fase 3: Entrenamiento del Clasificador (SVM) ---")
    modelo = clasificador.entrenar_modelo(X_train, y_train, metodo='SVM')
    print("Modelo entrenado.")

    # --- Fase 4: Evaluacion ---
    print("\n--- Fase 4: Evaluacion de Resultados ---")
    y_pred = clasificador.predecir(modelo, X_test)
    
    acc = evaluacion.calcular_accuracy(y_test, y_pred)
    print(f"PRECISION GLOBAL (Accuracy): {acc*100:.2f}%")
    
    print("Generando graficos...")
    # Solo pasamos nombres de clases que existan en el test para evitar errores de indices
    lista_clases_presentes = np.unique(y) # Todas las posibles
    
    # Visualizacion Matplotlib (Matriz)
    evaluacion.mostrar_confusion_matrix(y_test, y_pred, labels=[NOMBRES_CLASES[c] for c in CLASES])
    
    # Visualizacion OpenCV (Ejemplos) - NUEVO REQUISITO
    evaluacion.visualizar_resultados(imgs_test, y_test, y_pred, NOMBRES_CLASES)

if __name__ == "__main__":
    main()
