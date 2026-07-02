<div align="center">

# Sistema de Clasificación de Señales de Tráfico (Visión Clásica)

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

*Un pipeline completo de Visión por Computador y Machine Learning tradicional para la detección y clasificación de señales de tráfico del dataset GTSRB.*

</div>

---

## Descripción del Proyecto

Este proyecto implementa un sistema de reconocimiento visual capaz de identificar señales de tráfico específicas a partir de imágenes en crudo. A diferencia de los enfoques modernos de Deep Learning (como CNNs), este proyecto se construye desde cero utilizando **técnicas clásicas de Visión por Computador**, demostrando un profundo entendimiento de la adquisición, preprocesado, extracción de características y clasificación de patrones.

El sistema está entrenado para clasificar 5 tipos de señales del dataset **GTSRB (German Traffic Sign Recognition Benchmark)**:
- **14:** Stop
- **13:** Ceda el Paso
- **17:** Prohibido
- **33:** Giro a la Derecha
- **38:** Mantener la Derecha

## Autores

- **Rodrigo Briz Ramos**
- **Isaac Pérez Mascaró**

---

## Metodología y Pipeline

El proyecto sigue una arquitectura secuencial (pipeline) donde cada módulo es responsable de una etapa crítica del proceso de reconocimiento:

### 1. Preprocesamiento (`preprocesamiento.py`)
- **Redimensionado (Resize):** Todas las imágenes se estandarizan a un tamaño de `64x64` píxeles para asegurar consistencia en los descriptores.
- **Suavizado Gaussiano:** Aplicación de un filtro para la reducción de ruido de alta frecuencia.
- **Espacios de Color:** Conversión de BGR a **HSV** para aislar el color de los cambios de iluminación ambiental.

### 2. Segmentación (`segmentacion.py`)
- **Filtros de Color (Cromaticidad):** Búsqueda de colores específicos (rojos y azules) en el espacio HSV para aislar la señal del fondo.
- **Morfología Matemática:** 
  - *Apertura (Erosión + Dilatación)* para eliminar el ruido de "sal".
  - *Cierre (Dilatación + Erosión)* para rellenar los huecos negros y formar una región sólida.

### 3. Extracción de Características (`descriptores.py`)
- **Descriptores HOG (Histogram of Oriented Gradients):** Extracción de la "forma" de las señales. HOG es una técnica excepcionalmente buena para capturar los bordes y geometrías locales de un objeto, ignorando variaciones de luz e imperfecciones.

### 4. Clasificación y Entrenamiento (`clasificador.py`)
- División del dataset en **80% Entrenamiento** y **20% Prueba** (Test) de forma estratificada.
- **Máquinas de Vectores de Soporte (SVM):** El modelo principal utilizado es un SVM con kernel lineal, ideal para espacios de alta dimensionalidad como los generados por HOG. (También soporta KNN).

### 5. Evaluación Visual (`evaluacion.py`)
- Cálculo automático de **Accuracy** (Precisión global).
- Generación de **Matriz de Confusión** con Matplotlib.
- Visualización interactiva con OpenCV mostrando un mosaico de ejemplos predichos correctamente e incorrectamente (para facilitar el análisis de errores).

---

## Estructura del Repositorio

```text
├── Informe.pdf              # Documentación detallada del desarrollo
├── codigo/                  # Código fuente principal
│   ├── main.py              # Script orquestador del pipeline
│   ├── preprocesamiento.py  # Módulo de carga y filtrado de imagen
│   ├── segmentacion.py      # Módulo de máscaras y morfología
│   ├── descriptores.py      # Módulo de extracción HOG
│   ├── clasificador.py      # Módulo de entrenamiento SVM/KNN
│   └── evaluacion.py        # Módulo de métricas y visualización (Matplotlib/OpenCV)
├── dataset/                 # Carpetas numéricas con las imágenes GTSRB (13, 14, 17, 33, 38)
└── README.md                # Este archivo
```

---

## Instalación y Uso

### Requisitos Previos

Asegúrate de tener instalado Python 3.x y las siguientes dependencias. Puedes instalarlas ejecutando:

```bash
pip install numpy opencv-python scikit-learn matplotlib
```

### Ejecución

1. Clona el repositorio:
```bash
git clone https://github.com/TU_USUARIO/traffic-sign-classifier-cv.git
cd traffic-sign-classifier-cv/codigo
```

2. Ejecuta el pipeline completo:
```bash
python main.py
```

El script procesará el dataset de forma automática, entrenará el modelo e imprimirá por terminal las métricas de rendimiento. Además, abrirá ventanas interactivas mostrando la Matriz de Confusión y ejemplos visuales de las predicciones.
