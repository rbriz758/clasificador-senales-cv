import cv2
import numpy as np

def extraer_hog(img_gray):
    """
    Extrae descriptores HOG (Histogram of Oriented Gradients) usando OpenCV.
    """
    # Verificacion de dimensiones (Critico para cv2.HOGDescriptor)
    # La ventana de deteccion (winSize) debe coincidir con el tamaño de la imagen de entrada (64x64)
    winSize = (64, 64) 
    
    # Parametros estandar para captura de forma en señales de trafico:
    blockSize = (16, 16) # Tamaño del bloque de normalizacion
    blockStride = (8, 8) # Paso del bloque (overlap del 50%)
    cellSize = (8, 8)    # Tamaño de la celda donde se calculan los histogramas
    nbins = 9            # Numero de orientaciones (bins del histograma)
    
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    
    # Calculo del descriptor. 
    # Flatten() convierte la matriz multidimensional en un vector caracteristico unidimensional (Feature Vector).
    features = hog.compute(img_gray).flatten()
    
    return features
