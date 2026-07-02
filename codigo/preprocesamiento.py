import cv2
import numpy as np

def cargar_imagen(path):
    """
    Carga una imagen desde disco.
    """
    img = cv2.imread(path)
    return img

def resize_imagen(img, size=(64, 64)):
    """
    Redimensiona la imagen al tamaño especificado (por defecto 64x64).
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def suavizado_gaussiano(img, kernel_size=(5, 5)):
    """
    Aplica suavizado Gaussiano para reducir ruido.
    """
    # SigmaX = 0 implica que se calcula automaticamente desde el tamaño del kernel
    return cv2.GaussianBlur(img, kernel_size, 0)

def convertir_hsv(img):
    """
    Convierte de espacio BGR (Blue-Green-Red) a HSV (Hue-Saturation-Value).
    Objetivo: Facilitar la segmentacion basada en cromaticidad (Color).
    Tema 4: Espacios de color.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
