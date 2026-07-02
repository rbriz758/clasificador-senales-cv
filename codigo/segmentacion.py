import cv2
import numpy as np

def segmentar_por_color(hsv_img, color='rojo'):
    """
    Segmenta la imagen por color utilizando rangos HSV especificos.
    """
    mask = None
    
    # Definicion de rangos de color en HSV
    # HSV es robusto a cambios de iluminacion (Componente V separada de H y S)
    if color == 'rojo':
        # El matiz (Hue) rojo cae en el inicio (0-10) y fin (170-180) del espectro angular
        lower1 = np.array([0, 70, 50])
        upper1 = np.array([10, 255, 255])
        
        lower2 = np.array([170, 70, 50])
        upper2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv_img, lower1, upper1)
        mask2 = cv2.inRange(hsv_img, lower2, upper2)
        
        # Combinamos ambos rangos con OR
        mask = cv2.bitwise_or(mask1, mask2)
        
    elif color == 'azul': 
        # Rango para señales azules (Clases 33, 38)
        # Valores aproximados de matiz azul
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        
        mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    
    else:
         mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)

    # MORFOLOGIA OBLIGATORIA
    # Apertura para quitar ruido, Cierre para rellenar
    kernel = np.ones((5,5), np.uint8)
    
    # 1. Apertura (Erosion -> Dilatacion):
    # Objetivo: Eliminar ruido externo ("sal") o pequeños puntos blancos que no son señal.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 2. Cierre (Dilatacion -> Erosion):
    # Objetivo: Rellenar huecos negros dentro de la señal ("pimienta") para tener una region solida.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask
