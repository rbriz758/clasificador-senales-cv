import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import cv2
import numpy as np
import random

def calcular_accuracy(y_true, y_pred):
    """
    Calcula Accuracy.
    """
    return accuracy_score(y_true, y_pred)

def mostrar_confusion_matrix(y_true, y_pred, labels=None):
    """
    Genera y muestra visualmente la matriz de confusion.
    Requisito: Visualizacion con Matplotlib.
    Modificado: Usamos ConfusionMatrixDisplay de sklearn.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Crear display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # Plotear
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    
    plt.title('Matriz de Confusion')
    plt.show() 
    
    try:
        plt.savefig("confusion_matrix.png")
        print("Matriz guardada como 'confusion_matrix.png'")
    except Exception:
        pass

def visualizar_resultados(imgs_test, y_test, y_pred, nombres_clases):
    """
    Muestra ventanas de OpenCV con ejemplos de clasificacion exitosa y fallida.
    """
    indices_correctos = np.where(y_test == y_pred)[0]
    indices_incorrectos = np.where(y_test != y_pred)[0]
    
    print(f"\nVisualizando resultados: {len(indices_correctos)} aciertos, {len(indices_incorrectos)} fallos.")
    
    # Funcion auxiliar para crear mosaico
    def crear_mosaico(indices, titulo_ventana):
        if len(indices) == 0:
            return
            
        seleccion = indices[:5] if len(indices) > 5 else indices # Mostrar hasta 5 ejemplos
        mosaico = []
        
        for idx in seleccion:
            img = imgs_test[idx].copy()
            img = cv2.resize(img, (128, 128)) # Un poco mas grande para ver bien
            
            real = nombres_clases[y_test[idx]]
            pred = nombres_clases[y_pred[idx]]
            
            color = (0, 255, 0) if y_test[idx] == y_pred[idx] else (0, 0, 255)
            
            # Texto en la imagen
            cv2.putText(img, f"R:{real}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(img, f"P:{pred}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            mosaico.append(img)
            
        # Concatenar horizontalmente
        if mosaico:
            img_final = cv2.hconcat(mosaico)
            cv2.imshow(titulo_ventana, img_final)

    # Mostrar Correctos (Shuffle para variedad)
    indices_correctos_list = list(indices_correctos)
    random.shuffle(indices_correctos_list)
    crear_mosaico(indices_correctos_list, "Ejemplos Correctos (Verde)")
    
    # Mostrar Incorrectos
    if len(indices_incorrectos) > 0:
        indices_incorrectos_list = list(indices_incorrectos)
        random.shuffle(indices_incorrectos_list)
        crear_mosaico(indices_incorrectos_list, "Ejemplos Fallidos (Rojo)")
    else:
        print("¡Perfecto! No hubo fallos para mostrar.")

    print("Presiona cualquier tecla en las ventanas de OpenCV para cerrar y terminar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
