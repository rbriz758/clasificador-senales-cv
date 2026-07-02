from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def entrenar_modelo(X_train, y_train, metodo='SVM'):
    """
    Entrena el modelo Scikit-Learn.
    Tema 7/8: Reconocimiento.
    """
    modelo = None
    if metodo == 'SVM':
        # SVC es C-Support Vector Classification
        modelo = SVC(kernel='linear', C=1.0, random_state=42)
    elif metodo == 'KNN':
        modelo = KNeighborsClassifier(n_neighbors=5)
    
    if modelo:
        modelo.fit(X_train, y_train)
        
    return modelo

def predecir(modelo, X_test):
    """
    Realiza predicciones.
    """
    return modelo.predict(X_test)
