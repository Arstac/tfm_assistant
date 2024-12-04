import joblib
import os

# Ruta al modelo guardado
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelo_rf.pkl')

# Cargar el modelo
model = joblib.load(MODEL_PATH)

def predict(features):
    """
    Realiza una predicción utilizando el modelo cargado.
    :param features: Lista de características [x1, x2, ..., xn].
    :return: Predicción del modelo.
    """
    prediction = model.predict([features])
    return prediction[0]