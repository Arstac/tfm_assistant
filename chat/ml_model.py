import joblib
import os
import pandas as pd

# Ruta al modelo guardado
MODEL_PATH_RISK = os.path.join(os.path.dirname(__file__), '../modelo_xgb_risk.pkl')
MODEL_PATH_COST = os.path.join(os.path.dirname(__file__), '../modelo_lstm.pkl')

model_cost = joblib.load(MODEL_PATH_COST)
model_risk = joblib.load(MODEL_PATH_RISK)

def predict(features):
    """
    Realiza una predicción utilizando el modelo cargado.
    :param features: Lista de características [x1, x2, ..., xn].
    :return: Predicción del modelo.
    """
    prediction = MODEL_PATH_RISK.predict([features])
    return prediction[0]

# FEATURE_NAMES = [
#         "importe_presupuestado_x",
#         "duracion_trabajo_dias",
#         "año_presupuesto",
#         "mes_presupuesto",
#         "dia_semana_presupuesto",
#         "sector_industria",
#         "segmento_cliente",
#         "categoria_licitada",
#     ]


# def predict_satisfaccion_cliente(features):
#     """
#     Predice la Satisfacción del Cliente.
#     """
#     data = resultados_cargados["Satisfaccion_Cliente"]
#     modelo, encoder, scaler = data["modelo"], data["encoder"], data["scaler"]

#     features_scaled = scaler.transform([features]) if scaler else [features]
#     prediction = modelo.predict(features_scaled)

#     # Decodificar si es categórico
#     if encoder:
#         prediction = encoder.inverse_transform(prediction.reshape(-1, 1)).flatten()[0]

#     return prediction

def predict_cost(features):
    prediction = model_cost.predict([features])

    return prediction[0]