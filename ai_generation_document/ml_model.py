import joblib
import os
import pandas as pd
import numpy as np
from django.http import JsonResponse
from .class_model import CostVariance

MODEL_PATH_COST = os.path.join(os.path.dirname(__file__), '../modelo_lstm.pkl')

model_cost = joblib.load(MODEL_PATH_COST)

FEATURE_NAMES = [
        'sector_industria', 'segmento_cliente', 'importe_presupuestado_x', 'dias_ejecucion_real'
    ]

def predict_cost_variance(cost: CostVariance):
    """
    Predice el Costo Final del proyecto.
    """
    input_data = np.array([
            cost.importe_presupuestado_x,
            cost.dias_ejecucion_real,
            cost.sector_industria,
            cost.segmento_cliente,
        ]).reshape(1, -1)
    
     # Normalizar datos
    features_scaled = model_cost.transform(input_data)

    # Crear secuencia (LSTM necesita secuencias)
    seq_length = 5  # La misma longitud usada en el entrenamiento
    input_sequence = np.array([features_scaled] * seq_length).reshape(1, seq_length, cost.shape[1])

        # Hacer la predicción
    desviacion_predicha = model_cost.predict(input_sequence)[0][0]

    return JsonResponse({"cost_variance": desviacion_predicha})