import joblib
import os
import pandas as pd


# Ruta al modelo guardado
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelos_entrenados_xgb.pkl")

# Cargar el modelo con todas las predicciones
resultados_cargados = joblib.load(MODEL_PATH)

FEATURE_NAMES = [
        "Superficie_Construida",
        "Complejidad_Diseño",
        "Indice_Riesgo",
        "Experiencia_Contratista",
        "Costo_Total_Planificado",
        "Cantidad_Turnos_Trabajo",
        "Rating_Encargado",
        "Tipo_Proyecto",
        "Tipo_Estructura",
        "Disponibilidad_Materiales",
        "Disponibilidad_Subcontratistas",
        "Zona_Sismica",
        "Tipo_Suelo",
        "Clima"
    ]

def predict_costo_final(features):
    """
    Predice el Costo Final del proyecto.
    """
    data = resultados_cargados["Costo_Final"]
    modelo, encoder, scaler = data["modelo"], data["encoder"], data["scaler"]
    print("🚀 Features recibidas:", features)
    print("📊 Columnas esperadas por StandardScaler:", scaler.feature_names_in_)
    features_df = pd.DataFrame([features], columns=FEATURE_NAMES)   

    features_scaled = scaler.transform(features_df)
    prediction = modelo.predict(features_scaled)

    return prediction[0]

def predict_duracion_real(features):
    """
    Predice la Duración Real del proyecto.
    """
    data = resultados_cargados["Duracion_Real"]
    modelo, encoder, scaler = data["modelo"], data["encoder"], data["scaler"]

    features_scaled = scaler.transform([features]) if scaler else [features]
    prediction = modelo.predict(features_scaled)

    return prediction[0]

def predict_satisfaccion_cliente(features):
    """
    Predice la Satisfacción del Cliente.
    """
    data = resultados_cargados["Satisfaccion_Cliente"]
    modelo, encoder, scaler = data["modelo"], data["encoder"], data["scaler"]

    features_scaled = scaler.transform([features]) if scaler else [features]
    prediction = modelo.predict(features_scaled)

    # Decodificar si es categórico
    if encoder:
        prediction = encoder.inverse_transform(prediction.reshape(-1, 1)).flatten()[0]

    return prediction

def predict_desviacion_presupuestaria(features):
    """
    Predice la Desviación Presupuestaria.
    """
    data = resultados_cargados["Desviacion_Presupuestaria"]
    modelo, encoder, scaler = data["modelo"], data["encoder"], data["scaler"]

    features_scaled = scaler.transform([features]) if scaler else [features]
    prediction = modelo.predict(features_scaled)

    return prediction[0]