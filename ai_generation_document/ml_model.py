import joblib
import os
import pandas as pd

# Ruta al modelo guardado
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelos_entrenados_xgb.pkl")

# Cargar el modelo con todas las predicciones
resultados_cargados = joblib.load(MODEL_PATH)

FEATURE_NAMES = [
        "desviacion_coste",  
        "desviacion_tiempo",
        "categoria_licitada",
        "complejidad_general",
        "categoria_licitada_Automatico",
        "categoria_licitada_Balsa",
        "categoria_licitada_Distribución",
        "categoria_licitada_Infraestructura",
        "categoria_licitada_Otros",
        "categoria_licitada_Parques",
        "categoria_licitada_Tanque",
        "categoria_licitada_Tratamiento",
        "complejidad_general_2",
        "complejidad_general_3",
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