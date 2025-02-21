import joblib
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
import os
from io import BytesIO
import seaborn as sns
from weasyprint import HTML
import pandas as pd
from ai_generation_document.agent_generation_risk import app_info_risk
from ai_generation_document.agent_generation_feasibility import app_info_feasibility
from .class_model import State
from django.http import JsonResponse, FileResponse
from .class_model import State, Feasibility, Risk
import numpy as np
import base64

def generate_feasibility_report(features: Feasibility) -> JsonResponse:
    try:
        MODEL_PATH_FEASIBILITY = os.path.join(os.path.dirname(__file__), '../modelo_feasibility.pkl')
        model_feasibility = joblib.load(MODEL_PATH_FEASIBILITY)

        input_data = np.array([
            features.categoria_licitada,
            features.complejidad_general,
            features.desviacion_coste,
            features.desviacion_tiempo,
        ])

        input_data = input_data.reshape(1, -1)
        feasibility_predict = model_feasibility.predict(input_data)
       
        response = app_info_feasibility.invoke({"messages": "Genera un informe de viabilidad", "feasibility": feasibility_predict})["md_content"]
      
        try:
            return JsonResponse({"data": response.md_content, "graphics": response.graphics})
        except Exception as e:
            print("Error al generar PDF:", str(e))
            return JsonResponse({"error": str(e)}, status=500)
    
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)
    

def generate_feasibility_report_pdf(features: Feasibility) -> JsonResponse:
    try:
        MODEL_PATH_FEASIBILITY = os.path.join(os.path.dirname(__file__), '../modelo_feasibility.pkl')
        model_feasibility = joblib.load(MODEL_PATH_FEASIBILITY)

        input_data = np.array([
            features.categoria_licitada,
            features.complejidad_general,
            features.desviacion_coste,
            features.desviacion_tiempo,
        ])

        input_data = input_data.reshape(1, -1)
        feasibility_predict = model_feasibility.predict(input_data)
        
        pdf_path = "informe_viabilidad.pdf"
        response = app_info_feasibility.invoke({"messages": "Genera un informe de viabilidad", "feasibility": feasibility_predict})["md_content"]
      
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template("informe_template.html")
        html_output = template.render(
            titulo="Informe de Viabilidad del Proyecto",
            contenido=response.md_content,
            grafico_viabilidad = base64.b64encode(response.graphics.getvalue()).decode('utf-8')
        )
        try:
            return JsonResponse({"data": html_output}, status=200)
        except Exception as e:
            print("Error al generar PDF:", str(e))
            return JsonResponse({"error": str(e)}, status=500)
    
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


def generate_risk_report(features: Risk) -> State:
    try:
        MODEL_PATH_RISK = os.path.join(os.path.dirname(__file__), '../modelo_xgb_risk.pkl')
        model_risk = joblib.load(MODEL_PATH_RISK)

        input_data = np.array([
            features.categoria_licitada,
            features.complejidad_general,
            features.beneficios_esperados,
            features.dias_ejecucion_real,
        ])

        input_data = input_data.reshape(1, -1)
        risk = model_risk.predict(input_data)
        risk_probability = model_risk.predict_proba(input_data)[0][1] 

        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template("informe_template.html")
        html_output = template.render(
            titulo="Informe de Evaluación de Riesgo del Proyecto",
            contenido=response
        )

        response = app_info_risk.invoke({"messages": "Genera un informe de riesgos", "risk": risk, "risk_probability": risk_probability})["md_content"]
        pdf_path = "informe_riesgo.pdf"

        try:
            HTML(string=html_output).write_pdf(pdf_path)
        except Exception as e:
            print("Error al generar PDF:", str(e))
            return JsonResponse({"error": str(e)}, status=500)

        return FileResponse(open(pdf_path, 'rb'), content_type='application/pdf')

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


