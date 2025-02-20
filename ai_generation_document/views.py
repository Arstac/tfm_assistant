from django.shortcuts import render
from django.http import JsonResponse
import pdfkit
import numpy as np
import joblib
from jinja2 import Environment, FileSystemLoader
from chat.agents import chat_feasibility
from django.http import JsonResponse, FileResponse
import os
import pdfplumber
import pandas as pd
from weasyprint import HTML
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
matplotlib.use('Agg')

def generate_risk_report(request):
    if request.method == "POST":

        try:
            MODEL_PATH_RISK = os.path.join(os.path.dirname(__file__), '../modelo_xgb_risk.pkl')
            model_risk = joblib.load(MODEL_PATH_RISK)
            pdf_file = request.FILES.get("file")
            if not pdf_file:
                return JsonResponse({"error": "No se subió ningún archivo."}, status=400)

            # Aquí debes implementar la lógica para extraer las características relevantes del texto
            # Por ejemplo, podrías utilizar expresiones regulares o técnicas de procesamiento de lenguaje natural
            # Para este ejemplo, asumiremos que ya tienes un diccionario 'data' con las características extraídas
            data = {
                "beneficios_esperados": 4500,  # Ejemplo de valor extraído del texto
                "duracion_plan": 120,          # Ejemplo de valor extraído del texto
                "complejidad_general": "Media",# Ejemplo de valor extraído del texto
                "sector_industria": "Construcción", # Ejemplo de valor extraído del texto
                "categoria_licitada": "Infraestructura" # Ejemplo de valor extraído del texto
            }

            # Crear un DataFrame con los datos extraídos
            input_df = pd.DataFrame([data])

            # Definir el preprocesador que se utilizó durante el entrenamiento
            preprocesador = ColumnTransformer([
                ('cat', OneHotEncoder(drop='first'), ['sector_industria', 'categoria_licitada']),
                ('num', StandardScaler(), ['beneficios_esperados', 'duracion_plan'])
            ])

            # Aplicar las transformaciones al DataFrame de entrada
            X_input = preprocesador.fit_transform(input_df) 

            # Realizar la predicción de riesgo
            riesgo_predicho = model_risk.predict(X_input)[0]
            probabilidad_riesgo = model_risk.predict_proba(X_input)[0][1]  # Probabilidad de riesgo alto

            # Construir la respuesta
            response = {
                "riesgo": "Alto" if riesgo_predicho == 1 else "Bajo",
                "probabilidad_riesgo": probabilidad_riesgo,
                "beneficios_esperados": data["beneficios_esperados"],
                "duracion_plan": data["duracion_plan"],
                "complejidad_general": data["complejidad_general"],
                "sector_industria": data["sector_industria"],
                "categoria_licitada": data["categoria_licitada"]
            }

            plt.figure(figsize=(6, 4))
            sns.histplot(df['riesgos'], bins=20, kde=True, color='orange', label='Histórico')
            plt.axvline(x=probabilidad_riesgo[0], color='red', linestyle='--', label='Predicción Actual')
            plt.title('Comparación del Riesgo Predicho con el Histórico')
            plt.xlabel('Índice de Riesgo')
            plt.ylabel('Frecuencia')
            plt.legend()
            buffer_riesgo = BytesIO()
            plt.savefig(buffer_riesgo, format='png')
            buffer_riesgo.seek(0)
            plt.close()

            # Generar el informe en HTML utilizando una plantilla
            env = Environment(loader=FileSystemLoader('.'))
            template = env.get_template("informe_template.html")
            html_output = template.render(
                titulo="Informe de Evaluación de Riesgo del Proyecto",
                contenido=response
            )

            # Convertir el HTML a PDF
            pdf_path = "informe_riesgo.pdf"
            try:
                HTML(string=html_output).write_pdf(pdf_path)
            except Exception as e:
                print("Error al generar PDF:", str(e))
                return JsonResponse({"error": str(e)}, status=500)

            # Retornar el PDF como respuesta
            return FileResponse(open(pdf_path, 'rb'), content_type='application/pdf')

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    else:
        return JsonResponse({"error": "Método no permitido"}, status=405)

def generate_feasibility_report(request):
    if request.method == "POST":

        try:
            MODEL_PATH_FEASIBILITY = os.path.join(os.path.dirname(__file__), '../modelo_feasibility.pkl')
            model_feasibility = joblib.load(MODEL_PATH_FEASIBILITY)
            pdf_file = request.FILES.get("file")
            if not pdf_file:
                return JsonResponse({"error": "No se subió ningún archivo."}, status=400)
    
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""

            try:
                data = {
                    "desviacion_coste": 0.05,  
                    "desviacion_tiempo": 0.1,
                    "categoria_licitada": 5,
                    "complejidad_general": 1,
                    "categoria_licitada_Automatico": 1,
                    "categoria_licitada_Balsa": 0,
                    "categoria_licitada_Distribución": 0,
                    "categoria_licitada_Infraestructura": 0,
                    "categoria_licitada_Otros": 0,
                    "categoria_licitada_Parques": 0,
                    "categoria_licitada_Tanque": 0,
                    "categoria_licitada_Tratamiento": 0,
                    "complejidad_general_2": 0,
                    "complejidad_general_3": 0,
                }

                features = pd.DataFrame([[
                    data["desviacion_coste"],
                    data["desviacion_tiempo"],
                    data["categoria_licitada_Automatico"],
                    data["categoria_licitada_Balsa"],
                    data["categoria_licitada_Distribución"],
                    data["categoria_licitada_Infraestructura"],
                    data["categoria_licitada_Otros"],
                    data["categoria_licitada_Parques"],
                    data["categoria_licitada_Tanque"],
                    data["categoria_licitada_Tratamiento"],
                    data["complejidad_general_2"],
                    data["complejidad_general_3"],
                ]], columns=[
                    'desviacion_coste',
                    'desviacion_tiempo',
                    'categoria_licitada_Automático',
                    'categoria_licitada_Balsa',
                    'categoria_licitada_Distribución',
                    'categoria_licitada_Infraestructura',
                    'categoria_licitada_Otros',
                    'categoria_licitada_Parques',
                    'categoria_licitada_Tanque',
                    'categoria_licitada_Tratamiento',
                    'complejidad_general_2',
                    'complejidad_general_3'
                ])

                feasibility_predict = model_feasibility.predict(features)
            except Exception as e:
                print("Error:", str(e))

            print("ALMAGRO12")
            response = {
                "viabilidad": "Viable" if feasibility_predict[0] == 1 else "No Viable",
                "desviacion_coste": data["desviacion_coste"],
                "desviacion_tiempo": data["desviacion_tiempo"]
            }
            
            # Cargar dataset histórico
            dataset_path = os.path.join(os.path.dirname(__file__), '../dataset.csv')
            df = pd.read_csv(dataset_path)
            print("ALMAGRO34")
            try:
                # Gráfico de viabilidad comparativa
                plt.figure(figsize=(5, 3))
                sns.countplot(x='viabilidad', hue='viabilidad', data=df, palette='viridis', legend=False)
                plt.axhline(y=df['viabilidad'].value_counts().get(feasibility_predict[0], 0), color='red', linestyle='--', label='Predicción Actual')
                plt.title('Comparación de Viabilidad con el Histórico')
                plt.xlabel('Estado de Viabilidad')
                plt.ylabel('Cantidad de Proyectos')
                plt.legend()
                buffer_viabilidad = BytesIO()
                plt.savefig(buffer_viabilidad, format='png')
                buffer_viabilidad.seek(0)
                plt.close()
            except Exception as e:
                print("Error al generar PDF:", str(e))

            html_analysis = chat_feasibility({"messages": [response]}, buffer_viabilidad)

            print("ALMAGRO1")
            pdf_path = "informe_viabilidad.pdf"
    
            try:
                HTML(string=html_analysis).write_pdf(pdf_path)
            except Exception as e:
                print("Error al generar PDF:", str(e))
                return JsonResponse({"error": str(e)}, status=500)
           
            return FileResponse(open(pdf_path, 'rb'), content_type='application/pdf')

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    else:
        return JsonResponse({"error": "Método no permitido"}, status=405)
