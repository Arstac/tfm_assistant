from typing import List
import json
import requests
import pandas as pd
import os
from django.conf import settings
from langchain_openai import ChatOpenAI  # OpenAI language model integration with LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
import openai
from io import BytesIO
import base64
import markdown
import joblib
import matplotlib.pyplot as plt
import os
from io import BytesIO
import seaborn as sns
from weasyprint import HTML
from .class_model import State, FeasibilityReport
import pandas as pd
import matplotlib
from .agent import llm

def chat_feasibility(state: State):
    """
    Función que analiza la viabilidad de un proyecto de construcción 
    utilizando un modelo de lenguaje basado en IA y genera un informe en HTML.
    """ 
    system = """Eres un asistente virtual experto en análisis de viabilidad de proyectos de construcción. 
Tu tarea es evaluar la viabilidad basándote en las características del proyecto y los riesgos identificados. 

Los datos que tienes incluyen:
- Viabilidad del proyecto (Viable / No Viable)
- Desviación en costos y tiempos.
- Factores como zona sísmica, tipo de suelo, disponibilidad de materiales y experiencia del contratista.

Tu respuesta debe:
1. Analizar la viabilidad del proyecto basándote en los datos proporcionados.
2. Explicar por qué el proyecto es viable o no, con referencia a los factores clave.
3. Proporcionar recomendaciones prácticas para mejorar la viabilidad del proyecto.
4. Ofrecer comparaciones con proyectos similares en base a riesgos y costos.

Formato de Respuesta Esperado:
- Un resumen de la evaluación del proyecto.
- Factores que afectan la viabilidad.
- Sugerencias concretas para mejorar la planificación.
- Si aplica, ejemplos de casos similares y cómo se resolvieron.
"""  
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | llm.with_structured_output(FeasibilityReport)

    print("--------- ENTRANDO EN ANALISIS VIABILIDAD ---------")
    print("Mensajes recibidos: ", state["messages"])
    print("--------- --------------------------------- ---------")

    try:
        response = chain.invoke({"messages":state["messages"]})
    except Exception as e:
        print(f"Error al invocar el modelo de lenguaje: {e}")
        return "Ocurrió un error al generar la respuesta del modelo de lenguaje."

    return {"md_content": response}

def generate_feasibility_chart(state: State):
    matplotlib.use('Agg')
    dataset_path = os.path.join(os.path.dirname(__file__), 'final_final_dataset.csv')
    df = pd.read_csv(dataset_path)
    buffer_viabilidad = BytesIO()

    print("respuesta: ", state)

    try:
        plt.figure(figsize=(5, 3))
        sns.countplot(x='viabilidad', hue='viabilidad', data=df, palette='viridis', legend=False)
        plt.axhline(y=df['viabilidad'].value_counts().get(state["feasibility"], 0), color='red', linestyle='--', label='Predicción Actual')
        plt.title('Comparación de Viabilidad con el Histórico')
        plt.xlabel('Estado de Viabilidad')
        plt.ylabel('Cantidad de Proyectos')
        plt.legend()
        plt.savefig(buffer_viabilidad, format='png')
        buffer_viabilidad.seek(0)
        plt.close()
    except Exception as e:
        print("Error al generar PDF:", str(e))

    return {"graphics": buffer_viabilidad}	

graph = StateGraph(State)
graph.add_node("chat_feasibility", chat_feasibility)
graph.add_node("generate_feasibility_chart", generate_feasibility_chart)

graph.add_edge(START, "chat_feasibility")
graph.add_edge("chat_feasibility", "generate_feasibility_chart")
graph.add_edge("generate_feasibility_chart", END)

app_info_feasibility = graph.compile()

