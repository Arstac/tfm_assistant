from typing import List
import pandas as pd
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from io import BytesIO
import markdown
import matplotlib.pyplot as plt
import os
from io import BytesIO
import seaborn as sns
from .class_model import State, RiskReport
import pandas as pd
import matplotlib
from .agent import llm

def chat_risk(state: State):
    """
    Función que analiza el riesgo de un proyecto de construcción
    utilizando un modelo de lenguaje basado en IA y genera un informe en HTML.
    """

    system = """Eres un asistente virtual experto en análisis de riesgo de proyectos de construcción.
Tu tarea es evaluar el nivel de riesgo basándote en las características del proyecto y los factores identificados.

Los datos que tienes incluyen:
- Nivel de riesgo (Alto / Moderado / Bajo).
- Desviación en costos y tiempos.
- Factores como zona sísmica, tipo de suelo, disponibilidad de materiales y experiencia del contratista.

Tu respuesta debe:
1. Analizar el nivel de riesgo del proyecto basándote en los datos proporcionados.
2. Explicar por qué el proyecto tiene ese nivel de riesgo, con referencia a los factores clave.
3. Proporcionar recomendaciones prácticas para mitigar los riesgos identificados.
4. Ofrecer comparaciones con proyectos similares en base a riesgos y costos.

Formato de Respuesta Esperado:
- Un resumen de la evaluación del riesgo del proyecto.
- Factores que afectan el nivel de riesgo.
- Sugerencias concretas para mitigar los riesgos.
- Si aplica, ejemplos de casos similares y cómo se gestionaron.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | llm.with_structured_output(RiskReport)

    print("--------- ENTRANDO EN ANÁLISIS DE RIESGO ---------")
    print("Mensajes recibidos: ", state["messages"])
    print("--------------------------------------------------")

    try:
        response = chain.invoke({"messages": state["messages"]})
    except Exception as e:
        print(f"Error al invocar el modelo de lenguaje: {e}")
        return "Ocurrió un error al generar la respuesta del modelo de lenguaje."

    print("Respuesta generada: ", response)

    return {"md_content": response}

def generate_risk_chart(state: State):
    matplotlib.use('Agg')
    dataset_path = os.path.join(os.path.dirname(__file__), 'final_final_dataset.csv')
    df = pd.read_csv(dataset_path)
    buffer_risk = BytesIO()

    try:
        plt.figure(figsize=(6, 4))
        sns.histplot(df['riesgos'], bins=20, kde=True, color='orange', label='Histórico')
        plt.axvline(x=state.state["risk"], color='red', linestyle='--', label='Predicción Actual')
        plt.title('Comparación del Riesgo Predicho con el Histórico')
        plt.xlabel('Índice de Riesgo')
        plt.ylabel('Frecuencia')
        plt.legend()
        buffer_risk = BytesIO()
        plt.savefig(buffer_risk, format='png')
        buffer_risk.seek(0)
        plt.close()
    except Exception as e:
        print("Error al generar PDF:", str(e))

    return {"graphics": buffer_risk}	

graph = StateGraph(State)
graph.add_node("chat_risk", chat_risk)
graph.add_node("generate_risk_chart", generate_risk_chart)

graph.add_edge(START, "chat_risk")
graph.add_edge("chat_risk", "generate_risk_chart")
graph.add_edge("generate_risk_chart", END)

app_info_risk = graph.compile()