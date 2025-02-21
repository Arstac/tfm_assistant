from typing import List, Optional
import json
from pydantic import BaseModel, Field
import requests
import pandas as pd
import os
from dotenv import load_dotenv

from django.conf import settings

from langchain_openai import ChatOpenAI  # OpenAI language model integration with LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langgraph.graph import StateGraph, START, END,MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
import openai
from jinja2 import Environment, FileSystemLoader

import numpy as np
def convert_to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # Convert NumPy scalar to native Python scalar
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to list
    elif isinstance(obj, pd.DataFrame):
        # Recursively convert DataFrame elements
        return obj.applymap(convert_to_serializable).to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        # Recursively convert Series elements
        return obj.apply(convert_to_serializable).tolist()
    elif isinstance(obj, pd.Index):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Timedelta):
        return str(obj)
    elif isinstance(obj, pd.Period):
        return str(obj)
    elif isinstance(obj, pd.api.extensions.ExtensionDtype):
        return str(obj)
    elif isinstance(obj, pd.api.extensions.ExtensionArray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert all keys to strings
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [convert_to_serializable(v) for v in obj]
    else:
        # Fallback: Convert to string if all else fails
        return str(obj)
# Define the state class

class State(MessagesState):
    pass


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# carga directorio de trabajo
df = pd.read_csv('Data/presupuestos_con_desviaciones.csv', delimiter=',')

import io
import base64
################################################################
# HISTOGRAM TOOL
################################################################
import uuid

def save_image_to_file(base64_content, file_path=None):
    # Generar un nombre único para cada imagen
    if not file_path:
        unique_name = f"temp_image_{uuid.uuid4().hex}.png"
        file_path = unique_name
    
    # Ruta completa para guardar en la carpeta estática
    full_path = os.path.join(settings.STATICFILES_DIRS[0], file_path)
    with open(full_path, "wb") as f:
        f.write(base64.b64decode(base64_content))
    # Devuelve la URL relativa
    return f"{settings.STATIC_URL}{file_path}"

################################################################
# CSV AGENT AS TOOL
################################################################

custom_suffix = """Instrucciones adicionales:
Las columnas del dataframe son: 
{dfcolumns}"""
custom_suffix = custom_suffix.format(dfcolumns=df.columns)

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True,
    suffix=custom_suffix,
    include_df_in_prompt=True,
)

s = agent.as_tool(
    name="csv_agent",
    description="Agent for csv file",
)

################################################################
# Generate chart tool
################################################################

class CodeOutputStructure(BaseModel):
    """
    Estructura de salida para el código generado por la función generate_chart.
    """
    chart_type: str = Field(description="Tipo de gráfico generado.")
    x_column: str = Field(description="Columna del eje X.")
    y_column: Optional[str] = Field(description="Columna del eje Y.")
    code: str = Field(description="Código generado para generar el gráfico.")

@tool
def generate_any_chart(chart_type: str, filepath_csv: str, x_column: str, y_column: str = None) -> dict:
    """
    Generates a chart based on the specified chart type and columns.
    """
    prompt_generate_any_chart = """
    You are an expert in Python code development.
    You need to generate the code to create a {chart_type} chart based on the data in the file {filepath_csv}.
    The chart should have {x_column} on the x-axis and {y_column} on the y-axis.
    You should return the code to:
    - Read the data from the CSV file.
    - Generate the chart based on the specified chart type.
    - Return the chart data as a dictionary.
    - Never plot the chart, just generate the data. (NEVER use plt.show())
    - Always import pandas as pd, numpy as np, matplotlib.pyplot as plt
    
    Code exaples:
    <code>
        if chart_type == "histogram":
        data = df[x_column].dropna()
        counts, bins = np.histogram(data, bins=30)
        chart_data = {{
            "x": bins[:-1].tolist(),
            "y": counts.tolist(),
            "type": "bar",
            "name": f"Histograma de {x_column}"
        }}

        if chart_type == "scatter":
            if y_column is None:
                return {{"error": "Se requiere una columna Y para gráficos de dispersión."}}
            x_data = df[x_column].dropna()
            y_data = df[y_column].dropna()
            chart_data = {{
                "x": x_data.tolist(),
                "y": y_data.tolist(),
                "type": "scatter",
                "mode": "markers",
                "name": f"{x_column} vs {y_column}"
            }}

        if chart_type == "line":
            if y_column is None:
                return {{"error": "Se requiere una columna Y para gráficos de líneas."}}
            x_data = df[x_column].dropna()
            y_data = df[y_column].dropna()
            chart_data = {{
                "x": x_data.tolist(),
                "y": y_data.tolist(),
                "type": "line",
                "name": f"{x_column} vs {y_column}"
            }}
    </code>
    """
    print("Generating chart of type:", chart_type)
    print("-----------------------------------")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_generate_any_chart),
            ("user", "Generate the chart."),
        ]
    )
    
    chain = prompt | llm.with_structured_output(CodeOutputStructure)
    
    response = chain.invoke({"chart_type": chart_type, "x_column": x_column, "y_column": y_column, "filepath_csv": filepath_csv})
    
    code = response.model_dump()['code']
    print("Code response: ", code)


# Check execution
    try:
        namespace = {}
        exec(code, namespace)
        
        chart_data = namespace.get('chart_data', None)
        
        chart_data = convert_to_serializable(chart_data)
        
        print("RESULTADO:", chart_data)
        
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error_message = HumanMessage(content=f"Your solution failed the code execution test: {e}")
        print("---ERROR MESSAGE---", error_message)
        messages += [error_message]
        return {"chart_data": {}, "title": "Error en la generación del gráfico"}

    return {"chart_data": chart_data, "title": f"{chart_type.capitalize()} de {x_column} {f'vs {y_column}' if y_column else ''}"}


HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "LangChain-Predictor",
    "Authorization": f"Token d20eaaba74be7fe262fa6a997da0b6bd29091031"
}
BASE_URL = "http://localhost:8000/predict/"  # Base URL para las predicciones

@tool
def api_predict_final_price(features) -> float:
    """
    Predice el Costo Final de un proyecto de construcción.
    """
    print("API FEATURES: ", features)
    print("API FEATURES: ")
    url = f"{BASE_URL}costo_final/"
    return call_prediction_api(url, features)

@tool
def api_predict_customer_satisfaction(features) -> float:
    """
    Predice la Duración Real de un proyecto de construcción.
    """
    print("API FEATURES: ", features)
    url = f"{BASE_URL}duracion_real/"
    return call_prediction_api(url, features)

@tool
def api_predict_customer_satisfaction(features) -> float:
    """
    Predice la Satisfacción del Cliente en un proyecto de construcción.
    """
    url = f"{BASE_URL}satisfaccion_cliente/"
    return call_prediction_api(url, features)

@tool
def api_predict_budget_deviation(features) -> float:
    """
    Predice la Desviación Presupuestaria en un proyecto de construcción.
    """
    url = f"{BASE_URL}desviacion_presupuestaria/"
    return call_prediction_api(url, features)

def call_prediction_api(url, features):
    """
    Realiza la llamada a la API de predicción y maneja errores.
    """
    print(f"🔍 Enviando solicitud a {url} con features: {features}")

    payload = {"features": features}

    try:
        response = requests.post(url, json=payload, headers=HEADERS)
        response_data = response.json()

        if response.status_code == 200:
            print(f"✅ Predicción recibida: {response_data}")
            return response_data
        else:
            print(f"⚠️ Error en la API ({response.status_code}): {response_data}")
            return {"error": f"API Error ({response.status_code}): {response_data}"}

    except requests.exceptions.RequestException as e:
        print(f"🚨 Fallo en la conexión con la API: {str(e)}")
        return {"error": f"Fallo en la conexión con la API: {str(e)}"}
    
# # Initialize the language model with the specified model name
llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)

tools = [api_predict_final_price, api_predict_customer_satisfaction, api_predict_customer_satisfaction, api_predict_budget_deviation, call_prediction_api, generate_any_chart]


################################################################
# CHAT NODE
################################################################

def chat(state: State):
    """
    Chat function that takes a state object as input and returns a response.
    """
    system = """Eres un asistente virtual especializado en la gestión de proyectos de construcción. 
Tu propósito principal es ofrecer soporte inteligente a una empresa mediante la consulta de datos históricos de obras ejecutadas, la estimación de riesgos y desviaciones en presupuestos, y la recomendación de estrategias para optimizar la planificación y minimizar riesgos.
Los datos se encuentran en el directorio: "Data/presupuestos_con_desviaciones.csv".

Dispones de una serie de funciones que puedes utilizar para resolver ciertas tareas:
Herramientas del Asistente:
    1- Prediccion de presupuesto real: utiliza la funcion api_predict para predecir el presupuesto real de un proyecto. Debes pasarle una lista de features con los valores correspondientes.
    Las features son:
        - feature_1: presupuesto_adjudicacion
        - feature_2: plazo_entrega_real
        - feature_3: plazo_entrega_adjudicacion
        Return: [presupuesto_adjudicacion, plazo_entrega_real, plazo_entrega_adjudicacion] (eje: [1000000, 12, 10])
 
    2- csv_agent: agente para interactuar con un archivo csv. Puedes utilizarlo para realizar consultas.
    3- generate_chart: función que genera un gráfico interactivo basado en los datos del archivo csv. 
    
 Interacción:
	1.	Entrada del Usuario:
        •	El usuario puede:
        •	Describir una necesidad o plantear preguntas como:
        •	“¿Qué desviaciones puedo esperar para un proyecto de construcción de viviendas con $500,000 en Madrid?”
        •	“Muéstrame obras similares ejecutadas el último año.”
        •	Usar un formulario guiado para introducir datos de un nuevo proyecto.
	2.	Salida del Asistente:
        •	Ofrece respuestas en formatos como:
        •	Texto explicativo.
        •	Gráficos interactivos.
        •	Tablas comparativas.
        •	Proporciona siempre una justificación clara basada en datos y predicciones.
    3. Si el ultimo mensaje es un ToolMessage donde hay una imagen codificada en base64, el devolverá la imagen.
 
 Formato de Respuesta Esperado:
	•	Comienza con un resumen de la consulta del usuario.
	•	Presenta la información más relevante y estructurada:
	•	Si es una consulta histórica, ofrece datos organizados y filtrados.
	•	Si es una predicción, incluye porcentajes de riesgo y gráficos.
	•	Si son recomendaciones, sugiere pasos específicos y justifica con datos.
	•	Cierra ofreciendo ayuda adicional o aclaraciones si es necesario.
 
 Instrucciones al Asistente:
	•	Mantén un tono profesional, preciso y accesible.
	•	Si la consulta no es clara, solicita educadamente información adicional.
	•	Garantiza que las visualizaciones sean claras y fáciles de entender.
	•	Respeta la confidencialidad y privacidad de los datos.
 
 Ejemplo de Solicitud:
	“¿Qué riesgos puedo prever para un proyecto de infraestructura vial con un presupuesto de $1,000,000 en Barcelona con un plazo estimado de 12 meses?”
	Respuesta esperada:
        •	Porcentaje de desviación estimada en costos y tiempos.
        •	Factores de riesgo específicos basados en datos históricos.
        •	Recomendaciones para mitigar los riesgos.
 """
           
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    chain = prompt | llm.bind_tools(tools)
    
    print("--------- ENTERTING CHAT NODE ---------")
    # print("Messages received: ", state["messages"])
    print("--------- ------------------ ---------")
    
    # Get 2 last messages
    last_messages = state["messages"]
    # Generate a response using the language model
    response = chain.invoke({"messages": last_messages})
    
    print("Response: ", response)
    # Return the response to the user
    return {"messages":response}

def chat_feasibility(state):
    """
    Función que analiza la viabilidad de un proyecto de construcción 
    utilizando un modelo de lenguaje basado en IA y genera un informe en HTML.
    """

    system = """Eres un asistente virtual experto en análisis de viabilidad de proyectos de construcción. 
Tu tarea es evaluar la viabilidad basándote en las características del proyecto y los riesgos identificados. 

Los datos que tienes incluyen:
- Viabilidad del proyecto (Viable / No Viable)
- Desviación en costos y tiempos.
- Nivel de riesgo (Alto / Moderado / Bajo).
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

    # Generamos el prompt con el contexto del sistema y los últimos mensajes
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Cadena que conecta el prompt con el modelo de lenguaje
    chain = prompt | llm.bind_tools(tools)


    print("--------- ENTRANDO EN ANALISIS VIABILIDAD ---------")
    print("Mensajes recibidos: ", state["messages"])
    print("--------- --------------------------------- ---------")
    
    # Obtener los últimos mensajes del usuario
    last_messages = {
        'viabilidad': 'Viable',
        'desviacion_coste': 0.05,
        'desviacion_tiempo': 0.1,
        'riesgo': 'Alto'
    }
    last_messages = ["'viabilidad': 'Viable','desviacion_coste': 0.05,'desviacion_tiempo': 0.1,'riesgo': 'Alto'"]

    # Generar una respuesta usando el modelo de IA
    try:
        response = chain.invoke({"messages": last_messages})
    except Exception as e:
        print(f"Error al invocar el modelo de lenguaje: {e}")
        return "Ocurrió un error al generar la respuesta del modelo de lenguaje."

    print("Respuesta generada: ", response)

    # Cargar la plantilla HTML para generar el informe
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("templates/informe_template.html")

    html_content = markdown.markdown(response.content)
   
    # Renderizar el HTML con los datos de la respuesta
    html_output = template.render(
        titulo="Informe de Viabilidad del Proyecto",
        contenido=html_content
    )

    return html_output

def chat_risk_analysis(state):
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

    # Generamos el prompt con el contexto del sistema y los últimos mensajes
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Cadena que conecta el prompt con el modelo de lenguaje
    chain = prompt | llm.bind_tools(tools)

    print("--------- ENTRANDO EN ANÁLISIS DE RIESGO ---------")
    print("Mensajes recibidos: ", state["messages"])
    print("--------------------------------------------------")

    # Obtener los últimos mensajes del usuario
    last_messages = state["messages"]

    # Generar una respuesta usando el modelo de IA
    try:
        response = chain.invoke({"messages": last_messages})
    except Exception as e:
        print(f"Error al invocar el modelo de lenguaje: {e}")
        return "Ocurrió un error al generar la respuesta del modelo de lenguaje."

    print("Respuesta generada: ", response)

    # Cargar la plantilla HTML para generar el informe
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("templates/informe_template.html")

    # Convertir la respuesta en formato Markdown a HTML
    html_content = markdown.markdown(response.content)

    # Renderizar el HTML con los datos de la respuesta
    html_output = template.render(
        titulo="Informe de Análisis de Riesgo del Proyecto",
        contenido=html_content
    )

    return html_output


################################################################
# GRAPH
################################################################

def need_tool(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we respond to the user
    if last_message.tool_calls:
        print("Tool call detected.")
        return "tool"
    else:
        print("No tool call detected.")
        return "end"
    
graph = StateGraph(State)
graph.add_node("chat", chat)
graph.add_node("tool", ToolNode(tools))
graph.add_edge(START, "chat")
graph.add_conditional_edges(
    "chat",
    need_tool,
    {
        "tool": "tool",
        "end": END,
    },
)
graph.add_edge("tool", "chat")
config = {"configurable": {"thread_id": "user_id"}}
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

################################################################
