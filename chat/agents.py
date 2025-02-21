from typing import List
import json
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
from io import BytesIO
import base64


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

@tool
def generate_histogram(column_name: str) -> str:
    """
    Genera un histograma de la columna especificada en el dataframe y retorna el path de la imagen codificada en base64.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    print("Generating histogram for column:", column_name)
    try:
        plt.figure(figsize=(10, 6))
    except Exception as e:
        print("Error: ", e)
    print("Plotting histogram...")
    
    plt.hist(df[column_name].dropna(), bins=30, color='blue', alpha=0.7)
    plt.title(f'Histograma de {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frecuencia')
    plt.grid(axis='y', alpha=0.75)
    print("Saving histogram...")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    print("Encoding histogram image...")
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    print("Histogram image encoded successfully.")
    
    img_path = save_image_to_file(image_base64)
    
    print("Image saved to: ", img_path)
    return img_path

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


@tool
def generate_chart(chart_type: str, x_column: str, y_column: str = None) -> dict:
    """
    Genera gráficos basados en el tipo solicitado (histograma, dispersión, líneas, etc.) y las columnas especificadas.
    Retorna un diccionario con los datos para un gráfico interactivo.
    
    Parámetros:
    - chart_type: Tipo de gráfico ('histogram', 'scatter', 'line', etc.).
    - x_column: Columna del eje X.
    - y_column: Columna del eje Y (opcional, para gráficos como dispersión o líneas).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    print(f"Generating {chart_type} for columns: {x_column} {f'vs {y_column}' if y_column else ''}")

    if x_column not in df.columns or (y_column and y_column not in df.columns):
        return {"error": f"Las columnas especificadas no existen: {x_column}, {y_column}"}

    chart_data = {}

    if chart_type == "histogram":
        data = df[x_column].dropna()
        counts, bins = np.histogram(data, bins=30)
        chart_data = {
            "x": bins[:-1].tolist(),
            "y": counts.tolist(),
            "type": "bar",
            "name": f"Histograma de {x_column}"
        }

    elif chart_type == "scatter":
        if y_column is None:
            return {"error": "Se requiere una columna Y para gráficos de dispersión."}
        x_data = df[x_column].dropna()
        y_data = df[y_column].dropna()
        chart_data = {
            "x": x_data.tolist(),
            "y": y_data.tolist(),
            "type": "scatter",
            "mode": "markers",
            "name": f"{x_column} vs {y_column}"
        }

    elif chart_type == "line":
        if y_column is None:
            return {"error": "Se requiere una columna Y para gráficos de líneas."}
        x_data = df[x_column].dropna()
        y_data = df[y_column].dropna()
        chart_data = {
            "x": x_data.tolist(),
            "y": y_data.tolist(),
            "type": "line",
            "name": f"{x_column} vs {y_column}"
        }

    elif chart_type == "box":
        if y_column is None:
            return {"error": "Se requiere una columna Y para gráficos de caja."}
        x_data = df[x_column].dropna()
        y_data = df[y_column].dropna()
        chart_data = {
            "x": x_data.tolist(),
            "y": y_data.tolist(),
            "type": "box",
            "name": f"{x_column} vs {y_column}"
        }
        
    elif chart_type == "bar":
        if y_column is None:
            return {"error": "Se requiere una columna Y para gráficos de barras."}
        x_data = df[x_column].dropna()
        y_data = df[y_column].dropna()
        chart_data = {
            "x": x_data.tolist(),
            "y": y_data.tolist(),
            "type": "bar",
            "name": f"{x_column} vs {y_column}"
        }
        
    else:
        return {"error": f"Tipo de gráfico no soportado: {chart_type}"}

    print(f"{chart_type} data generated successfully.")
    return {"chart_data": chart_data, "title": f"{chart_type.capitalize()} de {x_column} {f'vs {y_column}' if y_column else ''}"}


################################################################
# PREDICT PRESUPUESTO REAL API
################################################################

@tool
def api_predict(features) -> float:
    """
    Chat function that makes an API call to a moodel to predict a response.
    Las features son:
    - feature_1: presupuesto_adjudicacion
    - feature_2: plazo_entrega_real
    - feature_3: plazo_entrega_adjudicacion
    """
    print("API FEATURES: ", features)
    url = "http://localhost:8000/predict/"

    payload = {"features": features}
    payload_str = json.dumps(payload, indent=4)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "insomnia/10.1.1",
        "Authorization": "Token d20eaaba74be7fe262fa6a997da0b6bd29091031"
    }

    response = requests.request("POST", url, data=payload_str, headers=headers)

    print(response.text)
    
    return response.json()["prediction"]


# # Initialize the language model with the specified model name
llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)

# tools = [api_predict, csv_agent_as_tool, generate_chart]
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "LangChain-Predictor",
    "Authorization": f"Token d20eaaba74be7fe262fa6a997da0b6bd29091031"
}
BASE_URL = "http://localhost:8000/predict/"  # Base URL para las predicciones

tools = [api_predict, generate_chart]


################################################################
# CHAT NODE
################################################################

def chat(state: State):
    """
    Chat function that takes a state object as input and returns a response.
    """
    system = """Eres un asistente virtual especializado en la gestión de proyectos de construcción. 
Tu propósito principal es ofrecer soporte inteligente a una empresa mediante la consulta de datos históricos de obras ejecutadas, la estimación de riesgos y desviaciones en presupuestos, y la recomendación de estrategias para optimizar la planificación y minimizar riesgos.
Los datos se encuentran en el archivo: o2.csv.

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
    print("Messages received: ", state["messages"])
    print("--------- ------------------ ---------")
    
    # Get 2 last messages
    last_messages = state["messages"]
    # Generate a response using the language model
    response = chain.invoke({"messages": last_messages})
    
    print("Response: ", response)
    # Return the response to the user
    return {"messages":response}

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
