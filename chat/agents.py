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


# Define the state class
class State(MessagesState):
    pass


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


df = pd.read_csv('o2.csv', delimiter=',')

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
    Genera un histograma de la columna especificada en el dataframe y retorna la imagen codificada en base64.
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

    return img_path

################################################################
# CSV AGENT AS TOOL
################################################################
custom_suffix = """Instrucciones adicionales:
- Usa siempre este fragmento de codigo: 
    import matplotlib
    matplotlib.use('agg')
- Al generar gráficos con Matplotlib, no utilices plt.show(). En su lugar, guarda el gráfico en un buffer en memoria y retorna el gráfico codificado en base64.
- Proporciona el string base64 del gráfico en tu respuesta, indicando claramente que es una imagen codificada.
"""

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True,
)

csv_agent_as_tool = agent.as_tool(
    name="csv_agent",
    description="Agent for csv file",
)

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


# Initialize the language model with the specified model name
llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)

tools = [api_predict, csv_agent_as_tool, generate_histogram]


################################################################
# CHAT NODE
################################################################

def chat(state: State):
    """
    Chat function that takes a state object as input and returns a response.
    """
    system = """Eres un asistente virtual especializado en la gestión de proyectos de construcción. 
Tu propósito principal es ofrecer soporte inteligente a una empresa mediante la consulta de datos históricos de obras ejecutadas, la estimación de riesgos y desviaciones en presupuestos, y la recomendación de estrategias para optimizar la planificación y minimizar riesgos.

Dispones de una serie de funciones que puedes utilizar para resolver ciertas tareas:
Herramientas del Asistente:
    1- Prediccion de presupuesto real: utiliza la funcion api_predict para predecir el presupuesto real de un proyecto. Debes pasarle una lista de features con los valores correspondientes.
    Las features son:
        - feature_1: presupuesto_adjudicacion
        - feature_2: plazo_entrega_real
        - feature_3: plazo_entrega_adjudicacion
        Return: [presupuesto_adjudicacion, plazo_entrega_real, plazo_entrega_adjudicacion] (eje: [1000000, 12, 10])
 
    2- csv_agent: agente para interactuar con un archivo csv. Puedes utilizarlo para realizar consultas.
    3- generate_histogram: función que genera un histograma de una columna específica en el dataframe y retorna la imagen codificada en base64.
    
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
    # Generate a response using the language model
    response = chain.invoke({"messages": state["messages"]})
    
    # Return the response to the user
    return {"messages":response}


################################################################
# GRAPH
################################################################

def need_tool(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    print("last_message: ", last_message, "\n\n")
    # If there is no function call, then we respond to the user
    if last_message.tool_calls:
        return "tool"
    else:
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