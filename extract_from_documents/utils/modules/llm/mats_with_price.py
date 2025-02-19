from langgraph.graph import StateGraph, START, END

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


from modules.load_prompts import prompt_extraccion_precios
from .class_models import State, MatPricesOutStr
from .load_llm_models import llm
import pandas as pd

# config = {
#         "configurable": {
#             "thread_id": "123" ,
#         }
#     }


@tool
def buscador_precio(diametro: int, cantidad):
    """Busca el precio de un material en base a su diámetro
    Abre el archivo Data/pn16.csv ->
    diametro;material;maquinaria;mano obra;total
    20;0,72;0;1;1,72
    25;1,06;0;1,27;2,33
    32;1,72;0;1,5;3,22
    40;2,64;0;1,78;4,42
    50;4,09;0;2;6,09
    
    Luego, calcula el precio total segun las cantidades
    ...
    """
    prices = pd.read_csv("Data/pn16.csv", sep=";", decimal=",")
    price = prices[(prices["diametro"] == diametro)]
    # print(f"Precio unitario: {price['total'].values[0]}")
    # print(f"Cantidad: {cantidad}")
    # print(f"Precio total: {price['total'].values[0] * cantidad}")
    
    return {"precio": price["mano obra"].values[0] * cantidad,
            "cantidad": cantidad,
            "precio_unitario": price["mano obra"].values[0]}    
    
tools = [buscador_precio]

def load_materiales_licitacion(state: State):
    # doc_name = state['doc'].split("/")[-1].split(".")[0]
    materiales_licitacion = pd.read_csv(f"Data/materiales_lic_2.csv", sep=";")
    
    # Convertir a diccionario  -> pd df a dict:
    materiales = materiales_licitacion.to_dict(orient="records")
    
    # print(f"Materiales: {materiales}")
    
    return {"materiales": materiales}
    
def estructurar_materiales(state: State):
    prompt_est_materiales = """En base al siguiente mensaje, extrae de forma estructurada los materiales y sus precios.
    
    Mensaje:
    
   {input} 
    """
    
    messages = state["messages"]
    message = messages[-1].content
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_est_materiales),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    chain = prompt | llm.with_structured_output(MatPricesOutStr)
    
    response = chain.invoke({"messages":state["messages"], "input": message})
    # print("-------------------")
    # print(f"Response mat-prices: {response.model_dump()['materiales']}")
    # print("-------------------")
    return {"mat_prices": response.model_dump()['materiales']}

def mats_to_excel(state: State):
    """Converts the structured output of the material extraction process to an Excel file"""

    # Extract the materials data from the state
    materials_data = state["mat_prices"]

    # Create a DataFrame from the materials data
    df = pd.DataFrame(materials_data)

    # doc_name = state['doc'].split("/")[-1].split(".")[0]
    # Define the output Excel file path
    output_file = f"Data/materiales_lic_2_prices.xlsx"

    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False)

    return {"excel_file_path": output_file}
    

def extraccion_precios(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_extraccion_precios),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    chain = prompt | llm.bind_tools(tools)
    
    response = chain.invoke({"messages":state["messages"], "listado_materiales": state["materiales"]})
    
    # print(f"Response: {response}")
    
    return {"messages": response}


def need_tool(state: State):
   messages = state["messages"]
   last_message = messages[-1]
   if last_message.tool_calls:
       return "TOOL"
   else:
       return "NO_TOOL"


graph_price = StateGraph(State)
graph_price.add_node("load_materiales_licitacion", load_materiales_licitacion)
graph_price.add_node("extraccion_precios", extraccion_precios)
graph_price.add_node("estructurar_materiales", estructurar_materiales)
graph_price.add_node("mats_to_excel", mats_to_excel)
graph_price.add_node("tool", ToolNode(tools))


graph_price.add_edge(START, "load_materiales_licitacion")
graph_price.add_edge("load_materiales_licitacion", "extraccion_precios")
graph_price.add_conditional_edges(
    "extraccion_precios", 
    need_tool,
    {
        "TOOL": "tool",
        "NO_TOOL": "estructurar_materiales"
    }
)
graph_price.add_edge("tool", "extraccion_precios")
graph_price.add_edge("estructurar_materiales", "mats_to_excel")
graph_price.add_edge("mats_to_excel", END)

#          START
#            |
#            V
# load_materiales_licitacion
#            |
#            V
# extraccion_precios -> need_tool -> TOOL
#            |     ^                  |       
#            V     |_ _ _ _ _ _ __ _ _|
# estructurar_materiales
#            |
#            V
#      mats_to_excel
#            |
#            V
#           END


app_price = graph_price.compile()
