import pandas as pd

from langgraph.graph import StateGraph, START, END

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from markitdown import MarkItDown
import time

from modules.load_prompts import prompt_extraccion_materiales
from .class_models import MatOutStr, State
from .load_llm_models import llm

#from modules.utils import convert_to_md

MAX_TOKENS = 500

# config = {
#         "configurable": {
#             "thread_id": "123" ,
#         }
#     }

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=500,
    separators=["\n\n", "\n", "\. ", "Página \d+ de \d+"],
    keep_separator=True
)


def convert_to_md(state: State):
    md = MarkItDown()
    result = md.convert(f"{state['doc']}")
    
    print(f"Length: {len(result.text_content)}")

    return {"md_content": result.text_content}

def mats_to_excel(state: State):
    """Converts the structured output of the material extraction process to an Excel file"""

    # Extract the materials data from the state
    materials_data = state["materiales"]

    # Create a DataFrame from the materials data
    df = pd.DataFrame(materials_data)


    doc_name = state['doc'].split("/")[-1].split(".")[0]
    # Define the output Excel file path
    output_file = f"Data/materiales_{doc_name}.xlsx"

    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False)

    return {"excel_file_path": output_file}
    

def extraccion_materiales(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_extraccion_materiales),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    chain = prompt | llm.with_structured_output(MatOutStr)
    
    response = chain.invoke({"messages":state["messages"], "pliego": state["md_content"]})
    
    print(f"Response: {response.model_dump()['materiales']}")
    
    return {"materiales": response.model_dump()["materiales"]}


def extraccion_materiales(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_extraccion_materiales),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    chunks = text_splitter.split_text(state["md_content"])
    all_materials = []

    chain = prompt | llm.with_structured_output(MatOutStr)
    
    # Procesar chunks con ventana deslizante
    for i, chunk in enumerate(chunks):
        # Añadir contexto de paginación
        context = f"CONTEXTO ACTUAL: Fragmento {i+1}/{len(chunks)}\n"
        
        # Mantener encabezados de sección entre chunks
        if i > 0:
            prev_chunk = chunks[i-1][-200:]
            context += f"CONTEXTO ANTERIOR: {prev_chunk}\n"
        
        full_chunk = context + chunk
        
        # Procesar con reintentos
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response =chain.invoke({"messages":state["messages"], "pliego": full_chunk})
                all_materials.extend(response.model_dump()["materiales"])
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
    
    # Post-procesamiento para elementos fragmentados
    # consolidated = []
    # buffer = ""
    # for material in all_materials:
    #     if not material["descripcion"].endswith((".", ":")) and len(material["descripcion"]) < 100:
    #         buffer += material["descripcion"] + " "
    #     else:
    #         if buffer:
    #             material["descripcion"] = buffer + material["descripcion"]
    #             buffer = ""
    #         consolidated.append(material)
    
    return {"materiales": all_materials}

graph_mats = StateGraph(State)
graph_mats.add_node("convert_to_md", convert_to_md)
graph_mats.add_node("extraccion_materiales", extraccion_materiales)
graph_mats.add_node("mats_to_excel", mats_to_excel)

graph_mats.add_edge(START, "convert_to_md")
graph_mats.add_edge("convert_to_md", "extraccion_materiales")
graph_mats.add_edge("extraccion_materiales", "mats_to_excel")
graph_mats.add_edge("mats_to_excel", END)

app_mats = graph_mats.compile()
