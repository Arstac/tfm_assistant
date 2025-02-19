import pandas as pd

from langgraph.graph import StateGraph, START, END

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from markitdown import MarkItDown

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



def convert_to_md(state: State):
    md = MarkItDown()
    result = md.convert(f"{state['doc']}")
    #En state['doc'] se guarda el path del archivo pdf, quiero extraer el nombre del archivo para guardarlo en el estado
    
    
    print(f"Length: {len(result.text_content)}")

    return {"md_content": result.text_content}

def mats_to_excel(state: State):
    """Converts the structured output of the material extraction process to an Excel file"""

    # Extract the materials data from the state
    materials_data = state["materiales"]

    # Create a DataFrame from the materials data
    df = pd.DataFrame(materials_data)

    # doc_name = state['doc'].split('/')[-1].split(".")[0]
    
    # Define the output Excel file path
    output_file = f"Data/materiales_lic_2.xlsx"

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

graph_mats = StateGraph(State)
graph_mats.add_node("convert_to_md", convert_to_md)
graph_mats.add_node("extraccion_materiales", extraccion_materiales)
graph_mats.add_node("mats_to_excel", mats_to_excel)

graph_mats.add_edge(START, "convert_to_md")
graph_mats.add_edge("convert_to_md", "extraccion_materiales")
graph_mats.add_edge("extraccion_materiales", "mats_to_excel")
graph_mats.add_edge("mats_to_excel", END)

app_mats = graph_mats.compile()
