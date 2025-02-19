from langgraph.graph import StateGraph, START, END

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from markitdown import MarkItDown

from modules.load_prompts import prompt_extraccion_datos
from .class_models import OutStr, State
from .load_llm_models import llm

# config = {
#         "configurable": {
#             "thread_id": "123" ,
#         }
#     }

def get_current_step(state: State):
    # Lógica para obtener el paso actual
    return state["current_step"]

def convert_to_md(state: State):
    md = MarkItDown()
    result = md.convert(f"{state['doc']}")

    return {"md_content": result.text_content}


def extraccion_datos(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_extraccion_datos),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    chain = prompt | llm.with_structured_output(OutStr)
    
    response = chain.invoke({"messages":state["messages"], "md_content": state["md_content"]})
    
    print(f"Response: {response.model_dump()}")
    
    return {"DataInfo": response}

graph = StateGraph(State)
graph.add_node("convert_to_md", convert_to_md)
graph.add_node("extraccion_datos", extraccion_datos)

graph.add_edge(START, "convert_to_md")
graph.add_edge("convert_to_md", "extraccion_datos")
graph.add_edge("extraccion_datos", END)

app_info = graph.compile()
