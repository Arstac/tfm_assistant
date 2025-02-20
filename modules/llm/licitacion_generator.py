import pandas as pd

from langgraph.graph import StateGraph, START, END

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tika import parser

from modules.load_prompts import prompt_extraccion_materiales
from .class_models import MatOutStr, State
from .load_llm_models import llm
from pydantic import BaseModel, Field
from .class_models import MatLicitacion, MatLicitationOutStr, StateLic



prompt_licitacion_generator = """
Eres un experto en licitaciones hidráulicas de instalación de tubería de fundición dúctil y polietileno junto con todos sus accesorios.

Debes generar una lista de materiales para la licitación ficticia que simule un caso real.

A continuación de proporciono un listado de material junto con sus descripciones para que puedas generar la lista de materiales:
<unidades>m. Tubería de fundición dúctil, ø <diametro> mm, clase C40, colocada. Tubería de fundición dúctil, 300
mm de diámetro nominal, y Clase de Presión C 40, fabricada según norma UNE EN 545:2011
con revestimiento exterior de cinc metálico, cubierto por una capa de acabado de un producto
bituminoso o de resina sintética compatible con el cinc, revestida interiormente con mortero de
cemento, colocada y montada en obra, incluye p.p. de unión flexible cuyos materiales
elastoméricos se ajusten a los requisitos de la norma EN 681-1. No se incluyen piezas
especiales, ni excavación, ni cama, ni extendido y relleno de la tierra procedente de la
excavación. Tubería y pruebas a cargo de Tragsa.
<unidades>m. Tubería PVC lisa saneamiento ø diametro mm, rig.4 kN/m², coloc. Tubería lisa de saneamiento
de PVC de 200 mm de diámetro nominal y 4 kN/m² de rigidez, unión con junta elástica,
incluyendo materiales a pie de obra, montaje, colocación. No incluye la excavación de la zanja,
ni el extendido y relleno de la tierra procedente de la misma, ni la cama, ni el material
seleccionado, ni su compactación. Tubería y pruebas a cargo de Tragsa.
<unidades>m. Tubería PVC orientado, ø diametro mm, 1,25 MPa, junta goma, colocada. Tubería de PVC
orientado de 200 mm de diámetro y 1,25 MPa de presión de servicio y unión por junta de goma,
incluyendo materiales a pie de obra, montaje, colocación y pruebas. No incluye las piezas
especiales ni la excavación de la zanja, ni el extendido y relleno de la tierra procedente de la
misma, ni la cama, ni el material seleccionado, ni su compactación. Tubería a cargo de Tragsa.
<unidades>ud. VÁLVULA DE COMPUERTA EMBRIDADA diametro mm PN-16 ENT. Montaje de válvula de
compuerta embridada de cierre elástico, diametro PN10/16. Accionada mediante eje de extensión
telescópico de la Serie 04/04-001, cuadradillo 23-32 de long. 1700-2900 mm, con todos los
materiales necesarios para la completa maniobra de la válvula enterrada, con parte proporcional
de juntas, tornillería y calderería y accesorios de unión a la tubería. Unidad totalmente montada
ejecutada y probada. Los materiales a cargo de Tragsa. Montaje y medios auxiliares por parte
del adjudicatario.
<unidades> ud. VÁLVULA MARIPOSA CONCENTRICA DN-<DN> PN-16 ENT. Montaje de válvula de
compuerta embridada de cierre elástico, DN<DN> PN10/16. Accionamiento mediante reductor y columna de maniobra de 2 m. montaje con el eje horizontal. 
Unidadmontada con parte proporcional de juntas, tornillería y calderería y accesorios de unión a la
tubería. Unidad totalmente montada ejecutada y probada. Los materiales a cargo de Tragsa.
Montaje y medios auxiliares por parte del adjudicatario.
<unidades>  m Montaje de tubería PE100, ø <diametro> mm,PN10, colocada. Los trabajos
incluyen la puesta en obra de la tubería, el tendido y distribución de la
misma en las zanjas ejecutadas por TRAGSA, así como la soldadura e
identificación de la misma. Los materiales serán suministrados por TRASA
en su almacén de obra.
<unidades> m Montaje de tubería PE100, ø <diametro> mm, PN10 colocada. Los trabajos
incluyen la puesta en obra de la tubería, el tendido y distribución de la
misma en las zanjas ejecutadas por TRAGSA, así como la soldadura e
identificación de la misma. Los materiales serán suministrados por TRASA
en su almacén de obra.
<unidades>  m Montaje de tubería PE100, ø <diametro> mm, PN16, colocada. Los trabajos
incluyen la puesta en obra de la tubería, el tendido y distribución de la
misma en las zanjas ejecutadas por TRAGSA, así como la soldadura e
identificación de la misma. Los materiales serán suministrados por TRASA
en su almacén de obra.

Con el mismo formato de descripcion, puedes crear nuevos materiales segun el titulo de la licitacion para relacioanrlos con la licitacion.

Un ejemplo sería:

Titulo: Montaje de tubería de polietileno alta densidad en la red secundaria de la obra de modernización de
regadío en la colectividad de Cuevas del Campo (Granada) Fases I y II, Cofinanciado con Fondos
FEADE

Lista de materiales:
Nº Uds. Ud. Descripción
<unidades>  m Montaje de tubería PE100, ø <diametro> mm, PN<PN>, colocada. Los trabajos
incluyen la puesta en obra de la tubería, el tendido y distribución de la
misma en las zanjas ejecutadas por TRAGSA, así como la soldadura e
identificación de la misma. Los materiales serán suministrados por TRASA
en su almacén de obra
<unidades>  m Montaje de tubería PE100, ø 32 mm, PN10, colocada. Los trabajos
incluyen la puesta en obra de la tubería, el tendido y distribución de la
misma en las zanjas ejecutadas por TRAGSA, así como la soldadura e
identificación de la misma. Los materiales serán suministrados por TRASA
en su almacén de obra.
<unidades>  m Montaje de tubería PE100, ø 40 mm, PN10, colocada. Los trabajos
incluyen la puesta en obra de la tubería, el tendido y distribución de la
misma en las zanjas ejecutadas por TRAGSA, así como la soldadura e
identificación de la misma. Los materiales serán suministrados por TRASA
en su almacén de obra.
<unidades>  m Montaje de tubería PE100, ø 50 mm, PN10, colocada. Los trabajos
incluyen la puesta en obra de la tubería, el tendido y distribución de la
misma en las zanjas ejecutadas por TRAGSA, así como la soldadura e
identificación de la misma. Los materiales serán suministrados por TRASA
en su almacén de obra.
<unidades> m Montaje de tubería PE100, ø 63 mm, PN10 colocada. Los trabajos
incluyen la puesta en obra de la tubería, el tendido y distribución de la
misma en las zanjas ejecutadas por TRAGSA, así como la soldadura e
identificación de la misma. Los materiales serán suministrados por TRASA
en su almacén de obra.
<unidades>  m Montaje de tubería PE100, ø 75 mm, PN10, colocada. Los trabajos
incluyen la puesta en obra de la tubería, el tendido y distribución de la
misma en las zanjas ejecutadas por TRAGSA, así como la soldadura e
identificación de la misma. Los materiales serán suministrados por TRASA
en su almacén de obra.
<unidades> m Montaje de tubería PE100, ø 90 mm, PN10, colocada. Los trabajos
incluyen la puesta en obra de la tubería, el tendido y distribución de la
misma en las zanjas ejecutadas por TRAGSA, así como la soldadura e
identificación de la misma. Los materiales serán suministrados por TRASA
en su almacén de obra.

Las unidades de medida son las siguientes:
m: metros
ud: unidades

Las cantidades deben ser proporcionales a la longitud de la tubería y a la cantidad de accesorios necesarios para su instalación.


Titulo de la licitación: {titulo_licitacion}
"""

from langgraph.graph import MessagesState





def licitation_generator(state: StateLic):
    """Generates a list of materials for a fictitious tender based on the title and material list provided"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_licitacion_generator),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    chain = prompt | llm.with_structured_output(MatLicitationOutStr)
    
    response = chain.invoke({"messages":state["messages"], "titulo_licitacion": state["titulo_licitacion"]})
    
    print(f"Response: {response.model_dump()['materiales']}")
    
    return {"materiales": response.model_dump()["materiales"]}
    
def mats_to_excel(state: StateLic):
    """Converts the structured output of the material extraction process to an Excel file"""

    # Extract the materials data from the state
    materials_data = state["materiales"]

    # Create a DataFrame from the materials data
    df = pd.DataFrame(materials_data)

    # Define the output Excel file path
    output_file = f"Data/materiales_{state['titulo_licitacion']}.xlsx"

    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False)

    return {"excel_file_path": output_file}
    

graph_lic = StateGraph(StateLic)
graph_lic.add_node("licitation_generator", licitation_generator)
graph_lic.add_node("mats_to_excel", mats_to_excel)
graph_lic.add_edge(START, "licitation_generator")
graph_lic.add_edge("licitation_generator", "mats_to_excel")
graph_lic.add_edge("mats_to_excel", END)

app_lic = graph_lic.compile()
