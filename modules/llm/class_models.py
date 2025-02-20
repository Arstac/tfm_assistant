from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

class OutStr(BaseModel):
    # - Titulo
    # - Importe total
    # - Plazo ejecución
    # - Tipo de contrato
    # - Subtipo de contrato
    # - Contrato mixto
    # - Lugar de ejecución
    # - Clasificación CPV
    # - Entidad Adjudicadora
    titulo: str = Field(None, title="Titulo", description="Titulo del proyecto")
    importe_total: str = Field(None, title="Importe total", description="Importe total")
    plazo_ejecucion: str = Field(None, title="Plazo de ejecución", description="Plazo de ejecución")
    tipo_contrato: str = Field(None, title="Tipo de contrato", description="Tipo de contrato")
    subtipo_contrato: str = Field(None, title="Subtipo de contrato", description="Subtipo de contrato")
    contrato_mixto: str = Field(None, title="Contrato mixto", description="Contrato mixto")
    lugar_ejecucion: str = Field(None, title="Lugar de ejecución", description="Lugar de ejecución")
    clasificacion_CPV: str = Field(None, title="Clasificación CPV", description="Clasificación CPV")
    entidad_adjudicadora: str = Field(None, title="Entidad Adjudicadora", description="Entidad Adjudicadora")
    
class MatLicitacion(BaseModel):
    descripcion: str = Field(None, title="Material", description="Material")
    cantidad: str = Field(None, title="Cantidad", description="Cantidad")
    unidad: str = Field(None, title="Unidad", description="Unidad de medida")

class MatLicitationOutStr(BaseModel):
    materiales: list[MatLicitacion] = Field(None, title="Materiales", description="Materiales")



class Mat(BaseModel):
    descripcion: str = Field(None, title="Material", description="Material")
    cantidad: str = Field(None, title="Cantidad", description="Cantidad")
    unidad: str = Field(None, title="Unidad", description="Unidad")
    # precio_unitario: float = Field(None, title="Precio unitario", description="Precio unitario")
    # importe: float = Field(None, title="Importe", description="Importe")
    ubicaion: str = Field(None, title="Ubicación", description="Ubicación dento del documento")
    referencia: str = Field(None, title="Referencia", description="Codigo de referencia del material")
    
class MatOutStr(BaseModel):
    materiales: list[Mat] = Field(None, title="Materiales", description="Materiales")
   
class MatPrice(BaseModel):
    descripcion: str = Field(None, title="Material", description="Material")
    cantidad: str = Field(None, title="Cantidad", description="Cantidad")
    unidad: str = Field(None, title="Unidad", description="Unidad")
    precio_unitario: float = Field(None, title="Precio unitario", description="Precio unitario")
    importe: float = Field(None, title="Importe", description="Importe")
    
    
class MatPricesOutStr(BaseModel):
    materiales: list[MatPrice] = Field(None, title="Materiales", description="Materiales")
    
class State(MessagesState):
    DataInfo: OutStr = Field(None, title="DataInfo", description="Información de los datos")
    md_content: str = Field(None, title="md_content", description="Contenido en markdown")
    md_content_parts: list[str] = Field(None, title="md_content_parts", description="Contenido en markdown dividido en partes")
    materiales: list[Mat] = Field(None, title="Materiales", description="Materiales")
    doc: str = Field(None, title="Documento", description="Documento")    
    mat_prices: list[MatPrice] = Field(None, title="MatPrices", description="Materiales con precios")

class StateLic(MessagesState):
    materiales: list[MatLicitacion] = Field(None, title="Materiales", description="Materiales")
    titulo_licitacion: str = Field(None, titule="Titulo", description="Titulo de la licitación" )
