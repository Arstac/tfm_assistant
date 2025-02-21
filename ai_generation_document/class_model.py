from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from io import BytesIO

class FeasibilityReport(BaseModel):
    md_content: str = Field(None, title="Contenido Markdown", description="Informe de viabilidad en formato Markdown")
    feasibility: str = Field(None, title="Viabilidad", description="Valor viabilidad")

class RiskReport(BaseModel):
    md_content: str = Field(None, title="Contenido Markdown", description="Informe de viabilidad en formato Markdown")
    risk: str = Field(None, title="Riesgo", description="Valor riesgo")
    
class Feasibility(BaseModel):
    desviacion_coste: int = Field(None, title="Desviación de costes", description="Desviación de costes")
    desviacion_tiempo: int = Field(None, title="Desviación de tiempo", description="Desviación de tiempo")
    categoria_licitada: int = Field(None, title="Categoría licitada", description="Categoría licitada")
    complejidad_general: int = Field(None, title="Complejidad general", description="Complejidad general")
    viabilidad: int = Field(None, title="viabilidad", description="Viabilidad")

class Risk(BaseModel):
    beneficios_esperados: float = Field(None, title="Beneficios esperados", description="Beneficios esperados")
    dias_ejecucion_real: int = Field(None, title="Dias de ejecucion", description="Dias de ejecucion")
    categoria_licitada: int = Field(None, title="Categoría licitada", description="Categoría licitada")
    complejidad_general: int = Field(None, title="Complejidad general", description="Complejidad general")

class CostVariance(BaseModel):
    importe_presupuestado_x: float = Field(None, title="Beneficios esperados", description="Beneficios esperados")
    dias_ejecucion_real: int = Field(None, title="Dias de ejecucion", description="Dias de ejecucion")
    segmento_cliente: int = Field(None, title="Segmento cliente", description="Segmento cliente")
    sector_industria: int = Field(None, title="Sector industria", description="Sector industria")

class State(MessagesState):
    feasibility_content: Feasibility = Field(None, title="feasibility", description="Viabilidad")
    risk_content: Risk = Field(None, title="risk", description="Riesgo")
    md_content: str = Field(None, title="md_content", description="Contenido en markdown")
    prediction: int = Field(None, title="prediction", description="Contenido prediccion")
    graphics: BytesIO = Field(None, title="graphics", description="Grafico")