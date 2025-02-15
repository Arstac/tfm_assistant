import pdfplumber
import re
import spacy
from typing import Dict

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrae texto de un PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def segment_text(text: str) -> Dict[str, str]:
    """Divide el texto usando los encabezados reales del PDF."""
    sections = {
        "Contrato": "",
        "Proceso": "",
        "Entidad": ""
    }
    current_section = None
    headers = [
        "# Anuncio de licitación",
        "## Contrato Sujeto a regulación armonizada",
        "## Proceso de Licitación",
        "## Entidad Adjudicadora",
    ]
    
    for line in text.split("\n"):
        if any(header in line for header in headers):
            current_section = next(key for key, val in sections.items() if val == "")
        elif current_section:
            sections[current_section] += line + "\n"
    return sections

def extract_entities(text: str) -> Dict[str, list]:
    """Extrae entidades con spaCy."""
    nlp = spacy.load("es_core_news_md")
    doc = nlp(text)
    entities = {"FECHAS": [], "MONTOS": [], "ENTIDADES": [], "CPV": []}
    for ent in doc.ents:
        if ent.label_ == "DATE":
            entities["FECHAS"].append(ent.text)
        elif ent.label_ == "MONEY":
            entities["MONTOS"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["ENTIDADES"].append(ent.text)
    return entities

def extract_data_with_regex(text: str) -> Dict[str, str]:
    """Extrae datos con regex específicos para el formato del PDF."""
    # Función auxiliar para evitar errores si no hay coincidencia
    def safe_search(pattern, text):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    return {
        "expediente": safe_search(r"Número de Expediente\s+(\d+)", text),
        "valor_estimado": safe_search(r"Valor estimado del contrato\s+([\d.,]+\s+EUR)", text),
        "cpv": safe_search(r"Clasificación CPV\s+→\s+(\d{8})", text),
        "plazo_ejecucion": safe_search(r"Plazo de Ejecución\s+(\d+)\s*Mes(?:\(es\))?", text),
        "fecha_presentacion": safe_search(r"Plazo de Presentación de Oferta\s+→\s+Hasta el (\d{2}/\d{2}/\d{4})", text),
    }

def enhanced_segmenter(text: str) -> Dict[str, str]:
    """Segmenta el texto usando reglas semánticas y NLP."""
    sections = {
        "metadata": "",         # N° expediente, fechas, entidad
        "objeto_contrato": "",  # Descripción y CPV
        "condiciones": "",      # Plazos, presupuesto, garantías
        "proceso": "",          # Licitación, pliegos, apertura
        "criterios": ""         # Adjudicación, ponderaciones
    }

    # Reglas semánticas (regex mejorados + keywords)
    patterns = {
        "metadata": r"(Número de Expediente|Publicado en la Plataforma|Entidad Adjudicadora)",
        "objeto_contrato": r"(Valor estimado del contrato)",
        "condiciones": r"(Plazo de Ejecución)",
        "proceso": r"(Clasificación CPV)",
        "criterios": r"(Criterios de Adjudicación|Ponderación|Juicio de valor)"
    }

    current_section = None
    for line in text.split("\n"):
        # Buscar secciones por patrones semánticos
        for section, pattern in patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                current_section = section
                break
        if current_section:
            sections[current_section] += line + "\n"

    return sections