from dotenv import load_dotenv

import gradio as gr
import pandas as pd
import os
import logging

load_dotenv()

from modules import app_info, app_mats, app_price


# #Prueba de funcionamiento de PDF a información
# response_data = app.invoke({"messages": "Extrae la info, porfavor", "doc": doc})

# print(response_data["DataInfo"])


# #Prueba de funcionamiento de PDF a información
# response_mats = app_mats.invoke({"messages": "Extrae la info, porfavor", "doc": doc})

# print(response_mats["materiales"])

# response_price = app_price.invoke({"messages": "Extrae la info, porfavor", "doc": doc})

# print(response_price["mat_prices"])

def extraer_info(file):
    print(f"extrayendo info de file: {file}")
    response_data = app_info.invoke({"messages": "Extrae la info, porfavor", "doc": file})
    df = pd.DataFrame(response_data["DataInfo"])
    df.columns = df.columns.astype(str) 
    output_path = "output.csv"
    df.to_csv(output_path, index=False)
    return output_path, df

# def extraer_materiales(file):
#     response_mats = app_mats.invoke({"messages": "Extrae la info, porfavor", "doc": file})
#     df = pd.DataFrame(response_mats["materiales"])
#     output_path = "materiales.csv"
#     df.to_csv(output_path, index=False)
#     return output_path, df

def buscar_precios():
    response_price = app_price.invoke({"messages": "Extrae la info, porfavor", "doc": "lic_2"})
    df = pd.DataFrame(response_price["mat_prices"])
    output_path = "precios.csv"
    df.to_csv(output_path, index=False)
    return output_path




# def main():
    
#     doc = "lic_2"
#     try:
#         with gr.Blocks() as app:
#             gr.Markdown("## Procesador de PDFs para Licitaciones")
            
#             with gr.Row():
#                 file1 = gr.File(label="Anuncio Lic 1 (PDF)")
#                 file2 = gr.File(label="Lic 1 (PDF)")
            
#             extract_info_btn = gr.Button("Extraer Info")
#             extract_materials_btn = gr.Button("Extraer Materiales")
#             output_csv = gr.File(label="Output CSV")
#             output_table = gr.Dataframe()
#             materials_csv = gr.File(label="Materiales CSV")
#             materials_table = gr.Dataframe()
            
#             extract_info_btn.click(extraer_info, inputs=[file1], outputs=[output_csv, output_table])
#             extract_materials_btn.click(extraer_materiales, inputs=[file2], outputs=[materials_csv, materials_table])
            
#             buscar_btn = gr.Button("Buscar Precios")
#             prices_csv = gr.File(label="Precios CSV")
            
#             buscar_btn.click(buscar_precios, inputs=[], outputs=prices_csv)

#         app.launch()

#     except Exception as e:
#         logging.error(f"Error al procesar el documento {doc}: {e}")


# if __name__ == "__main__":
#     main()


licitaciones = [
    "Ejecución de la red de polietileno PN10 y montaje de válvulas de fundición dúctil para la mejora del riego en la Vega de Granada",
    "Instalación de tubería de polietileno alta densidad PN16 y accesorios de fundición dúctil en la Comunidad de Regantes de Montijo (Badajoz)",
    "Proyecto de ampliación de la red con tuberías PEAD PN10 y collarines de fundición dúctil en el término municipal de La Rinconada (Sevilla)",
    "Suministro y soldadura de tubería de polietileno PN16 y válvulas compuerta de fundición para la modernización de riego en la zona regable del Bajo Guadalquivir",
    "Trabajos de instalación de tubería PEAD PN10 y bocas de riego de fundición dúctil en la red secundaria de la Comunidad de Regantes de Segorbe (Castellón)",
    "Renovación del sistema de riego mediante conducciones de polietileno PN16 y accesorios de fundición dúctil en la comarca de la Ribera del Júcar (Valencia)",
    "Montaje de tuberías de polietileno PN10 y válvulas de compuerta de fundición en la comarca de Sierra de Cádiz para optimizar la distribución de agua",
    "Reforma de la infraestructura de regadío con tubería de polietileno PN16 y piezas especiales de fundición dúctil en el Canal de la Margen Derecha del Ebro (Zaragoza)",
    "Ejecución de la red de polietileno PN10 y montaje de hidrantes de fundición dúctil en la Mancomunidad de Regantes del Alagón (Cáceres)",
    "Instalación de tubería PEAD PN16 y válvulas de compuerta de fundición para la modernización de la red de riego de la comunidad de regantes de Baza (Granada)",
    "Ampliación de la red secundaria con conducciones de polietileno PN10 y equipamiento de fundición dúctil en la cuenca del Tajo (Toledo)",
    "Montaje y soldadura de tuberías PEAD PN16 y collarines de fundición en la Mancomunidad de Usuarios del Bajo Duero (Valladolid)",
    "Reforma integral del sistema de riego con tubería de polietileno PN10 y válvulas compuerta de fundición en la Comunidad de Regantes de Alarcos (Ciudad Real)",
    "Proyecto de sustitución de tuberías de polietileno PN16 y piezas de fundición dúctil en la zona regable del Canal de Estremera (Madrid)",
    "Trabajos de renovación de conducciones de polietileno PN10 y montaje de hidrantes de fundición en la vega del río Eume (A Coruña)",
    "Instalación de tubería PEAD PN16 y elementos de fundición dúctil para la mejora del regadío en el Valle de Olid (Valladolid)",
    "Montaje de red principal de polietileno PN10 y bocas de riego de fundición en la Comunidad de Regantes de Salvatierra (Álava)",
    "Ejecución de la red de polietileno PN16 y válvulas de fundición dúctil en el proyecto de modernización de riego de la Cuenca del Genil (Córdoba)",
    "Reforma del sistema de riego mediante tubería PEAD PN10 y accesorios de fundición dúctil en la Comunidad de Regantes de Requena (Valencia)",
    "Instalación de conducciones de polietileno PN16 y montaje de collarines de fundición para la red de abastecimiento agrícola en la comarca del Mar Menor (Murcia)",
    "Modernización de la red de regadío con tubería de polietileno PN10 y bocas de riego de fundición dúctil en la zona de La Nava (Huelva)",
    "Proyecto de mejora de la red secundaria usando polietileno PN16 y válvulas de compuerta de fundición en la Comunidad de Regantes de Valderredible (Cantabria)",
    "Trabajos de canalización con tubería PEAD PN10 y accesorios de fundición dúctil para la optimización de riego en la comarca de Urgell (Lleida)",
    "Renovación integral de la infraestructura de riego con polietileno PN16 y arquetas de fundición en la zona regable de la Sierra de Alcaraz (Albacete)",
    "Ejecución de la red de polietileno PN10 y válvulas de compuerta de fundición para la Comunidad de Regantes de la Hoya de Huesca (Huesca)",
    "Instalación de tubería PEAD PN16 y bocas de riego de fundición dúctil para la ampliación de la red de abastecimiento en la Vega Baja (Alicante)",
    "Montaje y soldadura de tubería de polietileno PN10 y equipamiento de fundición en la Comunidad de Regantes del Bajo Ebro (Tarragona)",
    "Proyecto de sustitución de tubería PEAD PN16 y válvulas compuerta de fundición para la modernización de la red de riego de la comarca de Antequera (Málaga)",
    "Obras de refuerzo de la red secundaria con polietileno PN10 y accesorios de fundición dúctil en la Comunidad de Regantes de Medina de Rioseco (Valladolid)",
    "Renovación de la línea principal con tubería de polietileno PN16 y colocación de collarines de fundición en la comarca de La Sotonera (Huesca)",
    "Instalación de conducciones PEAD PN10 y sustitución de hidrantes de fundición en la Comunidad de Regantes de Elda (Alicante)",
    "Modernización del sistema de riego con tuberías de polietileno PN16 y válvulas de compuerta de fundición en la zona regable del Río Orbigo (León)",
    "Ejecución de la red secundaria mediante tubería PEAD PN10 y equipamiento de fundición dúctil en la Vega del Guadiana (Badajoz)",
    "Ampliación de la red de riego con tubería de polietileno PN16 y bocas de riego de fundición en la comunidad de regantes de Benavente (Zamora)",
    "Trabajos de sustitución de tubería PEAD PN10 y montaje de válvulas de compuerta de fundición para optimizar la red de riego en la comarca de Uribe (Vizcaya)",
    "Proyecto de refuerzo de la red principal con polietileno PN16 y collarines de fundición en la Mancomunidad del Alto Guadiana (Ciudad Real)",
    "Renovación de las conducciones de polietileno PN10 y piezas de fundición dúctil en la Comunidad de Regantes de la Ribera Baja (Navarra)",
    "Instalación de tubería PEAD PN16 y bocas de riego de fundición para la modernización de los regadíos del Canal del Páramo (León)",
    "Montaje y soldadura de polietileno PN10 y válvulas compuerta de fundición en la red de la Acequia Real del Júcar (Valencia)",
    "Proyecto de sustitución integral de tuberías PEAD PN16 y accesorios de fundición dúctil en la zona regable de la Campiña Sur (Badajoz)",
    "Obras de instalación de tubería de polietileno PN10 y montaje de hidrantes de fundición para la Comunidad de Regantes de Montsià (Tarragona)",
    "Renovación de la red con conducciones PEAD PN16 y piezas especiales de fundición en la mancomunidad de regadíos de Santomera (Murcia)",
    "Ejecución de la red de polietileno PN10 y elementos de fundición dúctil en la modernización de la Comunidad de Regantes del Bajo Adaja (Ávila)",
    "Instalación de tuberías PEAD PN16 y válvulas de compuerta de fundición para la mejora de la red de riego en la comarca de Campo de Calatrava (Ciudad Real)",
    "Trabajos de canalización con polietileno PN10 y bocas de riego de fundición en la comunidad de regantes de Alcarràs (Lleida)",
    "Proyecto de refuerzo del sistema de riego con tubería PEAD PN16 y accesorios de fundición dúctil en la zona regable del Bajo Tormes (Salamanca)",
    "Montaje y soldadura de tubería de polietileno PN10 y válvulas compuerta de fundición en la vega del Río Segura (Murcia)",
    "Renovación de conducciones PEAD PN16 y colocación de collarines de fundición para la mejora del sistema de riego en la comarca de La Serena (Badajoz)",
    "Instalación de la red secundaria con tubería de polietileno PN10 y bocas de riego de fundición en la comunidad de regantes de Padul (Granada)",
    "Modernización de la infraestructura de regadío mediante tubería PEAD PN16 y accesorios de fundición en la Vega del Carrión (Palencia)",
    "Proyecto de mejora de la red principal con tubería de polietileno PN10 y válvulas de compuerta de fundición en la Mancomunidad de Riegos de Calasparra (Murcia)",
    "Montaje de conducciones PEAD PN16 y piezas especiales de fundición para la optimización del regadío en la comarca de Somontano (Huesca)",
    "Renovación de la red de polietileno PN10 y colocación de hidrantes de fundición en la Comunidad de Regantes del Bajo Bidasoa (Navarra)",
    "Instalación de tubería PEAD PN16 y válvulas compuerta de fundición en la modernización de la zona regable del Guadarrama (Madrid)",
    "Proyecto de canalización con polietileno PN10 y bocas de riego de fundición para la reestructuración hídrica en la comarca de Tierra de Pinares (Valladolid)",
    "Trabajos de montaje de tubería PEAD PN16 y accesorios de fundición dúctil en la Comunidad de Regantes de la Alpujarra Baja (Granada)",
    "Ejecución de la red secundaria con polietileno PN10 y collarines de fundición en la Mancomunidad de Regantes del Bajo Aragón-Caspe (Zaragoza)",
    "Instalación de tuberías de polietileno PN16 y elementos de fundición para la modernización de la comunidad de regadíos de L'Horta Sud (Valencia)",
    "Montaje y soldadura de tubería PEAD PN10 y válvulas de compuerta de fundición para la mejora del abastecimiento agrícola en la comarca de Montilla (Córdoba)",
    "Reforma integral de la red de riego con polietileno PN16 y accesorios de fundición dúctil en la Comunidad de Regantes de Sierra de Gata (Cáceres)",
    "Proyecto de sustitución de conducciones de polietileno PN10 y montaje de hidrantes de fundición en la zona regable del Embalse de Guadalcacín (Cádiz)",
    "Modernización de las infraestructuras de riego con tubería PEAD PN16 y válvulas compuerta de fundición en la Mancomunidad de Regantes del Río Turia (Valencia)",
    "Instalación de conducciones de polietileno PN10 y piezas especiales de fundición para la mejora hídrica en la vega del Río Tajuña (Madrid)"
]

for licitacion in licitaciones:
    response = app_lic.invoke({"messages":"Genera una licitacion con materiales y unidades realistas en base a este titulo de licitacion proporcionado.", "titulo_licitacion": licitacion})
    print(response)
response = app_lic.invoke({"messages":"Genera una licitacion", "titulo_licitacion": "INSTALACIÓN DE TUBERÍA Y PIEZAS DE POLIETILENO DE ALTA DENSIDAD PARA LA MEJORA DEL SISTEMA DE REGADÍO DE VALLS(CATALUÑA)"})