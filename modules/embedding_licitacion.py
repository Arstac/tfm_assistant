from sentence_transformers import SentenceTransformer
import pandas as pd
from tika import parser
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
from spacy.lang.es import Spanish
from langchain_core.prompts import ChatPromptTemplate
from modules.llm.load_llm_models import llm


def text_chunking(text, chunk_size=5, overlap=1):
    """Divide el texto en chunks semánticos con ventana deslizante"""
    nlp = Spanish()
    nlp.add_pipe("sentencizer")
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    chunks = []
    start = 0
    while start < len(sentences):
        end = min(start + chunk_size, len(sentences))
        chunks.append(" ".join(sentences[start:end]))
        start += chunk_size - overlap
    return chunks

def calculate_cosine_similarity(pregunta_usuario, path_pliego):
    pregunta_usuario = pregunta_usuario.lower()
    # Modelo de embeddings
    print("cargando modelo")
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    pregunta_usuario_embeddings = model.encode(pregunta_usuario)

    print("pregunta del usuario codificada")
    # Procesar PDF con chunking mejorado
    ocr_pliego = parser.from_file(path_pliego)
    ocr_text = " ".join(ocr_pliego['content'].split())

    #ocr_pliego_chunks = ocr_pliego['content'].split("\n")
    ocr_pliego_chunks = text_chunking(ocr_text)  # Chunks semánticos

    # Generar embeddings
    pliego_embeddings = model.encode(ocr_pliego_chunks)

    print("pliego codificado")
    
    THRESHOLD = 0.7
    resultados = []

    # for normativa in normativas_embeddings:
    #     similarities = model.similarity(normativa, pliego_embeddings)
    #     print(max(similarities))

    similitudes = cosine_similarity(
        pregunta_usuario_embeddings.reshape(1, -1),
        pliego_embeddings
    )
    
    print("similitudes calculadas")
    max_similitud = np.max(similitudes)

    # Opcional: muestra cuál chunk logró la similitud máxima
    best_chunk_index = np.argmax(similitudes)
    best_chunk_text = ocr_pliego_chunks[best_chunk_index]
    
    # los 3 mejores textos
    best_chunks_indices = np.argsort(similitudes[0])[::-1][:3]
    best_chunks_texts = [ocr_pliego_chunks[i] for i in best_chunks_indices]

    resultados.append({
        "Pregunta": pregunta_usuario,
        "Similitud": max_similitud,
        "Mejor chunk": best_chunk_text,
        "Mejores chunks": best_chunks_texts
    })
    
    
    prompt_QA ="""
    Eres experto en question answering. Dada la siguiente informacion encontrada en un pliego de licitacion, dar respuesta a la pregunta del usuario:
    Pregunta: {pregunta_usuario}
    
    Informacion del pliego:
    {best_chunks_texts}
    """
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_QA),
        ("user", "Generate the chart."),
    ]
    )
    chain = prompt | llm
    
    response = chain.invoke({"pregunta_usuario":pregunta_usuario, "pliego": path_pliego})
    
    print(f"Response: {response.content}")
    
    return response.content
    