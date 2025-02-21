from django.shortcuts import render

from modules import app_info, app_mats

import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def extract_info(request):
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido"}, status=405)

    try:
        request_data = json.loads(request.body.decode("utf-8"))  # Decodifica JSON correctamente
        document = request_data.get('path')

        if not document:
            return JsonResponse({"error": "No se recibió un documento válido"}, status=400)

        response = app_info.invoke({"doc": document, "messages": "Busca la información en la licitacion"})['DataInfo']
        return JsonResponse({"data": response})

    except json.JSONDecodeError:
        return JsonResponse({"error": "El cuerpo de la solicitud no es un JSON válido"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

from modules import calculate_cosine_similarity
@csrf_exempt
def question_answering(request):
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido"}, status=405)

    try:
        request_data = json.loads(request.body.decode("utf-8"))  # Decodifica JSON correctamente
        document = request_data.get('path')
        question = request_data.get('question')

        if not document:
            return JsonResponse({"error": "No se recibió un documento válido"}, status=400)

        if not question:
            return JsonResponse({"error": "No se recibió una pregunta válida"}, status=400)

        response = calculate_cosine_similarity(question, document)
        
        return JsonResponse({"data": response})

    except json.JSONDecodeError:
        return JsonResponse({"error": "El cuerpo de la solicitud no es un JSON válido"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    
    
    
    
@csrf_exempt
def extract_mats(request):
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido"}, status=405)

    try:
        request_data = json.loads(request.body.decode("utf-8"))  # Decodifica JSON correctamente
        document = request_data.get('path')

        if not document:
            return JsonResponse({"error": "No se recibió un documento válido"}, status=400)

        response = app_mats.invoke({"doc": document, "messages": "Busca la información en la licitacion"})['materiales']
        return JsonResponse({"data": response})

    except json.JSONDecodeError:
        return JsonResponse({"error": "El cuerpo de la solicitud no es un JSON válido"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
