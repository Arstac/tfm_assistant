# Path: chat/views.py
import json
import requests  # For making HTTP requests to external APIs
from .serializers import ChatMessageSerializer, ConversationSerializer  # Serializers to convert model instances to JSON
from django.views.decorators.csrf import csrf_exempt  # Decorator to exempt views from CSRF verification
from rest_framework.decorators import authentication_classes, permission_classes, api_view  # Decorators for REST framework views
from rest_framework.authentication import TokenAuthentication  # Token-based authentication system
from rest_framework.permissions import IsAuthenticated  # Permission class to ensure the user is authenticated
from rest_framework.parsers import JSONParser  # For parsing JSON data from requests
from django.http import JsonResponse  # To send JSON responses

from langchain_core.messages import HumanMessage

from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory  # In-memory chat history for LangChain
from langchain.memory import ConversationBufferMemory  # Memory buffer for storing conversation history
from langchain.chains import ConversationChain  # Chain to handle conversational interactions

from .agents import app, config
from .models import ChatMessage, Conversation  # Importing the ChatMessage and Conversation models
from rest_framework.response import Response
from .ml_model import (
    predict_costo_final,
    predict_duracion_real,
    predict_satisfaccion_cliente,
    predict_desviacion_presupuestaria
)

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
import os
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .utils.extract import extract_text_from_pdf, enhanced_segmenter, extract_data_with_regex
from .utils.bert_qa import BERTQA  # Asumiendo que tienes esta clase en utils



# API details for the title generator (using a Hugging Face model)
API_URL = "https://api-inference.huggingface.co/models/czearing/article-title-generator"
headers = {"Authorization": f"Bearer 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"}

def generate_title(payload):
    """
    Function to generate a title using an external API.
    It sends a POST request to the Hugging Face model with the input payload.
    """
    response = requests.post(API_URL, headers=headers, json=payload)
    print("RESPONSE", response.json())
    # Extract the generated text (title) from the response
    return response.json()[0]['generated_text']

def retrieve_conversation(title, user):
    """
    Retrieve the last few messages of a conversation from the database.
    This helps provide context to the model while being memory efficient.
    """
    # Number of recent messages to retrieve
    num_recent_conversations = 4

    # Get the conversation object for the given title and user
    conversation_obj = Conversation.objects.get(title=title, user=user)
    conversation_id = getattr(conversation_obj, 'id')
    
    # Retrieve the most recent conversation messages in reverse chronological order
    conversation_context = ChatMessage.objects.filter(
        conversation_id=conversation_id
    ).order_by('-timestamp')[:num_recent_conversations:-1]
    
    # Prepare a list to store input and output messages
    lst = []
    for msg in conversation_context:
        input_msg = getattr(msg, 'user_response')  # User's message
        output_msg = getattr(msg, 'ai_response')   # AI's response
        lst.append({"input": input_msg, "output": output_msg})
    
    # Save the retrieved messages into the conversation memory
    for x in lst:
        inputs = {"input": x["input"]}
        outputs = {"output": x["output"]}
        memory.save_context(inputs, outputs)
    
    # Create a ChatMessageHistory object with the messages in memory
    retrieved_chat_history = ChatMessageHistory(
        messages=memory.chat_memory.messages
    )

    return retrieved_chat_history

def store_message(user_response, ai_response, conversation_id):
    """
    Function to store a chat message in the database.
    It saves both the user's message and the AI's response.
    """
    ChatMessage.objects.create(
        user_response=user_response,
        ai_response=ai_response,
        conversation_id=conversation_id,
    )

def store_title(title, user):
    """
    Function to create a new Conversation in the database with the given title and user.
    """
    Conversation.objects.create(
        title=title,
        user=user
    )

@csrf_exempt  # Exempt the view from CSRF verification (use with caution)
@api_view(['POST', 'GET'])  # Allow GET and POST requests
@authentication_classes([TokenAuthentication])  # Use token-based authentication
@permission_classes([IsAuthenticated])  # Only allow access to authenticated users
def chat(request):
    """
    View function to handle chat interactions.
    Handles both fetching chat history (GET) and sending new messages (POST).
    """
    if request.method == 'GET':
        # Retrieve chat history
        request_data = JSONParser().parse(request)
        provided_title = request_data.get('title')  # Get the conversation title from the request
        user = request.user  # Get the authenticated user
        if provided_title:
            # Get the conversation object for the given title and user
            conversation_title = Conversation.objects.get(
                title=provided_title, user=user)
            conversation_id = getattr(conversation_title, 'id')
            # Retrieve all messages for the conversation, ordered by timestamp
            ChatObj = ChatMessage.objects.filter(
                conversation_id=conversation_id).order_by('timestamp')
            # Serialize the chat messages into JSON format
            Chat = ChatMessageSerializer(ChatObj, many=True)
            # Return the chat messages as a JSON response
            return JsonResponse(Chat.data, safe=False)
        else:
            # If no title is provided, return an error message
            return JsonResponse({'error': 'Title not provided'}, status=400)

    elif request.method == 'POST':
        # Handle sending a new message or starting a new conversation
        request_data = JSONParser().parse(request)
        prompt = request_data.get('prompt')  # Get the user's message from the request
        user = request.user  # Get the authenticated user
        provided_title = request_data.get('title')  # Get the conversation title (if provided)

        if provided_title:
            # Continue an existing conversation
            retrieved_chat_history = retrieve_conversation(
                provided_title, user)
            title = provided_title
        else:
            # Start a new conversation
            memory.clear()  # Clear any existing memory
            retrieved_chat_history = ChatMessageHistory(messages=[])
            title = None  # Title will be generated after getting the AI's response

        # Create a conversation chain with the language model and memory
        reloaded_chain = ConversationChain(
            llm=llm,
            memory=ConversationBufferMemory(
                chat_memory=retrieved_chat_history),
            verbose=True
        )

        # Generate the AI's response to the user's message
        #response = reloaded_chain.predict(input=prompt)
        response = app.invoke({"messages":prompt}, config=config)
        if not title:
            # If no title was provided, generate one based on the user's message
            title = generate_title({
                "inputs": prompt  # Alternatively, you could use the AI's response
            })
            # Store the new conversation with the generated title
            store_title(title, user)

        # Get or create the conversation in the database
        conversation_title, created = Conversation.objects.get_or_create(
            title=title, user=user)
        conversation_id = getattr(conversation_title, 'id')

        # Store the user's message and the AI's response in the database
        store_message(prompt, response, conversation_id)

        # Return the AI's response and the conversation title as a JSON response
        return JsonResponse({
            'ai_response': response,
            'title': title
        }, status=201)

@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])  
def get_title(request):
    """
    View function to retrieve all conversation titles for the authenticated user.
    """
    user = request.user  # Get the authenticated user
    titles = Conversation.objects.filter(user=user)  # Get all conversations for the user
    serialized = ConversationSerializer(titles, many=True)  # Serialize the data
    return JsonResponse(serialized.data, safe=False)  # Return the titles as a JSON response

@csrf_exempt   
@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated]) 
def delete_conversation(request):
    """
    View function to delete a conversation by its title.
    """
    user = request.user  # Get the authenticated user
    data = JSONParser().parse(request)
    title = data.get('title')  # Get the title of the conversation to delete
    obj = Conversation.objects.get(user=user, title=title)  # Get the conversation object
    obj.delete()  # Delete the conversation
    return JsonResponse("Deleted successfully", safe=False)

@csrf_exempt   
@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def get_data(request):
    """
    View function to retrieve chat messages for a specific conversation.
    """
    request_data = JSONParser().parse(request)
    provided_title = request_data.get('title')  # Get the conversation title from the request
    user = request.user  # Get the authenticated user
    if provided_title:
        # Get the conversation object for the given title and user
        conversation_title = Conversation.objects.get(
            title=provided_title, user=user)
        conversation_id = getattr(conversation_title, 'id')
        # Retrieve all messages for the conversation, ordered by timestamp
        ChatObj = ChatMessage.objects.filter(
            conversation_id=conversation_id).order_by('timestamp')
        # Serialize the chat messages
        Chat = ChatMessageSerializer(ChatObj, many=True)
        # Return the chat messages as a JSON response
        return JsonResponse(Chat.data, safe=False)
    else:
        # If no title is provided, return an error message
        return JsonResponse({'error': 'Title not provided'}, status=400)
    

@api_view(['POST'])
def chatbot_api(request):
    user_message = request.data.get('message')

    if not user_message:
        return Response({'error': 'No message provided'}, status=status.HTTP_400_BAD_REQUEST)
    
    chart_type = request.data.get('chart_type', 'scatter')  # Por defecto, dispersión
    x_column = request.data.get('x_column')
    y_column = request.data.get('y_column')
    # Aquí puedes manejar el historial de mensajes si es necesario
    # Para este ejemplo, simplemente enviamos el mensaje al modelo y obtenemos una respuesta

    #response = llm.invoke(user_message)
    response = app.invoke({"messages":[HumanMessage(content=user_message)],
                            "chart_type": chart_type,
                            "x_column": x_column,
                            "y_column": y_column
                        }, config=config)
    ai_message = response["messages"][-1].content
    tool_msg = response["messages"][-2].content
    print("AI Response:", ai_message)
    print("Tool Message:", tool_msg)
    
    if_tool_msg_dict = isinstance(tool_msg, dict)
    print("Is Tool Message a dictionary?", if_tool_msg_dict)
    
    is_chart_in_tool_msg = 'chart_data' in tool_msg
    print("Is there a chart in Tool Message?", is_chart_in_tool_msg)
    # si tool_msg es un diccionario y hay "chart_data" en él, entonces es un gráfico    
    if  is_chart_in_tool_msg:
        print("Tool Message is a chart")
        return Response({'response': tool_msg, 'type': 'chart'}, status=status.HTTP_200_OK)
    else:
        print("Tool Message is a text")
        return Response({'response': ai_message, 'type': 'text'}, status=status.HTTP_200_OK)

@api_view(['POST'])
def predict_view(request):
    if request.method == 'POST':
        try:
            # Carga los datos enviados por el cliente
            data = json.loads(request.body)
            features = data.get('features')  # Debe ser una lista de características

            if features is None or not isinstance(features, list):
                return JsonResponse({'error': 'Invalid input. "features" should be a list.'}, status=400)

            # Realiza la predicción
            result = predict(features)

            # Devuelve la predicción como respuesta JSON
            return JsonResponse({'prediction': result})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Only POST method is allowed.'}, status=405)

@api_view(["POST"])
def predict_costo_final_view(request):
    """API para predecir el Costo Final."""
    return _handle_prediction(request, predict_costo_final, "Costo_Final")

@api_view(["POST"])
def predict_duracion_real_view(request):
    """API para predecir la Duración Real."""
    return _handle_prediction(request, predict_duracion_real, "Duracion_Real")

@api_view(["POST"])
def predict_satisfaccion_cliente_view(request):
    """API para predecir la Satisfacción del Cliente."""
    return _handle_prediction(request, predict_satisfaccion_cliente, "Satisfaccion_Cliente")

@api_view(["POST"])
def predict_desviacion_presupuestaria_view(request):
    """API para predecir la Desviación Presupuestaria."""
    return _handle_prediction(request, predict_desviacion_presupuestaria, "Desviacion_Presupuestaria")

def _handle_prediction(request, prediction_function, variable_name):
    """Maneja las predicciones para una variable específica."""
    try:
        data = request.data.get("features", [])
        if not data:
            return Response({"error": "No se proporcionaron características"}, status=400)

        # Obtener la predicción específica llamando a la función correspondiente
        prediccion = prediction_function(data)

        return Response({variable_name: prediccion})

    except Exception as e:
        return Response({"error": str(e)}, status=500)

class UploadPDFView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        # 1. Obtener y guardar el archivo PDF
        pdf_file = request.FILES.get('file')
        if not pdf_file:
            return Response({"error": "No se subió ningún archivo."}, status=400)

        temp_file_path = f"/tmp/{pdf_file.name}"
        with open(temp_file_path, 'wb') as temp_file:
            for chunk in pdf_file.chunks():
                temp_file.write(chunk)

        try:
            # 2. Extraer texto del PDF
            pdf_text = extract_text_from_pdf(temp_file_path)
            if not pdf_text:
                return Response({"error": "No se pudo extraer texto del PDF."}, status=400)

            # 3. Segmentar el texto en secciones clave
            sections = enhanced_segmenter(pdf_text)

            # 4. Extracción estructurada con regex
            regex_data = extract_data_with_regex(pdf_text)

            # 5. Cargar modelo BERT
            try:
                bert_qa = BERTQA(use_finetuned=True)
            except Exception as e:
                bert_qa = BERTQA(use_finetuned=False)

            # 6. Preguntas clave con contexto optimizado
            questions = {
                "valor_estimado_bert": {
                    "pregunta": "¿Cuál es el valor estimado exacto del contrato en euros?",
                    "contexto": sections.get("objeto_contrato", pdf_text)
                },
                "plazo_ejecucion_bert": {
                    "pregunta": "¿Cuál es el plazo total de ejecución en meses?",
                    "contexto": sections.get("condiciones", pdf_text)
                },
                "clasificacion_cpv": {
                    "pregunta": "¿Cuál es el código CPV completo de la clasificación?",
                    "contexto": sections.get("proceso", pdf_text)
                }
            }

            # 7. Procesar preguntas con BERT
            bert_responses = {}
            for key, config in questions.items():
                answer = bert_qa.answer(
                    context=config["contexto"],
                    question=config["pregunta"]
                )
                bert_responses[key] = {
                    "respuesta": answer.get("answer", "No encontrado"),
                    "confianza": f"{answer.get('score', 0):.1%}" if 'score' in answer else "N/A"
                }

            # 8. Consolidar datos
            final_data = {
                **regex_data,
                "valor_bert": bert_responses["valor_estimado_bert"]["respuesta"],
                "plazo_bert": bert_responses["plazo_ejecucion_bert"]["respuesta"],
                "cpv_bert": bert_responses["clasificacion_cpv"]["respuesta"]
            }

            # 9. Guardar resultados en un archivo CSV
            os.makedirs("output", exist_ok=True)
            df = pd.DataFrame([final_data])
            csv_path = f"output/{pdf_file.name}_analizada.csv"
            df.to_csv(csv_path, index=False)

            return Response({
                "message": "Datos extraídos correctamente.",
                # "data": final_data,
                # "csv_path": csv_path
            }, status=200)

        except Exception as e:
            return Response({"error": f"Hubo un problema al procesar el archivo: {str(e)}"}, status=500)

        finally:
            # 10. Eliminar archivo temporal
            os.remove(temp_file_path)
