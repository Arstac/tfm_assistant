# from langgraph.graph import StateGraph, START, END

# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from markitdown import MarkItDown

# from modules.load_prompts import prompt_extraccion_datos
# from .class_models import OutStr, State
# from .load_llm_models import llm

# # config = {
# #         "configurable": {
# #             "thread_id": "123" ,
# #         }
# #     }

# def get_current_step(state: State):
#     # Lógica para obtener el paso actual
#     return state["current_step"]

# def convert_to_md(state: State):
#     md = MarkItDown()
#     result = md.convert(f"{state['doc']}")

#     return {"md_content": result.text_content}


# def extraccion_datos(state: State):
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", prompt_extraccion_datos),
#             MessagesPlaceholder(variable_name="messages"),
#         ]
#     )
    
#     chain = prompt | llm.with_structured_output(OutStr)
    
#     response = chain.invoke({"messages":state["messages"], "md_content": state["md_content"]})
    
#     print(f"Response: {response.model_dump()}")
    
#     return {"DataInfo": response}

# graph = StateGraph(State)
# graph.add_node("convert_to_md", convert_to_md)
# graph.add_node("extraccion_datos", extraccion_datos)

# graph.add_edge(START, "convert_to_md")
# graph.add_edge("convert_to_md", "extraccion_datos")
# graph.add_edge("extraccion_datos", END)

# app_info = graph.compile()

from django.shortcuts import render

from modules import app_info
# Create your views here.

# @csrf_exempt  # Exempt the view from CSRF verification (use with caution)
# @api_view(['POST', 'GET'])  # Allow GET and POST requests
# @authentication_classes([TokenAuthentication])  # Use token-based authentication
# @permission_classes([IsAuthenticated])  # Only allow access to authenticated users
# def chat(request):
#     """
#     View function to handle chat interactions.
#     Handles both fetching chat history (GET) and sending new messages (POST).
#     """
#     if request.method == 'GET':
#         # Retrieve chat history
#         request_data = JSONParser().parse(request)
#         provided_title = request_data.get('title')  # Get the conversation title from the request
#         user = request.user  # Get the authenticated user
#         if provided_title:
#             # Get the conversation object for the given title and user
#             conversation_title = Conversation.objects.get(
#                 title=provided_title, user=user)
#             conversation_id = getattr(conversation_title, 'id')
#             # Retrieve all messages for the conversation, ordered by timestamp
#             ChatObj = ChatMessage.objects.filter(
#                 conversation_id=conversation_id).order_by('timestamp')
#             # Serialize the chat messages into JSON format
#             Chat = ChatMessageSerializer(ChatObj, many=True)
#             # Return the chat messages as a JSON response
#             return JsonResponse(Chat.data, safe=False)
#         else:
#             # If no title is provided, return an error message
#             return JsonResponse({'error': 'Title not provided'}, status=400)

#     elif request.method == 'POST':
#         # Handle sending a new message or starting a new conversation
#         request_data = JSONParser().parse(request)
#         prompt = request_data.get('prompt')  # Get the user's message from the request
#         user = request.user  # Get the authenticated user
#         provided_title = request_data.get('title')  # Get the conversation title (if provided)

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
