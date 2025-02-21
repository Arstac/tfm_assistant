from django.shortcuts import render
from django.http import JsonResponse
import pdfkit
import numpy as np
import joblib
from jinja2 import Environment, FileSystemLoader
from django.http import JsonResponse, FileResponse
import os
import pdfplumber
import pandas as pd
from weasyprint import HTML
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
import json
from .generation_document import generate_feasibility_report, generate_risk_report, generate_feasibility_report_pdf
from .class_model import State, Feasibility, Risk, CostVariance
import base64
import io
from .serializers import FeasibilitySerializer, RiskSerializer, CostVarianceSerializer
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser
from .ml_model import predict_cost_variance


@api_view(['POST'])
@parser_classes([JSONParser, FormParser, MultiPartParser])
def evaluate_feasibility(request):
    if request.method != "POST":
        return Response({"error": "Método no permitido"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

    serializer = FeasibilitySerializer(data=request.data)
    if serializer.is_valid():
        feasibility_content = Feasibility(**serializer.validated_data)
        result = generate_feasibility_report(feasibility_content)
    
        return result
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
@api_view(['POST'])
@parser_classes([JSONParser, FormParser, MultiPartParser])
def evaluate_feasibility_pdf(request):
    if request.method != "POST":
        return Response({"error": "Método no permitido"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

    serializer = FeasibilitySerializer(data=request.data)
    if serializer.is_valid():
        feasibility_content = Feasibility(**serializer.validated_data)
        result = generate_feasibility_report_pdf(feasibility_content)
    
        return result
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(["POST"])
@parser_classes([JSONParser, FormParser, MultiPartParser])
def evaluate_risk(request):
    if request.method != "POST":
        return Response({"error": "Método no permitido"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

    serializer = RiskSerializer(data=request.data)
    if serializer.is_valid():
        risk_content = Risk(**serializer.validated_data)
        result = generate_risk_report(risk_content)
        return result

    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(["POST"])
def cost_variance(request):
    if request.method == 'POST':
        try:
           
            serializer = CostVarianceSerializer(data=request.data)
            if serializer.is_valid():
                cost_content = CostVariance(**serializer.validated_data)
                result = predict_cost_variance(cost_content)
            
                return result
            else:
                return JsonResponse(serializer.errors, status=400)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Only POST method is allowed.'}, status=405)

