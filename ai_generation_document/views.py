from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from .models import Report, Budget
from jinja2 import Template
import pdfkit
import csv
import json
import numpy as np
import joblib
from jinja2 import Environment, FileSystemLoader


def generate_report(request):
    """Vista para generar un informe de proyecto."""
    # Ejemplo de lógica para generación de informes
    project_name = request.GET.get('project_name', 'Proyecto Desconocido')
    report = Report.objects.create(
        project_name=project_name,
        total_cost=1000000,
        risks_summary="Resumen de riesgos...",
        viability="Viabilidad alta",
    )
    return JsonResponse({"message": "Informe generado", "report_id": report.id})

def generate_budget(request):
    """Vista para generar un presupuesto de proyecto."""
    project_name = request.GET.get('project_name', 'Proyecto Desconocido')
    budget_item = Budget.objects.create(
        project_name=project_name,
        item="Cemento",
        quantity=100,
        unit_price=50.00,
        total_price=5000.00
    )
    return JsonResponse({"message": "Presupuesto generado", "budget_id": budget_item.id})

def integrate_predictions_to_report(request):
    """Genera un informe que incluye predicciones del modelo."""
    project_name = request.GET.get('project_name', 'Proyecto Desconocido')
    # Aquí se integrarían los resultados del modelo predictivo
    model_results = {
        "Costo_Final_Predicho": 1200000,
        "Duracion_Real_Predicha": "14 meses",
        "Riesgos_Identificados": "Alto riesgo por disponibilidad de materiales."
    }

    report = Report.objects.create(
        project_name=project_name,
        total_cost=model_results["Costo_Final_Predicho"],
        risks_summary=model_results["Riesgos_Identificados"],
        viability="Moderada"
    )

    return JsonResponse({
        "message": "Informe generado con predicciones.",
        "report_id": report.id,
        "predicciones": model_results
    })

def integrate_predictions_to_budget(request):
    """Genera un presupuesto que incluye datos basados en predicciones del modelo."""
    project_name = request.GET.get('project_name', 'Proyecto Desconocido')
    # Supongamos que los modelos predicen cantidades ajustadas
    predicted_budget = [
        {"item": "Cemento", "quantity": 110, "unit_price": 55, "total_price": 6050},
        {"item": "Acero", "quantity": 220, "unit_price": 105, "total_price": 23100},
    ]

    # Guardar en la base de datos y generar respuesta
    for budget_item in predicted_budget:
        Budget.objects.create(
            project_name=project_name,
            item=budget_item["item"],
            quantity=budget_item["quantity"],
            unit_price=budget_item["unit_price"],
            total_price=budget_item["total_price"]
        )

    return JsonResponse({
        "message": "Presupuesto generado con predicciones.",
        "project_name": project_name,
        "predicciones": predicted_budget
    })

def generate_report_pdf(request):
    """Genera un informe en formato PDF basado en un reporte existente."""
    report_id = request.GET.get('report_id')
    report = Report.objects.filter(id=report_id).first()

    if not report:
        return JsonResponse({"error": "Informe no encontrado."}, status=404)

    # Plantilla para PDF
    template = Template('''
        <h1>Informe del Proyecto</h1>
        <h2>Datos Generales</h2>
        <ul>
          <li><strong>Nombre del Proyecto:</strong> {{ report.project_name }}</li>
          <li><strong>Costo Total:</strong> {{ report.total_cost }}</li>
          <li><strong>Resumen de Riesgos:</strong> {{ report.risks_summary }}</li>
          <li><strong>Viabilidad:</strong> {{ report.viability }}</li>
        </ul>
    ''')

    rendered_html = template.render(report=report)

    # Generar PDF con pdfkit
    pdf = pdfkit.from_string(rendered_html, False)
    response = HttpResponse(pdf, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="Informe_{report.id}.pdf"'
    return response

def project_dashboard(request):
    """Renderiza un panel para gestionar proyectos y visualizar predicciones."""
    reports = Report.objects.all()
    budgets = Budget.objects.all()
    context = {
        "reports": reports,
        "budgets": budgets,
    }
    return render(request, "dashboard.html", context)

def evaluate_feasibility(request):
    if request.method == "POST":
        try:
            modelo_viabilidad = joblib.load("modelo_viabilidad.pkl")
            data = json.loads(request.body)
            
            # Extraer características relevantes para la predicción
            features = np.array([
                data["desviacion_coste"],
                data["desviacion_tiempo"],
                data["Indice_Riesgo"],
                data["Experiencia_Contratista"],
                data["Zona_Sismica"],
                data["Tipo_Suelo"],
                data["Disponibilidad_Materiales_Actual"],
                data["Turnos_Trabajo_Actual"]
            ]).reshape(1, -1)
            
            # Hacer la predicción con el modelo
            viabilidad_predicha = modelo_viabilidad.predict(features)[0]
            riesgo = "Alto" if data["Indice_Riesgo"] > 0.7 else "Moderado"
            
            # Construir la respuesta
            response = {
                "viabilidad": "Viable" if viabilidad_predicha == 1 else "No Viable",
                "desviacion_coste": data["desviacion_coste"],
                "desviacion_tiempo": data["desviacion_tiempo"],
                "riesgo": riesgo
            }
            
            env = Environment(loader=FileSystemLoader('.'))

            # Generar informe PDF automáticamente
            template = env.get_template("informe_template.html")
            html_output = template.render(
                proyecto=data.get("proyecto", "Proyecto Desconocido"),
                coste_total=data["importe_presupuestado"],
                duracion=data["duracion_planificada"],
                riesgos=[riesgo],
                viabilidad=response["viabilidad"],
                conclusion="Revisión recomendada debido a riesgos elevados." if riesgo == "Alto" else "Proyecto viable con consideraciones estándar."
            )
            
            pdf_path = "informe_viabilidad.pdf"
            pdfkit.from_string(html_output, pdf_path)
            
            return JsonResponse(response, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    else:
        return JsonResponse({"error": "Método no permitido"}, status=405)
