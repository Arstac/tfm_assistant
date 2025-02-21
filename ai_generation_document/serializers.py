from rest_framework import serializers
from django.core.validators import MaxValueValidator, MinValueValidator

class FeasibilitySerializer(serializers.Serializer):
    desviacion_coste = serializers.IntegerField(
        required=True,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
    )
    desviacion_tiempo = serializers.IntegerField(
        required=True,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
    )
    categoria_licitada = serializers.IntegerField(
        required=True,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
    )
    complejidad_general = serializers.IntegerField(
        required=True,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
    )

class RiskSerializer(serializers.Serializer):
    beneficios_esperados = serializers.FloatField(
        required=True,
        validators=[MinValueValidator(0), MaxValueValidator(1000000)],
    )
    dias_ejecucion_real = serializers.IntegerField(
        required=True,
        validators=[MinValueValidator(0), MaxValueValidator(1000)],
    )
    complejidad_general = serializers.IntegerField(
        required=True,
        validators=[MinValueValidator(0), MaxValueValidator(10)],
    )
    categoria_licitada = serializers.IntegerField(
        required=True,
        validators=[MinValueValidator(0), MaxValueValidator(10)],
    )

class CostVarianceSerializer(serializers.Serializer):
    importe_presupuestado_x = serializers.FloatField(
        required=True,
        validators=[MinValueValidator(0), MaxValueValidator(1000000)],
    )
    dias_ejecucion_real = serializers.IntegerField(
        required=True,
        validators=[MinValueValidator(0), MaxValueValidator(1000)],
    )
    segmento_cliente = serializers.IntegerField(
        required=True,
        validators=[MinValueValidator(0), MaxValueValidator(10)],
    )
    sector_industria = serializers.IntegerField(
        required=True,
        validators=[MinValueValidator(0), MaxValueValidator(10)],
    )