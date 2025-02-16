from django.db import models

class Report(models.Model):
    project_name = models.CharField(max_length=255)
    total_cost = models.DecimalField(max_digits=12, decimal_places=2)
    risks_summary = models.TextField()
    viability = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.project_name

class Budget(models.Model):
    project_name = models.CharField(max_length=255)
    item = models.CharField(max_length=255)
    quantity = models.IntegerField()
    unit_price = models.DecimalField(max_digits=12, decimal_places=2)
    total_price = models.DecimalField(max_digits=12, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.project_name}: {self.item}"
    
class Feasibility(models.Model):
    project_name = models.CharField(max_length=255)
    item = models.CharField(max_length=255)
    quantity = models.IntegerField()
    unit_price = models.DecimalField(max_digits=12, decimal_places=2)
    total_price = models.DecimalField(max_digits=12, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.project_name}: {self.item}"
