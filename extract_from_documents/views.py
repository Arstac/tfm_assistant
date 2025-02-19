from django.shortcuts import render

# Create your views here.

def extract_info(request):
    return render(request, 'extract_from_documents/extract_info.html')
   