from django.shortcuts import render
from django.http import HttpResponse
from .models import Article


# Create your views here.

def index(response, title):
    Article.objects.get(title=title)
    return HttpResponse('hi')
