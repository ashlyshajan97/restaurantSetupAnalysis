from django.urls import path

from . import views
from search import ranking2

urlpatterns = [
    path('', views.index, name='index'),
    path('search', views.search, name='search'),
    path('rest/<str:loc>/', views.rest, name='rest'),
    path('insights/<str:loc>/', views.insights, name='insights')
]
