from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ValidatedImageViewSet

router = DefaultRouter()
router.register(r'validated-images', ValidatedImageViewSet)

urlpatterns = [
    path('', include(router.urls)),
]