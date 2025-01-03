from rest_framework import serializers
from .models import ValidatedImage

class ValidatedImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ValidatedImage
        fields = ['id', 'image', 'is_valid', 'validation_message', 'uploaded_at']
        read_only_fields = ['is_valid', 'validation_message']