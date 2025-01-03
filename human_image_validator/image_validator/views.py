# views.py
from rest_framework import viewsets, status
from rest_framework.response import Response
from .models import ValidatedImage
from .serializers import ValidatedImageSerializer
from .utils import validate_human_image

class ValidatedImageViewSet(viewsets.ModelViewSet):
    queryset = ValidatedImage.objects.all()
    serializer_class = ValidatedImageSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        
        if serializer.is_valid():
            image_file = request.FILES.get('image')
            is_valid, message, face_encoding = validate_human_image(image_file)
            
            # Only save to database if the image is valid (human face and not duplicate)
            if is_valid:
                validated_image = serializer.save(
                    is_valid=True,
                    validation_message=message
                )
                
                # Save face encoding
                if face_encoding is not None:
                    validated_image.set_face_encoding(face_encoding)
                    validated_image.save()
                
                return Response(
                    ValidatedImageSerializer(validated_image).data,
                    status=status.HTTP_201_CREATED
                )
            else:
                # Return error response without saving to database
                return Response({
                    'is_valid': False,
                    'validation_message': message
                }, status=status.HTTP_400_BAD_REQUEST)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)