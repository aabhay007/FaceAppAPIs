# views.py
from rest_framework import viewsets, status
from rest_framework.response import Response
from .models import ValidatedImage
from .serializers import ValidatedImageSerializer
from .utils import validate_human_image
import base64
from io import BytesIO
from PIL import Image
from rest_framework.decorators import action
import numpy as np
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
    
    @action(detail=False, methods=['post'])
    def webcam(self, request, *args, **kwargs):
        """
        Handle image data sent from webcam (base64 or raw bytes).
        """
        image_data = request.data.get('image_data')  # Expecting base64 string

        if not image_data:
            return Response(
                {"error": "No image data provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(",")[1])
            image = Image.open(BytesIO(image_bytes)).convert('RGB')

            # Convert PIL image to numpy array
            img_array = np.array(image)

            # Validate the image
            is_valid, message, face_encoding = validate_human_image(BytesIO(image_bytes))

            if is_valid:
                # Save validated image to database
                validated_image = ValidatedImage.objects.create(
                    is_valid=True,
                    validation_message=message
                )
                validated_image.set_face_encoding(face_encoding)
                validated_image.save()

                return Response(
                    ValidatedImageSerializer(validated_image).data,
                    status=status.HTTP_201_CREATED
                )
            else:
                return Response({
                    'is_valid': False,
                    'validation_message': message
                }, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )