from django.db import models
import json
import numpy as np 

class ValidatedImage(models.Model):
    image = models.ImageField(upload_to='validated_images/')
    is_valid = models.BooleanField(default=False)
    validation_message = models.CharField(max_length=255, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    face_encoding = models.TextField(blank=True, null=True)  # Store face encoding as JSON

    def set_face_encoding(self, encoding):
        if encoding is not None:
            self.face_encoding = json.dumps(encoding.tolist())
    
    def get_face_encoding(self):
        if self.face_encoding:
            return np.array(json.loads(self.face_encoding))
        return None

    def __str__(self):
        return f"Image {self.id} - Valid: {self.is_valid}"