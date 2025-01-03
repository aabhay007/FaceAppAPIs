# utils.py
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import face_recognition
import io
import logging
import traceback
from .models import ValidatedImage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_face_encoding(image_array):
    """Get face encoding using face_recognition library"""
    try:
        # Resize image to improve performance
        height, width = image_array.shape[:2]
        max_size = 800
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            image_array = cv2.resize(image_array, new_size)
        
        # Convert to RGB (face_recognition requires RGB)
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Detect face locations first
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        if not face_locations:
            logger.info("No faces found in the image during encoding")
            return None
            
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if face_encodings:
            logger.info("Successfully generated face encoding")
            return face_encodings[0]
        
        logger.info("No face encodings generated")
        return None
        
    except Exception as e:
        logger.error(f"Error in get_face_encoding: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def check_duplicate_face(new_encoding, threshold=0.6):
    """Check if face already exists in database"""
    try:
        if new_encoding is None:
            logger.info("No encoding provided for duplicate check")
            return False, None
        
        existing_images = ValidatedImage.objects.exclude(face_encoding='')
        
        for existing_image in existing_images:
            try:
                existing_encoding = existing_image.get_face_encoding()
                if existing_encoding is not None:
                    face_distance = face_recognition.face_distance([existing_encoding], new_encoding)[0]
                    logger.info(f"Face distance with image {existing_image.id}: {face_distance}")
                    if face_distance < threshold:
                        return True, existing_image
            except Exception as e:
                logger.error(f"Error comparing with image {existing_image.id}: {str(e)}")
                continue
        
        return False, None
        
    except Exception as e:
        logger.error(f"Error in check_duplicate_face: {str(e)}")
        logger.error(traceback.format_exc())
        return False, None

def validate_human_image(image_file):
    """Validate if image contains a human face and check for duplicates"""
    mp_face_detection = None
    try:
        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

        # Read and convert image
        image = Image.open(image_file)
        image = image.convert('RGB')
        img_array = np.array(image)
        
        # Process with MediaPipe
        results = mp_face_detection.process(img_array)
        
        if not results.detections:
            logger.info("No faces detected by MediaPipe")
            return False, "No human face detected in the image", None
        
        if len(results.detections) > 1:
            logger.info("Multiple faces detected by MediaPipe")
            return False, "Multiple faces detected. Please upload an image with exactly one person", None
        
        logger.info("Single face detected by MediaPipe, proceeding to face encoding")
        
        # Convert image for face_recognition
        face_encoding = get_face_encoding(img_array)
        if face_encoding is None:
            logger.info("Could not generate face encoding")
            return False, "Could not generate face encoding. Please try a clearer image", None
            
        # Check for duplicates
        is_duplicate, duplicate_image = check_duplicate_face(face_encoding)
        if is_duplicate:
            logger.info(f"Duplicate face found with image ID: {duplicate_image.id}")
            return False, f"Duplicate face found! This person was already uploaded (Image ID: {duplicate_image.id})", None
        
        logger.info("Image validation successful")
        return True, "Valid human image detected", face_encoding
        
    except Exception as e:
        logger.error(f"Error in validate_human_image: {str(e)}")
        logger.error(traceback.format_exc())
        return False, f"Error processing image: {str(e)}", None
        
    finally:
        if mp_face_detection:
            mp_face_detection.close()