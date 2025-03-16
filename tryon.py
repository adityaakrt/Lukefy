import cv2
import numpy as np
import uuid
import os
from PIL import Image

def process_image(user_image_path, clothing_image_path, position=None):
    """
    Process the user image and clothing image to create a virtual try-on effect.
    
    Args:
        user_image_path: Path to the user's uploaded image
        clothing_image_path: Path to the clothing item image
        position: Dictionary with positioning parameters (x, y, scale)
        
    Returns:
        Path to the resulting image
    """
    # Default position if not provided
    if position is None:
        position = {'x': 0, 'y': 0, 'scale': 1.0}
    
    # Read the user image
    user_img = cv2.imread(user_image_path)
    if user_img is None:
        raise ValueError(f"Could not read user image from {user_image_path}")
    
    # Read the clothing image (with alpha channel for transparency)
    clothing_img = cv2.imread(clothing_image_path, cv2.IMREAD_UNCHANGED)
    if clothing_img is None:
        raise ValueError(f"Could not read clothing image from {clothing_image_path}")
    
    # Ensure clothing image has alpha channel
    if clothing_img.shape[2] < 4:
        # Create alpha channel if it doesn't exist
        b, g, r = cv2.split(clothing_img)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        clothing_img = cv2.merge((b, g, r, alpha))
    
    # For this simple version, we'll just do a basic overlay
    # In a real app, you would detect body pose and fit the clothing properly
    
    # Resize user image to a standard size (keeping aspect ratio)
    height, width = user_img.shape[:2]
    max_dim = 800
    if height > max_dim or width > max_dim:
        if height > width:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        else:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        
        user_img = cv2.resize(user_img, (new_width, new_height))
        height, width = user_img.shape[:2]
    
    # Resize clothing image to fit user (this is a simplification)
    # In a real app, you would size it based on detected body proportions
    clothing_height, clothing_width = clothing_img.shape[:2]
    target_width = int(width * 0.7 * position['scale'])  # 70% of user width, adjustable with scale
    target_height = int(clothing_height * (target_width / clothing_width))
    
    clothing_img = cv2.resize(clothing_img, (target_width, target_height))
    
    # Position the clothing on the user image
    # This is a basic centered overlay, adjustable with x, y position
    x_offset = int((width - target_width) / 2) + position['x']
    y_offset = int(height * 0.3) + position['y']  # Place at 30% from top, adjustable with y
    
    # Make sure the offsets are within bounds
    x_offset = max(0, min(width - target_width, x_offset))
    y_offset = max(0, min(height - target_height, y_offset))
    
    # Create a region of interest (ROI) for placing the clothing
    roi = user_img[y_offset:y_offset+target_height, x_offset:x_offset+target_width]
    
    # Separate the alpha channel from the color channels
    b, g, r, alpha = cv2.split(clothing_img)
    clothing_rgb = cv2.merge((b, g, r))
    
    # Create a mask from the alpha channel
    alpha = alpha / 255.0
    
    # Apply the alpha mask to blend the images
    for c in range(0, 3):
        roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * clothing_rgb[:, :, c]
    
    # Save the result image
    result_path = f"temp/result_{uuid.uuid4()}.jpg"
    cv2.imwrite(result_path, user_img)
    
    return result_path

def detect_face(image):
    """
    Detect the face in an image to help with positioning.
    
    Args:
        image: Image array
        
    Returns:
        Coordinates of face rectangle (x, y, width, height) or None if no face detected
    """
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Return the first face found, or None
    if len(faces) > 0:
        return faces[0]  # Returns (x, y, width, height)
    else:
        return None