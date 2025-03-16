from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import os
import uuid
import cv2
import numpy as np
from PIL import Image
import io
import cloudinary
import cloudinary.uploader
import cloudinary.api
from tryon import process_image

# Load environment variables
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create temp directory if it doesn't exist
os.makedirs('temp', exist_ok=True)

# Clothing items data (in a real app, this would come from a database)
CLOTHING_ITEMS = {
    1: {
        'name': 'T-Shirt',
        'image_path': 'clothes/tshirt.png',
        'position': {'x': 0, 'y': 0, 'scale': 1.0}
    },
    2: {
        'name': 'Jacket',
        'image_path': 'clothes/jacket.png',
        'position': {'x': 0, 'y': 0, 'scale': 1.0}
    },
    3: {
        'name': 'Sweater',
        'image_path': 'clothes/sweater.png',
        'position': {'x': 0, 'y': 0, 'scale': 1.0}
    }
}

@app.route('/api/try-on', methods=['POST'])
def try_on():
    try:
        # Check if files and data are present
        if 'user_image' not in request.files:
            return jsonify({'error': 'No user image provided'}), 400
        
        clothing_id = request.form.get('clothing_id')
        if not clothing_id:
            return jsonify({'error': 'No clothing item selected'}), 400
        
        clothing_id = int(clothing_id)
        if clothing_id not in CLOTHING_ITEMS:
            return jsonify({'error': 'Invalid clothing item'}), 400
        
        # Get the clothing item data
        clothing_item = CLOTHING_ITEMS[clothing_id]
        
        # Save user image temporarily
        user_image = request.files['user_image']
        user_image_path = f"temp/user_{uuid.uuid4()}.jpg"
        user_image.save(user_image_path)
        
        # Get clothing image path
        clothing_image_path = os.path.join('assets', clothing_item['image_path'])
        
        # Process images
        result_image_path = process_image(
            user_image_path, 
            clothing_image_path,
            position=clothing_item['position']
        )
        
        # Return processed image
        return send_file(result_image_path, mimetype='image/jpeg')
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Failed to process image'}), 500
    
    finally:
        # Clean up temporary files
        if 'user_image_path' in locals() and os.path.exists(user_image_path):
            os.remove(user_image_path)
        if 'result_image_path' in locals() and os.path.exists(result_image_path):
            # Keep file for now since we're sending it, but you might want to implement
            # a cleanup schedule or delete after sending
            pass

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True)