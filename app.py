from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Update path to your model if necessary

# Load class names
class_names = model.names  # Mapping from class IDs to class names

# Hardcoded nutritional data
nutritional_data = {
    "apple": {"calories": 52, "carbs": 14, "protein": 0.3, "fat": 0.2},
    "banana": {"calories": 89, "carbs": 23, "protein": 1.1, "fat": 0.3},
    "orange": {"calories": 47, "carbs": 12, "protein": 0.9, "fat": 0.1},
    "grape": {"calories": 67, "carbs": 17, "protein": 0.6, "fat": 0.2},
    # Add more as needed
}

# List to store prediction history
prediction_history = []

@app.route('/')
def index():
    return render_template('index.html')  # Render the index.html file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is provided
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({"error": "No file uploaded"}), 400

        # Read image
        file = request.files['file']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400

        # Run YOLO prediction
        results = model.predict(image, conf=0.5)
        predictions = results[0].boxes.data.tolist() if results[0].boxes else []

        if not predictions:
            return jsonify({"predictions": [], "image": None}), 200

        # Annotate the image with bounding boxes and class names
        for box in predictions:
            x1, y1, x2, y2, conf, cls = map(float, box[:6])
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # Convert to integers
            class_name = class_names[int(cls)]  # Get class name
            label = f'{class_name} ({conf:.2f})'

            # Draw the bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode the image as base64
        _, img_encoded = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        # Prepare predictions with class names and nutritional info
        predictions_with_details = [
            {
                "class_name": class_names[int(box[5])],
                "confidence": box[4],
                "bounding_box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "nutrition": nutritional_data.get(class_names[int(box[5])], "Nutritional info not available")
            }
            for box in predictions
        ]

        # Add predictions to history
        prediction_history.append(predictions_with_details)

        # Return predictions and annotated image
        return jsonify({"predictions": predictions_with_details, "image": img_base64})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

# Route to fetch prediction history
@app.route('/history', methods=['GET'])
def history():
    if prediction_history:
        return jsonify({"history": prediction_history})
    else:
        return jsonify({"message": "No prediction history available."})

if __name__ == '__main__':
    app.run(debug=True)
