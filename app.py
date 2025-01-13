from flask import Flask, request, jsonify, render_template, Response
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load the YOLO model
model = YOLO('yolov8n.pt')  # model path

# sve yolo klase
class_names = model.names  

# Hardcoded 
nutritional_data = {
    "apple": {"calories": 52, "carbs": 14, "protein": 0.3, "fat": 0.2},
    "banana": {"calories": 89, "carbs": 23, "protein": 1.1, "fat": 0.3},
    "orange": {"calories": 47, "carbs": 12, "protein": 0.9, "fat": 0.1},
    "grape": {"calories": 67, "carbs": 17, "protein": 0.6, "fat": 0.2},
    # moremo dodat još
}

prediction_history = []

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400

        results = model.predict(image, conf=0.5) # more i strože
        predictions = results[0].boxes.data.tolist() if results[0].boxes else []

        if not predictions:
            return jsonify({"predictions": [], "image": None}), 200
        
        for box in predictions:
            x1, y1, x2, y2, conf, cls = map(float, box[:6])
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2)) 
            class_name = class_names[int(cls)] 
            label = f'{class_name} ({conf:.2f})'

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, img_encoded = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        predictions_with_details = [
            {
                "class_name": class_names[int(box[5])],
                "confidence": box[4],
                "bounding_box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "nutrition": nutritional_data.get(class_names[int(box[5])], "Nutritional info not available")
            }
            for box in predictions
        ]
        prediction_history.append(predictions_with_details)

        return jsonify({"predictions": predictions_with_details, "image": img_base64})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/history', methods=['GET'])
def history():
    if prediction_history:
        return jsonify({"history": prediction_history})
    else:
        return jsonify({"message": "No prediction history available."})


def generate_frames():
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    fruit_classes = ["apple", "banana", "orange", "grape", "pineapple"]  # Add more fruits as needed

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, conf=0.5)
        predictions = results[0].boxes.data.tolist() if results[0].boxes else []
        for box in predictions:
            x1, y1, x2, y2, conf, cls = map(float, box[:6])
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  
            class_name = class_names[int(cls)]  

            if class_name in fruit_classes:
                label = f'{class_name} ({conf:.2f})'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/live_feed', methods=['GET', 'POST'])
def live_feed():
    print('Live feed started')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live', methods=['GET', 'POST'])
def live():
    return render_template('live.html')


if __name__ == '__main__':
    app.run(port = 5097, debug=True) # ne pitajte zaš koristin port 5097
