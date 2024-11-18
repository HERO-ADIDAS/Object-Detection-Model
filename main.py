import io
import base64
import numpy as np
import cv2
from PIL import Image as PILImage
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from matplotlib import cm

app = FastAPI()

class ImageRequest(BaseModel):
    image: str  # Base64-encoded image string

class ObjectDetectionSystem:
    def __init__(self, model_path='Model/yolov8l.pt'):
        self.yolo_model = YOLO(model_path)
        self.color_map = cm.get_cmap("tab20", len(self.yolo_model.names))  # Generate distinct colors

    def detect_objects(self, image_base64: str, confidence_threshold: float = 0.5):
        try:
            # Decode Base64 and convert to NumPy array
            image_data = base64.b64decode(image_base64)
            pil_image = PILImage.open(io.BytesIO(image_data)).convert("RGB")
            image_np = np.array(pil_image)

            # Resize image for consistent processing
            input_size = 640  # Example size for YOLO models
            height, width, _ = image_np.shape
            scale = input_size / max(height, width)
            resized_image = cv2.resize(image_np, (int(width * scale), int(height * scale)))

            # Perform object detection
            results = self.yolo_model(resized_image)

            # Draw results on image
            output_image = resized_image.copy()
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id]

                    if confidence > confidence_threshold:
                        # Generate color for this class
                        color = tuple(int(c * 255) for c in self.color_map(class_id)[:3])

                        # Draw bounding box
                        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

                        # Add label with confidence
                        label = f'{class_name} ({confidence:.2f})'
                        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        label_y = max(y1, label_size[1] + 10)
                        cv2.rectangle(output_image, (x1, label_y - label_size[1] - 10),
                                      (x1 + label_size[0], label_y + baseline - 10), color, -1)
                        cv2.putText(output_image, label, (x1, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Convert back to Base64
            output_pil = PILImage.fromarray(output_image)
            buffered = io.BytesIO()
            output_pil.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return encoded_image

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Initialize the detector with a more advanced model
detector = ObjectDetectionSystem('Model/yolov8l.pt')

@app.post("/detect")
async def detect_objects(request: ImageRequest):
    try:
        result_image = detector.detect_objects(request.image)
        return {"processed_image": result_image}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Object Detection API is running"}
