from django.shortcuts import render
import cv2
import threading
import numpy as np
from django.http import StreamingHttpResponse, JsonResponse
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
from django.views.decorators.csrf import csrf_exempt

# Initialize Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="WOcnEgGkFTPZf6g1hyUH"
)
MODEL_ID = "smart_cart-mio9y-7gylq/1"

# Class Colors & Prices
CLASS_COLORS = {
    "Ariel": (0, 255, 0), "Coke": (255, 0, 0), "Colgate": (255, 0, 255),
    "Colin": (0, 255, 255), "Dettol": (0, 128, 0), "Harpic": (0, 0, 255),
    "Ketchup": (255, 182, 193), "Oreo": (25, 25, 112), "Patanjali Dish Soap": (173, 255, 47),
    "WaiWai": (255, 215, 0)
}
CLASS_PRICES = {
    "Coke": 50, "Ariel": 120, "Colgate": 45, "Colin": 60, "Dettol": 80,
    "Harpic": 70, "Ketchup": 55, "Oreo": 40, "Patanjali Dish Soap": 30, "WaiWai": 20,
}

# Store detected items
latest_items = {}

# Video Capture
cap = cv2.VideoCapture(0)
latest_frame = None
latest_result = None
lock = threading.Lock()

def infer():
    """Continuously perform object detection and update detected items."""
    global latest_frame, latest_result, latest_items

    while True:
        with lock:
            if latest_frame is None:
                continue
            frame_copy = latest_frame.copy()

        # Convert to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
        pil_image.save("temp_frame.jpg")

        # Run inference
        result = CLIENT.infer("temp_frame.jpg", model_id=MODEL_ID)
        predictions = result.get("predictions", [])

        with lock:
            latest_result = predictions
            latest_items = {}  # ✅ Reset detected items before each detection
            
            for detection in predictions:
                class_name = detection["class"]
                if class_name in CLASS_PRICES:
                    if class_name in latest_items:
                        latest_items[class_name]["count"] += 1
                    else:
                        latest_items[class_name] = {"count": 1, "price": CLASS_PRICES[class_name]}

        print("📌 Updated Items:", latest_items)  # ✅ Debugging log


# Start inference thread
thread = threading.Thread(target=infer, daemon=True)
thread.start()

@csrf_exempt
def get_detected_items(request):
    """API endpoint to get detected items"""
    with lock:
        items_list = [{"name": key, "count": value["count"], "price": value["price"]} for key, value in latest_items.items()]
    
    print("Detected Items:", items_list)  # ✅ Debugging Log
    return JsonResponse({"items": items_list})


def generate_frames():
    """Generate video frames with bounding boxes for streaming."""
    global latest_frame, latest_result

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        with lock:
            latest_frame = frame.copy()

        if latest_result:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            for detection in latest_result:
                x, y, w, h = detection["x"], detection["y"], detection["width"], detection["height"]
                class_name = detection["class"]
                confidence = detection["confidence"]

                x0, y0 = int(x - w / 2), int(y - h / 2)
                x1, y1 = int(x + w / 2), int(y + h / 2)

                # Draw bounding box
                color = CLASS_COLORS.get(class_name, (255, 255, 255))
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                draw.text((x0, y0 - 10), f"{class_name} {confidence:.2f}", fill=color)

            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

def video_feed(request):
    """Django view that streams video."""
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, "index.html")
