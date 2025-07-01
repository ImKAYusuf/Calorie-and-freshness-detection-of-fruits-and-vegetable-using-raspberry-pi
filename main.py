import re
import cv2
import base64
import requests
from time import sleep
from json import loads
from ultralytics import YOLO
from picamera2 import Picamera2

cam = Picamera2()
height = 480
width = 640
cam.configure(cam.create_video_configuration(main={"format": 'RGB888', "size": (width, height)}))
model = YOLO("yolov5mu.pt")

api_key = "MONaUie6cItOnhKm2ljhj4i69udW5rIb"
vlm_model = "pixtral-12b-2409"
shud_sleep = False

def encode_image_frame(image_frame):
    try:

        _, buffer = cv2.imencode('.jpg', image_frame)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return encoded_image

    except Exception as e:
        print(f"Error: {e}")

def vlm_response(api_key, model, messages):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    data = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")

cam.start()
while True:
    
    try:

        frame = cam.capture_array()
        #cv2.imshow('camera_feed', frame)
        # uncomment the above line to display the camera feed

        results = model(frame)
        boxes = results[0].boxes

        class_names = results[0].names
        for box in boxes:
            for i in box.cls:
                i = int(re.findall(r'\d+', str(i))[0])
            conf = box.conf[0]
            if i in [41, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]:
                shud_sleep = True
                b64_image = encode_image_frame(frame)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Check if the image shows an edible food item. If yes, provide the calorie count. Else set False & 0. {`edible`:<True/False>, `calories`:<calories>}"
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{b64_image}"
                            }
                        ]
                    }
                ]

                chat_response = vlm_response(api_key, vlm_model, messages)

                if chat_response:
                    model_response = loads(chat_response.get("choices")[0]['message']['content'])
                    if str(model_response.get('edible')).lower() == "true":
                        print(f"Edible Food Item. Approx Calories: {model_response.get('calories')} kcal")

            print(f"Class ID: {i} Class: {class_names.get(i)} Confidence: {conf:.2f}")

        if shud_sleep:
            sleep(10)
            shud_sleep = False


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        break

#{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
