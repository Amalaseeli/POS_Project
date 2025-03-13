from ultralytics import YOLO
import cv2
import time
import numpy as np
import json
import requests
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import subprocess

# -----------------------------
# OpenCV and Model Setup
# -----------------------------
cap = cv2.VideoCapture(0)  # Change to 0 if using default webcam
cap.set(3,1280)
cap.set(4,780)
#cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 60)  # Increase FPS
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

frame_count = 0
frame_skip= 2
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 15.0, (1280, 720))

model = YOLO('latest.pt')

# Product Data
product_data = {
    'coke-bottle': {'price': '£1.25', 'barcode': '1234567890'},
    'coke-can': {'price': '£1.05', 'barcode': '0987654321'},
    'lemon puff': {'price': '75p', 'barcode': '1111222233'},
    'pepsi-bottle': {'price': '£1.09', 'barcode': '4444555566'},
    'pepsi-can': {'price': '£1.00', 'barcode': '7777888899'}
}

classNames = ['Unknown', 'coke-bottle', 'coke-can', 'lemon puff', 'pepsi-bottle', 'pepsi-can']

# Fixed ROI for product placement
PAD = 50
area = [(847, 135), (360, 153), (385, 499), (850, 488)]
polygon = np.array(area, np.int32).reshape((-1, 1, 2))

def parse_price(price_str):
    price_str = price_str.lower().strip()
    if 'p' in price_str:
        return float(price_str.replace('p', '').strip()) / 100
    elif '£' in price_str:
        return float(price_str.replace('£', '').strip())
    return 0.0

def draw_text_with_pillow(image, text, position, font_size=32, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("arial.ttf", font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    x, y = position
    draw.rectangle([x, y, x + bbox[2] + 10, y + bbox[3] + 10], fill=bg_color)
    draw.text((x + 5, y), text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def save_to_json(detected_items):
    with open("detected_items.json", "w") as f:
        json.dump(detected_items, f, indent=4)

def send_detection_to_api(detected_items):
    url = "http://localhost:5000/api/products"
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=detected_items, headers=headers)
        if response.status_code != 200:
            print("API Error:", response.status_code)
    except Exception as e:
        print("API Exception:", e)

# -----------------------------
# OpenCV Detection Loop
# -----------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        time.sleep(0.03)
        continue
    found_unknown = False
    # Draw fixed ROI on frame
    cv2.polylines(frame, [polygon], True, (255, 0, 255), 2)

    # Get bounding box for ROI
    x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(polygon)
    padded_x, padded_y = max(0, x_roi - PAD), max(0, y_roi - PAD)
    padded_x2, padded_y2 = min(frame.shape[1], x_roi + w_roi + PAD), min(frame.shape[0], y_roi + h_roi + PAD)
    padded_roi = frame[padded_y:padded_y2, padded_x:padded_x2]

    # frame_count += 1
    # if frame_count % frame_skip != 0:  
        #continue 
    # Run YOLO detection
    results = model(padded_roi, iou=0.5, conf=0.5)

    alert_message = ""
    detected_products = []

    if results and hasattr(results[0], 'boxes'):
        fr_x, fr_y, fr_w, fr_h = x_roi - padded_x, y_roi - padded_y, w_roi, h_roi
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            className = classNames[class_id] if class_id < len(classNames) else "Unknown"
            confidence = float(box.conf[0])

            if confidence < 0.6:
                continue
            
            if className == "Unknown":
                found_unknown = True

            fully_inside = (x1 >= fr_x and y1 >= fr_y and x2 <= fr_x + fr_w and y2 <= fr_y + fr_h)
            box_color = (0, 255, 0) if fully_inside else (0, 0, 255)
            if fully_inside:
                detected_products.append(className)

            price = parse_price(product_data.get(className, {}).get("price", "N/A"))
            label = f"{className} {price}"
            cv2.rectangle(padded_roi, (x1, y1), (x2, y2), box_color, 3)
            cv2.putText(padded_roi, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            if not fully_inside:
                alert_message = f"Place {className} fully inside ROI!"
    
    if alert_message:
        cv2.putText(padded_roi, alert_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    if found_unknown:
        alert_text = "Do you want to run the self-training pipeline now?"
        cv2.putText(frame, alert_text, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),3 )
        cv2.imshow("Object Detection", frame)
        cv2.waitKey(2000)

        subprocess.run(["python", "pipeline.py"])


    # Calculate total price
    product_counts = Counter(detected_products)
    total_price = sum(
        parse_price(product_data[p]['price']) * count
        for p, count in product_counts.items() if p in product_data
    )
    total_price_label = f"Total: £{total_price:.2f}"
    padded_roi = draw_text_with_pillow(padded_roi, total_price_label, (10, 50), font_size=40, bg_color=(50, 50, 50))

    out.write(padded_roi)

    # Display the frame
    cv2.imshow("Object Detection", padded_roi)
    #cv2.moveWindow("Object Detection", 0, 0)

    # Save detected items
    detected_items = [{"name": name, "barcode": product_data[name]["barcode"], "count": count}
                      for name, count in product_counts.items() if name in product_data]
    save_to_json(detected_items)
    send_detection_to_api(detected_items)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
