from ultralytics import YOLO
import cv2
import time
import numpy as np
import datetime
import subprocess
import tkinter as tk
from tkinter import Label, Entry, Button
from PIL import Image, ImageDraw, ImageFont, ImageTk
from collections import Counter
import json
import threading
import requests  
import tkinter.messagebox as messagebox
import logging
import timeit
# -----------------------------
# Tkinter UI Setup - Control Panel
# -----------------------------
import sys, os

# os.chdir(sys._MEIPASS)
# # Configure logging
# logging.basicConfig(filename="debug_log.txt", level=logging.DEBUG, 
#                     format="%(asctime)s - %(levelname)s - %(message)s")

# Root Window Configuration
root = tk.Tk()
root.title("Live Object Detection - Setup")
root.geometry("435x300")  # Set window size
root.maxsize(390, 300)
root.configure(bg='#FBFCEA')

# Root Grid Configuration for expansion
root.grid_rowconfigure(0, weight=1)  # Main row with control_frame

def center_window(width = 550, height = 550):
    #get screen width and height
    width = int(width)
    height = int(height)

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    #calculate position of x and y coordinates
    center_x = int(screen_width/2 - width / 2)
    center_y = int(screen_height/2 - height / 2)
    root.geometry(f'{width}x{height}+{center_x}+{center_y}')

# Initialize window size
center_window(390, 300)
# Control Frame
control_frame = tk.Frame(root, bg='#3acfcc')
control_frame.grid(row=0, column=0, padx=5, pady=5, columnspan=4, sticky="nsew")

# Control Frame Grid Configuration
control_frame.grid_rowconfigure(0, weight=1)  # Heading area
control_frame.grid_rowconfigure(1, weight=1)  # Text widget area
control_frame.grid_rowconfigure(2, weight=1)  # Input fields area
control_frame.grid_rowconfigure(3, weight=1)  # Button area

control_frame.grid_columnconfigure(0, weight=1)
control_frame.grid_columnconfigure(1, weight=1)
control_frame.grid_columnconfigure(2, weight=1)
control_frame.grid_columnconfigure(3, weight=1)

# Logo Setup
#logo = Image.open('VBR_logo.PNG')
#logo = logo.resize((75, 50), Image.LANCZOS)
#logo_img = ImageTk.PhotoImage(logo)
#root.iconphoto(True, logo_img)

#ogo_label = tk.Label(control_frame, image=logo_img, bg="#3acfcc")
#logo_label.grid(row=0, column=0, sticky="ew")

# Heading Label (adjusted for proper alignment)
heading_label = tk.Label(control_frame, text="User Configuration Setup File", font=("Courier", 12), bg="#3acfcc")
heading_label.grid(row=0, column=1, columnspan=3, pady=5, sticky="ew", padx=2)

# Text Widget for Instructions
T = tk.Text(control_frame, height=1, width=20, wrap="word", bg="#fbfcea", pady=20, fg="#052600")
T.grid(row=1, column=0, columnspan=4, pady=(7, 0), sticky="nsew")

# Insert Default Message
message = "Please define the width, height, and position."
T.insert(tk.END, message)
T.tag_configure("center", justify="center")
T.tag_add("center", "1.0", "end")
T.config(state="disabled")

# Input Labels and Entries
tk.Label(control_frame, text="X:").grid(row=2, column=0, sticky="e", padx=5)
x_entry = tk.Entry(control_frame, width=5)
x_entry.grid(row=2, column=1, sticky="w")

tk.Label(control_frame, text="Y:").grid(row=2, column=2, sticky="e", padx=5)
y_entry = tk.Entry(control_frame, width=5)
y_entry.grid(row=2, column=3, sticky="w")

tk.Label(control_frame, text="Width:").grid(row=3, column=0, sticky="e", padx=5)
width_entry = tk.Entry(control_frame, width=5)
width_entry.grid(row=3, column=1, sticky="w")

tk.Label(control_frame, text="Height:").grid(row=3, column=2, sticky="e", padx=5)
height_entry = tk.Entry(control_frame, width=5)
height_entry.grid(row=3, column=3, sticky="w")

# Start Button
start_button = tk.Button(control_frame, text="Set Position & Start", bd=2, command=lambda: start_stream(), bg="green", fg="white")
start_button.grid(row=4, column=1, columnspan=2, pady=10)

# Canvas for Video Display (after control panel is closed)
canvas = tk.Label(root)



# -----------------------------
# Global Variables and Settings
# -----------------------------
# These values will be set by user input.
user_x, user_y = 100, 100
user_width, user_height = 640, 360

stream_started = False  # Ensure the stream starts only once
PAD = 10                # Padding (in pixels) around the fixed ROI

# -----------------------------
# OpenCV and Model Setup
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Original resolution width
cap.set(4, 720)   # Original resolution height

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 15.0, (1280, 720))

model = YOLO('latest.pt')

prices = {
    'coke-bottle': '£1.25',
    'coke-can': '£1.05',
    'lemon puff': '75p',
    'pepsi-bottle': '£1.09',
    'pepsi-can': '£1.00'
}

product_data = {
    'coke-bottle': {'price': '£1.25', 'barcode': '1234567890'},
    'coke-can': {'price': '£1.05', 'barcode': '0987654321'},
    'lemon puff': {'price': '75p', 'barcode': '1111222233'},
    'pepsi-bottle': {'price': '£1.09', 'barcode': '4444555566'},
    'pepsi-can': {'price': '£1.00', 'barcode': '7777888899'}
}

classNames = ['Unknown', 'coke-bottle', 'coke-can', 'lemon puff', 'pepsi-bottle', 'pepsi-can']

def parse_price(price_str):
    price_str = price_str.lower().strip()
    if 'p' in price_str:
        return float(price_str.replace('p','').strip()) / 100
    elif '£' in price_str:
        return float(price_str.replace('£','').strip())
    return 0.0

def draw_text_with_pillow(image, text, position, font_path="arial.ttf", font_size=32,
                          text_color=(255,255,255), bg_color=(0,0,0)):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)
    bbox = draw.textbbox((0,0), text, font=font)
    x, y = position
    draw.rectangle([x, y, x+bbox[2]+10, y+bbox[3]+10], fill=bg_color)
    draw.text((x+5, y), text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Global list for current detections (names) for each frame.
current_detected_products = []

# -----------------------------
# Fixed ROI Setup (remains constant)
# -----------------------------
# Fixed ROI polygon in original 1280x720 coordinates.
area = [(847,135), (360,153), (385,499), (850,488)]
polygon = np.array(area, np.int32)
pts = polygon.reshape((-1,1,2))  # For drawing

# -----------------------------
# Tkinter & OpenCV Integration
# -----------------------------
def start_stream():
    """Read user inputs (position and size), set window geometry, destroy control panel, and start detection."""
    global user_x, user_y, user_width, user_height, stream_started
    try:
        logging.info("Starting stream setup...")
        user_x = int(x_entry.get())
        user_y = int(y_entry.get())
        user_width = int(width_entry.get())
        user_height = int(height_entry.get())
        root.geometry(f"{user_width}x{user_height}+{user_x}+{user_y}")
        root.maxsize(user_width, user_height)
        control_frame.destroy()
        canvas.pack(fill="both", expand=True)
        canvas.bind("<Button-1>", on_canvas_click)
        if not stream_started:
            stream_started = True
            threading.Thread(target=run_detection_loop, daemon=True).start()
    except ValueError as e:
        print("Invalid input! Please enter numbers for X, Y, Width, and Height.")
        logging.error(f"Invalid input! {e}")

unknown_boxes = []
product_boxes = []

def run_detection_loop():
    """Run detection in a separate thread and update the Tkinter canvas."""
    global current_detected_products, unknown_boxes, product_boxes
    start_time = timeit.default_timer()
    logging.info("Detection loop started.")
    unknown_boxes = []
    product_boxes = []
    while stream_started:
        success, frame = cap.read()
        if not success:
            logging.warning("Frame capture failed.")
            time.sleep(0.03)
            continue

        # Draw fixed ROI polygon on the full-resolution frame for reference.
        cv2.polylines(frame, [pts], True, (255, 0, 255), 2)

        # Compute the bounding rectangle of the fixed ROI in the original frame.
        x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(pts)
        # Add padding around the ROI.
        padded_x = max(0, x_roi - PAD)
        padded_y = max(0, y_roi - PAD)
        padded_x2 = min(frame.shape[1], x_roi + w_roi + PAD)
        padded_y2 = min(frame.shape[0], y_roi + h_roi + PAD)
        padded_roi = frame[padded_y:padded_y2, padded_x:padded_x2]

        # Fixed ROI relative to padded_roi.
        fixed_roi_rel = (x_roi - padded_x, y_roi - padded_y, w_roi, h_roi)

        # Run YOLO detection on the padded ROI.
        results = model(padded_roi, iou=0.5, conf=0.5)
        logging.info(f"Detection results: {results}")

        alert_message = ""
        current_detected_products = []  # Reset for this frame
        product_boxes = []
        if results and hasattr(results[0], 'boxes'):
            fr_x, fr_y, fr_w, fr_h = fixed_roi_rel
            for i, box in enumerate(results[0].boxes):
                # Detection coordinates are relative to padded_roi.
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                className = classNames[class_id] if class_id < len(classNames) else "Unknown"
                
                
                confidence = float(box.conf[0])
                if confidence < 0.6:
                    continue

                # Check if detection is fully inside the fixed ROI.
                fully_inside = (x1 >= fr_x and y1 >= fr_y and x2 <= fr_x + fr_w and y2 <= fr_y + fr_h)
                if fully_inside:
                    box_color = (0, 255, 0)
                    
                    if className != "Unknown":
                        current_detected_products.append(className)
                    else:
                        #unknown_boxes.append((x1, y1, x2, y2))
                        product_boxes.append((x1,y1, x2, y2, className))
                        
                else:
                    box_color = (0, 0, 255)
                    alert_message = f"Place {className} fully inside ROI!"

                price = parse_price(prices.get(className, "N/A"))
                label = f"{className} {price}"
                cv2.rectangle(padded_roi, (x1, y1), (x2, y2), box_color, 3)
                cv2.putText(padded_roi, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

                
                    

        if alert_message:
            cv2.putText(padded_roi, alert_message, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        # Compute total price based on current detections.
        product_counts = Counter(current_detected_products)
        total_price = sum(
            float(product_data[p]['price'].replace('£','').replace('p','')) /
            (100 if 'p' in product_data[p]['price'] else 1) * count
            for p, count in product_counts.items() if p in product_data
        )
        total_price_label = f"Total: £{total_price:.2f}"
        padded_roi = draw_text_with_pillow(padded_roi, total_price_label, (10,50), font_size=40, bg_color=(50,50,50))
    
            
        out.write(padded_roi)

        # Resize the padded ROI (with overlays) to fit the Tkinter window.
        display_frame = cv2.resize(padded_roi, (user_width, user_height), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        global pil_image
        pil_image = Image.fromarray(frame_rgb)
        update_canvas_with_detections(pil_image)
        root.after(0, update_canvas_with_detections, pil_image)

        # Prepare detected items with count.
        product_counts = Counter(current_detected_products)
        detected_items = [{"name": name, "barcode": product_data[name]["barcode"], "count": count}
                          for name, count in product_counts.items() if name in product_data]
        save_to_json(detected_items)
        send_detection_to_api(detected_items)    

        time.sleep(0.03)
        end_time = timeit.default_timer()
        elapsed_time = end_time - start_time
        print(elapsed_time)

def update_canvas_with_detections(pil_image_param):
    """Update canvas with the current frame."""
    global pil_image  # You can also use global if necessary
    pil_image = pil_image_param  # Store the image globally

    # Update canvas with the current frame
    image = ImageTk.PhotoImage(pil_image)
    canvas.img = image
    canvas.configure(image=image)

    # Bind the event to the canvas
    canvas.bind("<Button-1>", on_canvas_click)

   
def on_canvas_click(event):
    """Handle mouse click on canvas."""
    # Get mouse coordinates
    global product_boxes

    x, y = event.x, event.y

    # Find scaling factor
    scale_x = user_width / float(pil_image.width)
    scale_y = user_height / float(pil_image.height)

    # Adjust mouse click coordinates according to the scaling
    adjusted_x = x / scale_x
    adjusted_y = y / scale_y

   
     # Check for clicks within product boxes
    for (ux1, uy1, ux2, uy2, label) in product_boxes:
        if ux1 <= adjusted_x <= ux2 and uy1 <= adjusted_y <= uy2:
            # Show the confirmation dialog
            result = messagebox.askyesno("Run Training Module", f"Do you want to run the training module now for {label}?")
            if result:
                print(f"Running pipeline.py for {label}...")
                logging.info(f"Starting self-training module for {label}.")
                try:
                    subprocess.Popen(["python", "pipeline.py"])
                    logging.info(f"Self-training module started successfully for {label}.")
                except Exception as e:
                    logging.error(f"Error starting self-training module for {label}: {e}")
            break


def save_to_json(detected_items):
    with open("detected_items.json", "w", encoding="utf-8") as f:
        json.dump(detected_items, f, indent=4, ensure_ascii=False)

def send_detection_to_api(detected_items):
    url = "http://localhost:5000/api/products"  # Change to your API endpoint URL.
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=detected_items, headers=headers)
        if response.status_code != 200:
            print("API Error:", response.status_code)
            logging.error(f"API Error: {response.status_code}")
    except Exception as e:
        print("API Exception:", e)
        logging.error(f"API Exception: {e}")


logging.info("Script started.")
root.mainloop()
cap.release()
out.release()
cv2.destroyAllWindows()
logging.info("Script terminated.")
