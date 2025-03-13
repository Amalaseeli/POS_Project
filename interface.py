import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import cv2
import yaml
from tqdm import tqdm
from tkinter import PhotoImage
from PIL import Image, ImageTk
import threading
import datetime
import random
import shutil
import numpy as np
from ultralytics import YOLO
from tkinter import ttk

BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, "train/images")
TRAIN_LABEL_DIR = os.path.join(DATASET_DIR, "train/labels")
VAL_IMAGE_DIR = os.path.join(DATASET_DIR, "val/images")
VAL_LABEL_DIR = os.path.join(DATASET_DIR, "val/labels")
AUGMENT_DIR = os.path.join(DATASET_DIR, "augmented")
TEST_IMAGE_DIR = os.path.join(DATASET_DIR, "test/images")
TEST_LABEL_DIR = os.path.join(DATASET_DIR, "test/labels")
LOG_FILE = os.path.join(BASE_DIR, "log.txt")



for path in [TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, VAL_IMAGE_DIR, VAL_LABEL_DIR, AUGMENT_DIR, TEST_IMAGE_DIR, TEST_LABEL_DIR]:
    os.makedirs(path, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "latest.pt") 
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

default_classNames = ['Unknown', 'coke-bottle', 'coke-can', 'lemon puff', 'pepsi-bottle', 'pepsi-can']

root = tk.Tk()
root.title("Self-training module")
#root.geometry("400x400+50+50")
root.minsize(200,200)
root.maxsize(550,520)


root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)


icon_image = PhotoImage(file="VBR_logo.PNG")
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
center_window(550, 520)
#root.iconbitmap('visual_business_retail_limited_logo.ico')
img = Image.open('VBR_logo.png')

# Resize the image (optional, adjust size as needed)
img = img.resize((32, 32))  # Resize to 32x32 pixels

# Convert it to a Tkinter-compatible image
icon = ImageTk.PhotoImage(img)
root.iconphoto(True, icon)
root.configure(bg='#FBFCEA')

main_frame = tk.Frame(root, padx = 10, pady = 10, bg='#FBFCEA', width=550, height=500, highlightbackground="#eff299", highlightthickness=2)
main_frame.grid(row=0, column=0, sticky="nsew")

main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)
# logo = img.resize((75, 50), Image.LANCZOS)
# logo_img = ImageTk.PhotoImage(logo)

# logo_label = tk.Label(main_frame, image=logo_img)
# logo_label.grid(row=0, column=0, padx=2, sticky="nsew")

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}{msg}]\n"

    with open(LOG_FILE, "a") as f:
        f.write(log_message)

def update_yaml_with_new_class(new_class):
    """Update dataset.yaml to include new class names."""
    data = {'train': TRAIN_IMAGE_DIR, 'val': VAL_IMAGE_DIR, 'nc': 0, 'names': []}

    if os.path.exists(DATA_YAML):
        with open(DATA_YAML, 'r') as f:
            try:
                loaded_data = yaml.safe_load(f)
                if isinstance(loaded_data, dict) and 'names' in loaded_data:
                    data = loaded_data
            except yaml.YAMLError as e:
                log(f"Error reading YAML: {e}")

    if new_class not in data['names']:
        data['names'].append(new_class)
        data['nc'] = len(data['names'])

    with open(DATA_YAML, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    return data

cap_live = None             
live_feed_running = False
def start_live():
    global cap_live, live_feed_running
    cap_live = cv2.VideoCapture(0)
    if not cap_live.isOpened():
        messagebox.showerror("Error", "Unable to access webcam.")
        return
    live_feed_running = True
    log("Live feed started.")
    update_live_feed()

def update_live_feed():
    global current_image
    if not live_feed_running:
        return
    ret, frame = cap_live.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        current_image = pil_img
        disp_img = pil_img.resize((300, 250))
        tk_img = ImageTk.PhotoImage(disp_img)
        live_feed_label.config(image=tk_img)
        live_feed_label.image = tk_img
    live_feed_label.after(30, update_live_feed)

def stop_live():
    global live_feed_running, cap_live
    live_feed_running = False
    if cap_live is not None:
        cap_live.release()
    live_feed_label.config(image='')
    log("Live feed stopped.")

current_image = None
captured_counts = {}
def capture_and_annotate():
    """Captures an image from the live feed and allows annotation."""
    global current_image

    if current_image is None:
        messagebox.showwarning("No Image", "No image available to capture.")
        return

    new_class = class_entry.get().strip()
    if not new_class:
        messagebox.showwarning("No Name", "Enter a product name.")
        return

    
    update_yaml_with_new_class(new_class)

    image_dir = TRAIN_IMAGE_DIR 
    os.makedirs(image_dir, exist_ok=True)

    if new_class not in captured_counts:
        captured_counts[new_class] = 0
    preview_image(current_image, new_class, image_dir)

def preview_image(image, new_class, image_dir):
    """Show a preview of the captured image and ask user to save or discard it."""
    # create a window to show the captured image
    preview_window = tk.Toplevel()
    preview_window.title("Preview Image")
    preview_window.maxsize(300,300)

    # Convert the PIL image to a Tkinter-compatible photo image
    tk_img = ImageTk.PhotoImage(image.resize((300, 200)))
    # Display the image in the new window
    image_label = tk.Label(preview_window, image=tk_img)
    image_label.image = tk_img
    image_label.pack(padx=10, pady=10)

     # Ask the user if they want to save or discard the image
    def on_save():
        # Generate the filename and save the image
        image_filename = f"{new_class}_{captured_counts[new_class]}.jpg"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path)
        captured_counts[new_class] += 1

        log(f"Image saved: {image_path}")
        annotate_image(image_path, new_class)
        update_gallery(new_class)

        preview_window.destroy()  # Close the preview window

    def on_discard():
        log("Image capture discarded.")
        preview_window.destroy()  # Close the preview window

    # Add save/discard buttons
    save_button = tk.Button(preview_window, text="Save Image", command=on_save, bg="green", fg="white")
    save_button.pack(side="left", padx=20, pady=10)

    discard_button = tk.Button(preview_window, text="Discard Image", command=on_discard, bg="red", fg="white")
    discard_button.pack(side="right", padx=20, pady=10)   


def annotate_image(image_path, class_label):
    global bounding_boxes
    bounding_boxes = []  # Reset current annotations
    
    messagebox.showinfo("Instructions", "Please draw a box around the product by clicking and dragging.")

    annot_window = tk.Toplevel(root, bg='#FBFCEA')
    annot_window.title("Annotate Image")
    annot_window.maxsize(300, 350)
    img = Image.open(image_path)
    disp_width, disp_height = 300, 300
    img_resized = img.resize((disp_width, disp_height))
    tk_img = ImageTk.PhotoImage(img_resized)
    
    canvas = tk.Canvas(annot_window, width=disp_width, height=disp_height, bg="white")
    canvas.grid(row=0, column=0, columnspan=1)
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
    canvas.img = tk_img  # Prevent garbage collection
    
    start_coords = {}
    box_rect_id = None
    
    def on_mouse_press(event):
        start_coords['x'] = event.x
        start_coords['y'] = event.y
    
    def on_mouse_release(event):
        nonlocal box_rect_id
        canvas.delete(box_rect_id)  # Remove previous box if exists
        
        end_x, end_y = event.x, event.y
        x1, x2 = sorted([start_coords['x'], end_x])
        y1, y2 = sorted([start_coords['y'], end_y])
        box_rect_id = canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
        
        norm_x_center = ((x1 + x2) / 2) / disp_width
        norm_y_center = ((y1 + y2) / 2) / disp_height
        norm_width = (x2 - x1) / disp_width
        norm_height = (y2 - y1) / disp_height
        
        bounding_boxes.clear()
        bounding_boxes.append((norm_x_center, norm_y_center, norm_width, norm_height))
        
        confirm = messagebox.askyesno("Confirm", "Is the box correctly drawn around the product?")
        if not confirm:
            canvas.delete(box_rect_id)  # Remove incorrect bounding box
            bounding_boxes.clear()
    
    canvas.bind("<ButtonPress-1>", on_mouse_press)
    canvas.bind("<ButtonRelease-1>", on_mouse_release)
    
    def save_and_close():
        if not bounding_boxes:
            messagebox.showerror("Error", "No box drawn. Please draw before saving.")
            return
        
        try:
            class_id = default_classNames.index(class_label)
        except ValueError:
            class_id = len(default_classNames)
        
        save_annotations(image_path, bounding_boxes, class_label)
        log(f"Annotations saved for {os.path.basename(image_path)}")
        augment_images(class_label)

        annot_window.destroy()
        
    
    btn_save = tk.Button(annot_window, text="Save Annotations", command=save_and_close, bg="blue", fg="white")
    btn_save.grid(row=1, column=0, pady=1, columnspan=2, sticky="n")

def save_annotations(image_file, boxes, class_name):
    """Save annotations in YOLO format inside train/labels or val/labels."""
    label_dir = TRAIN_LABEL_DIR
    os.makedirs(label_dir, exist_ok=True)

    # Load class ID from dataset.yaml
    with open(DATA_YAML, 'r') as f:
        data = yaml.safe_load(f)
        print("Class names from YAML:", data['names'])
    class_id = data['names'].index(class_name)
    # Save annotation in YOLO format
    annotation_filename = os.path.splitext(os.path.basename(image_file))[0] + ".txt"
    annotation_path = os.path.join(label_dir, annotation_filename)

    with open(annotation_path, "w") as f:
        for box in boxes:
            x_center, y_center, width, height = box
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    log(f"Annotation saved at {annotation_path}")

def update_gallery(new_class):
    for widget in gallery_scrollable_frame.winfo_children():
        widget.destroy()

    image_dir = TRAIN_IMAGE_DIR  # Keep images directly here

    if os.path.exists(image_dir):
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        global image_refs  # Store image references to prevent garbage collection
        image_refs = []

        col, row = 0, 0
        for img_file in image_files:
            try:
                img = Image.open(img_file).copy()  # Prevent modification issues
                img.thumbnail((100, 100))
                tk_img = ImageTk.PhotoImage(img)

                lbl = tk.Label(gallery_scrollable_frame, image=tk_img, borderwidth=2, relief="groove")
                lbl.image = tk_img
                lbl.grid(row=row, column=col, padx=5, pady=5)

                image_refs.append(tk_img)  # Store references to prevent garbage collection

                col += 1
                if col >= 5:
                    col = 0
                    row += 1
            except Exception as e:
                log("Error loading image: " + str(e))


def upload_and_annotate():
    global current_image
    new_class = class_entry.get().strip()
    if not new_class:
        messagebox.showwarning("No Name", "Enter a product name.")
        return
    file_path = filedialog.askopenfilename(title="Select Image", 
                                           filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    current_image = Image.open(file_path)

    update_yaml_with_new_class(new_class)
    image_dir = TRAIN_IMAGE_DIR 
    os.makedirs(image_dir, exist_ok=True)
    if new_class not in captured_counts:
        captured_counts[new_class] = 0

    image_filename = f"{new_class}_{captured_counts[new_class]}.jpg"
    image_path = os.path.join(image_dir, image_filename)
    current_image.save(image_path)
    captured_counts[new_class] += 1

    log(f"Image saved: {image_path} ")
    annotate_image(image_path, new_class)
    update_gallery(new_class)


def update_model_config_from_yaml(model, data_yaml):
    """Update the model's configuration (`nc`) based on the data.yaml file."""
    # Load the dataset configuration (data.yaml)
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    # Get the number of classes (nc) from the data.yaml
    num_classes = data['nc']
    class_names = data['names']
    
    # Update the model's configuration to reflect the new number of classes
    model.model.yaml['nc'] = num_classes  # Update the number of classes in the model config
    model.model.yaml['names'] = class_names  # Update the class names in the model config

    print(f"Updated model config: {num_classes} classes")

def train_model():
    """Trains the YOLO model using dataset and updates it with the new class dynamically."""
    def training_task():
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Model Error", "Model file not found. Ensure 'latest.pt' exists.")
            progress_bar.stop()
            return
    
        try:
            # Load the pretrained model with existing weights (for old classes)
            model = YOLO(MODEL_PATH)

            # Dynamically update the model's config based on the data.yaml file
            update_model_config_from_yaml(model, DATA_YAML)

            # Train the model with the updated dataset (including new class)
            model.train(data=DATA_YAML, imgsz=640, epochs=2, batch=16, project="training_results", name="yolo_model", exist_ok=True)

            backup_model_path = "training_results/yolo_model/weights/previous_version.pt"
            if os.path.exists(MODEL_PATH):
                os.replace(MODEL_PATH, backup_model_path)
            
            # Save the best model after training
            trained_model_path = "training_results/yolo_model/weights/best.pt"
            if os.path.exists(trained_model_path):
                os.replace(trained_model_path, MODEL_PATH)
                log("Model updated to latest.pt")

            messagebox.showinfo("Training Complete", "Model training completed and updated successfully!")

        except Exception as e:
            messagebox.showerror("Training Error", str(e))
        
        progress_bar.stop()
    progress_bar.start()


def augment_images(class_label):
    """Apply augmentation to images and labels for a specific class and store them in AUGMENT_DIR."""
    os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
    os.makedirs(VAL_IMAGE_DIR, exist_ok=True)
    os.makedirs(VAL_LABEL_DIR, exist_ok=True)
    os.makedirs(TEST_IMAGE_DIR, exist_ok=True)
    os.makedirs(TEST_LABEL_DIR, exist_ok=True)

    # Load class id mapping from the dataset.yaml
    with open(DATA_YAML, 'r') as f:
        data = yaml.safe_load(f)
    class_id = data['names'].index(class_label)

    # # Define augmentation pipeline using Albumentations
    # augmentations = A.Compose([
    #     A.HorizontalFlip(p=0.5),
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.RandomCrop(width=450, height=450, p=0.5),
    #     A.Rotate(limit=30, p=0.5),
    #     A.RandomSizedCrop(min_max_height=(200, 400),  size=(300, 300), p=0.5),
    #     A.CLAHE(p=0.5)
    # ])

    def random_crop(image, bboxes):
        h, w = image.shape[:2]
        x = random.randint(0, w // 4)
        y = random.randint(0, h // 4)
        cropped = image[y:h - y, x:w - x]
        new_bboxes = []
        for bbox in bboxes:
            class_id, x_center, y_center, bw, bh = bbox
            x_center = (x_center * w - x) / (w - 2*x)
            y_center = (y_center * h - y) / (h - 2*y)
            if 0 <= x_center <= 1 and 0 <= y_center <= 1:
                new_bboxes.append([class_id, x_center, y_center, bw, bh])
        return cv2.resize(cropped, (w, h)), new_bboxes

    def random_brightness(image, bboxes):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value = random.randint(-50, 50)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)  # Fix to avoid overflow
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), bboxes

    def random_flip(image, bboxes):
        flip_code = random.choice([-1, 0, 1])
        flipped = cv2.flip(image, flip_code)
        new_bboxes = []
        for bbox in bboxes:
            class_id, x_center, y_center, bw, bh = bbox
            if flip_code == 1:  # Horizontal Flip
                x_center = 1 - x_center
            elif flip_code == 0:  # Vertical Flip
                y_center = 1 - y_center
            new_bboxes.append([class_id, x_center, y_center, bw, bh])
        return flipped, new_bboxes

    def random_rotate(image, bboxes):
        angle = random.randint(-30, 30)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h))

        new_bboxes = []
        for bbox in bboxes:
            class_id, x_center, y_center, bw, bh = bbox
            x = int(x_center * w)
            y = int(y_center * h)
            new_x, new_y = np.dot(rotation_matrix, np.array([x, y, 1]))
            new_bboxes.append([class_id, new_x / w, new_y / h, bw, bh])

        return rotated, new_bboxes

    def random_translation(image, bboxes):
        h, w = image.shape[:2]
        max_shift = min(h, w) // 10
        tx, ty = random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(image, translation_matrix, (w, h))

        new_bboxes = []
        for bbox in bboxes:
            class_id, x_center, y_center, bw, bh = bbox
            x_center = (x_center * w + tx) / w
            y_center = (y_center * h + ty) / h
            if 0 <= x_center <= 1 and 0 <= y_center <= 1:
                new_bboxes.append([class_id, x_center, y_center, bw, bh])
        return translated, new_bboxes
    

    for img_name in tqdm(os.listdir(TRAIN_IMAGE_DIR)):
        img_path = os.path.join(TRAIN_IMAGE_DIR, img_name)

        # Skip images that are not related to the target class
        if class_label not in img_name:
            continue  # Skip images that don't belong to the class

        # Read the image
        image = cv2.imread(img_path)
        if image is None:
            continue

        # Construct the label path from the train/labels directory
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(TRAIN_LABEL_DIR, label_name)
        
        # Check if the label file exists
        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            labels = [list(map(float, line.strip().split())) for line in f.readlines()]

        # Filter labels that correspond to the target class
        labels_for_class = [label for label in labels if label[0] == class_id]

        # If there are no labels for the class in this image, skip
        if not labels_for_class:
            continue
        
        for i in range(20):  # Generate 20 augmentations per image
            augmented = image.copy()

            # Apply transformations
            augmented, aug_labels = random_crop(augmented, labels_for_class)
            augmented, aug_labels = random_flip(augmented, aug_labels)
            augmented, aug_labels = random_rotate(augmented, aug_labels)
            augmented, aug_labels = random_translation(augmented, aug_labels)

            # Determine the split (train, val, test) for each augmented image
            split_choice = random.random()

            if split_choice < 0.7:
                dest_img_dir = TRAIN_IMAGE_DIR
                dest_label_dir = TRAIN_LABEL_DIR
            elif split_choice < 0.9:
                dest_img_dir = VAL_IMAGE_DIR
                dest_label_dir = VAL_LABEL_DIR
            else:
                dest_img_dir = TEST_IMAGE_DIR
                dest_label_dir = TEST_LABEL_DIR

            # Save the augmented image in the respective directory
            aug_img_name = f"{os.path.splitext(img_name)[0]}_augmented_{i}.jpg"
            aug_img_path = os.path.join(dest_img_dir, aug_img_name)
            cv2.imwrite(aug_img_path, augmented)

            # Save the updated annotation file for the augmented image
            aug_label_name = os.path.splitext(label_name)[0] + f"_augmented_{i}.txt"
            aug_label_path = os.path.join(dest_label_dir, aug_label_name)
            with open(aug_label_path, "w") as f:
                for label in aug_labels:
                    f.write(" ".join(map(str, label)) + "\n")

    log("Dataset successfully split into train, val, and test sets.")
    log(f"Augmentation completed for class: {class_label}")

def start_inference():
    try:
        log("Starting inference... Running object_tracking.py")
        subprocess.run(["python", "object_tracking.py"], check=True)
        log("Inference completed.")
    except subprocess.CalledProcessError as e:
        log(f"Error running object_tracking.py: {e}")

#live feed frame
Live_feed_frame = tk.Frame(main_frame, bd=1, relief=tk.SUNKEN, bg='#FBFCEA', highlightbackground="#eff299", highlightthickness=2)
Live_feed_frame.pack(padx=10, pady=5)
live_feed_label = tk.Label(Live_feed_frame, bg='#FBFCEA')
live_feed_label.pack()

# Controls Frame
control_frame = tk.Frame(main_frame, bg='#FBFCEA', highlightbackground="#eff299", bd=2)
control_frame.pack(pady=5)
tk.Label(control_frame, text="Enter product name:", bg='#FBFCEA').grid(row=0, column=0, padx=5)
class_entry = tk.Entry(control_frame, width=30)
class_entry.grid(row=0, column=1, padx=5)

btn_frame = tk.Frame(main_frame, bg='#FBFCEA',highlightbackground="#eff299", bd=2)
btn_frame.pack(pady=5, fill="x")

# Allow all columns to expand equally
for i in range(4):  
    btn_frame.grid_columnconfigure(i, weight=1)  # Equal expansion

# Buttons with 'sticky="ew"' for expansion
tk.Button(btn_frame, text="Start Live Feed", command=start_live, bg="green", fg="white").grid(row=0, column=0, padx=2, sticky="ew")
tk.Button(btn_frame, text="Stop Live Feed", command=stop_live, bg="red", fg="white").grid(row=0, column=1, padx=2, sticky="ew")
tk.Button(btn_frame, text="Capture", command=capture_and_annotate, bg="blue", fg="white").grid(row=0, column=2, padx=2, sticky="ew")
tk.Button(btn_frame, text="Upload", command=upload_and_annotate, bg="blue", fg="white").grid(row=0, column=3, padx=2, sticky="ew")
tk.Button(btn_frame, text="Train Model", command=lambda: threading.Thread(target=train_model, daemon=True).start(), bg="green", fg="white").grid(row=1, column=1, padx=10,pady=3,columnspan=2, sticky="ew")
tk.Button(btn_frame, text="Start Inference", command=lambda: threading.Thread(target=start_inference, daemon=True).start(), bg="blue", fg="white").grid(row=1, column=3, padx=10,pady=3, sticky="ew", columnspan=2)

progress_bar = ttk.Progressbar(main_frame, orient="horizontal", mode="indeterminate", length=300)
progress_bar.pack(pady=10)
# Gallery Frame for Saved Images
gallery_label = tk.Label(main_frame, text="Class Gallery:", font=("Arial", 12), bg='#FBFCEA')
gallery_label.pack(pady=(10,0))
gallery_frame = tk.Frame(main_frame, bd=2, relief=tk.SUNKEN, width=500, height=200)
gallery_frame.pack(padx=10, pady=5, fill="both", expand=True)
gallery_frame.pack_propagate(False)
gallery_canvas = tk.Canvas(gallery_frame)
gallery_scrollbar = tk.Scrollbar(gallery_frame, orient="vertical", command=gallery_canvas.yview)
gallery_scrollable_frame = tk.Frame(gallery_canvas)
gallery_scrollable_frame.bind("<Configure>", lambda e: gallery_canvas.configure(scrollregion=gallery_canvas.bbox("all")))
gallery_canvas.create_window((0,0), window=gallery_scrollable_frame, anchor="nw")
gallery_canvas.configure(yscrollcommand=gallery_scrollbar.set)
gallery_canvas.pack(side="left", fill="both", expand=True)
gallery_scrollbar.pack(side="right", fill="y")

T = tk.Text(main_frame, bg='#f5f5ce')
T.pack()

# Insert default message
message = f"Thank you for using Visual AI POS system!\nFor technical assistance contact info@visualbusineessretail.com\nFor sales department contact sales@visualbusineessretail.com"
T.insert(tk.END, message)
T.tag_configure("center", justify="center")
T.tag_add("center", "1.0", "end")


def on_closing():
    global live_feed_running, cap_live
    live_feed_running = False
    if cap_live is not None:
        cap_live.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
